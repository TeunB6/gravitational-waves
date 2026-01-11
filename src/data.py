import torch
from torch.utils.data import Dataset
import pycbc.catalog as catalog
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
from itertools import islice
import numpy as np
from tqdm import tqdm
from pathlib import Path

from typing import Generator, Tuple

MASS_DIST = lambda: np.random.uniform(10, 30) # solar masses
FULL_ANGLE_DIST = lambda: np.random.uniform(0, 2*np.pi) # radians
HALF_ANGLE_DIST = lambda: np.random.uniform(0, np.pi) # radians

TIME_SHIFT_DIST = lambda: np.random.uniform(-0.4, 0.4) # seconds

SAMPLING_FREQ = 2048  # Hz

SAVE_PATH_BASE = "data/gravitational_wave_dataset_{samples}_{duration}_{whitened}_{bandpass}.pt"
ROOT_DIR = Path(__file__).parent.parent

BUFFER = 4.0  # seconds of standard buffer time around signal


# Logging
import logging
logging.basicConfig(filename=ROOT_DIR / 'logs/data_loading.log', encoding='utf-8', filemode='w', format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GravWaveDataset(Dataset):
    def __init__(self, num_samples: int, sample_duration: float, overwrite: bool = False, whitened: bool = False, bandpass: Tuple[float, float] | None = (30.0, 256.0)):
        """
        Class to generate a dataset of noisy gravitational wave signals.

        Args:
            num_samples (int): Number of samples to generate in the dataset.
            sample_duration (float): Duration of each sample in seconds.
            overwrite (bool, optional): Whether to overwrite an existing dataset file. Defaults to False.
            whitened (bool, optional): Whether to whiten the signals. Defaults to False.
            bandpass (Tuple[float, float] | None, optional): Bandpass filter range. Defaults to (30.0, 512.0).
        """
        save_path = ROOT_DIR / SAVE_PATH_BASE.format(samples=num_samples, duration=sample_duration, whitened=whitened, bandpass=bandpass)
        
        # Configure class
        self.num_samples = num_samples
        self.sample_duration = sample_duration
        self.whitened = whitened
        self.bandpass = bandpass
        
        # Load if allowed
        if not overwrite and Path(save_path).exists():
            logger.info(f"Loading dataset from {save_path}")
            self._data = torch.load(save_path)
            self.num_samples = len(self._data)
            self.sample_duration = sample_duration
            return
        
        # Generate dataset
        logger.info(f"Generating dataset with {num_samples} samples of duration {sample_duration} seconds")
        data = list(self.signal_generator(num_samples, sample_duration))
        self._data = torch.Tensor(data)
        
        # Save dataset
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving dataset to {save_path}")
        torch.save(self._data, save_path)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: The sample at the specified index.
        """
        return self._data[idx]
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return self.num_samples
     
    def signal_generator(self, num_samples: int, sample_duration: float) -> Generator[Tuple[TimeSeries, TimeSeries], None, None]:
        """
        Generator that yields noisy gravitational wave signals.
        
        Args:
            num_samples (int): Number of samples to generate.
            sample_duration (float): Duration of each sample in seconds.
            
        Yields:
            Tuple[TimeSeries, TimeSeries]: A tuple containing the noisy signal and the pure signal
        """
        noise_gen = self._noise_generator(num_samples, sample_duration)
        signal_gen = self._wave_generator(num_samples)
        
        
        for (noise, time), signal in tqdm(zip(noise_gen, signal_gen), total=num_samples, desc="Generating dataset"):
            # Time shift signal randomly within the noise segment
            shift = TIME_SHIFT_DIST()
            logger.debug(f"Time shifting signal by {shift} seconds")
            signal.start_time += time + sample_duration / 2 + shift + BUFFER # center signal in noise segment with shift
            
            # Should be impossible due to choice of random bounds but log just in case
            if sample_duration + shift < 0:
                logger.warning(f"Signal start out of bounds after shift: start_time={signal.start_time}, shift={shift}")
            if signal.end_time - time > sample_duration:
                logger.warning(f"Signal end out of bounds after shift: end_time={signal.end_time}, shift={shift}")
                
            noisy = noise.inject(signal)
            print(noisy.start_time, noisy.end_time, signal.start_time, signal.end_time)
            # Preprocessing: need padding for whitening, so slice with buffer first
            if self.whitened:
                print(time - BUFFER > noisy.start_time)
                noisy_padded = noisy.time_slice(time - BUFFER / 2, time + sample_duration + BUFFER / 2)
                noisy = noisy_padded.whiten(BUFFER / 2, BUFFER / 2)
            
            # Now slice to exact duration and apply bandpass
            if self.bandpass is not None:
                noisy = noisy.time_slice(time - BUFFER, time + sample_duration + BUFFER)
                noisy = noisy.highpass_fir(self.bandpass[0], 512).lowpass_fir(self.bandpass[1], 512) if self.bandpass is not None else noisy
            
            noisy = noisy.time_slice(time, time + sample_duration)
            # Create clean signal of same length for consistency
            clean = TimeSeries(np.zeros_like(noisy), delta_t=1/SAMPLING_FREQ, epoch=noisy.start_time).inject(signal)
            
            yield noisy, clean
            
    
    def _noise_generator(self, num_samples: int, sample_duration: float) -> Generator[TimeSeries, None, None]:
        """
        Generator that yields noise segments from real detector data.

        Args:
            num_samples (int): Number of noise segments to generate.
            sample_duration (float): Duration of each noise segment in seconds.

        Yields:
            TimeSeries: A noise segment.
            float: Time to place the signal within the noise segment.
        """
        merger_names = [m for m in catalog.Catalog('gwtc-3').mergers]
        merger_idx = 0
        for _ in range(num_samples):
        
            def get_next_merger():
                nonlocal merger_idx
                # Filter out mergers without L1 data
                while True:
                    merger = catalog.Merger(merger_names[merger_idx], source="gwtc-3")
                    if not any(fdict['detector'] == 'L1' for fdict in merger.data['strain']):
                        merger_idx = (merger_idx + 1) % len(merger_names)
                        logger.info(f"Skipping merger {merger_names[merger_idx]} without L1 data")
                    else:
                        break
                merger_idx = (merger_idx + 1) % len(merger_names)
                logger.debug(f"Merger loaded: {merger_names[merger_idx]}")
                return merger
            
            merger = get_next_merger()
            strain = merger.strain('L1')
            strain = strain.resample(1/SAMPLING_FREQ)
            strain = strain.time_slice(strain.start_time, merger.time - 2*sample_duration)
            time = np.random.uniform(strain.start_time + BUFFER, strain.end_time - 2*sample_duration - 2 * BUFFER)
            yield strain, time
    
    def _wave_generator(self, num_samples: int) -> Generator[TimeSeries, None, None]:
        """
        Generator that yields gravitational wave signals. Based on SEOBNRv4 waveform model. 
        Using randomized parameters for masses, inclination, coalescence phase, and polarization.
        
        Args:
            num_samples (int): Number of waveforms to generate.
        Yields:
            TimeSeries: A gravitational wave signal.
        """
        det = Detector("L1")
        dec = 0.5333 # not sure what this should be
        ra = 2 # not sure what this should be
        
        def _calculate_wave():
            params = {"mass1" : MASS_DIST(),
                            "mass2" : MASS_DIST(),
                            "inclination" : HALF_ANGLE_DIST(),
                            "coa_phase" : FULL_ANGLE_DIST(),
                            "delta_t" : 1.0/SAMPLING_FREQ
            }                    
            hp, hc = get_td_waveform(approximant="SEOBNRv4",
                            f_lower=40,
                            **params
                            )
            logger.debug(f"Generated waveform with params: {params}")
            signal = det.project_wave(hp, hc, ra, dec, polarization=FULL_ANGLE_DIST())
            return signal
        
        for _ in range(num_samples):   
            yield _calculate_wave()
        