from torch import Tensor
from torch.utils.data import Dataset
from pycbc.noise import noise_from_psd
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import numpy as np

MASS_DIST = lambda: np.random.uniform(10, 30)
FULL_ANGLE_DIST = lambda: np.random.uniform(0, 2*np.pi)
HALF_ANGLE_DIST = lambda: np.random.uniform(0, np.pi)

TIME_SHIFT_DIST = lambda: np.random.uniform(-0.4, 0.4) # seconds

SAMPLING_FREQ = 4096

class GravWaveDataset(Dataset):
     
    def _generate_L2_noise(num_samples: int, sample_duration: float):
        """For now with a bunch of hardcoded parameters, will change when things are better understood"""
                
        flow = 30.0
        delta_f = 1.0 / 16
        flen = int(2048 / delta_f) + 1
        psd = aLIGOZeroDetHighPower(flen, delta_f, flow)
        
        delta_t = 1.0 / SAMPLING_FREQ
        tsamples = int(sample_duration * num_samples / delta_t)
        ts = noise_from_psd(tsamples, delta_t, psd) # one long string of noise
        
        ts =  Tensor(ts).reshape(num_samples, sample_duration * SAMPLING_FREQ)
        return ts
    
    def _generate_BBH_signals(num_samples: int, sample_duration: float):
        det = Detector("L1")
        dec = 0.533
        ra = 0 # not sure what this should be
        def _generate_signal():       
            hp, hc = get_td_waveform(approximant="SEOBNRv4",
                            mass1=MASS_DIST(),
                            mass2=MASS_DIST(),
                            inclination=HALF_ANGLE_DIST(),
                            coa_phase=FULL_ANGLE_DIST(),
                            delta_t=1.0/SAMPLING_FREQ,
                            f_lower=40
                            )
            
            signal = det.project_wave(hp, hc, ra, dec, polarization=FULL_ANGLE_DIST())
            return Tensor(signal)
            
        return Tensor([_generate_signal() for _ in range(num_samples)])
        