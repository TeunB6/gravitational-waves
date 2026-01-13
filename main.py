from src.data import GravWaveDataset
from src.models import AutoEncoder
import matplotlib.pyplot as plt
from time import sleep

dataset = GravWaveDataset(num_samples=1000, sample_duration=1.0, overwrite=True, whitened=False, bandpass=None)
print(f"Dataset size: {len(dataset)} samples")  # Should print 1000 samples
print(f"Shape of first noisy signal: {dataset[0][0].shape}")  # Shape of the first noisy signal

for idx in range(100):
    plt.plot(dataset[idx][0].numpy(), label='Noisy Signal')
    plt.plot(dataset[idx][1].numpy(), label='Clean Signal')
    plt.legend()
    plt.show()
    sleep(0.2)
    plt.close('all')