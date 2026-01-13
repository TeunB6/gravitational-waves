from src.data import GravWaveDataset
from src.models import AutoEncoder
import matplotlib.pyplot as plt
from time import sleep
from torch.utils.data import DataLoader, random_split
from src.train import Trainer
from torch.optim import Adam
from torch import save
from pathlib import Path


dataset = GravWaveDataset(
    num_samples=1000, sample_duration=1.0, overwrite=False, whitened=True
)
print(f"Dataset size: {len(dataset)} samples")  # Should print 1000 samples
print(
    f"Shape of first noisy signal: {dataset[0][0].shape}"
)  # Shape of the first noisy signal

for idx in range(100):
    plt.plot(dataset[idx][0].numpy(), label="Noisy Signal")
    plt.plot(dataset[idx][1].numpy(), label="Clean Signal")
    plt.legend()
    plt.show()
    sleep(0.2)
    plt.close("all")

model = AutoEncoder()
train_data, val_data = random_split(dataset, [800, 200])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=Adam(model.parameters(), lr=0.001),
)

trainer.train(epochs=20)

Path("models").mkdir(parents=True, exist_ok=True)
save(model.state_dict(), "models/autoencoder_model.pth")