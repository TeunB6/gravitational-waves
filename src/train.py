from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from pathlib import Path
import logging

UPDATE_FREQ = 5  # Frequency of progress bar updates

ROOT_DIR = Path(__file__).parent.parent

# Logging setup
(ROOT_DIR  / "logs").mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    (ROOT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(
        ROOT_DIR / "logs/train.log",
        encoding="utf-8",
        mode="w"
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion=nn.MSELoss(),
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def validate(self, dataloader: DataLoader, criterion=nn.MSELoss()) -> float:
        """
        Validate the model.

        Args:
            dataloader (DataLoader): DataLoader for validation data.
            criterion: Loss function to use for validation.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        return val_loss / len(dataloader)

    def train(self, epochs: int = 5) -> dict:
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int, optional): Amount of epochs to train the model. Defaults to 5.

        Returns:
            dict: Training history containing training and validation losses.
        """
        tqdm_iter = tqdm(range(epochs), desc="Training Epochs", unit="epoch")

        history = {"train_loss": [], "val_loss": []}

        for epoch in tqdm_iter:
            self.train_epoch()

            if epoch % UPDATE_FREQ == 0 or epoch == epochs - 1:
                val_loss = self.validate(self.val_loader, self.criterion)
                training_loss = self.validate(self.train_loader, self.criterion)
                logging.info(
                    f"Epoch {epoch+1}/{epochs}, Training Loss: {training_loss:.6f}, Validation Loss: {val_loss:.6f}"
                )

                history["train_loss"].append(training_loss)
                history["val_loss"].append(val_loss)

                tqdm_iter.set_postfix(
                    {"Training Loss": training_loss, "Validation Loss": val_loss}
                )

        return history
