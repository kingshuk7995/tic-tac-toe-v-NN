from typing import Tuple
import torch
from torch import Tensor, nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

class TicTacToeNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(27, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def load_data(path: str, device: torch.device) -> Tuple[Tensor, Tensor]:
    df = pd.read_csv(path, header=None)
    x_data: np.ndarray = df.iloc[:, :-1].values.astype(np.float32)
    y_data: np.ndarray = df.iloc[:, -1].values.astype(np.int64)

    x_tensor: Tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_tensor: Tensor = torch.tensor(y_data, dtype=torch.long).to(device)
    return x_tensor, y_tensor


def train_model(model: nn.Module, data: Tuple[Tensor, Tensor], device: torch.device) -> None:
    epochs = 30
    x, y = data
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss: float = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred: Tensor = model(xb)
            loss: Tensor = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "tictactoe_model.pt")
    print("Model saved to tictactoe_model.pt")


def main() -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model: TicTacToeNN = TicTacToeNN().to(device)
    data: Tuple[Tensor, Tensor] = load_data("tictactoe_data.csv", device)
    train_model(model, data, device)

if __name__ == "__main__":
    main()
