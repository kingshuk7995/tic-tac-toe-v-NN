import torch
import torch.nn as nn
import numpy as np
from typing import Optional


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Game:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.board = [[0 for _ in range(3)] for _ in range(3)]  # 0: unused, 1: X (AI), 2: O (player)
        self.model = model
        self.device = device

    def encode(self) -> np.ndarray:
        encoded = []
        for row in self.board:
            for cell in row:
                encoded.append(1.0 if cell == 0 else 0.0)  # unused
                encoded.append(1.0 if cell == 1 else 0.0)  # X
                encoded.append(1.0 if cell == 2 else 0.0)  # O
        return np.array(encoded, dtype=np.float32)

    def ai_move(self) -> None:
        x = torch.tensor(self.encode().reshape(1, 27)).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze()
        for idx in probs.argsort(descending=True):
            r, c = divmod(idx.item(), 3)
            if self.board[r][c] == 0:
                self.board[r][c] = 1
                print(f"AI plays at: ({r}, {c})")
                return

    def user_move(self, r: int, c: int) -> bool:
        if 0 <= r < 3 and 0 <= c < 3 and self.board[r][c] == 0:
            self.board[r][c] = 2
            return True
        return False

    def display(self) -> None:
        symbols = {0: ".", 1: "X", 2: "O"}
        print("\nBoard:")
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print()

    def check_winner(self) -> Optional[int]:
        for i in range(3):
            if self.board[i][0] != 0 and self.board[i][0] == self.board[i][1] == self.board[i][2]:
                return self.board[i][0]
            if self.board[0][i] != 0 and self.board[0][i] == self.board[1][i] == self.board[2][i]:
                return self.board[0][i]
        if self.board[0][0] != 0 and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]
        if self.board[0][2] != 0 and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]
        return None

    def is_draw(self) -> bool:
        return all(cell != 0 for row in self.board for cell in row)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TicTacToeNN().to(device)
    model.load_state_dict(torch.load("tictactoe_model.pt", map_location=device))
    model.eval()

    game = Game(model, device)
    game.display()

    while True:
        while True:
            try:
                r, c = map(int, input("Your move (row col): ").split())
                if game.user_move(r, c):
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Enter row and col as two integers (0-2).")

        game.display()

        winner = game.check_winner()
        if winner == 2:
            print("You win!")
            break
        if game.is_draw():
            print("It's a draw!")
            break

        game.ai_move()
        game.display()

        winner = game.check_winner()
        if winner == 1:
            print("AI wins!")
            break
        if game.is_draw():
            print("It's a draw!")
            break


if __name__ == "__main__":
    main()
