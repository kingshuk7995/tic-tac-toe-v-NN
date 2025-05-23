#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <random>
#include <tuple>
#include <sstream>
using namespace std;

enum Cell { UNUSED = 0, X = 1, O = 2 };

struct Grid {
    int board[3][3];

    Grid() {
        for (auto &row : board)
            for (int &cell : row)
                cell = UNUSED;
    }

    vector<pair<int, int>> available_moves() const {
        vector<pair<int, int>> moves;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (board[i][j] == UNUSED)
                    moves.emplace_back(i, j);
        return moves;
    }

    bool is_terminal(int &winner) const {
        for (int i = 0; i < 3; ++i) {
            if (board[i][0] != UNUSED && board[i][0] == board[i][1] && board[i][1] == board[i][2]) {
                winner = board[i][0]; return true;
            }
            if (board[0][i] != UNUSED && board[0][i] == board[1][i] && board[1][i] == board[2][i]) {
                winner = board[0][i]; return true;
            }
        }
        if (board[0][0] != UNUSED && board[0][0] == board[1][1] && board[1][1] == board[2][2]) {
            winner = board[0][0]; return true;
        }
        if (board[0][2] != UNUSED && board[0][2] == board[1][1] && board[1][1] == board[2][0]) {
            winner = board[0][2]; return true;
        }
        for (auto &row : board)
            for (int cell : row)
                if (cell == UNUSED)
                    return false;

        winner = 0;
        return true;
    }

    int minimax(bool is_maximizing) {
        int winner;
        if (is_terminal(winner)) {
            if (winner == X) return 1;
            if (winner == O) return -1;
            return 0;
        }

        int best = is_maximizing ? -999 : 999;
        for (auto [x, y] : available_moves()) {
            board[x][y] = is_maximizing ? X : O;
            int score = minimax(!is_maximizing);
            board[x][y] = UNUSED;
            best = is_maximizing ? max(best, score) : min(best, score);
        }
        return best;
    }

    pair<int, int> best_move() {
        int best_score = -999;
        pair<int, int> move = {-1, -1};
        for (auto [x, y] : available_moves()) {
            board[x][y] = X;
            int score = minimax(false);
            board[x][y] = UNUSED;
            if (score > best_score) {
                best_score = score;
                move = {x, y};
            }
        }
        return move;
    }

    vector<float> flatten() const {
        vector<float> flat;
        for (auto &row : board)
            for (int cell : row) {
                flat.push_back(cell == UNUSED ? 1.0f : 0.0f);
                flat.push_back(cell == X ? 1.0f : 0.0f);
                flat.push_back(cell == O ? 1.0f : 0.0f);
            }
        return flat;
    }
};

mutex mtx;

void generate_games(int games, int thread_id, ofstream &out) {
    random_device rd;
    mt19937 gen(rd());
    ostringstream buffer;

    for (int g = 0; g < games; ++g) {
        Grid grid;
        while (true) {
            auto move = grid.best_move();
            if (move.first == -1) break;

            auto flat = grid.flatten();
            for (float f : flat)
                buffer << f << ",";
            int idx = move.first * 3 + move.second;
            buffer << idx << "\n";

            grid.board[move.first][move.second] = X;
            int winner;
            if (grid.is_terminal(winner)) break;

            auto moves = grid.available_moves();
            if (moves.empty()) break;
            auto random_move = moves[gen() % moves.size()];
            grid.board[random_move.first][random_move.second] = O;
            if (grid.is_terminal(winner)) break;
        }
    }

    lock_guard<mutex> lock(mtx);
    out << buffer.str();
}

int main() {
    const int total_games = 10000;
    const int num_threads = thread::hardware_concurrency();
    const int games_per_thread = total_games / num_threads;

    ofstream out("tictactoe_data.csv");
    vector<thread> threads;

    for (int i = 0; i < num_threads; ++i)
        threads.emplace_back(generate_games, games_per_thread, i, ref(out));

    for (auto &t : threads)
        t.join();

    cout << "Multithreaded data generation complete.\n";
    return 0;
}
