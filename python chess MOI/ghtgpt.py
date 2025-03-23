import chess
import random

class ChessAI:
    def __init__(self, depth):
        self.depth = depth

    def get_best_move(self, board):
        _, best_move = self.minimax(board, self.depth, True, float("-inf"), float("inf"))
        return best_move

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board), None

        legal_moves = list(board.legal_moves)
        if maximizing_player:
            max_eval = float("-inf")
            best_move = None
            for move in legal_moves:
                if not self.is_own_piece_captured(board, move):
                    board.push(move)
                    eval, _ = self.minimax(board, depth - 1, False, alpha, beta)
                    board.pop()
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval, best_move
        else:
            min_eval = float("inf")
            best_move = None
            for move in legal_moves:
                if not self.is_own_piece_captured(board, move):
                    board.push(move)
                    eval, _ = self.minimax(board, depth - 1, True, alpha, beta)
                    board.pop()
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval, best_move

    def evaluate(self, board):
        evaluation = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = self.get_piece_value(piece)
            if piece.color == chess.WHITE:
                evaluation += value
            else:
                evaluation -= value
        return evaluation

    def get_piece_value(self, piece):
        if piece.piece_type == chess.PAWN:
            return 1
        elif piece.piece_type == chess.KNIGHT:
            return 3
        elif piece.piece_type == chess.BISHOP:
            return 3
        elif piece.piece_type == chess.ROOK:
            return 5
        elif piece.piece_type == chess.QUEEN:
            return 9
        elif piece.piece_type == chess.KING:
            return 1000
        else:
            return 0

    def is_own_piece_captured(self, board, move):
        if board.piece_at(move.to_square) is not None:
            return board.piece_at(move.to_square).color == board.turn
        return False

def main():
    board = chess.Board()
    ai = ChessAI(depth=3)  # Depth of search set to 3 for this example

    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            move = input("Your move: ")
            try:
                board.push_san(move)
            except ValueError:
                print("Invalid move. Please try again.")
        else:
            best_move = ai.get_best_move(board)
            board.push(best_move)
            print("AI plays:", best_move)

    print("Game over")
    print("Result:", board.result())

if __name__ == "__main__":
    main()

