import torch
import chess
import random  # added for emergency fallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name: str = "Laurencia"):
        super().__init__(name)
        # check GPU or Cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        # use model to test
        self.model_id = "Qwen/Qwen2.5-0.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        # read current chessboard state
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves) 
        # if already been killed and have no legal steps left, return None
        if not legal_moves:
            return None

        # Filter 1: win immediately if possible
        for move in legal_moves:
            board.push(move)
            is_mate = board.is_checkmate)aa
            board.pop()
            if is_mate:
                return move.uci()
                
        # Filter 2: avoid moving into squares attacked by the opponent
        opponent_color = not board.turn
        safe_moves = []
        for move in legal_moves:
            # a move is safe if it is a capture itself, or if the destination square is not attacked
            if board.is_capture(move) or not board.is_attacked_by(opponent_color, move.to_square):
                safe_moves.append(move)             
        # if danger filter removes all moves, fall back to all legal moves
        candidate_moves = safe_moves if len(safe_moves) > 0 else legal_moves

        # wrap the entire prediction process in try-except to prevent runtime crashes
        try:
            best_move = None
            best_score = -float('inf')
            # tell model what the current chessboard looks like
            prompt = (
                "You are a Grandmaster chess engine. "
                f"Analyze this board position in FEN: {fen}\n"
                "What is the single best and winning UCI move? "
                "The best move is: "
            ) 
            # let model rate only the safe candidate moves and select
            with torch.no_grad():
                for move in candidate_moves: # only evaluate filtered candidate_moves now
                    move_uci = move.uci()
                    text = f"{prompt}{move_uci}"
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    score = -outputs.loss.item() # the smaller the loss, the better
                    
                    if score > best_score:
                        best_score = score
                        best_move = move_uci
            # double check if a valid move is found
            if best_move is not None:
                return best_move
            else:
                return random.choice(candidate_moves).uci()
                
        except Exception as e:
            # bulletproof fallback: if anything goes wrong, play a random safe move
            return random.choice(candidate_moves).uci()
