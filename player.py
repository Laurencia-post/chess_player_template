import torch
import chess
from transformers import AutoModelForCausalLM, AutoTokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name: str = "Laurencia"):
        super().__init__(name)
       # check GPU or Cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # use gpt2 to test
        self.model_id = "gpt2" 
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        # read current chessboard state
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        # if already been killed and have no legal steps left, return None
        if not legal_moves:
            return None
        best_move = None
        best_score = -float('inf')
        
        # tell model what the current chessboard looks like
        prompt = f"Chess board position in FEN: {fen}\nThe best next move is "
        # let gpt2 rate each legal move and select
        with torch.no_grad():
            for move in legal_moves:
                move_uci = move.uci()
                text = f"{prompt}{move_uci}"
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                score = -outputs.loss.item() # the smaller the loss, the better; adding negative sign means the higher the score, the better
                if score > best_score:
                    best_score = score
                    best_move = move_uci
                    
        return best_move
