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
        
        # piece values for evaluating captures
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }

    def get_move(self, fen: str) -> Optional[str]:
        # read current chessboard state
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)   
        # if no legal steps left, return None
        if not legal_moves:
            return None
            
        # Filter 1: win immediately if possible
        for move in legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move.uci()
                
        # Filter 2: avoid moving into attacked squares, avoid BAD trades, and avoid STALEMATE (Teacher's Hint)
        opponent_color = not board.turn
        safe_moves = []
        for move in legal_moves:
            board.push(move)
            is_stalemate = board.is_stalemate()
            board.pop()
            
            # Avoid Stalemate
            if is_stalemate:
                continue 

            is_attacked = board.is_attacked_by(opponent_color, move.to_square)
            
            if not is_attacked:
                safe_moves.append(move)
            elif board.is_capture(move):
                # Avoid Bad Trades
                attacker_piece = board.piece_at(move.from_square)
                victim_piece = board.piece_at(move.to_square)
                attacker_val = self.piece_values.get(attacker_piece.piece_type, 0) if attacker_piece else 0
                if board.is_en_passant(move):
                    victim_val = 1
                else:
                    victim_val = self.piece_values.get(victim_piece.piece_type, 0) if victim_piece else 0
                # only when pieces eaten are greater than or equal to our pieces can it be considered a safe move
                if victim_val >= attacker_val:
                    safe_moves.append(move)              
        
        candidate_moves = safe_moves if len(safe_moves) > 0 else legal_moves

        # Filter 3: if we can safely eat a high-value piece, do it
        best_capture_val = 0
        best_capture_move = None
        for move in candidate_moves:
            if board.is_capture(move):
                if board.is_en_passant(move):
                    val = 1
                else:
                    captured_piece = board.piece_at(move.to_square)
                    val = self.piece_values.get(captured_piece.piece_type, 0) if captured_piece else 0
                
                if val > best_capture_val:
                    best_capture_val = val
                    best_capture_move = move
                    
        if best_capture_move and best_capture_val > 0:
            return best_capture_move.uci()

        # wrap the LLM prediction in try-except to prevent runtime crashes
        try:
            best_move = None
            best_score = -float('inf')
            
            # grandmaster prompt to guide the LLM
            prompt = (
                "You are a Grandmaster chess engine. "
                f"Analyze this board position in FEN: {fen}\n"
                "What is the single best and winning UCI move? "
                "The best move is: "
            ) 
            
            # let LLM rate only the filtered candidate moves
            with torch.no_grad():
                for move in candidate_moves:
                    move_uci = move.uci()
                    text = f"{prompt}{move_uci}"
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    
                    # calculate cross-entropy loss (lower loss = higher probability)
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    score = -outputs.loss.item() 
                    
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
