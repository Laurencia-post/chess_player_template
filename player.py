import time
import torch
import chess
import random  
from transformers import AutoModelForCausalLM, AutoTokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name: str = "Laurencia_GodMode"):
        super().__init__(name)
        # check GPU or CPU
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
        # set an upper limit on thinking time
        self.max_think_time = 7.5  

    def get_move(self, fen: str) -> Optional[str]:
        start_time = time.time()
        
        # First layer of defense, prevent receiving damaged FEN strings from causing a crash
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)   
        except ValueError:
            return None
            
        if not legal_moves:
            return None
            
        # wrap the core logic in try-except to prevent runtime crashes
        try:
            # Filter 1: win immediately if possible
            for move in legal_moves:
                board.push(move)
                is_mate = board.is_checkmate()
                board.pop()
                if is_mate:
                    return move.uci()
                    
            # Filter 2: avoid attacked squares, BAD trades, STALEMATE, and OPPONENT MATE
            opponent_color = not board.turn
            safe_moves = []
            for move in legal_moves:
                board.push(move)
                is_stalemate = board.is_stalemate()
                
                # Opponent Mate Sniffer
                # If I make this move, can opponent checkmate me immediately on the next turn?
                opp_can_mate = any(board.is_checkmate() for opp_move in board.legal_moves)
                
                board.pop()
                
                # avoid Stalemate and avoid giving the opponent a forced mate
                if is_stalemate or opp_can_mate:
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

            # Pawn Promotion
            for move in candidate_moves:
                if move.promotion == chess.QUEEN:
                    return move.uci()

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

            # LLM prediction
            best_move = None
            best_score = -float('inf')        
            prompt = (
                "You are a Grandmaster chess engine. "
                f"Analyze this board position in FEN: {fen}\n"
                "What is the single best and winning UCI move? "
                "The best move is: "
            )  
            with torch.no_grad():
                for move in candidate_moves:
                    # if thinking time is almost up, force an interruption
                    if time.time() - start_time > self.max_think_time:
                        break
                        
                    move_uci = move.uci()
                    text = f"{prompt}{move_uci}"
                    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    # fix length bias of large models
                    seq_len = inputs["input_ids"].size(1)
                    score = -(outputs.loss.item() * seq_len)
                    
                    if score > best_score:
                        best_score = score
                        best_move = move_uci
                        
            if best_move is not None:
                return best_move
            else:
                return random.choice(candidate_moves).uci()
                
        except Exception as e:
            # Second layer of defense
            return random.choice(legal_moves).uci()
                
        except Exception as e:
            # Second layer of defense: if it is a tactical calculation or model reports an error, randomly take legal steps
            return random.choice(legal_moves).uci()
