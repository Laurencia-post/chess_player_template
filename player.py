import time
import torch
import chess
import random  
from transformers import AutoModelForCausalLM, AutoTokenizer
from chess_tournament import Player
from typing import Optional

class TransformerPlayer(Player):
    def __init__(self, name: str = "Laurencia_Apex"):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  
        self.model_id = "Qwen/Qwen2.5-0.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval() 
        self.piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
        }
        self.max_think_time = 7.5  

    def get_move(self, fen: str) -> Optional[str]:
        start_time = time.time()
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)   
        except ValueError:
            return None   
        if not legal_moves:
            return None
            
        try:
            # Filter 1: win immediately if possible
            for move in legal_moves:
                board.push(move)
                is_mate = board.is_checkmate()
                board.pop()
                if is_mate:
                    return move.uci()
                    
            our_color = board.turn
            opponent_color = not board.turn
            safe_moves = []
            
            for move in legal_moves:
                board.push(move)
                is_stalemate = board.is_stalemate()
                is_draw = board.can_claim_draw()
                opp_can_mate = False
                for opp_move in board.legal_moves:
                    board.push(opp_move)
                    if board.is_checkmate():
                        opp_can_mate = True
                        board.pop()
                        break
                    board.pop()
                board.pop()
                
                if is_stalemate or opp_can_mate or is_draw:
                    continue           
                is_attacked = board.is_attacked_by(opponent_color, move.to_square)
                if not is_attacked:
                    safe_moves.append(move)
                elif board.is_capture(move):
                    attacker_piece = board.piece_at(move.from_square)
                    victim_piece = board.piece_at(move.to_square)
                    attacker_val = self.piece_values.get(attacker_piece.piece_type, 0) if attacker_piece else 0
                    vic_val = 1 if board.is_en_passant(move) else (self.piece_values.get(victim_piece.piece_type, 0) if victim_piece else 0)    
                    if vic_val >= attacker_val:
                        safe_moves.append(move)
                else:
                    is_defended = board.is_attacked_by(our_color, move.to_square)
                    if is_defended:
                        piece = board.piece_at(move.from_square)
                        piece_val = self.piece_values.get(piece.piece_type, 0) if piece else 0
                        if piece_val <= 3:
                            safe_moves.append(move)
            candidate_moves = safe_moves if len(safe_moves) > 0 else legal_moves 
            if len(candidate_moves) == 1:
                return candidate_moves[0].uci()  
            for move in candidate_moves:
                if move.promotion == chess.QUEEN:
                    return move.uci()

            best_capture_val = 0
            best_capture_move = None
            for move in candidate_moves:
                if board.is_capture(move):
                    attacker_piece = board.piece_at(move.from_square)
                    victim_piece = board.piece_at(move.to_square)
                    att_val = self.piece_values.get(attacker_piece.piece_type, 0) if attacker_piece else 0
                    vic_val = 1 if board.is_en_passant(move) else (self.piece_values.get(victim_piece.piece_type, 0) if victim_piece else 0)
                    is_attacked = board.is_attacked_by(opponent_color, move.to_square)
                    is_profitable = (not is_attacked) or (vic_val > att_val) 
                    if is_profitable and vic_val > best_capture_val:
                        best_capture_val = vic_val
                        best_capture_move = move            
            if best_capture_move and best_capture_val > 0:
                return best_capture_move.uci()

            # Heuristic Move Ordering
            def move_heuristic(m):
                score = 0
                if board.gives_check(m): score += 50
                if board.is_castling(m): score += 40
                if m.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}: score += 10
                return score
            # even if 7.5-second circuit breaker occurs, the best move has already been overrated
            candidate_moves.sort(key=move_heuristic, reverse=True)
            # Tensor-Level Concatenation
            best_move_uci = None
            best_score = -float('inf')           
            turn_str = "White" if our_color == chess.WHITE else "Black"
            candidate_san_list = [board.san(m) for m in candidate_moves]
            candidate_str = ", ".join(candidate_san_list)
            
            prompt = (
                "You are a Grandmaster chess engine. "
                f"It is {turn_str}'s turn to move. "
                f"Analyze this board position in FEN: {fen}\n"
                f"The safe candidate moves are: {candidate_str}\n"
                "What is the single best and winning move in Standard Algebraic Notation (SAN)?\n"
                "The best move is: "
            )  
            
            # Independently code Prompt
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            prompt_len = prompt_ids.size(1)   
            with torch.no_grad():
                for i, move in enumerate(candidate_moves):
                    if time.time() - start_time > self.max_think_time:
                        break        
                    move_san = candidate_san_list[i]  
                    # Independently code Move (add_special_tokens=False)
                    move_ids = self.tokenizer.encode(move_san, add_special_tokens=False, return_tensors="pt").to(self.device) 
                    # Force stitching in Tensor dimension to prevent incorrect merging of tokens
                    input_ids = torch.cat([prompt_ids, move_ids], dim=1)  
                    labels = input_ids.clone()
                    labels[0, :prompt_len] = -100 
                    outputs = self.model(input_ids=input_ids, labels=labels)  
                    move_len = move_ids.size(1)
                    if move_len > 0:
                        score = -(outputs.loss.item() * move_len)
                        
                        if board.is_castling(move): score += 2.0
                        elif board.gives_check(move): score += 1.0
                        elif move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}: score += 0.5 
                        if score > best_score:
                            best_score = score
                            best_move_uci = move.uci()        
                        
            if best_move_uci is not None:
                return best_move_uci
            else:
                return random.choice(candidate_moves).uci()        
        except Exception as e:
            return random.choice(legal_moves).uci()
