import pandas as pd
import numpy as np
import ast
import chess
import chess.pgn
import io
from tqdm import tqdm

# ── CONSTANTS ───────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0
}

SHARP_ECO_PREFIXES = [
    "B2", "B3", "B4", "B7", "B8", "B9",  # Sicilian lines
    "E6", "E7", "E8", "E9",               # King's Indian
    "C1", "C2", "C3", "C4",               # Open games
]

# ── TIMING FEATURES ─────────────────────────────────────────────────
def get_player_side(row: pd.Series, username: str) -> str:
    if str(row["white_user"]).lower() == username.lower():
        return "white"
    return "black"

def did_player_win(row: pd.Series, side: str) -> int:
    if row["winner"] == side:
        return 1
    elif pd.isna(row["winner"]):
        return 0
    return -1

def compute_panic_score(clocks: list, player_side: str) -> float:
    """
    late game time per move / early game time per move.
    Higher = more time pressure late in game.
    """
    if not clocks or len(clocks) < 6:
        return np.nan
    
    if player_side == "white":
        player_clocks = clocks[0::2]
    else:
        player_clocks = clocks[1::2]
    
    if len(player_clocks) < 6:
        return np.nan
    
    time_per_move = [
        max(player_clocks[i] - player_clocks[i+1], 0)
        for i in range(len(player_clocks) - 1)
    ]
    
    n = len(time_per_move)
    cutoff = max(1, int(n * 0.3))
    early_avg = np.mean(time_per_move[:cutoff])
    late_avg  = np.mean(time_per_move[n - cutoff:])
    
    if early_avg == 0:
        return np.nan
    return late_avg / early_avg

def compute_time_per_move(clocks: list, player_side: str) -> float:
    """Average seconds spent per move across the whole game."""
    if not clocks or len(clocks) < 2:
        return np.nan
    if player_side == "white":
        player_clocks = clocks[0::2]
    else:
        player_clocks = clocks[1::2]
    if len(player_clocks) < 2:
        return np.nan
    time_per_move = [
        max(player_clocks[i] - player_clocks[i+1], 0)
        for i in range(len(player_clocks) - 1)
    ]
    return np.mean(time_per_move)

# ── MOVE-LEVEL FEATURES ─────────────────────────────────────────────
def parse_move_features(moves_str: str, player_side: str) -> dict:
    """
    Replay the game move by move using python-chess.
    Extract capture rate, sacrifice rate, check rate,
    early aggression, castling timing, piece activity.
    """
    empty = {
        "capture_rate":       np.nan,
        "sacrifice_rate":     np.nan,
        "check_rate":         np.nan,
        "early_aggression":   np.nan,
        "castle_move":        np.nan,
        "piece_activity":     np.nan,
        "avg_material_diff":  np.nan,
    }
    
    if not moves_str or pd.isna(moves_str):
        return empty
    
    try:
        # Build a PGN string python-chess can parse
        pgn_string = f"[Event \"?\"]\n\n{moves_str}"
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        
        if game is None:
            return empty
        
        board = game.board()
        player_color = chess.WHITE if player_side == "white" else chess.BLACK
        
        # Tracking variables
        total_player_moves  = 0
        captures            = 0
        sacrifices          = 0
        checks_given        = 0
        early_aggression    = 0   # captures + checks in first 10 player moves
        castle_move_num     = np.nan
        piece_moves_early   = 0   # non-pawn moves in first 10 player moves
        material_diffs      = []
        move_number         = 0

        for move in game.mainline_moves():
            current_color = board.turn
            
            if current_color == player_color:
                total_player_moves += 1
                move_number += 1
                is_early = move_number <= 10

                # ── Capture detection ──
                if board.is_capture(move):
                    captures += 1
                    if is_early:
                        early_aggression += 1
                    
                    # ── Sacrifice detection ──
                    # Get value of piece being moved
                    moving_piece = board.piece_at(move.from_square)
                    captured_piece = board.piece_at(move.to_square)
                    
                    if moving_piece and captured_piece:
                        attacker_val = PIECE_VALUES.get(moving_piece.piece_type, 0)
                        defender_val = PIECE_VALUES.get(captured_piece.piece_type, 0)
                        
                        # Sacrifice = giving up more valuable piece
                        if attacker_val > defender_val:
                            sacrifices += 1

                # ── Castling detection ──
                if board.is_castling(move) and np.isnan(castle_move_num):
                    castle_move_num = move_number

                # ── Piece activity (development speed) ──
                if is_early:
                    moving_piece = board.piece_at(move.from_square)
                    if moving_piece and moving_piece.piece_type != chess.PAWN:
                        piece_moves_early += 1

                # Push move to update board
                board.push(move)

                # ── Check detection (after pushing) ──
                if board.is_check():
                    checks_given += 1
                    if is_early:
                        early_aggression += 1

                # ── Material difference ──
                material_diff = compute_material_balance(board, player_color)
                material_diffs.append(material_diff)

            else:
                board.push(move)

        if total_player_moves == 0:
            return empty

        return {
            "capture_rate":      captures   / total_player_moves,
            "sacrifice_rate":    sacrifices / total_player_moves,
            "check_rate":        checks_given / total_player_moves,
            "early_aggression":  early_aggression,
            "castle_move":       castle_move_num,
            "piece_activity":    piece_moves_early,
            "avg_material_diff": np.mean(material_diffs) if material_diffs else 0,
        }

    except Exception as e:
        return empty


def compute_material_balance(board: chess.Board, player_color: int) -> int:
    """
    Count total material on board for player minus opponent.
    Positive = player is ahead in material.
    """
    player_material   = 0
    opponent_material = 0
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            value = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == player_color:
                player_material += value
            else:
                opponent_material += value
    
    return player_material - opponent_material


# ── OPENING FEATURES ────────────────────────────────────────────────
def compute_gambit_tendency(opening_name: str) -> int:
    if pd.isna(opening_name):
        return 0
    return int("gambit" in opening_name.lower())

def compute_sharp_opening(opening_eco: str) -> int:
    if pd.isna(opening_eco):
        return 0
    return int(any(opening_eco.startswith(p) for p in SHARP_ECO_PREFIXES))


# ── MASTER FEATURE BUILDER ───────────────────────────────────────────
def build_feature_matrix(df: pd.DataFrame, username: str) -> pd.DataFrame:
    """
    Build full feature matrix — one row per game.
    Combines timing + move-level + opening features.
    """
    features = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Engineering features"):
        side = get_player_side(row, username)
        
        # Parse clocks
        clocks = row["clocks"]
        if isinstance(clocks, str):
            try:
                clocks = ast.literal_eval(clocks)
            except:
                clocks = []

        # Timing features
        panic  = compute_panic_score(clocks, side)
        tpm    = compute_time_per_move(clocks, side)

        # Move-level features
        move_feats = parse_move_features(row["moves"], side)

        f = {
            "game_id":         row["game_id"],
            "side":            side,
            "speed":           row["speed"],
            "num_moves":       row["num_moves"],
            "result":          did_player_win(row, side),

            # Timing
            "panic_score":     panic,
            "time_per_move":   tpm,

            # Move-level
            "capture_rate":    move_feats["capture_rate"],
            "sacrifice_rate":  move_feats["sacrifice_rate"],
            "check_rate":      move_feats["check_rate"],
            "early_aggression":move_feats["early_aggression"],
            "castle_move":     move_feats["castle_move"],
            "piece_activity":  move_feats["piece_activity"],
            "avg_material_diff":move_feats["avg_material_diff"],

            # Opening
            "gambit_tendency": compute_gambit_tendency(row["opening_name"]),
            "sharp_opening":   compute_sharp_opening(row["opening_eco"]),
            "decisive_game":   int(row["status"] != "draw"),

            # Raw info
            "opening_name":    row["opening_name"],
            "opening_eco":     row["opening_eco"],
            "status":          row["status"],
        }
        features.append(f)

    feature_df = pd.DataFrame(features)
    
    # Print summary
    move_cols = ["capture_rate", "sacrifice_rate", "check_rate",
                 "early_aggression", "castle_move", "piece_activity",
                 "avg_material_diff"]
    
    print(f"\nFeature matrix shape: {feature_df.shape}")
    print(f"\nMove-level feature summary:")
    print(feature_df[move_cols].describe().round(3))
    
    return feature_df


if __name__ == "__main__":
    df_raw = pd.read_csv("data/games_raw.csv")
    USERNAME = "DrNykterstein"
    
    feature_df = build_feature_matrix(df_raw, USERNAME)
    feature_df.to_csv("data/features.csv", index=False)
    
    print("\nSample move-level features:")
    print(feature_df[["game_id", "capture_rate", "sacrifice_rate",
                       "check_rate", "early_aggression", 
                       "castle_move", "piece_activity"]].head(10))