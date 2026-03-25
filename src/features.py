import pandas as pd
import numpy as np
import ast
import chess
import chess.pgn
import io
from tqdm import tqdm
from scipy.stats import entropy

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
    "B2", "B3", "B4", "B7", "B8", "B9",
    "E6", "E7", "E8", "E9",
    "C1", "C2", "C3", "C4",
]

# ── KING SAFETY HELPERS ─────────────────────────────────────────────
def get_pawn_shield_squares(king_sq: int, player_color: int) -> list:
    """
    Returns the 3 squares directly in front of the king —
    the classic pawn shield zone.
    For white: one rank up. For black: one rank down.
    We check the king's file and one file either side.
    """
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    shield_rank = king_rank + 1 if player_color == chess.WHITE else king_rank - 1

    # Stay on the board
    if shield_rank < 0 or shield_rank > 7:
        return []

    squares = []
    for f in [king_file - 1, king_file, king_file + 1]:
        if 0 <= f <= 7:
            squares.append(chess.square(f, shield_rank))
    return squares


def compute_pawn_shield(board: chess.Board, player_color: int) -> float:
    """
    Count pawns on the 3 squares in front of the king.
    Returns a score from 0.0 (no shield) to 1.0 (full shield).
    If king is in center (never castled), penalize with 0.0.
    """
    king_sq = board.king(player_color)
    if king_sq is None:
        return 0.0

    king_file = chess.square_file(king_sq)

    # King in center files (c,d,e,f = files 2,3,4,5) = no shield
    if 2 <= king_file <= 5:
        return 0.0

    shield_squares = get_pawn_shield_squares(king_sq, player_color)
    if not shield_squares:
        return 0.0

    pawn_count = sum(
        1 for sq in shield_squares
        if board.piece_at(sq) == chess.Piece(chess.PAWN, player_color)
    )
    return pawn_count / 3.0


# ── CONSTANTS ───────────────────────────────────────────────────────
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
    if not clocks or len(clocks) < 6:
        return np.nan
    player_clocks = clocks[0::2] if player_side == "white" else clocks[1::2]
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
    if not clocks or len(clocks) < 2:
        return np.nan
    player_clocks = clocks[0::2] if player_side == "white" else clocks[1::2]
    if len(player_clocks) < 2:
        return np.nan
    time_per_move = [
        max(player_clocks[i] - player_clocks[i+1], 0)
        for i in range(len(player_clocks) - 1)
    ]
    return np.mean(time_per_move)


def compute_material_balance(board: chess.Board, player_color: int) -> int:
    player_mat, opp_mat = 0, 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            val = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == player_color:
                player_mat += val
            else:
                opp_mat += val
    return player_mat - opp_mat


def parse_move_features(moves_str: str, player_side: str) -> dict:
    empty = {
        "capture_rate":       np.nan,
        "sacrifice_rate":     np.nan,
        "check_rate":         np.nan,
        "checks_received_rate": np.nan,
        "pawn_shield_score":  np.nan,
        "early_aggression":   np.nan,
        "castle_move":        np.nan,
        "piece_activity":     np.nan,
        "avg_material_diff":  np.nan,
    }
    if not moves_str or pd.isna(moves_str):
        return empty
    try:
        pgn_string = f"[Event \"?\"]\n\n{moves_str}"
        game = chess.pgn.read_game(io.StringIO(pgn_string))
        if game is None:
            return empty

        board = game.board()
        player_color = chess.WHITE if player_side == "white" else chess.BLACK

        total_moves      = 0
        opponent_moves   = 0
        captures         = 0
        sacrifices       = 0
        checks_given     = 0
        checks_received  = 0
        early_aggression = 0
        piece_moves_early = 0
        castle_move_num  = np.nan
        material_diffs   = []
        move_number      = 0

        # Pawn shield snapshot at move 20
        pawn_shield_score = 0.0
        shield_captured   = False

        for move in game.mainline_moves():
            current_color = board.turn

            if current_color == player_color:
                # ── OUR MOVE ──────────────────────────────────────
                total_moves += 1
                move_number += 1
                is_early = move_number <= 10

                if board.is_capture(move):
                    captures += 1
                    if is_early:
                        early_aggression += 1
                    moving_piece   = board.piece_at(move.from_square)
                    captured_piece = board.piece_at(move.to_square)
                    if moving_piece and captured_piece:
                        if PIECE_VALUES.get(moving_piece.piece_type, 0) > \
                           PIECE_VALUES.get(captured_piece.piece_type, 0):
                            sacrifices += 1

                if board.is_castling(move) and np.isnan(castle_move_num):
                    castle_move_num = move_number

                if is_early:
                    mp = board.piece_at(move.from_square)
                    if mp and mp.piece_type != chess.PAWN:
                        piece_moves_early += 1

                board.push(move)

                # Check given — detected after our move
                if board.is_check():
                    checks_given += 1
                    if is_early:
                        early_aggression += 1

                material_diffs.append(
                    compute_material_balance(board, player_color)
                )

                # Pawn shield snapshot at our move 20
                if move_number == 20 and not shield_captured:
                    pawn_shield_score = compute_pawn_shield(board, player_color)
                    shield_captured = True

            else:
                # ── OPPONENT'S MOVE ───────────────────────────────
                opponent_moves += 1
                board.push(move)

                # Check received — detected after opponent's move
                if board.is_check():
                    checks_received += 1

        # If game ended before move 20, take final position
        if not shield_captured and total_moves > 0:
            pawn_shield_score = compute_pawn_shield(board, player_color)

        if total_moves == 0:
            return empty

        return {
            "capture_rate":         captures      / total_moves,
            "sacrifice_rate":       sacrifices    / total_moves,
            "check_rate":           checks_given  / total_moves,
            "checks_received_rate": checks_received / max(opponent_moves, 1),
            "pawn_shield_score":    pawn_shield_score,
            "early_aggression":     early_aggression,
            "castle_move":          castle_move_num,
            "piece_activity":       piece_moves_early,
            "avg_material_diff":    np.mean(material_diffs) if material_diffs else 0,
        }
    except Exception:
        return empty


def build_game_features(df: pd.DataFrame) -> pd.DataFrame:
    features = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Game features"):
        username = row["username"]
        side     = get_player_side(row, username)

        clocks = row["clocks"]
        if isinstance(clocks, str):
            try:
                clocks = ast.literal_eval(clocks)
            except:
                clocks = []

        move_feats = parse_move_features(row["moves"], side)

        f = {
            "username":             username,
            "game_id":              row["game_id"],
            "side":                 side,
            "num_moves":            row["num_moves"],
            "result":               did_player_win(row, side),
            "panic_score":          compute_panic_score(clocks, side),
            "time_per_move":        compute_time_per_move(clocks, side),
            "capture_rate":         move_feats["capture_rate"],
            "sacrifice_rate":       move_feats["sacrifice_rate"],
            "check_rate":           move_feats["check_rate"],
            "checks_received_rate": move_feats["checks_received_rate"],
            "pawn_shield_score":    move_feats["pawn_shield_score"],
            "early_aggression":     move_feats["early_aggression"],
            "castle_move":          move_feats["castle_move"],
            "piece_activity":       move_feats["piece_activity"],
            "avg_material_diff":    move_feats["avg_material_diff"],
            "gambit_tendency":      int("gambit" in str(row["opening_name"]).lower()),
            "opening_eco":          str(row["opening_eco"]),
            "sharp_opening":        int(any(
                str(row["opening_eco"]).startswith(p)
                for p in SHARP_ECO_PREFIXES
            )),
            "decisive_game":        int(row["status"] != "draw"),
        }
        features.append(f)

    return pd.DataFrame(features)


def aggregate_to_player_level(game_df: pd.DataFrame) -> pd.DataFrame:
    players = []

    for username, grp in tqdm(
            game_df.groupby("username"),
            desc="Aggregating players"):

        n_games = len(grp)

        # Castle features
        castled_games = grp["castle_move"].dropna()
        castle_rate   = len(castled_games) / n_games
        # avg_castle_move DROPPED — noisy signal, castle_rate is sufficient

        # Opening diversity
        eco_counts      = grp["opening_eco"].value_counts()
        opening_entropy = entropy(eco_counts)
        top_opening_pct = eco_counts.iloc[0] / n_games if len(eco_counts) > 0 else 0
        unique_openings = grp["opening_eco"].nunique()

        # Composite king safety score
        # Combines castle rate, pawn shield, and checks received
        # Higher = safer king management
        checks_received_mean = grp["checks_received_rate"].mean()
        pawn_shield_mean     = grp["pawn_shield_score"].mean()
        king_safety_score    = (
            castle_rate          * 0.4
            + pawn_shield_mean   * 0.4
            - checks_received_mean * 0.2
        )

        p = {
            "username":               username,
            "n_games":                n_games,

            # Timing
            "panic_score_mean":       grp["panic_score"].mean(),
            "panic_score_std":        grp["panic_score"].std(),
            "time_per_move_mean":     grp["time_per_move"].mean(),

            # Move-level
            "capture_rate_mean":      grp["capture_rate"].mean(),
            "capture_rate_std":       grp["capture_rate"].std(),
            "sacrifice_rate_mean":    grp["sacrifice_rate"].mean(),
            "check_rate_mean":        grp["check_rate"].mean(),
            "checks_received_mean":   checks_received_mean,
            "early_aggression_mean":  grp["early_aggression"].mean(),
            "piece_activity_mean":    grp["piece_activity"].mean(),
            "avg_material_diff":      grp["avg_material_diff"].mean(),

            # King safety (replaces avg_castle_move)
            "castle_rate":            castle_rate,
            "pawn_shield_mean":       pawn_shield_mean,
            "king_safety_score":      king_safety_score,

            # Opening
            "gambit_rate":            grp["gambit_tendency"].mean(),
            "sharp_opening_rate":     grp["sharp_opening"].mean(),
            "opening_entropy":        opening_entropy,
            "top_opening_pct":        top_opening_pct,
            "unique_openings":        unique_openings,

            # Outcomes
            # decisive_rate DROPPED — mathematically = 1 - draw_rate
            "win_rate":               (grp["result"] == 1).mean(),
            "draw_rate":              (grp["result"] == 0).mean(),
            "avg_game_length":        grp["num_moves"].mean(),
        }
        players.append(p)

    player_df = pd.DataFrame(players)

    print(f"\nPlayer-level feature matrix: {player_df.shape}")
    print(f"\nFeature summary:")
    print(player_df.drop(columns=["username", "n_games"]).describe().round(3))

    return player_df


# ── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_raw = pd.read_csv("data/games_raw.csv")

    game_counts = df_raw.groupby("username").size()
    valid       = game_counts[game_counts >= 30].index
    df_raw      = df_raw[df_raw["username"].isin(valid)]

    print(f"Loaded {len(df_raw)} games for "
          f"{df_raw['username'].nunique()} players "
          f"(30+ games each)")

    print("\nStep 1: Computing game-level features...")
    game_df = build_game_features(df_raw)
    game_df.to_csv("data/features_game_level.csv", index=False)
    print(f"Saved: data/features_game_level.csv {game_df.shape}")

    print("\nStep 2: Aggregating to player level...")
    player_df = aggregate_to_player_level(game_df)

    before = len(player_df)
    player_df = player_df[player_df["capture_rate_mean"] > 0.10]
    print(f"\nOutlier filter: removed {before - len(player_df)} players "
          f"(capture_rate < 0.10)")
    print(f"Remaining: {len(player_df)} players")

    player_df.to_csv("data/features_player_level.csv", index=False)
    print(f"Saved: data/features_player_level.csv {player_df.shape}")

    print("\nPlayer profiles:")
    print(player_df[[
        "username", "capture_rate_mean", "sacrifice_rate_mean",
        "gambit_rate", "castle_rate", "king_safety_score",
        "opening_entropy", "win_rate", "avg_game_length"
    ]].to_string())