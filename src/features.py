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

# ── GAME-LEVEL FEATURES ─────────────────────────────────────────────
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
        "capture_rate":      np.nan,
        "sacrifice_rate":    np.nan,
        "check_rate":        np.nan,
        "early_aggression":  np.nan,
        "castle_move":       np.nan,
        "piece_activity":    np.nan,
        "avg_material_diff": np.nan,
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

        total_moves = captures = sacrifices = checks = 0
        early_aggression = piece_moves_early = 0
        castle_move_num = np.nan
        material_diffs = []
        move_number = 0

        for move in game.mainline_moves():
            current_color = board.turn
            if current_color == player_color:
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

                if board.is_check():
                    checks += 1
                    if is_early:
                        early_aggression += 1

                material_diffs.append(
                    compute_material_balance(board, player_color)
                )
            else:
                board.push(move)

        if total_moves == 0:
            return empty

        return {
            "capture_rate":      captures   / total_moves,
            "sacrifice_rate":    sacrifices / total_moves,
            "check_rate":        checks     / total_moves,
            "early_aggression":  early_aggression,
            "castle_move":       castle_move_num,
            "piece_activity":    piece_moves_early,
            "avg_material_diff": np.mean(material_diffs) if material_diffs else 0,
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
            "username":          username,
            "game_id":           row["game_id"],
            "side":              side,
            "num_moves":         row["num_moves"],
            "result":            did_player_win(row, side),
            "panic_score":       compute_panic_score(clocks, side),
            "time_per_move":     compute_time_per_move(clocks, side),
            "capture_rate":      move_feats["capture_rate"],
            "sacrifice_rate":    move_feats["sacrifice_rate"],
            "check_rate":        move_feats["check_rate"],
            "early_aggression":  move_feats["early_aggression"],
            "castle_move":       move_feats["castle_move"],
            "piece_activity":    move_feats["piece_activity"],
            "avg_material_diff": move_feats["avg_material_diff"],
            "gambit_tendency":   int("gambit" in str(row["opening_name"]).lower()),
            "opening_eco":       str(row["opening_eco"]),
            "sharp_opening":     int(any(
                str(row["opening_eco"]).startswith(p)
                for p in SHARP_ECO_PREFIXES
            )),
            "decisive_game":     int(row["status"] != "draw"),
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
        castled_games   = grp["castle_move"].dropna()
        castle_rate     = len(castled_games) / n_games
        avg_castle_move = castled_games.mean() if len(castled_games) > 0 else np.nan

        # Opening diversity features
        eco_counts      = grp["opening_eco"].value_counts()
        opening_entropy = entropy(eco_counts)
        top_opening_pct = eco_counts.iloc[0] / n_games if len(eco_counts) > 0 else 0
        unique_openings = grp["opening_eco"].nunique()

        p = {
            "username":              username,
            "n_games":               n_games,

            # Timing
            "panic_score_mean":      grp["panic_score"].mean(),
            "panic_score_std":       grp["panic_score"].std(),
            "time_per_move_mean":    grp["time_per_move"].mean(),

            # Move-level
            "capture_rate_mean":     grp["capture_rate"].mean(),
            "capture_rate_std":      grp["capture_rate"].std(),
            "sacrifice_rate_mean":   grp["sacrifice_rate"].mean(),
            "check_rate_mean":       grp["check_rate"].mean(),
            "early_aggression_mean": grp["early_aggression"].mean(),
            "piece_activity_mean":   grp["piece_activity"].mean(),
            "avg_material_diff":     grp["avg_material_diff"].mean(),

            # Castle
            "castle_rate":           castle_rate,
            "avg_castle_move":       avg_castle_move,

            # Opening
            "gambit_rate":           grp["gambit_tendency"].mean(),
            "sharp_opening_rate":    grp["sharp_opening"].mean(),
            "opening_entropy":       opening_entropy,
            "top_opening_pct":       top_opening_pct,
            "unique_openings":       unique_openings,

            # Outcomes
            "win_rate":              (grp["result"] == 1).mean(),
            "draw_rate":             (grp["result"] == 0).mean(),
            "decisive_rate":         grp["decisive_game"].mean(),
            "avg_game_length":       grp["num_moves"].mean(),
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

    # Filter players with 30+ games
    game_counts = df_raw.groupby("username").size()
    valid       = game_counts[game_counts >= 30].index
    df_raw      = df_raw[df_raw["username"].isin(valid)]

    print(f"Loaded {len(df_raw)} games for "
          f"{df_raw['username'].nunique()} players "
          f"(30+ games each)")

    # Step 1: game-level features
    print("\nStep 1: Computing game-level features...")
    game_df = build_game_features(df_raw)
    game_df.to_csv("data/features_game_level.csv", index=False)
    print(f"Saved: data/features_game_level.csv {game_df.shape}")

    # Step 2: aggregate to player level
    print("\nStep 2: Aggregating to player level...")
    player_df = aggregate_to_player_level(game_df)

    # Step 3: remove statistical outliers
    # Players with capture_rate < 0.10 are likely bots or test accounts
    before = len(player_df)
    player_df = player_df[player_df["capture_rate_mean"] > 0.10]
    print(f"\nOutlier filter: removed {before - len(player_df)} players "
          f"(capture_rate < 0.10)")
    print(f"Remaining: {len(player_df)} players")

    player_df.to_csv("data/features_player_level.csv", index=False)
    print(f"Saved: data/features_player_level.csv {player_df.shape}")

    # Print player profiles
    print("\nPlayer profiles:")
    print(player_df[[
        "username", "capture_rate_mean", "sacrifice_rate_mean",
        "gambit_rate", "castle_rate", "avg_castle_move",
        "opening_entropy", "top_opening_pct", "unique_openings",
        "win_rate", "avg_game_length"
    ]].to_string())