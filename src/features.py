import pandas as pd
import numpy as np
import ast

def get_player_side(row: pd.Series, username: str) -> str:
    """Returns 'white' or 'black' for the target player in this game."""
    if str(row["white_user"]).lower() == username.lower():
        return "white"
    return "black"

def did_player_win(row: pd.Series, side: str) -> int:
    """1 = win, 0 = draw, -1 = loss"""
    if row["winner"] == side:
        return 1
    elif pd.isna(row["winner"]):
        return 0
    return -1

def compute_panic_score(clocks: list, player_side: str) -> float:
    """
    Panic score = avg time per move in last 30% of game
                  / avg time per move in first 30% of game
    
    Clocks alternate: white move 1, black move 1, white move 2...
    So we filter to only the target player's clock entries.
    """
    if not clocks or len(clocks) < 6:
        return np.nan
    
    # Filter to player's moves only (white = even indices, black = odd)
    if player_side == "white":
        player_clocks = clocks[0::2]
    else:
        player_clocks = clocks[1::2]
    
    if len(player_clocks) < 6:
        return np.nan
    
    # Time spent per move = difference between consecutive clock values
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

def compute_gambit_tendency(opening_name: str) -> int:
    """1 if player chose a gambit opening, 0 otherwise."""
    if pd.isna(opening_name):
        return 0
    return int("gambit" in opening_name.lower())

def compute_decisive_tendency(status: str, winner: str, side: str) -> int:
    """
    1 = decisive result (win or loss — player plays for outcome)
    0 = draw
    """
    return int(status != "draw")

def compute_aggression_proxy(opening_eco: str) -> int:
    """
    Sharp/aggressive openings by ECO code.
    B = Sicilian and other sharp semi-open games
    E = King's Indian, Grunfeld — sharp and dynamic
    A45/A46 = Trompowsky, London — more positional
    """
    if pd.isna(opening_eco):
        return 0
    sharp_prefixes = ["B2", "B3", "B4", "B7", "B8", "B9",  # Sicilian lines
                      "E6", "E7", "E8", "E9",                # King's Indian
                      "C1", "C2", "C3", "C4",                # open games
                      ]
    return int(any(opening_eco.startswith(p) for p in sharp_prefixes))

def build_feature_matrix(df: pd.DataFrame, username: str) -> pd.DataFrame:
    """
    Takes raw games DataFrame, returns one row per game
    with all engineered features for the target player.
    """
    features = []

    for _, row in df.iterrows():
        side = get_player_side(row, username)
        
        # Parse clocks from string if needed
        clocks = row["clocks"]
        if isinstance(clocks, str):
            try:
                clocks = ast.literal_eval(clocks)
            except:
                clocks = []

        f = {
            "game_id":           row["game_id"],
            "side":              side,
            "speed":             row["speed"],
            "num_moves":         row["num_moves"],
            
            # Target player outcome
            "result":            did_player_win(row, side),
            
            # Style features
            "panic_score":       compute_panic_score(clocks, side),
            "gambit_tendency":   compute_gambit_tendency(row["opening_name"]),
            "sharp_opening":     compute_aggression_proxy(row["opening_eco"]),
            "decisive_game":     compute_decisive_tendency(
                                     row["status"], row["winner"], side),
            
            # Raw info we'll use later
            "opening_name":      row["opening_name"],
            "opening_eco":       row["opening_eco"],
            "status":            row["status"],
        }
        features.append(f)

    feature_df = pd.DataFrame(features)
    print(f"Built feature matrix: {feature_df.shape}")
    print(feature_df[["panic_score", "gambit_tendency", 
                       "sharp_opening", "decisive_game"]].describe())
    return feature_df


if __name__ == "__main__":
    df_raw = pd.read_csv("data/games_raw.csv")
    
    USERNAME = "DrNykterstein"
    feature_df = build_feature_matrix(df_raw, USERNAME)
    
    feature_df.to_csv("data/features.csv", index=False)
    print("\nFeature matrix saved to data/features.csv")
    print(feature_df.head())