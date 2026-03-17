import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import fetch_games_for_player, parse_games
from features import build_game_features, aggregate_to_player_level
from clustering import CLUSTER_NAMES, CLUSTER_DESCRIPTIONS

def predict_player_style(username: str,
                          model_path: str = "data/kmeans_model.pkl",
                          scaler_path: str = "data/scaler.pkl",
                          pca_path: str   = "data/pca_model.pkl") -> dict:
    """
    Full pipeline for a new user:
    1. Fetch their recent rapid games
    2. Engineer features
    3. Aggregate to player level
    4. Scale + PCA transform
    5. Predict cluster
    6. Return style profile
    """
    print(f"Analyzing {username}...")

    # Step 1: Fetch games
    games_raw = fetch_games_for_player(username, max_games=50)
    if not games_raw:
        return {"error": f"No rapid games found for {username}"}

    df_raw = parse_games(games_raw, username)
    if len(df_raw) < 10:
        return {"error": f"Only {len(df_raw)} games found. Need at least 10."}

    print(f"  Fetched {len(df_raw)} games")

    # Step 2: Game-level features
    game_df = build_game_features(df_raw)

    # Step 3: Player-level aggregation
    player_df = aggregate_to_player_level(game_df)

    if player_df.empty:
        return {"error": "Could not compute features"}

    # Step 4: Load saved models and transform
    try:
        kmeans = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pca    = joblib.load(pca_path)
    except FileNotFoundError:
        return {"error": "Models not found. Run clustering.py first."}

    from clustering import FEATURE_COLS
    
    # Handle missing features
    for col in FEATURE_COLS:
        if col not in player_df.columns:
            player_df[col] = 0

    X = scaler.transform(player_df[FEATURE_COLS])
    X_pca = pca.transform(X)

    # Step 5: Predict cluster
    cluster_id = int(kmeans.predict(X_pca)[0])

    # Step 6: Build profile
    row = player_df.iloc[0]
    profile = {
        "username":        username,
        "cluster_id":      cluster_id,
        "style":           CLUSTER_NAMES[cluster_id],
        "description":     CLUSTER_DESCRIPTIONS[cluster_id],
        "games_analyzed":  len(df_raw),
        "stats": {
            "win_rate":          round(float(row.get("win_rate", 0)), 3),
            "castle_rate":       round(float(row.get("castle_rate", 0)), 3),
            "avg_castle_move":   round(float(row.get("avg_castle_move", 0)), 1),
            "gambit_rate":       round(float(row.get("gambit_rate", 0)), 3),
            "sacrifice_rate":    round(float(row.get("sacrifice_rate_mean", 0)), 3),
            "avg_game_length":   round(float(row.get("avg_game_length", 0)), 1),
            "opening_entropy":   round(float(row.get("opening_entropy", 0)), 2),
            "panic_score":       round(float(row.get("panic_score_mean", 0)), 2),
        }
    }

    return profile


if __name__ == "__main__":
    username = sys.argv[1] if len(sys.argv) > 1 else "DrNykterstein"
    profile  = predict_player_style(username)
    
    if "error" in profile:
        print(f"Error: {profile['error']}")
    else:
        print(f"\n{'='*50}")
        print(f"STYLE PROFILE: {profile['username']}")
        print(f"{'='*50}")
        print(f"Style:       {profile['style']}")
        print(f"Description: {profile['description']}")
        print(f"\nKey Stats:")
        for k, v in profile["stats"].items():
            print(f"  {k:<20} {v}")