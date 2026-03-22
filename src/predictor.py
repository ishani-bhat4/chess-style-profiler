import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import fetch_games_for_player, parse_games
from features import build_game_features, aggregate_to_player_level
from clustering import FEATURE_COLS

CLUSTER_NAMES = {
    0: "Chaotic Attacker",
    1: "Solid Strategist",
    2: "Time Pressure Wildcard",
    3: "Passive Defender"
}

CLUSTER_DESCRIPTIONS = {
    0: "You play aggressively and create complications, but inconsistent king safety costs you games. Your aggression is your strength. Structure it.",
    1: "You play principled, well-rounded chess. You castle reliably, manage time well, and have a broad opening repertoire. The hallmark of a complete player.",
    2: "Your biggest challenge is time management. You slow down dramatically in critical positions. Practice faster decision-making in time pressure situations.",
    3: "You play cautiously and avoid complications. This is safe but limits your winning chances. Try introducing more active piece play and tactical patterns."
}


def predict_player_style(
        username: str,
        model_path:  str = "data/kmeans_model.pkl",
        scaler_path: str = "data/scaler.pkl",
        pca_path:    str = "data/pca_model.pkl") -> dict:
    """
    Full pipeline for a new user:
    1. Fetch their recent rapid games
    2. Engineer features
    3. Aggregate to player level
    4. Scale + PCA transform
    5. Predict cluster
    6. Return style profile + puzzles
    """
    print(f"Analyzing {username}...")

    # ── Step 1: Fetch games ───────────────────────────────────────
    games_raw = fetch_games_for_player(username, max_games=50)
    if not games_raw:
        return {"error": f"No rapid games found for {username}"}

    df_raw = parse_games(games_raw, username)
    if len(df_raw) < 10:
        return {"error": f"Only {len(df_raw)} games found. Need at least 10."}

    print(f"  Fetched {len(df_raw)} games")

    # ── Step 2: Game-level features ───────────────────────────────
    game_df = build_game_features(df_raw)

    # ── Step 3: Player-level aggregation ─────────────────────────
    player_df = aggregate_to_player_level(game_df)

    if player_df.empty:
        return {"error": "Could not compute features"}

    # ── Step 4: Load saved models and transform ───────────────────
    try:
        kmeans = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pca    = joblib.load(pca_path)
    except FileNotFoundError:
        return {"error": "Models not found. Run clustering.py first."}

    # Handle missing features
    for col in FEATURE_COLS:
        if col not in player_df.columns:
            player_df[col] = 0

    X     = scaler.transform(player_df[FEATURE_COLS])
    X_pca = pca.transform(X)

    # ── Step 5: Predict cluster ───────────────────────────────────
    cluster_id = int(kmeans.predict(X_pca)[0])
    row        = player_df.iloc[0]

    # ── Step 6: Estimate player rating ───────────────────────────
    try:
        win_rate = float(row.get("win_rate", 0.5))
        if win_rate > 0.5:
            player_rating = int(df_raw["white_rating"].median())
        else:
            player_rating = int(df_raw["black_rating"].median())
        # Sanity clamp
        player_rating = max(400, min(3000, player_rating))
    except Exception:
        player_rating = 1500

    # ── Step 7: Get puzzle recommendations ───────────────────────
    try:
        from puzzle_loader import get_puzzles_for_player
        puzzles = get_puzzles_for_player(
            cluster_id=cluster_id,
            player_rating=player_rating,
            n_puzzles=6,
            puzzle_db_path="data/lichess_db_puzzle.csv"
        )
    except Exception as e:
        print(f"  Puzzle loading failed: {e}")
        puzzles = []

    # ── Step 8: Build profile ─────────────────────────────────────
    profile = {
        "username":      username,
        "cluster_id":    cluster_id,
        "style":         CLUSTER_NAMES[cluster_id],
        "description":   CLUSTER_DESCRIPTIONS[cluster_id],
        "games_analyzed": len(df_raw),
        "player_rating": player_rating,
        "puzzles":       puzzles,
        "stats": {
            "win_rate":        round(float(row.get("win_rate", 0)), 3),
            "castle_rate":     round(float(row.get("castle_rate", 0)), 3),
            "avg_castle_move": round(float(row.get("avg_castle_move", 0)), 1),
            "gambit_rate":     round(float(row.get("gambit_rate", 0)), 3),
            "sacrifice_rate":  round(float(row.get("sacrifice_rate_mean", 0)), 3),
            "avg_game_length": round(float(row.get("avg_game_length", 0)), 1),
            "opening_entropy": round(float(row.get("opening_entropy", 0)), 2),
            "panic_score":     round(float(row.get("panic_score_mean", 0)), 2),
        }
    }

    return profile


if __name__ == "__main__":
    username = sys.argv[1] if len(sys.argv) > 1 else "Kurald_Galain"
    profile  = predict_player_style(username)

    if "error" in profile:
        print(f"Error: {profile['error']}")
    else:
        print(f"\n{'='*50}")
        print(f"STYLE PROFILE: {profile['username']}")
        print(f"{'='*50}")
        print(f"Style:         {profile['style']}")
        print(f"Description:   {profile['description']}")
        print(f"Player Rating: {profile['player_rating']}")
        print(f"\nKey Stats:")
        for k, v in profile["stats"].items():
            print(f"  {k:<22} {v}")
        print(f"\nPuzzles ({len(profile['puzzles'])}):")
        for i, p in enumerate(profile["puzzles"], 1):
            themes = ", ".join(
                t for t in p["themes"]
                if t not in ["crushing", "advantage", "long", "short", "veryLong"]
            )[:60]
            print(f"  {i}. Rating {p['rating']:4d} | {themes}")
            print(f"     {p['url']}")