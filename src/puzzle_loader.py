import pandas as pd
import numpy as np
import os

# ── THEME MAPPING PER CLUSTER ────────────────────────────────────
CLUSTER_THEMES = {
    0: {  # Chaotic Attacker — needs defensive skills + endgame
        "primary":   ["defensiveMove", "endgame", "quietMove"],
        "secondary": ["zugzwang", "kingsideAttack", "pawnEndgame"],
        "label":     "Chaotic Attacker"
    },
    1: {  # Solid Strategist — needs endgame conversion
        "primary":   ["endgame", "zugzwang", "quietMove"],
        "secondary": ["bishopEndgame", "rookEndgame", "pawnEndgame"],
        "label":     "Solid Strategist"
    },
    2: {  # Time Pressure Wildcard — needs fast pattern recognition
        "primary":   ["mateIn1", "mateIn2", "fork"],
        "secondary": ["pin", "skewer", "attackingF7"],
        "label":     "Time Pressure Wildcard"
    },
    3: {  # Passive Defender — needs tactical sharpness
        "primary":   ["fork", "pin", "sacrifice"],
        "secondary": ["skewer", "attackingF7", "middlegame", "clearance"],
        "label":     "Passive Defender"
    },
}


def load_puzzles(
        path: str = "data/puzzles_filtered.csv",
        max_puzzles: int = 200_000) -> pd.DataFrame:
    """
    Load the Lichess puzzle database.
    Columns: PuzzleId, FEN, Moves, Rating, RatingDeviation,
             Popularity, NbPlays, Themes, GameUrl, OpeningTags
    """
    print(f"Loading puzzle database from {path}...")

    df = pd.read_csv(path, nrows=max_puzzles)

    print(f"Loaded {len(df):,} puzzles")
    print(f"Columns: {list(df.columns)}")
    print(f"Rating range: {df['Rating'].min()} - {df['Rating'].max()}")

    # Themes is a space-separated string — split into list
    df["ThemeList"] = df["Themes"].fillna("").str.split()

    return df


def filter_puzzles_for_player(
        df: pd.DataFrame,
        cluster_id: int,
        player_rating: int,
        n_puzzles: int = 10,
        rating_window: int = 200) -> pd.DataFrame:
    """
    Filter puzzles for a specific player based on:
    1. Their style cluster (determines puzzle themes)
    2. Their rating (determines puzzle difficulty)
    3. Scaffolding: start easy, increase difficulty
    """
    themes = CLUSTER_THEMES[cluster_id]
    all_themes = themes["primary"] + themes["secondary"]

    # Rating window: player_rating ± window
    effective_rating = min(player_rating, 2000)
    rating_min = effective_rating - rating_window
    rating_max = effective_rating + rating_window
    # Filter by rating range
    rating_mask = (
        (df["Rating"] >= rating_min) &
        (df["Rating"] <= rating_max)
    )
    rating_filtered = df[rating_mask].copy()

    print(f"\nPuzzles in rating range {rating_min}-{rating_max}: "
          f"{len(rating_filtered):,}")

    # Score each puzzle by theme match
    def theme_score(theme_list):
        if not theme_list:
            return 0
        primary_matches   = sum(1 for t in theme_list
                                if t in themes["primary"])
        secondary_matches = sum(1 for t in theme_list
                                if t in themes["secondary"])
        # Primary themes worth 2x
        return primary_matches * 2 + secondary_matches

    rating_filtered["theme_score"] = (
        rating_filtered["ThemeList"].apply(theme_score)
    )

    # Keep only puzzles with at least one matching theme
    matched = rating_filtered[rating_filtered["theme_score"] > 0].copy()
    print(f"Puzzles matching themes for cluster {cluster_id} "
          f"({CLUSTER_THEMES[cluster_id]['label']}): {len(matched):,}")

    if len(matched) == 0:
        print("No theme matches — falling back to rating-only filter")
        matched = rating_filtered.copy()

    # Sort: theme score descending, then rating ascending (scaffolding)
    matched = matched.sort_values(
        ["theme_score", "Rating"],
        ascending=[False, True]
    )

    result = matched.head(n_puzzles).copy()
    print(f"Returning {len(result)} puzzles")
    return result


def format_puzzle_for_display(row: pd.Series) -> dict:
    """Convert a puzzle row into a display-ready dict."""
    return {
        "puzzle_id":  row["PuzzleId"],
        "rating":     int(row["Rating"]),
        "themes":     row["ThemeList"],
        "fen":        row["FEN"],
        "moves":      row["Moves"],
        "url":        f"https://lichess.org/training/{row['PuzzleId']}",
        "popularity": row.get("Popularity", 0),
    }


def get_puzzles_for_player(
        cluster_id:     int,
        player_rating:  int,
        n_puzzles:      int = 10,
        puzzle_db_path: str = "data/puzzles_filtered.csv") -> list[dict]:
    """
    Main entry point: given a cluster and rating,
    return n puzzles ready for display.
    """
    df = load_puzzles(puzzle_db_path)
    filtered = filter_puzzles_for_player(
        df, cluster_id, player_rating, n_puzzles
    )
    puzzles = [format_puzzle_for_display(row)
               for _, row in filtered.iterrows()]
    return puzzles


if __name__ == "__main__":
    # Test all 4 clusters
    test_cases = [
        (0, 1600, "Chaotic Attacker"),
        (1, 2000, "Solid Strategist"),
        (2, 1400, "Time Pressure Wildcard"),
        (3, 1500, "Passive Defender"),
    ]

    df = load_puzzles()

    for cluster_id, rating, label in test_cases:
        print(f"\n{'='*55}")
        print(f"Cluster {cluster_id}: {label} | Rating {rating}")
        print(f"{'='*55}")

        filtered = filter_puzzles_for_player(
            df, cluster_id, rating, n_puzzles=3
        )

        for _, row in filtered.iterrows():
            p = format_puzzle_for_display(row)
            print(f"\n  Puzzle {p['puzzle_id']}")
            print(f"  Rating  : {p['rating']}")
            print(f"  Themes  : {', '.join(p['themes'])}")
            print(f"  URL     : {p['url']}")
