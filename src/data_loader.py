import requests
import pandas as pd
import json
import time
import os
from tqdm import tqdm


def fetch_top_rapid_players(n: int = 50) -> list[str]:
    url = "https://lichess.org/api/player/top/50/rapid"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()
    usernames = [u["username"] for u in data["users"]]
    print(f"Found {len(usernames)} top rapid players from Lichess")
    return usernames[:n]


def fetch_tournament_players(tournament_id: str,
                              max_players: int = 100) -> list[tuple]:
    """
    Fetch players and their ratings from a Lichess arena tournament.
    Returns list of (username, rating) tuples.
    """
    url = f"https://lichess.org/api/tournament/{tournament_id}/results"
    headers = {"Accept": "application/x-ndjson"}
    params = {"nb": max_players}

    try:
        response = requests.get(
            url, headers=headers,
            params=params, stream=True, timeout=30
        )
        players = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                username = data.get("username")
                rating   = data.get("rating", 0)
                if username:
                    players.append((username, rating))
        print(f"  Tournament {tournament_id}: {len(players)} players")
        return players
    except Exception as e:
        print(f"  Tournament {tournament_id}: FAILED — {e}")
        return []


def fetch_games_for_player(
        username: str,
        max_games: int = 100,
        time_control: str = "rapid") -> list[dict]:
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max":      max_games,
        "clocks":   True,
        "opening":  True,
        "moves":    True,
        "perfType": time_control,
        "rated":    True,
    }
    headers = {"Accept": "application/x-ndjson"}

    try:
        response = requests.get(
            url, params=params, headers=headers,
            stream=True, timeout=30
        )
        games = []
        for line in response.iter_lines():
            if line:
                games.append(json.loads(line))
        print(f"  {username}: {len(games)} games fetched")
        return games
    except Exception as e:
        print(f"  {username}: FAILED — {e}")
        return []


def parse_games(games: list[dict], username: str) -> pd.DataFrame:
    rows = []
    for g in games:
        try:
            players = g.get("players", {})
            white   = players.get("white", {})
            black   = players.get("black", {})
            row = {
                "username":     username,
                "game_id":      g.get("id"),
                "rated":        g.get("rated"),
                "speed":        g.get("speed"),
                "time_control": g.get("clock", {}).get("initial"),
                "increment":    g.get("clock", {}).get("increment"),
                "white_user":   white.get("user", {}).get("name"),
                "black_user":   black.get("user", {}).get("name"),
                "white_rating": white.get("rating"),
                "black_rating": black.get("rating"),
                "winner":       g.get("winner"),
                "status":       g.get("status"),
                "opening_name": g.get("opening", {}).get("name"),
                "opening_eco":  g.get("opening", {}).get("eco"),
                "opening_ply":  g.get("opening", {}).get("ply"),
                "moves":        g.get("moves", ""),
                "num_moves":    len(g.get("moves", "").split()),
                "clocks":       g.get("clocks", []),
            }
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def fetch_all_players(
        players: list[str],
        max_games: int = 100) -> pd.DataFrame:
    all_dfs = []
    print(f"\nFetching rapid games for {len(players)} players...")
    print("=" * 50)

    for username in tqdm(players, desc="Players"):
        games = fetch_games_for_player(username, max_games)
        if games:
            df = parse_games(games, username)
            all_dfs.append(df)
        time.sleep(1.5)

    if not all_dfs:
        print("No games fetched!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'=' * 50}")
    print(f"Total games fetched : {len(combined)}")
    print(f"Players with data   : {combined['username'].nunique()}")
    print(f"\nGames per player:")
    print(combined.groupby("username").size()
          .sort_values(ascending=False).to_string())
    return combined


def sample_by_rating_band(players: list[tuple],
                           bands: list[tuple],
                           per_band: int = 20) -> list[str]:
    """Sample players deliberately across rating bands."""
    sampled = []
    for low, high in bands:
        band = [u for u, r in players if low <= r < high]
        sampled.extend(band[:per_band])
        print(f"  {low}-{high}: {len(band)} available, "
              f"took {min(len(band), per_band)}")
    return list(dict.fromkeys(sampled))


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # ── Step 1: Fetch players from diverse arena tournaments ───────
    TOURNAMENT_IDS = [
        "SopTH39C",  # Slav Defense Rapid Arena   — 196 players
        "DEHeCQNy",  # ≤1300 Rapid Arena           — 58  players (beginners)
        "HPkHFL2D",  # Hourly SuperBlitz Arena     — 204 players (mixed)
        "y9YmKzXm",  # Hourly Blitz Arena          — 140 players (mixed)
        "ZbKTAZNc",  # ≤1500 Blitz Arena           — 69  players (casual)
        "pARZoUJY",  # ≤1700 SuperBlitz Arena      — 102 players (mid)
    ]

    print("Fetching players from arena tournaments...")
    all_player_tuples = []
    for tid in TOURNAMENT_IDS:
        players = fetch_tournament_players(tid, max_players=150)
        all_player_tuples.extend(players)
        time.sleep(1)

    # Deduplicate by username
    seen = set()
    unique_players = []
    for u, r in all_player_tuples:
        if u not in seen:
            seen.add(u)
            unique_players.append((u, r))

    print(f"\nTotal unique players from tournaments: {len(unique_players)}")

    if unique_players:
        ratings = [r for _, r in unique_players]
        print(f"Rating distribution:")
        print(f"  Min: {min(ratings)}")
        print(f"  Max: {max(ratings)}")
        print(f"  Mean: {int(sum(ratings)/len(ratings))}")

    # ── Step 2: Sample across rating bands for diversity ──────────
    print(f"\nSampling across rating bands:")
    bands = [
        (600,  1200),  # beginner
        (1200, 1600),  # casual
        (1600, 2000),  # club player
        (2000, 2400),  # advanced
        (2400, 3000),  # elite
    ]
    tournament_players = sample_by_rating_band(
        unique_players, bands, per_band=20
    )
    print(f"Total sampled: {len(tournament_players)} players")

    # ── Step 3: Add elite leaderboard players ─────────────────────
    elite = fetch_top_rapid_players(20)
    all_players = list(dict.fromkeys(tournament_players + elite))
    print(f"\nFinal player pool: {len(all_players)} players")

    # ── Step 4: Fetch games ───────────────────────────────────────
    df = fetch_all_players(all_players, max_games=100)

    if not df.empty:
        # Filter to 50+ rapid games
        counts = df.groupby("username").size()
        valid  = counts[counts >= 50].index
        df     = df[df["username"].isin(valid)]

        print(f"\nAfter filtering (50+ games):")
        print(f"  Players: {df['username'].nunique()}")
        print(f"  Games:   {len(df)}")

        df.to_csv("data/games_raw.csv", index=False)
        print("Saved to data/games_raw.csv")