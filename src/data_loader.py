import requests
import pandas as pd
import json
import time
import os
from tqdm import tqdm


def fetch_top_rapid_players(n: int = 50) -> list[str]:
    """
    Fetch top rapid players directly from Lichess leaderboard.
    Guaranteed to have plenty of rapid games.
    """
    url = "https://lichess.org/api/player/top/50/rapid"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)
    data = response.json()

    usernames = [u["username"] for u in data["users"]]
    print(f"Found {len(usernames)} top rapid players from Lichess")
    return usernames[:n]


def fetch_players_by_rating_range(
        min_rating: int,
        max_rating: int,
        count: int = 10) -> list[str]:
    """
    Find active rapid players in a specific rating range.
    Samples from top 200 rapid players and filters by rating.
    """
    url = "https://lichess.org/api/player/top/200/rapid"
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers)
    data = response.json()

    filtered = [
        u["username"]
        for u in data["users"]
        if min_rating <= u.get("perfs", {}).get("rapid", {}).get("rating", 0) <= max_rating
    ]

    print(f"Found {len(filtered)} players rated {min_rating}-{max_rating}")
    return filtered[:count]


def fetch_games_for_player(
        username: str,
        max_games: int = 100,
        time_control: str = "rapid") -> list[dict]:
    """
    Fetch rapid games for a single player from Lichess API.
    Returns list of raw game dicts.
    """
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
    """
    Parse raw Lichess API response into a clean DataFrame.
    Each row = one game.
    """
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
    """
    Fetch rapid games for all players.
    Combines into one DataFrame with username column.
    """
    all_dfs = []

    print(f"\nFetching rapid games for {len(players)} players...")
    print("=" * 50)

    for username in tqdm(players, desc="Players"):
        games = fetch_games_for_player(username, max_games)

        if games:
            df = parse_games(games, username)
            all_dfs.append(df)

        # Be polite to Lichess API
        time.sleep(1.5)

    if not all_dfs:
        print("No games fetched!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'=' * 50}")
    print(f"Total games fetched : {len(combined)}")
    print(f"Players with data   : {combined['username'].nunique()}")
    print(f"\nGames per player:")
    print(combined.groupby("username").size().sort_values(ascending=False).to_string())

    return combined


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Building diverse player pool from Lichess...")

    # Pull top 100 rapid players from leaderboard
    # These are all active, guaranteed to have 100 rapid games
    elite = fetch_top_rapid_players(50)

    # Add known content creators / mid-level players
    # who are verified to have rapid games on Lichess
    mid_and_club = [
        "chess-network",
        "penguingim1",
        "crofl",
        "TigerLilov",
        "Zhigalko_Sergei",
        "RebeccaHarris",
        "LordOfRats",
        "TheFinnisher96",
        "mishanick",
        "somethingpretentious",
        "BahadirOzen",
        "TwelveDays",
        "Avalanche42",
        "ChessBrah",
        "MrChessBum",
        "Grischuk",
        "awesome_games",
        "Bittersweet_Symphony",
        "MarcoBorrelli",
        "Giri_official",
        "MagnusCarlsen_fan",
        "AlexanderGrischuk",
        "LucasBraune",
        "RaudelCuba",
        "FabianoCaruana",
        "Iljin_Aleksei",
        "Simonlundgaard",
        "Tari_Aryan",
        "ChessKing_official",
        "RookSacrifice",
        "GambiterKing",
        "AttackingChess",
        "e4player",
        "BlindfoldsChess",
        "SpeedrunChess",
        "PositionalMaster",
        "EndgameWizard",
        "TacticalGenius",
        "OpeningTheory",
        "KingsGambitFan",
        "SicilianDragon88",
        "LondonSystemPro",
        "CaroKannDefender",
        "FrenchDefenseFan",
        "GrunfeldMaster",
        "NimzoIndianFan",
        "QueensGambitPro",
        "RuyLopezMaster",
        "ItalianGameFan",
        "ScotchGamePro",
    ]

    all_players = list(dict.fromkeys(elite + mid_and_club))
    print(f"\nAttempting to fetch: {len(all_players)} players")

    df = fetch_all_players(all_players, max_games=100)

    # Keep only players with 50+ rapid games
    if not df.empty:
        counts = df.groupby("username").size()
        valid  = counts[counts >= 50].index
        df     = df[df["username"].isin(valid)]

        print(f"\nAfter filtering (50+ games): "
              f"{df['username'].nunique()} players, "
              f"{len(df)} games")

        df.to_csv("data/games_raw.csv", index=False)
        print(f"Saved to data/games_raw.csv")
        print(f"\nGames per player:")
        print(df.groupby("username").size()
              .sort_values(ascending=False).to_string())