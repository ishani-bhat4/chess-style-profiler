import requests
import pandas as pd
import chess.pgn
import io
from tqdm import tqdm

def fetch_games(username: str, max_games: int = 200) -> list[dict]:
    """
    Fetch games for a Lichess user via the Lichess API.
    Returns a list of dicts, one per game.
    """
    url = f"https://lichess.org/api/games/user/{username}"
    
    params = {
        "max": max_games,
        "clocks": True,      # include clock times per move
        "evals": False,      # skip engine evals for now (slow)
        "opening": True,     # include opening name
        "moves": True,       # include move list
    }
    
    headers = {
        "Accept": "application/x-ndjson"  # Lichess streams as newline-delimited JSON
    }
    
    response = requests.get(url, params=params, headers=headers, stream=True)
    
    games = []
    for line in tqdm(response.iter_lines(), desc="Fetching games", total=max_games):
        if line:
            import json
            games.append(json.loads(line))
    
    print(f"Fetched {len(games)} games for {username}")
    return games


def parse_games(games: list[dict]) -> pd.DataFrame:
    """
    Parse raw Lichess API response into a clean DataFrame.
    Each row = one game.
    """
    rows = []
    
    for g in games:
        try:
            players = g.get("players", {})
            white = players.get("white", {})
            black = players.get("black", {})
            
            row = {
                # Game metadata
                "game_id":        g.get("id"),
                "rated":          g.get("rated"),
                "speed":          g.get("speed"),        # bullet/blitz/rapid
                "time_control":   g.get("clock", {}).get("initial"),
                "increment":      g.get("clock", {}).get("increment"),

                # Players
                "white_user":     white.get("user", {}).get("name"),
                "black_user":     black.get("user", {}).get("name"),
                "white_rating":   white.get("rating"),
                "black_rating":   black.get("rating"),

                # Outcome
                "winner":         g.get("winner"),       # "white", "black", or None (draw)
                "status":         g.get("status"),       # "mate", "resign", "timeout" etc

                # Opening
                "opening_name":   g.get("opening", {}).get("name"),
                "opening_eco":    g.get("opening", {}).get("eco"),
                "opening_ply":    g.get("opening", {}).get("ply"),

                # Moves
                "moves":          g.get("moves", ""),    # space-separated move string
                "num_moves":      len(g.get("moves", "").split()),

                # Clock times (list of seconds remaining per move)
                "clocks":         g.get("clocks", []),
            }
            rows.append(row)
        except Exception as e:
            print(f"Skipping game due to error: {e}")
            continue
    
    df = pd.DataFrame(rows)
    print(f"Parsed {len(df)} games into DataFrame")
    print(df.dtypes)
    return df


if __name__ == "__main__":
    # Test it right now with Magnus
    USERNAME = "DrNykterstein"
    
    games_raw = fetch_games(USERNAME, max_games=100)
    df = parse_games(games_raw)
    
    # Save to data folder
    df.to_csv("data/games_raw.csv", index=False)
    print("\nSample:")
    print(df[["game_id", "speed", "white_user", "black_user", 
              "winner", "status", "opening_name", "num_moves"]].head())