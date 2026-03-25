# Chess Style Profiler

An unsupervised machine learning system that fingerprints chess playing 
styles from game history and recommends personalized practice puzzles.

Enter a Lichess username → get a style profile → get puzzles matched 
to your weaknesses.

---

## What It Does

- Fetches rapid games for any Lichess username via the public API
- Engineers 21 behavioral features from move sequences, clock data, 
  and opening choices
- Aggregates game-level features into a player profile
- Clusters players into style archetypes using KMeans, GMM, DBSCAN, 
  and Hierarchical clustering
- Recommends Lichess puzzles matched to detected weaknesses

---

## Features Engineered

| Category | Features |
|----------|----------|
| Timing | Panic score, time per move |
| Move-level | Capture rate, sacrifice rate, check rate, early aggression, piece activity, material difference |
| Castle | Castle rate, avg castle move number |
| Opening | Gambit rate, sharp opening rate, opening entropy, top opening %, unique openings |
| Outcomes | Win rate, draw rate, avg game length |

---

## Current Results

- **Dataset**: 97 players, 9,496 rapid games across all rating levels
- **Best model**: DBSCAN — silhouette 0.337, Davies-Bouldin 0.933
- **Clusters found**: 4 meaningful player archetypes
  - Cluster 1: Strong experienced players (win rate 0.62–0.97, castle rate 0.83–0.92)
  - Cluster 3: Developing players (win rate 0.23–0.50, castle rate 0.35–0.40)

---

## Key Design Decisions

- **Rapid over blitz**: Blitz games introduce time-pressure noise that 
  conflates panic behavior with style. Rapid games give players enough 
  time to execute their actual style.
- **Player-level aggregation**: Style is a property of a player, not a 
  single game. 100 games per player are aggregated into one feature row.
- **Arena tournament sampling**: Lichess leaderboard players are all 
  2000+ rated. Arena tournaments attract all rating levels, giving the 
  diversity needed for meaningful clustering.
- **Castle split**: `castle_move` was split into `castle_rate` + 
  `avg_castle_move` because a player who never castles is making a 
  deliberate stylistic choice — not a missing value.

---

## Stack
```
Python 3.11
scikit-learn    — clustering, PCA, scaling, metrics
pandas / numpy  — data manipulation
python-chess    — move parsing and board state replay
scipy           — opening entropy calculation
matplotlib      — visualization
Lichess API     — game data (open, no auth required)
streamlit       — web app (https://chess-style-profiler.streamlit.app/)
```

---

## Run It
```bash
pip install -r requirements.txt

# 1. Fetch games for diverse player pool
python src/data_loader.py

# 2. Engineer features
python src/features.py

# 3. Cluster and evaluate
python src/clustering.py
```

---

## Project Structure
```
chess-style-profiler/
├── data/
│   ├── games_raw.csv               # Raw API data
│   ├── features_game_level.csv     # Per-game features
│   ├── features_player_level.csv   # Aggregated player profiles
│   └── clustered_players.csv       # Final cluster assignments
├── src/
│   ├── data_loader.py              # Lichess API pipeline
│   ├── features.py                 # Feature engineering
│   ├── clustering.py               # Clustering + evaluation
│   └── recommender.py              # Puzzle matching (coming soon)
├── app.py                          # Streamlit app (coming soon)
└── requirements.txt
```

---

## Documentation

Full technical documentation including ML concepts, every design 
decision, and the complete iteration history is in 
`chess_style.pdf`.
