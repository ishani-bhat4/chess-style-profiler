# Chess Style Profiler

A data science project that fingerprints chess playing styles 
from game history and recommends personalized practice.

## What it does
- Pulls games from Lichess API for any username
- Engineers behavioral features (time pressure, opening tendencies)
- Clusters games into style profiles using KMeans
- (Coming soon) Recommends puzzles based on detected weaknesses

## Current Results
- Magnus Carlsen (DrNykterstein): 68% Positional Grinder
- Silhouette score: 0.776 (k=5)

## Stack
Python, scikit-learn, pandas, python-chess, Lichess API

## Run it
pip install -r requirements.txt
python src/data_loader.py
python src/features.py  
python src/clustering.py
