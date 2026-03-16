import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = ["panic_score", "gambit_tendency", "sharp_opening", "decisive_game"]

STYLE_LABELS = {
    0: "Aggressive Attacker",
    1: "Positional Grinder", 
    2: "Tactical Opportunist",
    3: "Defensive Fortress"
}

def prepare_features(df: pd.DataFrame) -> tuple:
    """Scale features for clustering."""
    # Drop rows with missing panic_score
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[FEATURE_COLS])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
    return X, df_clean, scaler

def find_optimal_k(X: np.ndarray, max_k: int = 8) -> None:
    """
    Plot elbow curve and silhouette scores to find optimal k.
    This is what you show in interviews — you didn't just pick k=4 randomly.
    """
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Elbow curve
    ax1.plot(k_range, inertias, 'bo-', linewidth=2)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Curve — Finding Optimal k")
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_range, silhouettes, 'ro-', linewidth=2)
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score — Higher is Better")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("data/optimal_k.png", dpi=150)
    plt.show()
    
    best_k = k_range[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette score: {best_k}")
    print(f"Silhouette scores: { {k: round(s, 3) for k, s in zip(k_range, silhouettes)} }")
    return best_k

def cluster_games(X: np.ndarray, df_clean: pd.DataFrame, 
                  n_clusters: int = 4) -> pd.DataFrame:
    """Fit KMeans and assign style labels."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean = df_clean.copy()
    df_clean["cluster"] = km.fit_predict(X)
    df_clean["style"] = df_clean["cluster"].map(STYLE_LABELS)
    
    print(f"\nCluster distribution:")
    print(df_clean["cluster"].value_counts().sort_index())
    
    print(f"\nCluster centroids (scaled):")
    centroids = pd.DataFrame(
        km.cluster_centers_,
        columns=FEATURE_COLS
    )
    print(centroids.round(3))
    
    return df_clean, km

def visualize_clusters(X: np.ndarray, df_clean: pd.DataFrame) -> None:
    """
    Use PCA to reduce to 2D and visualize clusters.
    Important: PCA here is just for visualization, not for clustering.
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    explained = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: {explained[0]:.1%} + {explained[1]:.1%} = {sum(explained):.1%}")
    
    plt.figure(figsize=(10, 7))
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    
    for cluster_id in sorted(df_clean["cluster"].unique()):
        mask = df_clean["cluster"] == cluster_id
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=colors[cluster_id % len(colors)],
            label=STYLE_LABELS.get(cluster_id, f"Cluster {cluster_id}"),
            alpha=0.7, s=80
        )
    
    plt.xlabel(f"PC1 ({explained[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained[1]:.1%} variance)")
    plt.title("Chess Playing Style Clusters\n(PCA visualization)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("data/clusters.png", dpi=150)
    plt.show()

def profile_player(df_clustered: pd.DataFrame, username: str) -> None:
    """Print a summary style profile for the target player."""
    print(f"\n{'='*50}")
    print(f"STYLE PROFILE: {username}")
    print(f"{'='*50}")
    
    style_counts = df_clustered["style"].value_counts()
    dominant_style = style_counts.index[0]
    dominant_pct = style_counts.iloc[0] / len(df_clustered) * 100
    
    print(f"Dominant style: {dominant_style} ({dominant_pct:.1f}% of games)")
    print(f"\nStyle breakdown:")
    for style, count in style_counts.items():
        pct = count / len(df_clustered) * 100
        bar = "█" * int(pct / 5)
        print(f"  {style:<25} {bar} {pct:.1f}%")
    
    print(f"\nKey stats:")
    print(f"  Avg panic score:    {df_clustered['panic_score'].mean():.2f}")
    print(f"  Gambit frequency:   {df_clustered['gambit_tendency'].mean():.1%}")
    print(f"  Sharp openings:     {df_clustered['sharp_opening'].mean():.1%}")
    print(f"  Decisive games:     {df_clustered['decisive_game'].mean():.1%}")


if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")
    
    # Step 1: Scale
    X, df_clean, scaler = prepare_features(df)
    
    # Step 2: Find optimal k
    print("Finding optimal number of clusters...")
    best_k = find_optimal_k(X)
    
    # Step 3: Cluster with best k
    print(f"\nClustering with k={best_k}...")
    df_clustered, model = cluster_games(X, df_clean, n_clusters=best_k)
    
    # Step 4: Visualize
    visualize_clusters(X, df_clustered)
    
    # Step 5: Profile
    profile_player(df_clustered, "DrNykterstein")
    
    # Save
    df_clustered.to_csv("data/clustered_games.csv", index=False)
    print("\nSaved to data/clustered_games.csv")