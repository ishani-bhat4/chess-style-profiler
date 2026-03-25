import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
# matplotlib imported lazily — not needed for prediction, only for training
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "panic_score_mean", "panic_score_std", "time_per_move_mean",
    "capture_rate_mean", "capture_rate_std", "sacrifice_rate_mean",
    "check_rate_mean", "checks_received_mean",
    "early_aggression_mean", "piece_activity_mean", "avg_material_diff",
    "castle_rate", "pawn_shield_mean", "king_safety_score",
    "gambit_rate", "sharp_opening_rate", "opening_entropy",
    "top_opening_pct", "unique_openings",
    "win_rate", "draw_rate", "avg_game_length",
]

# ── PREP ────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple:
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[FEATURE_COLS])
    print(f"Feature matrix: {X.shape}, dropped {len(df)-len(df_clean)} rows")
    return X, df_clean, scaler

# ── PCA ─────────────────────────────────────────────────────────────
def reduce_dimensions(X: np.ndarray,
                      variance_threshold: float = 0.90) -> tuple:
    pca = PCA(n_components=variance_threshold, random_state=42)
    X_reduced = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    print(f"\nPCA Dimensionality Reduction:")
    print(f"  Original dimensions : {X.shape[1]}")
    print(f"  Reduced dimensions  : {X_reduced.shape[1]}")
    print(f"  Variance retained   : {sum(explained):.1%}")
    print(f"  Per component       : {[round(e, 3) for e in explained]}")
    return X_reduced, pca

# ── KMEANS ──────────────────────────────────────────────────────────
def run_kmeans(X: np.ndarray, k: int = 5) -> tuple:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    print(f"\nKMeans (k={k})")
    print(f"  Silhouette score : {sil:.3f}  (higher = better, max 1.0)")
    print(f"  Davies-Bouldin   : {db:.3f}  (lower  = better, min 0.0)")
    print(f"  Cluster sizes    : {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, sil, db, model

# ── GMM ─────────────────────────────────────────────────────────────
def run_gmm(X: np.ndarray, k: int = 5) -> tuple:
    model = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    probs = model.predict_proba(X)
    avg_confidence = np.max(probs, axis=1).mean()
    print(f"\nGaussian Mixture Model (k={k})")
    print(f"  Silhouette score    : {sil:.3f}")
    print(f"  Davies-Bouldin      : {db:.3f}")
    print(f"  Avg confidence      : {avg_confidence:.3f}")
    print(f"  Cluster sizes       : {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, sil, db, model, probs

# ── DBSCAN WITH AUTO EPS TUNING ──────────────────────────────────────
def tune_dbscan(X: np.ndarray) -> tuple:
    """
    Find optimal DBSCAN eps using k-distance plot.
    Try multiple eps values and pick the one with best silhouette.
    """
    import matplotlib.pyplot as plt
    # Use 4-NN distances to find reasonable eps range
    nbrs = NearestNeighbors(n_neighbors=4).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = np.sort(distances[:, 3])

    # Plot k-distance curve
    plt.figure(figsize=(8, 4))
    plt.plot(distances, 'b-', linewidth=2)
    plt.title("K-Distance Plot (k=4) — Finding DBSCAN Epsilon")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("4-NN Distance")
    plt.grid(True, alpha=0.3)
    plt.savefig("data/dbscan_eps.png", dpi=150)
    plt.show()

    # Try eps at various percentiles
    eps_candidates = np.percentile(distances, [60, 65, 70, 75, 80, 85, 90])

    best_sil    = -1
    best_eps    = eps_candidates[4]  # default to 80th percentile
    best_labels = None

    print("\nDBSCAN eps tuning:")
    for eps in eps_candidates:
        model  = DBSCAN(eps=eps, min_samples=3)
        labels = model.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = (labels == -1).sum()

        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > n_clusters:
                sil = silhouette_score(X[mask], labels[mask])
                print(f"  eps={eps:.3f}: {n_clusters} clusters, "
                      f"{n_outliers} outliers, silhouette={sil:.3f}")
                if sil > best_sil:
                    best_sil    = sil
                    best_eps    = eps
                    best_labels = labels.copy()
        else:
            print(f"  eps={eps:.3f}: only {n_clusters} cluster — too tight")

    print(f"\nBest eps: {best_eps:.3f} (silhouette={best_sil:.3f})")
    return best_eps, best_labels

def run_dbscan(X: np.ndarray, eps: float = 0.8,
               min_samples: int = 3) -> tuple:
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()

    print(f"\nDBSCAN (eps={eps:.3f}, min_samples={min_samples})")
    print(f"  Clusters found : {n_clusters}")
    print(f"  Outliers found : {n_outliers}")

    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X[mask], labels[mask]) if mask.sum() > n_clusters else 0
        db  = davies_bouldin_score(X[mask], labels[mask]) if mask.sum() > n_clusters else 0
        print(f"  Silhouette score : {sil:.3f}")
        print(f"  Davies-Bouldin   : {db:.3f}")
        print(f"  Cluster sizes    : {pd.Series(labels).value_counts().sort_index().to_dict()}")
        return labels, sil, db, model
    else:
        print("  Warning: DBSCAN found only 1 cluster")
        return labels, 0, 0, model

# ── HIERARCHICAL ────────────────────────────────────────────────────
def run_hierarchical(X: np.ndarray, k: int = 5) -> tuple:
    model  = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    print(f"\nHierarchical Clustering (k={k}, ward linkage)")
    print(f"  Silhouette score : {sil:.3f}")
    print(f"  Davies-Bouldin   : {db:.3f}")
    print(f"  Cluster sizes    : {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, sil, db, model

# ── FIND OPTIMAL K ──────────────────────────────────────────────────
def find_optimal_k(X: np.ndarray, max_k: int = 8) -> int:
    import matplotlib.pyplot as plt
    inertias, silhouettes, bic_scores = [], [], []
    k_range = range(2, max_k + 1)

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(k_range, inertias,   'bo-', linewidth=2)
    axes[0].set_title("KMeans: Elbow Curve")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2)
    axes[1].set_title("KMeans: Silhouette Score")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Score (higher = better)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(k_range, bic_scores,  'go-', linewidth=2)
    axes[2].set_title("GMM: BIC Score")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("BIC (lower = better)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/optimal_k.png", dpi=150)
    plt.show()

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette: {best_k}")
    print(f"Best k by BIC (GMM):  {list(k_range)[np.argmin(bic_scores)]}")
    return best_k

# ── COMPARISON PLOT ─────────────────────────────────────────────────
def plot_comparison(X: np.ndarray, results: dict) -> None:
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Clustering Model Comparison — Chess Style Profiler",
                 fontsize=14, fontweight="bold")

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12",
              "#9B59B6", "#1ABC9C", "#E67E22"]

    for ax, (name, labels) in zip(axes.flatten(), results.items()):
        for i, label in enumerate(sorted(set(labels))):
            mask  = np.array(labels) == label
            color = "#AAAAAA" if label == -1 else colors[i % len(colors)]
            lab   = "Outlier" if label == -1 else f"Cluster {label}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color, label=lab, alpha=0.7, s=60)
        ax.set_title(name)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("data/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

# ── SUMMARY TABLE ───────────────────────────────────────────────────
def print_summary(scores: list) -> None:
    print("\n" + "="*65)
    print(f"{'MODEL COMPARISON SUMMARY':^65}")
    print("="*65)
    print(f"{'Model':<30} {'Silhouette':>12} {'Davies-Bouldin':>15}")
    print("-"*65)
    for name, sil, db in scores:
        print(f"{name:<30} {sil:>12.3f} {db:>15.3f}")
    print("-"*65)
    print("Silhouette: higher is better (max 1.0)")
    print("Davies-Bouldin: lower is better (min 0.0)")
    best = max(scores, key=lambda x: x[1])
    print(f"\nWinner: {best[0]} (silhouette={best[1]:.3f})")
# Add this to clustering.py

CLUSTER_NAMES = {
    0: "Chaotic Attacker",
    1: "Solid Strategist", 
    2: "Time Pressure Wildcard",
    3: "Passive Defender"
}

CLUSTER_DESCRIPTIONS = {
    0: "You play aggressively and create complications, but inconsistent king safety costs you games. Focus on castling early and converting your attacks.",
    1: "You play principled, well-rounded chess. You castle reliably, manage time well, and have a broad opening repertoire. Work on converting small advantages.",
    2: "Your biggest challenge is time management. You slow down dramatically in critical positions. Practice faster decision-making in time pressure situations.",
    3: "You play cautiously and avoid complications. This is safe but limits your winning chances. Try introducing more active piece play and tactical patterns."
}
# ── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/features_player_level.csv")
    X, df_clean, scaler = prepare_features(df)

    # PCA
    X_reduced, pca_model = reduce_dimensions(X, variance_threshold=0.90)

    # Find optimal k
    print("\nFinding optimal k...")
    best_k = find_optimal_k(X_reduced)

    # Tune DBSCAN eps
    best_eps, db_labels_tuned = tune_dbscan(X_reduced)

    # Run all models
    print(f"\nRunning all models with k={best_k}...")
    km_labels,  km_sil,  km_db,  km_model         = run_kmeans(X_reduced, best_k)
    gmm_labels, gmm_sil, gmm_db, gmm_model, probs  = run_gmm(X_reduced, best_k)
    db_labels,  db_sil,  db_db,  db_model          = run_dbscan(X_reduced, eps=best_eps)
    hc_labels,  hc_sil,  hc_db,  hc_model          = run_hierarchical(X_reduced, best_k)

    # Comparison plot
    plot_comparison(X_reduced, {
        f"KMeans (k={best_k})":       km_labels,
        f"GMM (k={best_k})":          gmm_labels,
        f"DBSCAN (eps={best_eps:.2f})": db_labels,
        f"Hierarchical (k={best_k})": hc_labels,
    })

    # Summary
    print_summary([
        ("KMeans",       km_sil,  km_db),
        ("GMM",          gmm_sil, gmm_db),
        ("DBSCAN",       db_sil,  db_db),
        ("Hierarchical", hc_sil,  hc_db),
    ])

    # Save results
    df_clean["cluster_kmeans"]       = km_labels
    df_clean["cluster_gmm"]          = gmm_labels
    df_clean["cluster_dbscan"]       = db_labels
    df_clean["cluster_hierarchical"] = hc_labels

    for i in range(probs.shape[1]):
        df_clean[f"gmm_prob_cluster_{i}"] = probs[:, i]

    df_clean.to_csv("data/clustered_players.csv", index=False)
    print("\nSaved to data/clustered_players.csv")

    print(f"\nPlayer cluster assignments (KMeans):")
    print(df_clean[["username", "cluster_kmeans"]].sort_values(
        "cluster_kmeans").to_string())
    # Save models for the predictor
    import joblib
    joblib.dump(km_model, "data/kmeans_model.pkl")
    joblib.dump(scaler,   "data/scaler.pkl")
    joblib.dump(pca_model,"data/pca_model.pkl")
    print("Models saved to data/")