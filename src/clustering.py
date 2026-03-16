import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "panic_score", "time_per_move",
    "capture_rate", "sacrifice_rate", "check_rate",
    "early_aggression", "castle_move", "piece_activity",
    "avg_material_diff", "gambit_tendency", "sharp_opening",
    "decisive_game"
]

# ── PREP ────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame) -> tuple:
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[FEATURE_COLS])
    print(f"Feature matrix: {X.shape}, dropped {len(df)-len(df_clean)} rows")
    return X, df_clean, scaler

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
    
    # GMM gives us soft probabilities — this is unique to GMM
    probs = model.predict_proba(X)
    avg_confidence = np.max(probs, axis=1).mean()
    
    print(f"\nGaussian Mixture Model (k={k})")
    print(f"  Silhouette score    : {sil:.3f}")
    print(f"  Davies-Bouldin      : {db:.3f}")
    print(f"  Avg confidence      : {avg_confidence:.3f}  (how certain assignments are)")
    print(f"  Cluster sizes       : {pd.Series(labels).value_counts().sort_index().to_dict()}")
    return labels, sil, db, model, probs

# ── DBSCAN ──────────────────────────────────────────────────────────
def run_dbscan(X: np.ndarray, eps: float = 0.8, min_samples: int = 3) -> tuple:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    
    print(f"\nDBSCAN (eps={eps}, min_samples={min_samples})")
    print(f"  Clusters found : {n_clusters}  (discovered automatically — no k needed)")
    print(f"  Outliers found : {n_outliers}  (games that fit no style)")
    
    if n_clusters > 1:
        # Only compute silhouette on non-outlier points
        mask = labels != -1
        sil = silhouette_score(X[mask], labels[mask]) if mask.sum() > 1 else 0
        db  = davies_bouldin_score(X[mask], labels[mask]) if mask.sum() > 1 else 0
        print(f"  Silhouette score : {sil:.3f}")
        print(f"  Davies-Bouldin   : {db:.3f}")
        print(f"  Cluster sizes    : {pd.Series(labels).value_counts().sort_index().to_dict()}")
        return labels, sil, db, model
    else:
        print("  Warning: DBSCAN found only 1 cluster — try adjusting eps")
        return labels, 0, 0, model

# ── HIERARCHICAL ────────────────────────────────────────────────────
def run_hierarchical(X: np.ndarray, k: int = 5) -> tuple:
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
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
    inertias, silhouettes, bic_scores = [], [], []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        # KMeans metrics
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km_labels))
        
        # GMM BIC — another way to choose k
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
    axes[0].set_title("KMeans: Elbow Curve")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2)
    axes[1].set_title("KMeans: Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score (higher = better)")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(k_range, bic_scores, 'go-', linewidth=2)
    axes[2].set_title("GMM: BIC Score")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("BIC (lower = better)")
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
    """
    Side-by-side PCA visualization of all 4 models.
    This is the money plot for your portfolio.
    """
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Clustering Model Comparison — Chess Style Profiler", 
                 fontsize=14, fontweight='bold')
    
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    
    for ax, (name, labels) in zip(axes.flatten(), results.items()):
        unique_labels = sorted(set(labels))
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            color = "#AAAAAA" if label == -1 else colors[i % len(colors)]
            lab = "Outlier" if label == -1 else f"Cluster {label}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=color, label=lab, alpha=0.7, s=60)
        ax.set_title(name)
        ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("data/model_comparison.png", dpi=150, bbox_inches='tight')
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

# ── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")
    X, df_clean, scaler = prepare_features(df)
    
    # Find optimal k first
    print("Finding optimal k...")
    best_k = find_optimal_k(X)
    
    # Run all models
    print(f"\nRunning all models with k={best_k}...")
    
    km_labels,  km_sil,  km_db,  km_model          = run_kmeans(X, best_k)
    gmm_labels, gmm_sil, gmm_db, gmm_model, probs   = run_gmm(X, best_k)
    db_labels,  db_sil,  db_db,  db_model           = run_dbscan(X)
    hc_labels,  hc_sil,  hc_db,  hc_model           = run_hierarchical(X, best_k)
    
    # Comparison plot
    plot_comparison(X, {
        f"KMeans (k={best_k})":        km_labels,
        f"GMM (k={best_k})":           gmm_labels,
        "DBSCAN (auto k)":             db_labels,
        f"Hierarchical (k={best_k})":  hc_labels,
    })
    
    # Summary
    print_summary([
        ("KMeans",        km_sil,  km_db),
        ("GMM",           gmm_sil, gmm_db),
        ("DBSCAN",        db_sil,  db_db),
        ("Hierarchical",  hc_sil,  hc_db),
    ])
    
    # Save best model results
    df_clean["cluster_kmeans"]      = km_labels
    df_clean["cluster_gmm"]         = gmm_labels
    df_clean["cluster_dbscan"]      = db_labels
    df_clean["cluster_hierarchical"]= hc_labels
    
    # GMM soft probabilities — unique value of GMM
    for i in range(probs.shape[1]):
        df_clean[f"gmm_prob_cluster_{i}"] = probs[:, i]
    
    df_clean.to_csv("data/clustered_games.csv", index=False)
    print("\nSaved all model results to data/clustered_games.csv")