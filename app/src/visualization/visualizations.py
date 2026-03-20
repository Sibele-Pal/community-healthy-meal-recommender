#!/usr/bin/env python3
"""
visualizations.py

Single-file visualization suite for the Healthy Food Recommender project.

Place at: src/visualization/visualizations.py

Saves PNGs to: outputs/visualizations/

Usage (example):
    python src/visualization/visualizations.py --sample 5000 --skip-mf False --train-log path/to/train_log.csv

Install requirements first:
    pip install pandas matplotlib networkx scikit-learn pyarrow torch

Author: generated for your project
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

# Optional
try:
    import torch
except Exception:
    torch = None

# ---- Configurable paths (relative to project root) ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "src", "models_artifacts")  # adjust if models stored elsewhere
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "visualizations")

# default parquet filenames your pipeline uses
DEFAULT_FILES = {
    "master": "master_food_catalog.parquet",
    "interactions": "user_interactions.parquet",
    "embeddings": "item_embeddings.parquet",
    "nutrition_features": "item_nutrition_features.parquet",
    "text_features": "item_text_features.parquet",
    "centrality": "item_centrality.parquet",
    "bipartite": "bipartite_edges.parquet",
}

# nutritional columns candidates (common names used in your project)
NUTRI_CANDIDATES = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
    "carbohydrates_100g", "sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g"
]

# ============================
#  MF MODEL LOSS VALUES
# ============================
mf_epochs = [1, 2, 3, 4, 5]
mf_losses = [2.2984, 2.0932, 1.9034, 1.7564, 1.6007]

# ============================
#  SEQUENCE MODEL LOSS VALUES
# ============================
seq_epochs = [1, 2, 3, 4, 5]
seq_losses = [11.5592, 11.4099, 11.1391, 10.7248, 10.0506]


# ============================
#  HYBRID (PSEUDO) LOSS CONFIG
# ============================
MF_WEIGHT = 0.6
SEQ_WEIGHT = 0.2


# ---- Helpers ----
def ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def try_read_parquet(path):
    if not os.path.exists(path):
        print(f"[WARN] missing file: {path}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[ERROR] cannot read parquet {path}: {e}")
        print("Install pyarrow (recommended): pip install pyarrow")
        return None

def save_fig(fig, fname):
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("Saved:", path)

def hist_plot(values, title, xlabel, fname, bins=40):
    fig = plt.figure(figsize=(8,4))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_fig(fig, fname)

# ---- Visualization routines ----
def plot_region_distribution(master_df):
    if master_df is None or "region" not in master_df.columns:
        print("[SKIP] region distribution (no master or region col)")
        return
    counts = master_df['region'].fillna("unknown").value_counts()
    fig = plt.figure(figsize=(10,5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Food Count by Region")
    plt.xlabel("Region")
    plt.ylabel("Count")
    save_fig(fig, "region_count_by_region.png")

def plot_nutrient_histograms(master_df, nutri_list):
    if master_df is None:
        return
    cols = [c for c in nutri_list if c in master_df.columns]
    if not cols:
        print("[SKIP] nutrient histograms (no candidate columns found)")
        return
    for c in cols:
        vals = pd.to_numeric(master_df[c].fillna(0), errors='coerce').values
        hist_plot(vals, f"Distribution of {c}", c, f"hist_{c.replace('/','_')}.png")

def plot_nutrition_corr(master_df, nutri_list):
    if master_df is None:
        return
    cols = [c for c in nutri_list if c in master_df.columns]
    if len(cols) < 2:
        return
    corr = master_df[cols].apply(pd.to_numeric, errors='coerce').corr()
    fig = plt.figure(figsize=(6,6))
    plt.imshow(corr.values, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha='right')
    plt.yticks(range(len(cols)), cols)
    plt.title("Nutrition Feature Correlation Heatmap")
    save_fig(fig, "nutrition_correlation_heatmap.png")

def plot_pca_on_columns(df, col_list, title, fname, sample=None):
    if df is None:
        return
    cols = [c for c in col_list if c in df.columns]
    if len(cols) < 2:
        return
    X = df[cols].fillna(0).values
    if sample and X.shape[0] > sample:
        idx = np.random.choice(X.shape[0], sample, replace=False)
        X = X[idx]
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(X)
    fig = plt.figure(figsize=(8,6))
    plt.scatter(XY[:,0], XY[:,1], s=6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(fig, fname)

def plot_item_embeddings_pca(emb_df, sample=None):
    if emb_df is None:
        return
    emb_cols = [c for c in emb_df.columns if str(c).startswith("emb_") or str(c).startswith("text_emb_") or str(c).startswith("dim_")]
    if not emb_cols:
        print("[SKIP] embeddings PCA (no embedding columns found)")
        return
    E = emb_df[emb_cols].fillna(0).values
    if sample and E.shape[0] > sample:
        idx = np.random.choice(E.shape[0], sample, replace=False)
        E = E[idx]
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(E)
    fig = plt.figure(figsize=(8,6))
    plt.scatter(XY[:,0], XY[:,1], s=6)
    plt.title("PCA (2D) of Item Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(fig, "pca_item_embeddings.png")

def plot_centrality_hist(centrality_df):
    if centrality_df is None:
        return
    col = None
    for c in ["degree_centrality","centrality","degree","num_users"]:
        if c in centrality_df.columns:
            col = c
            break
    if col is None:
        print("[SKIP] centrality hist (no known centrality col)")
        return
    vals = pd.to_numeric(centrality_df[col].fillna(0), errors='coerce').values
    hist_plot(vals, f"Item centrality ({col})", col, "item_centrality_hist.png")

def plot_popularity_activity(inter_df):
    if inter_df is None:
        return
    if "food_id" in inter_df.columns:
        fcounts = inter_df['food_id'].value_counts().values
        hist_plot(fcounts, "Food popularity (interactions per food)", "Interactions per food", "food_popularity_hist.png")
    if "user_id" in inter_df.columns:
        ucounts = inter_df['user_id'].value_counts().values
        hist_plot(ucounts, "User interaction counts", "Interactions per user", "user_interaction_counts_hist.png")

def plot_bipartite_sample(bip_df, sample_nodes=300):
    if bip_df is None or 'user_id' not in bip_df.columns or 'food_id' not in bip_df.columns:
        print("[SKIP] bipartite graph (missing bipartite file or columns)")
        return
    users = bip_df['user_id'].unique().tolist()
    foods = bip_df['food_id'].unique().tolist()
    half = sample_nodes//2
    s_users = users if len(users) <= half else random.sample(users, half)
    s_foods = foods if len(foods) <= half else random.sample(foods, half)
    sub = bip_df[bip_df['user_id'].isin(s_users) & bip_df['food_id'].isin(s_foods)]
    G = nx.Graph()
    for u in s_users:
        G.add_node(f"user_{int(u)}", bipartite=0)
    for f in s_foods:
        G.add_node(f"food_{int(f)}", bipartite=1)
    for row in sub.itertuples(index=False):
        G.add_edge(f"user_{int(row.user_id)}", f"food_{int(row.food_id)}")
    if len(G.nodes) == 0:
        print("[WARN] sampled bipartite graph empty")
        return
    users_b = [n for n,d in G.nodes(data=True) if d.get('bipartite',None)==0]
    try:
        pos = nx.bipartite_layout(G, users_b)
    except Exception:
        pos = nx.spring_layout(G, k=0.15, seed=42)
    fig = plt.figure(figsize=(12,8))
    nx.draw(G, pos, node_size=20, with_labels=False)
    plt.title("Sampled User-Food Bipartite Graph")
    save_fig(fig, "bipartite_graph_sample.png")

def inspect_mf_model(mf_path):
    if not os.path.exists(mf_path):
        print("[SKIP] MF model not found at " + mf_path)
        return
    if torch is None:
        print("[SKIP] torch not available; cannot inspect MF model")
        return
    try:
        ckpt = torch.load(mf_path, map_location='cpu')
    except Exception as e:
        print("[WARN] failed to load MF model:", e)
        return
    # Try heuristics to find 2D embedding matrices
    user_factors = None
    item_factors = None
    if isinstance(ckpt, dict):
        # try common keys
        for k,v in ckpt.items():
            try:
                if 'user' in k and v is not None and hasattr(v, 'numpy') or hasattr(v, 'cpu'):
                    arr = v.cpu().numpy() if hasattr(v, 'cpu') else np.array(v)
                    if arr.ndim==2 and arr.shape[0] > 1 and user_factors is None:
                        user_factors = arr
                if 'item' in k and v is not None and hasattr(v, 'numpy') or hasattr(v, 'cpu'):
                    arr = v.cpu().numpy() if hasattr(v, 'cpu') else np.array(v)
                    if arr.ndim==2 and arr.shape[0] > 1 and item_factors is None:
                        item_factors = arr
            except Exception:
                continue
        # fallback: take two largest 2D arrays
        if (user_factors is None or item_factors is None):
            arrays = []
            for k,v in ckpt.items():
                try:
                    arr = v.cpu().numpy() if hasattr(v, 'cpu') else np.array(v)
                    if arr.ndim==2 and min(arr.shape) > 1:
                        arrays.append((k, arr.size, arr))
                except Exception:
                    pass
            arrays = sorted(arrays, key=lambda x: x[1], reverse=True)
            if len(arrays) >= 2:
                user_factors = arrays[0][2]
                item_factors = arrays[1][2]
    if user_factors is None or item_factors is None:
        print("[WARN] Could not identify user/item factor matrices inside MF checkpoint")
        return
    # visualize norms & sample dot products
    u_norms = np.linalg.norm(user_factors, axis=1)
    v_norms = np.linalg.norm(item_factors, axis=1)
    hist_plot(u_norms, "User factor norms (MF)", "L2 norm", "mf_user_factor_norms.png")
    hist_plot(v_norms, "Item factor norms (MF)", "L2 norm", "mf_item_factor_norms.png")
    # sampled dot products
    n = min(500, user_factors.shape[0]*item_factors.shape[0])
    dots = []
    for _ in range(n):
        ui = random.randrange(user_factors.shape[0])
        ii = random.randrange(item_factors.shape[0])
        dots.append(np.dot(user_factors[ui], item_factors[ii]))
    hist_plot(dots, "Sampled MF user-item dot products", "dot(u,v)", "mf_dot_product_sample_hist.png")

def plot_training_logs(train_log_csv):
    """
    Accepts CSV with columns: epoch, train_loss, val_loss, train_metric, val_metric (optional)
    """
    if not train_log_csv or not os.path.exists(train_log_csv):
        print("[SKIP] training log plotting (no file)")
        return
    try:
        df = pd.read_csv(train_log_csv)
    except Exception as e:
        print("[WARN] could not read training log CSV:", e)
        return
    if 'epoch' not in df.columns:
        print("[WARN] training log CSV missing 'epoch' column")
        return
    epochs = df['epoch'].values
    fig = plt.figure(figsize=(8,5))
    if 'train_loss' in df.columns:
        plt.plot(epochs, df['train_loss'].values, label='train_loss')
    if 'val_loss' in df.columns:
        plt.plot(epochs, df['val_loss'].values, label='val_loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    save_fig(fig, "training_loss_curve.png")

    # optional metric plot
    metric_cols = [c for c in df.columns if 'metric' in c.lower()]
    if metric_cols:
        fig = plt.figure(figsize=(8,5))
        for c in metric_cols:
            plt.plot(epochs, df[c].values, label=c)
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("Training Metrics")
        plt.legend()
        save_fig(fig, "training_metrics.png")

def plot_simple_training_losses(mf_losses=None, seq_losses=None):
    """
    Plot simple loss-vs-epoch curves for MF and Sequence models
    when loss values are provided manually (e.g., from console output).
    """
    if mf_losses:
        epochs = list(range(1, len(mf_losses) + 1))
        fig = plt.figure(figsize=(8,5))
        plt.plot(epochs, mf_losses, marker='o')
        plt.title("MF Model Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        save_fig(fig, "mf_training_loss_manual.png")

    if seq_losses:
        epochs = list(range(1, len(seq_losses) + 1))
        fig = plt.figure(figsize=(8,5))
        plt.plot(epochs, seq_losses, marker='o')
        plt.title("Sequence Model Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        save_fig(fig, "sequence_training_loss_manual.png")

    if mf_losses and seq_losses:
        epochs_mf = list(range(1, len(mf_losses) + 1))
        epochs_seq = list(range(1, len(seq_losses) + 1))
        fig = plt.figure(figsize=(8,5))
        plt.plot(epochs_mf, mf_losses, marker='o', label="MF Loss")
        plt.plot(epochs_seq, seq_losses, marker='o', label="Sequence Loss")
        plt.title("MF vs Sequence Training Loss (Manual)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()
        save_fig(fig, "mf_vs_sequence_loss_manual.png")

def plot_hybrid_pseudo_loss(mf_losses, seq_losses):
    """
    Plot a pseudo hybrid loss curve derived as a weighted
    aggregation of MF and Sequence losses.
    This is a diagnostic visualization (not a trained loss).
    """
    if not mf_losses or not seq_losses:
        print("[SKIP] hybrid pseudo-loss (missing MF or Sequence losses)")
        return

    min_len = min(len(mf_losses), len(seq_losses))
    epochs = list(range(1, min_len + 1))

    hybrid_losses = [
        MF_WEIGHT * mf_losses[i] + SEQ_WEIGHT * seq_losses[i]
        for i in range(min_len)
    ]

    fig = plt.figure(figsize=(8,5))
    plt.plot(epochs, hybrid_losses, marker='o')
    plt.title("Hybrid Model Pseudo-Loss (Weighted MF + Sequence)")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Loss")
    plt.grid(True)
    save_fig(fig, "hybrid_pseudo_loss_curve.png")

def plot_all_loss_comparison(mf_losses, seq_losses):
    """
    Plot MF, Sequence, and Hybrid pseudo-loss together
    for comparative analysis.
    """
    if not mf_losses or not seq_losses:
        print("[SKIP] loss comparison plot (missing MF or Sequence losses)")
        return

    min_len = min(len(mf_losses), len(seq_losses))
    epochs = list(range(1, min_len + 1))

    hybrid_losses = [
        MF_WEIGHT * mf_losses[i] + SEQ_WEIGHT * seq_losses[i]
        for i in range(min_len)
    ]

    fig = plt.figure(figsize=(8,5))
    plt.plot(epochs, mf_losses[:min_len], marker='o', label="MF Loss")
    plt.plot(epochs, seq_losses[:min_len], marker='o', label="Sequence Loss")
    plt.plot(epochs, hybrid_losses, marker='o', label="Hybrid Pseudo-Loss")

    plt.title("Loss Comparison: MF vs Sequence vs Hybrid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    save_fig(fig, "mf_seq_hybrid_loss_comparison.png")


# ---- Main ----
def main(args):
    ensure_outdir()

    # build full paths
    files = {k: os.path.join(PROCESSED_DIR, v) for k,v in DEFAULT_FILES.items()}

    master = try_read_parquet(files['master'])
    interactions = try_read_parquet(files['interactions'])
    embeddings = try_read_parquet(files['embeddings'])
    nutrition_feat = try_read_parquet(files['nutrition_features'])
    text_feat = try_read_parquet(files['text_features'])
    centrality = try_read_parquet(files['centrality'])
    bipartite = try_read_parquet(files['bipartite'])

    # Stats & basic visuals
    plot_region_distribution(master)
    plot_nutrient_histograms(master, NUTRI_CANDIDATES)
    plot_nutrition_corr(master, NUTRI_CANDIDATES)
    plot_pca_on_columns(master, NUTRI_CANDIDATES, "PCA of Nutrition Features", "pca_nutrition_2d.png", sample=args.sample)
    plot_item_embeddings_pca(embeddings, sample=args.sample)
    plot_centrality_hist(centrality)
    plot_popularity_activity(interactions)
    plot_bipartite_sample(bipartite, sample_nodes=args.sample_nodes)

    # MF model inspection
    if not args.skip_mf:
        mf_path = os.path.join(MODELS_DIR, "mf_model.pt")
        inspect_mf_model(mf_path)

    # training logs plotting if provided
    if args.train_log:
        plot_training_logs(args.train_log)

    if not args.mf_losses:
        args.mf_losses = mf_losses
    if not args.seq_losses:
        args.seq_losses = seq_losses
    
    # Plot simple/manual training losses (MF + Sequence)
    if args.mf_losses or args.seq_losses:
        plot_simple_training_losses(args.mf_losses, args.seq_losses)
        plot_hybrid_pseudo_loss(args.mf_losses, args.seq_losses)
        plot_all_loss_comparison(args.mf_losses, args.seq_losses)

    print("\nAll done. Visualizations saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis visualizations for recommender project")
    parser.add_argument("--sample", type=int, default=None, help="max points for PCA scatter (random sample)")
    parser.add_argument("--sample-nodes", type=int, default=300, help="sample size for bipartite graph plotting")
    parser.add_argument("--skip-mf", action='store_true', help="skip MF model inspection")
    parser.add_argument("--train-log", type=str, default=None, help="path to training log CSV to plot loss/metrics")
    parser.add_argument("--mf-losses", nargs="*", type=float, default=None,
                    help="manual MF losses per epoch (space-separated)")
    parser.add_argument("--seq-losses", nargs="*", type=float, default=None,
                    help="manual Sequence model losses per epoch (space-separated)")

    args = parser.parse_args()
    main(args)

