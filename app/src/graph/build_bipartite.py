# src/graph/build_bipartite.py
import pandas as pd

from src.config import (
    MASTER_FOOD_CATALOG_PARQUET,
    USER_INTERACTIONS_PARQUET,
    BIPARTITE_EDGES_PARQUET,
    ITEM_CENTRALITY_PARQUET,
    USER_ID_COL,
)


def run():
    print("=== BUILD_BIPARTITE: creating edge list + item centrality ===")

    master = pd.read_parquet(MASTER_FOOD_CATALOG_PARQUET)
    interactions = pd.read_parquet(USER_INTERACTIONS_PARQUET)

    if interactions.empty:
        raise ValueError(
            f"user_interactions is empty at {USER_INTERACTIONS_PARQUET}. "
            "Check preprocess step."
        )

    # Edge list for bipartite graph: (user_id, food_id, rating)
    edges = interactions[[USER_ID_COL, "food_id", "rating"]].copy()
    edges.to_parquet(BIPARTITE_EDGES_PARQUET, index=False)
    print(f"  [graph] saved bipartite_edges.parquet with shape {edges.shape}")

    # Item degree centrality = #distinct users / max_users
    item_counts = (
        edges.groupby("food_id")[USER_ID_COL]
        .nunique()
        .reset_index(name="num_users")
    )

    max_users = item_counts["num_users"].max()
    if max_users and max_users > 0:
        item_counts["degree_centrality"] = item_counts["num_users"] / max_users
    else:
        item_counts["degree_centrality"] = 0.0

    # Optionally join food names for readability
    master_small = master[["food_id"]].copy()
    if "food_name" in master.columns:
        master_small["food_name"] = master.set_index("food_id")["food_name"]

    item_centrality = item_counts.merge(master_small, on="food_id", how="left")

    item_centrality.to_parquet(ITEM_CENTRALITY_PARQUET, index=False)
    print(
        f"  [graph] saved item_centrality.parquet with shape {item_centrality.shape}"
    )
    print("=== BUILD_BIPARTITE COMPLETE ===")


if __name__ == "__main__":
    run()
