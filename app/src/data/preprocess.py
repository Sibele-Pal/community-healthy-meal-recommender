# src/data/preprocess.py
import pandas as pd

from src.config import (
    MERGED_MASTER_PARQUET,
    MASTER_FOOD_CATALOG_PARQUET,
    USERS_PARQUET,
    USER_INTERACTIONS_PARQUET,
    FOOD_NAME_COL,
    REGION_COL,
    USER_ID_COL,
)


def find_column(df: pd.DataFrame, exact_candidates=None, contains=None):
    """
    Try to find a column in df:
    - first by exact name (case-insensitive) from exact_candidates
    - then by 'contains' substring (case-insensitive)
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    if exact_candidates:
        for cand in exact_candidates:
            cand_lower = cand.lower()
            if cand_lower in lower_map:
                return lower_map[cand_lower]

    if contains:
        for c in cols:
            if contains.lower() in c.lower():
                return c

    return None


def build_master_catalog() -> pd.DataFrame:
    print("  [master] Loading merged master parquet...")
    df = pd.read_parquet(MERGED_MASTER_PARQUET)
    print(f"  [master] merged_master shape = {df.shape}")

    # --- detect food name column ---
    food_col = find_column(
        df,
        exact_candidates=[
            "food_name",
            "Food_Name",
            "Food",
            "Food_Item",
            "item_name",
        ],
        contains="food",
    )
    if food_col is None:
        raise ValueError(
            f"Could not find a food-name column in merged_master columns: {list(df.columns)}"
        )

    # --- detect region / cuisine column ---
    region_col = find_column(
        df,
        exact_candidates=["food_cuisine", "cuisine", "region"],
        contains="cuisine",
    )

    # rename to standardized names
    rename_map = {food_col: FOOD_NAME_COL}
    if region_col is not None:
        rename_map[region_col] = REGION_COL

    df = df.rename(columns=rename_map)

    # if no region column found, create a dummy one
    if REGION_COL not in df.columns:
        df[REGION_COL] = "Unknown"

    # create a stable food_id (string)
    df["food_id"] = df.index.astype(int)

    # keep useful columns; don't know exact nutrient columns, so keep everything
    df.to_parquet(MASTER_FOOD_CATALOG_PARQUET, index=False)
    print(
        f"  [master] saved master_food_catalog to {MASTER_FOOD_CATALOG_PARQUET} "
        f"with columns: {list(df.columns)}"
    )
    return df


def build_user_interactions(master: pd.DataFrame) -> pd.DataFrame:
    print("  [interactions] Loading users parquet...")
    users = pd.read_parquet(USERS_PARQUET)
    print(f"  [interactions] users shape = {users.shape}")

    # detect user id column
    user_col = find_column(
        users,
        exact_candidates=["user_id", "User_ID", "UserId", "id"],
        contains="user",
    )
    if user_col is None:
        # if truly no user column, create synthetic
        users[USER_ID_COL] = users.index.astype(str)
    else:
        users = users.rename(columns={user_col: USER_ID_COL})

    # detect user region / cuisine preference
    user_region_col = find_column(
        users,
        exact_candidates=["region", "food_cuisine", "preferred_cuisine"],
        contains="cuisine",
    )
    if user_region_col is not None and user_region_col != REGION_COL:
        users = users.rename(columns={user_region_col: REGION_COL})
    elif REGION_COL not in users.columns:
        users[REGION_COL] = "Unknown"

    # Now create a simple interaction matrix:
    # for each user, connect to up to 20 foods from the same region.
    interactions_list = []

    for region, user_group in users.groupby(REGION_COL):
        foods_region = master[master[REGION_COL] == region]
        if foods_region.empty:
            foods_region = master

        foods_sample = foods_region[["food_id"]].head(20)
        if foods_sample.empty:
            continue

        tmp = (
            user_group[[USER_ID_COL]]
            .assign(key=1)
            .merge(foods_sample.assign(key=1), on="key")
            .drop(columns="key")
        )
        tmp["rating"] = 1.0  # implicit positive feedback
        interactions_list.append(tmp)

    if interactions_list:
        interactions = pd.concat(interactions_list, ignore_index=True)
    else:
        # fallback: empty table
        interactions = pd.DataFrame(columns=[USER_ID_COL, "food_id", "rating"])

    interactions.to_parquet(USER_INTERACTIONS_PARQUET, index=False)
    print(
        f"  [interactions] saved user_interactions to {USER_INTERACTIONS_PARQUET} "
        f"with shape {interactions.shape}"
    )
    return interactions


def run():
    print("=== PREPROCESS: build master catalog + user interactions ===")
    master = build_master_catalog()
    interactions = build_user_interactions(master)
    print("=== PREPROCESS COMPLETE ===")


if __name__ == "__main__":
    run()
