# src/config.py
from pathlib import Path

# -------- BASE DIRECTORIES --------
BASE_DIR = Path(__file__).resolve().parents[1]  # .../Project
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Make sure processed exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -------- RAW SUBFOLDERS --------
DIET_DIR = RAW_DIR / "DIET"
NUTRITION_DIR = RAW_DIR / "NUTRITION"
FOOD_SAFETY_DIR = RAW_DIR / "FOOD_SAFETY"
FOOD_CHOICES_DIR = RAW_DIR / "FOOD_CHOICES"
MERGED_DIR = RAW_DIR / "MERGED"

# -------- RAW CSV FILES (EXPECTED NAMES) --------
FINAL_DIET_CSV = DIET_DIR / "final_diet.csv"
MEAL_SUGGESTIONS_CSV = DIET_DIR / "meal_suggestions.csv"
MICRO_MACRO_CSV = DIET_DIR / "micro_macro_nutrients.csv"

NUTRITION_CSV = NUTRITION_DIR / "nutrition_cleaned_ready.csv"
FOOD_SAFETY_CSV = FOOD_SAFETY_DIR / "rasff_cleaned_ready.csv"
USERS_CSV = FOOD_CHOICES_DIR / "synthetic_users_cleaned_ready.csv"
MERGED_MASTER_CSV = MERGED_DIR / "master_food_safety_recommender.csv"

# -------- PROCESSED PARQUET OUTPUTS --------
DIET_FINAL_PARQUET = PROCESSED_DIR / "diet_final.parquet"
DIET_MEAL_PARQUET = PROCESSED_DIR / "diet_meal_suggestions.parquet"
DIET_MICRO_PARQUET = PROCESSED_DIR / "diet_micro_macro.parquet"

NUTRITION_PARQUET = PROCESSED_DIR / "nutrition.parquet"
FOOD_SAFETY_PARQUET = PROCESSED_DIR / "food_safety.parquet"
USERS_PARQUET = PROCESSED_DIR / "users.parquet"
MERGED_MASTER_PARQUET = PROCESSED_DIR / "merged_master.parquet"

# Our standardized catalog + interactions
MASTER_FOOD_CATALOG_PARQUET = PROCESSED_DIR / "master_food_catalog.parquet"
USER_INTERACTIONS_PARQUET = PROCESSED_DIR / "user_interactions.parquet"

# Graph outputs
BIPARTITE_EDGES_PARQUET = PROCESSED_DIR / "bipartite_edges.parquet"
ITEM_CENTRALITY_PARQUET = PROCESSED_DIR / "item_centrality.parquet"

# Text Embeddings output
TEXT_FEATURES_PATH = PROCESSED_DIR / "item_text_features.parquet"


# -------- STANDARD COLUMN NAMES WE WILL USE --------
FOOD_NAME_COL = "food_name"   # logical name for item
REGION_COL = "region"         # we will derive from 'food_cuisine'
USER_ID_COL = "user_id"       # logical user id column
