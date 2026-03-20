# =========================================================

# PROJECT EXECUTION GUIDE

# "A Novel Community and Family Centric

# Healthy Meal Planner Integrating

# Nutritional Information, User Preferences,

# and Regional Context" Prototype

# =========================================================

This document provides a complete, step-by-step guide to
build, run, and validate the project prototype.

This guide assumes:

* No source code modifications
* All artifacts are generated automatically by the code
* The project is a research prototype, not a production system

---

## ---------------------------------------------------------

## 1. PROJECT DIRECTORY STRUCTURE

## ---------------------------------------------------------

```bash
PROJECT/
│
├── .venv/                       -> Python virtual environment
├── data/
│   ├── raw/                     -> Original CSV datasets
│   └── processed/               -> Auto-generated parquet files
│
├── logs/                        -> Runtime logs (auto-created)
├── outputs/
│   └── visualizations/          -> Loss curves & graphs
│
├── src/
│   ├── data/
│   │   ├── ingest.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── nutritionfeatures.py
│   │   └── textfeatures.py
│   │
│   ├── embeddings/
│   │   └── item_embeddings.py
│   │
│   ├── graph/
│   │   └── build_bipartite.py
│   │
│   ├── models/
│   │   ├── transformer_sequence.py
│   │   └── attention_fusion.py
│   │
│   ├── models_artifacts/        -> Saved .npy and .pt files
│   │
│   ├── training/
│   │   ├── config.py
│   │   ├── trainer.py
│   │   └── sequence_trainer.py
│   │
│   ├── serving/
│   │   ├── recommend.py
│   │   └── api.py
│   │
│   ├── visualization/
│   │   └── visualizations.py
│   │
│   ├── __init__.py
│   ├── config.py
│   └── run_pipeline.py
│
└── requirements.txt
```

---

## ---------------------------------------------------------

## 2. ENVIRONMENT SETUP (ONE-TIME)

## ---------------------------------------------------------

### 2.1 Create Virtual Environment

```bash
python -m venv .venv
```

### 2.2 Activate Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 2.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ---------------------------------------------------------

## 3. DATA INGESTION & PREPROCESSING PIPELINE

## ---------------------------------------------------------

This step converts raw CSV datasets into structured,
cleaned parquet files and builds the interaction graph.

Run from PROJECT root directory:

```bash
python src/run_pipeline.py
```

This executes the following modules automatically:

### 3.1 ingest.py

* Reads all datasets from data/raw/
* Validates schema and column consistency
* Stores intermediate parquet files in data/processed/

### 3.2 preprocess.py

* Handles missing nutritional values
* Standardizes nutrient units
* Encodes categorical fields
* Assigns consistent food_id values

### 3.3 build_bipartite.py

* Constructs user–food bipartite graph
* Stores graph structure in processed data

After this step, data/processed/ will contain all required
parquet and graph files for feature extraction.

---

## ---------------------------------------------------------

## 4. FEATURE EXTRACTION (MANUAL EXECUTION)

## ---------------------------------------------------------

These scripts must be run after the pipeline completes.

### 4.1 Nutritional Feature Extraction

```bash
python src/features/nutritionfeatures.py
```

* Generates structured numerical nutrition vectors
* Output stored in data/processed/

### 4.2 Text Feature Extraction

```bash
python src/features/textfeatures.py
```

* Encodes food names/descriptions into vectors
* Output stored in data/processed/

---

## ---------------------------------------------------------

## 5. ITEM EMBEDDING GENERATION

## ---------------------------------------------------------

```bash
python src/embeddings/item_embeddings.py
```

* Combines nutrition, text, and popularity features
* Generates unified food embeddings
* Saves embeddings in src/models_artifacts/

---

## ---------------------------------------------------------

## 6. MATRIX FACTORIZATION MODEL TRAINING

## ---------------------------------------------------------

```bash
python src/training/trainer.py
```

* Trains user and food embeddings
* Captures long-term food preferences
* Optimizes interaction-based loss per epoch
* Saves trained embeddings in src/models_artifacts/
* Loss values are logged for visualization

---

## ---------------------------------------------------------

## 7. SEQUENCE & ATTENTION MODEL TRAINING

## ---------------------------------------------------------

### 7.1 Model Definitions

* transformer_sequence.py defines the Transformer architecture
* attention_fusion.py defines the score fusion mechanism

### 7.2 Train Sequence Model

```bash
python src/training/sequence_trainer.py
```

* Learns short-term eating behavior from sequences
* Uses attention-based modeling
* Saves trained model (.pt files) in src/models_artifacts/
* Loss decreases gradually across epochs

---

## ---------------------------------------------------------

## 8. HYBRID RECOMMENDATION GENERATION

## ---------------------------------------------------------

```bash
python src/serving/recommend.py
```

* Loads MF model, sequence model, and item embeddings
* Computes:

  * MF preference score
  * Sequence prediction score
  * Content similarity score
* Applies:

  * Health condition penalties
  * Allergy filtering
  * Food safety constraints
* Produces final ranked food recommendations

---

## ---------------------------------------------------------

## 9. API SERVING (OPTIONAL DEMO)

## ---------------------------------------------------------

```bash
uvicorn src.serving.api:app --reload
```

* Starts a local recommendation server
* Accepts JSON-based user input
* Returns top-K food recommendations

---

## ---------------------------------------------------------

## 10. VISUALIZATION & ANALYSIS

## ---------------------------------------------------------

```bash
python src/visualization/visualizations.py
```

* Generates loss curves for:

  * Matrix Factorization model
  * Transformer Sequence model
  * Hybrid fusion scores
* Saves plots in outputs/visualizations/

---

## ---------------------------------------------------------

## 11. CORRECT EXECUTION ORDER (SUMMARY)

## ---------------------------------------------------------

```bash
1. python src/run_pipeline.py
2. python src/features/nutritionfeatures.py
3. python src/features/textfeatures.py
4. python src/embeddings/item_embeddings.py
5. python src/training/trainer.py
6. python src/training/sequence_trainer.py
7. python src/serving/recommend.py
8. uvicorn src.serving.api:app --reload   (optional)
9. python src/visualization/visualizations.py
```

---

## ---------------------------------------------------------

## 12. IMPORTANT NOTES

## ---------------------------------------------------------

* No source files should be modified
* All outputs are auto-generated
* Model evaluation is loss-based (neural networks)
* Accuracy / Precision / Recall are NOT used
* This is a research prototype for academic evaluation

---

# =========================================================

# END OF DOCUMENT

# =========================================================
