# src/serving/recommend.py
import os
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch

from src.config import PROCESSED_DIR
from src.embeddings.item_embeddings import load_item_embeddings
from src.training.trainer import MatrixFactorization
from src.training.config import EMBEDDING_DIM
from src.training.sequence_trainer import SeqRecModel

MASTER_FOOD_CATALOG = os.path.join(PROCESSED_DIR, "master_food_catalog.parquet")
USER_INTERACTIONS_PATH = os.path.join(PROCESSED_DIR, "user_interactions.parquet")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models_artifacts"

# weights for hybrid scoring
MF_WEIGHT = 0.6
CONTENT_WEIGHT = 0.2
SEQ_WEIGHT = 0.2


# --------------------------------------------------------
# Safe utilities
# --------------------------------------------------------

def _safe(x) -> float:
    """Convert to a JSON-safe float: no NaN/inf/None."""
    try:
        if x is None:
            return 0.0
        if isinstance(x, (list, tuple, dict)):
            return 0.0
        if isinstance(x, (np.floating, float, int)):
            if np.isnan(x) or np.isinf(x):
                return 0.0
            return float(x)
        # anything else → try cast
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def _normalize(x: float) -> float:
    """Squash any real value to [0,1] using a sigmoid, safely."""
    x = _safe(x)
    try:
        return float(1.0 / (1.0 + np.exp(-x)))
    except Exception:
        return 0.0


# --------------------------------------------------------
# Load MF model
# --------------------------------------------------------

def _load_mf():
    user2idx = np.load(MODEL_DIR / "user2idx.npy", allow_pickle=True).item()
    item2idx = np.load(MODEL_DIR / "item2idx.npy", allow_pickle=True).item()
    idx2item = np.load(MODEL_DIR / "idx2item.npy", allow_pickle=True).item()

    model = MatrixFactorization(
        num_users=len(user2idx),
        num_items=len(item2idx),
        dim=EMBEDDING_DIM,
    )
    state = torch.load(MODEL_DIR / "mf_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, user2idx, item2idx, idx2item


# --------------------------------------------------------
# Load sequence model
# --------------------------------------------------------

def _load_seq():
    seq_user2idx = np.load(MODEL_DIR / "seq_user2idx.npy", allow_pickle=True).item()
    seq_food2idx = np.load(MODEL_DIR / "seq_foodid2idx.npy", allow_pickle=True).item()
    seq_idx2food = np.load(MODEL_DIR / "seq_idx2foodid.npy", allow_pickle=True).item()
    seq_reg2idx = np.load(MODEL_DIR / "seq_region2idx.npy", allow_pickle=True).item()

    num_items = len(seq_food2idx)
    num_items_with_pad = num_items + 1

    state = torch.load(MODEL_DIR / "seq_model.pt", map_location="cpu")
    max_len = state["encoder_model.pos_embedding.weight"].shape[0]

    model = SeqRecModel(
        num_items_with_pad=num_items_with_pad,
        num_regions=len(seq_reg2idx),
        num_users=len(seq_user2idx),
        d_model=EMBEDDING_DIM,
        n_heads=4,
        n_layers=2,
        max_len=max_len,
    )
    model.load_state_dict(state)
    model.eval()

    print(
        f"[SEQ] Model loaded. users={len(seq_user2idx)}, "
        f"items={num_items}, max_len={max_len}"
    )

    return model, seq_user2idx, seq_food2idx, seq_idx2food, seq_reg2idx, max_len, num_items_with_pad


# --------------------------------------------------------
# Allergy hard filter
# --------------------------------------------------------

STRICT_ALLERGEN_KEYWORDS = {
    "prawn": ["prawn", "shrimp"],
    "shellfish": ["prawn", "shrimp", "lobster", "crab"],
    "nuts": ["nut", "peanut", "almond", "cashew", "walnut", "hazelnut", "pistachio"],
    "peanut": ["peanut", "groundnut"],
    "dairy": ["milk", "cheese", "butter", "yoghurt", "yogurt", "cream"],
    # add more if you want
}


def _parse_allergies(allergies: Optional[str]) -> List[str]:
    """Parse a string like 'prawn; nuts' or 'prawn, nuts' -> ['prawn','nuts']."""
    if not allergies:
        return []
    toks: List[str] = []
    for tok in str(allergies).replace(",", ";").split(";"):
        tok = tok.strip().lower()
        if tok and tok not in ("none", "no", "nan"):
            toks.append(tok)
    return toks


def _build_food_text_index(master: pd.DataFrame) -> Dict[int, str]:
    """
    For each food_id, build a lowercase text blob from product_name / food_name / PROD_CAT.
    No reliance on a '__text' attribute on row objects.
    """
    # Start with empty text for every row
    text = pd.Series([""] * len(master), index=master.index, dtype="object")

    if "product_name" in master.columns:
        text = text + " " + master["product_name"].fillna("").astype(str)
    if "food_name" in master.columns:
        text = text + " " + master["food_name"].fillna("").astype(str)
    if "PROD_CAT" in master.columns:
        text = text + " " + master["PROD_CAT"].fillna("").astype(str)

    text = text.str.lower().fillna("")

    food_ids = master["food_id"].astype(int).values
    text_vals = text.values

    text_map: Dict[int, str] = {}
    for fid, t in zip(food_ids, text_vals):
        text_map[int(fid)] = str(t) if t is not None else ""
    return text_map


def _is_allergy_hit(food_text: str, allergy_tokens: List[str]) -> bool:
    """Returns True if this food_text should be banned because of allergies."""
    if not allergy_tokens:
        return False
    if not food_text:
        return False

    food_text = food_text.lower()

    for token in allergy_tokens:
        if token in STRICT_ALLERGEN_KEYWORDS:
            for kw in STRICT_ALLERGEN_KEYWORDS[token]:
                if kw in food_text:
                    return True
        else:
            if token and token in food_text:
                return True
    return False


# --------------------------------------------------------
# Health soft penalty (diabetes, hypertension, etc.)
# --------------------------------------------------------

def _health_penalty(row: pd.Series, health_condition: str, health_goal: str) -> float:
    """
    Returns a multiplier in [0,1] (or a small >1 boost). 1.0 = no penalty.
    Uses nutrition columns: sugars_100g, salt_100g, saturated-fat_100g, energy-kcal_100g.
    """

    def g(name: str, default: float = 0.0) -> float:
        if name not in row.index:
            return default
        try:
            return float(row[name])
        except Exception:
            return default

    sugar = g("sugars_100g")
    salt = g("salt_100g")
    satfat = g("saturated-fat_100g")
    energy = g("energy-kcal_100g")

    penalties: List[float] = []

    hc = (health_condition or "").lower()
    hg = (health_goal or "").lower()

    # Diabetes
    if "diabet" in hc:
        if sugar >= 20:
            penalties.append(0.1)
        elif sugar >= 15:
            penalties.append(0.3)
        elif sugar >= 10:
            penalties.append(0.5)
        elif sugar >= 5:
            penalties.append(0.8)
        else:
            penalties.append(1.0)

    # Hypertension / BP
    if "hypertens" in hc or "bp" in hc:
        if salt >= 1.5:
            penalties.append(0.2)
        elif salt >= 1.0:
            penalties.append(0.4)
        elif salt >= 0.6:
            penalties.append(0.7)
        else:
            penalties.append(1.0)

    # Cholesterol / heart
    if "cholesterol" in hc or "heart" in hc:
        if satfat >= 15:
            penalties.append(0.2)
        elif satfat >= 10:
            penalties.append(0.4)
        elif satfat >= 5:
            penalties.append(0.7)
        else:
            penalties.append(1.0)

    # Goal: weight loss / obesity
    if "weight_loss" in hg or "obesity" in hc:
        if energy >= 500:
            penalties.append(0.3)
        elif energy >= 350:
            penalties.append(0.5)
        elif energy >= 250:
            penalties.append(0.8)
        else:
            penalties.append(1.0)
    # Goal: muscle gain (mild boost for higher energy foods)
    elif "muscle_gain" in hg:
        if energy >= 400:
            penalties.append(1.05)
        else:
            penalties.append(1.0)

    if not penalties:
        return 1.0

    penalty = min(penalties)
    if penalty <= 0:
        penalty = 0.01
    if penalty > 1.2:
        penalty = 1.2
    return float(penalty)


# --------------------------------------------------------
# Sequence scores per user
# --------------------------------------------------------

def _seq_scores(user_id: int, master: pd.DataFrame, inter: pd.DataFrame) -> Dict[int, float]:
    try:
        model, u2i, f2i, i2f, r2i, max_len, padN = _load_seq()
    except Exception:
        return {}

    if user_id not in u2i:
        return {}

    hist = inter.loc[inter["user_id"] == user_id, "food_id"].astype(int)
    if len(hist) < 2:
        return {}

    hist_items = [f2i[f] for f in hist.values if f in f2i]
    if not hist_items:
        return {}

    if len(hist_items) > max_len:
        hist_items = hist_items[-max_len:]
        hist = hist.iloc[-max_len:]

    master_loc = master[["food_id", "region"]].astype({"food_id": int})
    master_loc["region"] = master_loc["region"].fillna("unknown").astype(str)
    food2reg = {
        int(r.food_id): r2i.get(str(r.region), 0)
        for r in master_loc.itertuples(index=False)
    }
    hist_reg = [food2reg.get(fid, 0) for fid in hist.values]

    dev = torch.device("cpu")
    items = torch.tensor([hist_items], dtype=torch.long, device=dev)
    regs = torch.tensor([hist_reg], dtype=torch.long, device=dev)
    lens = torch.tensor([len(hist_items)], dtype=torch.long, device=dev)
    uid = torch.tensor([u2i[user_id]], dtype=torch.long, device=dev)

    with torch.no_grad():
        logits = model(items, regs, uid, lens)[0].cpu().numpy()

    out: Dict[int, float] = {}
    cutoff = len(logits) - 1  # ignore PAD index
    for item_idx, food_id in i2f.items():
        if item_idx >= cutoff:
            continue
        raw = float(logits[item_idx])
        out[int(food_id)] = _normalize(_safe(raw))
    return out


# --------------------------------------------------------
# Main recommendation function
# --------------------------------------------------------

def recommend_for_user(
    user_id: int,
    top_k: int = 10,
    region: Optional[str] = None,
    allergies: Optional[str] = None,
    health_condition: Optional[str] = None,
    health_goal: Optional[str] = None,
    activity_level: Optional[str] = None,  # not yet used in scoring, but passed through
):
    """
    Main entry point used by the API.

    All user-specific info (allergies, health_condition, health_goal, activity_level)
    is passed directly as arguments (from JSON), not read from users.parquet.
    """

    # core tables
    master = pd.read_parquet(MASTER_FOOD_CATALOG)
    inter = pd.read_parquet(USER_INTERACTIONS_PATH)

    master["food_id"] = master["food_id"].astype(int)
    inter["food_id"] = inter["food_id"].astype(int)
    inter["user_id"] = inter["user_id"].astype(int)

    # build helper indices
    food_text = _build_food_text_index(master)
    # drop duplicates to avoid Series/DataFrame ambiguity
    master_by_id = master.drop_duplicates(subset=["food_id"]).set_index("food_id", drop=False)

    # user inputs
    allergy_tokens = _parse_allergies(allergies)
    hc = (health_condition or "").strip().lower()
    hg = (health_goal or "").strip().lower()
    # activity_level currently unused, but you can plug it into _health_penalty if needed

    # items already seen
    seen = set(inter.loc[inter["user_id"] == user_id, "food_id"].astype(int))

    # item embeddings
    emb_df, emb = load_item_embeddings()
    emb_df["food_id"] = emb_df["food_id"].astype(int)
    food_ids = emb_df["food_id"].values.astype(int)

    # region filter
    if region is not None and "region" in master.columns:
        region_set = set(
            master.loc[master["region"].astype(str) == str(region), "food_id"].astype(int)
        )
    else:
        region_set = set(food_ids)

    # ---------- MF ----------
    mf, u2i, i2i, i2f = _load_mf()
    if user_id not in u2i:
        raise ValueError(f"User {user_id} not found in MF model")

    uid = u2i[user_id]
    dev = torch.device("cpu")
    arr = np.array(sorted(i2i.values()))

    with torch.no_grad():
        umat = torch.tensor([uid] * len(arr), dtype=torch.long, device=dev)
        imat = torch.tensor(arr, dtype=torch.long, device=dev)
        mfs = torch.sigmoid(mf(umat, imat)).cpu().numpy()

    mf_scores = {i2f[i]: _normalize(_safe(s)) for i, s in zip(arr, mfs)}

    # ---------- Content ----------
    hist = inter.loc[inter["user_id"] == user_id, "food_id"].astype(int)
    if len(hist):
        mask = np.isin(food_ids, hist)
        H = emb[mask]
        if H.size:
            prof = H.mean(axis=0, keepdims=True)
            dot = emb @ prof.T
            norm = np.linalg.norm(emb, axis=1, keepdims=True) * np.linalg.norm(prof)
            norm[norm == 0] = 1.0
            cos = (dot / norm).flatten()
        else:
            cos = np.zeros(len(food_ids))
    else:
        cos = np.zeros(len(food_ids))
    cos = np.array([_normalize(_safe(float(x))) for x in cos])

    # ---------- Sequence ----------
    seq_scores = _seq_scores(user_id, master, inter)

    # ---------- Final scoring ----------
    items: List[int] = []
    scores: List[float] = []

    for i, fid in enumerate(food_ids):
        fid = int(fid)

        # already seen
        if fid in seen:
            continue

        # region filter
        if fid not in region_set:
            continue

        # allergy HARD FILTER
        text = food_text.get(fid, "")
        if _is_allergy_hit(text, allergy_tokens):
            continue

        # base hybrid score
        base_score = (
            MF_WEIGHT * mf_scores.get(fid, 0.0)
            + CONTENT_WEIGHT * cos[i]
            + SEQ_WEIGHT * seq_scores.get(fid, 0.0)
        )

        # health soft penalty
        row = master_by_id.loc[fid]
        if isinstance(row, pd.DataFrame):
            # if there were duplicates, take the first
            row = row.iloc[0]
        penalty = _health_penalty(row, hc, hg)

        final = _normalize(_safe(base_score * penalty))
        items.append(fid)
        scores.append(final)

    if not items:
        raise RuntimeError("No candidates after allergy/region/health filtering.")

    out = pd.DataFrame({"food_id": items, "score": scores})
    out = out.merge(master, on="food_id", how="left")
    out = out.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    # make sure there are no NaN/INF in the result (important for JSON)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.where(pd.notnull(out), None)

    return out


def demo():
    print("=== DEMO RECOMMEND (MF + content + seq + health/allergy) ===")
    user_id = 1
    try:
        df = recommend_for_user(
            user_id=user_id,
            top_k=10,
            region="mexican",
            allergies="prawn; nuts",
            health_condition="diabetes",
            health_goal="weight_loss",
            activity_level="active",
        )
        cols = [c for c in ["food_id", "product_name", "region", "score"] if c in df.columns]
        print(df[cols])
    except Exception as e:
        print("Error while recommending:", e)


if __name__ == "__main__":
    demo()
