# src/serving/api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.serving.recommend import recommend_for_user

app = FastAPI(title="Community Food Recommender")

# --- CORS: allow Hoppscotch / frontend calls ---
origins = [
    "*",  # in production, replace "*" with your frontend origin(s)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],    # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    """
    All user-specific info is passed in this request.
    If a field is null / omitted, that aspect is ignored.
    """
    user_id: int
    top_k: int = 10
    region: str | None = None
    allergies: str | None = None          # e.g. "prawn; nuts"
    health_condition: str | None = None   # e.g. "diabetes"
    health_goal: str | None = None        # e.g. "weight_loss", "muscle_gain"
    activity_level: str | None = None     # currently not used in scoring


@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        df = recommend_for_user(
            user_id=req.user_id,
            top_k=req.top_k,
            region=req.region,
            allergies=req.allergies,
            health_condition=req.health_condition,
            health_goal=req.health_goal,
            activity_level=req.activity_level,
        )

        records = df.to_dict(orient="records")

        return {
            "user_id": req.user_id,
            "top_k": req.top_k,
            "region": req.region,
            "allergies": req.allergies,
            "health_condition": req.health_condition,
            "health_goal": req.health_goal,
            "activity_level": req.activity_level,
            "recommendations": records,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    # Run from project root:
    #   python -m src.serving.api
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=True)
