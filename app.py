import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml_worker")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/tmp/ml_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "lgbm_edge_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

FEATURE_COLS = [
    "sma_20", "sma_50", "ema_9", "ema_21",
    "rsi_14", "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_middle", "bb_lower",
    "atr_14", "stoch_k", "stoch_d", "volume_ratio",
    "confidence",
    "regime_volatile",
]

model_state = {
    "model": None,
    "version": "untrained",
    "last_train_time": None,
    "train_count": 0,
    "feature_names": FEATURE_COLS,
}


def get_engine():
    url = os.getenv("POSTGRES_URL", "")
    if not url:
        raise RuntimeError("POSTGRES_URL environment variable is not set")
    return create_engine(url, pool_pre_ping=True, pool_size=3, max_overflow=5)


def load_model():
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            model_state["model"] = joblib.load(MODEL_PATH)
            meta = json.loads(META_PATH.read_text())
            model_state["version"] = meta.get("version", "unknown")
            model_state["last_train_time"] = meta.get("last_train_time")
            model_state["train_count"] = meta.get("train_count", 0)
            logger.info(f"Loaded model version={model_state['version']}")
        except Exception as e:
            logger.warning(f"Failed to load saved model: {e}")


def save_model():
    joblib.dump(model_state["model"], MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "version": model_state["version"],
        "last_train_time": model_state["last_train_time"],
        "train_count": model_state["train_count"],
        "feature_names": model_state["feature_names"],
    }))
    logger.info(f"Saved model version={model_state['version']}")


def fetch_trades(since: str | None = None) -> pd.DataFrame:
    engine = get_engine()
    query = """
        SELECT
            strategy_id,
            asset,
            entry_features,
            confidence,
            regime,
            pnl,
            pnl_percent,
            exit_reason,
            exit_time
        FROM trades
        WHERE status = 'closed'
          AND pnl IS NOT NULL
          AND entry_features IS NOT NULL
          AND entry_features != '{}'::jsonb
    """
    params = {}
    if since:
        query += " AND exit_time > :since"
        params["since"] = since

    query += " ORDER BY exit_time ASC"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for _, row in df.iterrows():
        ef = row["entry_features"] if isinstance(row["entry_features"], dict) else {}
        feat = {}
        for col in ["sma_20", "sma_50", "ema_9", "ema_21", "rsi_14",
                     "macd_line", "macd_signal", "macd_histogram",
                     "bb_upper", "bb_middle", "bb_lower",
                     "atr_14", "stoch_k", "stoch_d", "volume_ratio"]:
            val = ef.get(col)
            feat[col] = float(val) if val is not None else np.nan

        feat["confidence"] = float(row.get("confidence") or 0.5)
        feat["regime_volatile"] = 1.0 if (ef.get("regime") or row.get("regime")) == "volatile" else 0.0
        rows.append(feat)

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = df["pnl_percent"].astype(float)
    return X, y


def train_model(incremental: bool = True) -> dict:
    since = model_state["last_train_time"] if incremental and model_state["model"] else None
    df = fetch_trades(since=since)

    if len(df) < 20:
        return {"status": "skipped", "reason": f"Only {len(df)} trades available (need 20)"}

    X, y = prepare_features(df)

    valid_mask = X.notna().sum(axis=1) >= 8
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    if len(X) < 20:
        return {"status": "skipped", "reason": f"Only {len(X)} usable rows after filtering"}

    if incremental and model_state["model"] is not None:
        lgb_data = lgb.Dataset(X, label=y, free_raw_data=False)
        model_state["model"] = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 5,
                "verbose": -1,
            },
            lgb_data,
            num_boost_round=50,
            init_model=model_state["model"],
        )
    else:
        lgb_data = lgb.Dataset(X, label=y, free_raw_data=False)
        model_state["model"] = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 5,
                "verbose": -1,
            },
            lgb_data,
            num_boost_round=200,
        )

    model_state["train_count"] += 1
    now = datetime.now(timezone.utc).isoformat()
    model_state["last_train_time"] = now
    model_state["version"] = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-v{model_state['train_count']}"

    save_model()

    preds = model_state["model"].predict(X)
    rmse = float(np.sqrt(np.mean((preds - y.values) ** 2)))

    return {
        "status": "trained",
        "version": model_state["version"],
        "samples": len(X),
        "rmse": round(rmse, 4),
        "incremental": incremental and since is not None,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("ML Worker started")
    yield
    logger.info("ML Worker shutting down")


app = FastAPI(title="Trading ML Worker", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symbol: str
    strategy: str
    features: dict


class PredictResponse(BaseModel):
    predicted_edge: float
    model_version: str
    confidence_band: str


class TrainResponse(BaseModel):
    status: str
    version: str | None = None
    samples: int | None = None
    rmse: float | None = None
    incremental: bool | None = None
    reason: str | None = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_state["model"] is not None,
        "model_version": model_state["version"],
        "last_train_time": model_state["last_train_time"],
        "train_count": model_state["train_count"],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if model_state["model"] is None:
        return PredictResponse(
            predicted_edge=0.5,
            model_version="untrained",
            confidence_band="none",
        )

    feat = {}
    for col in ["sma_20", "sma_50", "ema_9", "ema_21", "rsi_14",
                 "macd_line", "macd_signal", "macd_histogram",
                 "bb_upper", "bb_middle", "bb_lower",
                 "atr_14", "stoch_k", "stoch_d", "volume_ratio"]:
        val = req.features.get(col)
        feat[col] = float(val) if val is not None else np.nan

    feat["confidence"] = float(req.features.get("confidence", 0.5))
    regime = req.features.get("regime", "normal")
    feat["regime_volatile"] = 1.0 if regime == "volatile" else 0.0

    X = pd.DataFrame([feat], columns=FEATURE_COLS)

    raw_pred = float(model_state["model"].predict(X)[0])

    edge = 1.0 / (1.0 + np.exp(-raw_pred / 2.0))
    edge = max(0.0, min(1.0, edge))

    if edge > 0.7:
        band = "high"
    elif edge > 0.5:
        band = "medium"
    else:
        band = "low"

    return PredictResponse(
        predicted_edge=round(edge, 4),
        model_version=model_state["version"],
        confidence_band=band,
    )


@app.post("/train", response_model=TrainResponse)
async def train_endpoint():
    try:
        result = train_model(incremental=True)
        return TrainResponse(**result)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/full", response_model=TrainResponse)
async def train_full_endpoint():
    try:
        result = train_model(incremental=False)
        return TrainResponse(**result)
    except Exception as e:
        logger.error(f"Full training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
