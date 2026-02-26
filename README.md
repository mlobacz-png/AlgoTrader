# ML Worker - LightGBM Trade Prediction Service

Self-learning prediction microservice that trains on your historical trades and predicts the edge of new signals before execution.

## Deploy to Railway (Recommended)

1. Push this `ml_worker` folder to a GitHub repository (can be the same repo or a separate one).

2. Go to [railway.app](https://railway.app) and sign in.

3. Click **New Project** > **Deploy from GitHub Repo** and select your repository.

4. Railway will auto-detect the Dockerfile. If prompted, set the root directory to `ml_worker`.

5. Go to **Variables** and add:

   ```
   POSTGRES_URL=postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
   ```

   Find this in your Supabase Dashboard > Settings > Database > Connection String (URI). Use the **Transaction pooler** URI.

6. Railway will build and deploy automatically. Copy the generated URL (e.g., `https://ml-worker-production-xxxx.up.railway.app`).

7. Back in the trading app, go to **Settings** > **ML Prediction** and paste the Railway URL.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check service status and model info |
| `/predict` | POST | Get predicted edge for a trade signal |
| `/train` | POST | Incremental training on new trades since last run |
| `/train/full` | POST | Full retrain on all historical trades |

### POST /predict

```json
{
  "symbol": "AAPL",
  "strategy": "ORB-15Min",
  "features": {
    "sma_20": 185.5,
    "rsi_14": 62.3,
    "confidence": 0.75,
    "regime": "normal"
  }
}
```

Response:

```json
{
  "predicted_edge": 0.78,
  "model_version": "2026-02-25-v3",
  "confidence_band": "high"
}
```

## Local Development

```bash
cd ml_worker
pip install -r requirements.txt
POSTGRES_URL="your-connection-string" uvicorn app:app --reload
```

## How It Works

- Reads closed trades from the `trades` table, flattening the `entry_features` JSONB
- Uses 17 features: 15 technical indicators + confidence + regime
- LightGBM regressor predicts expected PnL percentage
- Output is sigmoid-normalized to a 0-1 edge score
- Incremental training adds new trades without retraining from scratch
- Model persists to disk via joblib between restarts
