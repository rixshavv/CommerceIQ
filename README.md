# CommerceIQ — E-Commerce Intelligence & Review Analytics

A full-stack portfolio project that combines NLP sentiment classification, sales forecasting, a Flask REST API, database persistence, and a responsive analytics dashboard.

## Features

- Customer review sentiment classification: positive / neutral / negative.
- NLP preprocessing and TF-IDF feature extraction.
- Logistic Regression sentiment pipeline.
- Sales forecasting using a Random Forest Regressor.
- Flask JSON APIs with input validation.
- SQLite locally; PostgreSQL supported through `DATABASE_URL`.
- Review prediction history and sentiment analytics dashboard.
- Docker and Render deployment configuration.

## Important model transparency

The included sentiment model is trained on a small, bundled demonstration dataset. The sales model is trained on generated synthetic data. These are suitable for demonstrating the complete ML workflow, not for real business decisions. Replace the demo datasets with representative, licensed historical data, retrain, and evaluate on a held-out test set before production use. Do not claim business-grade accuracy without measured results.

## Run locally (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. On first startup, the app trains and saves the two demo models in `models/`. Model files are generated locally and intentionally excluded from Git.

## Run locally (macOS / Linux)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## API reference

### `GET /health`
Returns service status.

### `POST /api/sentiment`
```json
{"text":"The product quality is excellent and delivery was fast"}
```

Returns sentiment, confidence, class probabilities, and saved review ID.

### `POST /api/sales-prediction`
```json
{
  "price": 79.99,
  "discount": 0.15,
  "marketing_spend": 500,
  "rating": 4.2,
  "month": 11
}
```
`discount` must be a decimal between `0` and `1` (15% = `0.15`).

### `GET /api/overview`
Returns review counts, sentiment distribution, and the eight most recent predictions.

### `GET /api/reviews`
Returns up to 100 most recent stored predictions.

## Deploy to Render

1. Push this folder to a GitHub repository.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Render reads `render.yaml` to create the web service and PostgreSQL database.
4. Wait for the first deployment and open the generated `onrender.com` URL.
5. Verify `/health` returns `{"status":"ok", ...}`.

If deploying manually, use:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Add a PostgreSQL database and set `DATABASE_URL` to its internal connection string.

The free database/service availability and retention policies can change; check Render's current plan terms. For production, use a paid persistent database, set secrets through the hosting dashboard, and configure backups.

## Deploy with Docker

```bash
docker build -t commerceiq .
docker run --rm -p 5000:5000 commerceiq
```

For persistent local data, mount a volume for `/app/instance`. For hosted production, set `DATABASE_URL` to PostgreSQL.

## Retrain the models

- Replace `data/sample_reviews.csv` with a CSV containing `review` and `sentiment` columns.
- Delete `models/sentiment_pipeline.joblib` to force sentiment retraining on next startup.
- The sales model is currently generated from synthetic data in `ensure_models()` in `app.py`. Replace that section with a training pipeline based on your actual historical data. Keep the same five feature columns or update the prediction endpoint to match your new feature schema.
- Evaluate using a held-out test set. For classification, report precision, recall, F1 and confusion matrix. For regression, report MAE, RMSE and R².

## Suggested improvements

1. Add authentication and role-based access before exposing private customer data.
2. Add rate limiting, request logging, and monitoring.
3. Add data upload and a retraining workflow restricted to administrators.
4. Add model versioning and automated tests.
5. Add pagination and retention controls for review history.
6. Add a proper time-series forecasting model when you have dated sales history.

## Project structure

```text
commerceiq/
├── app.py
├── data/sample_reviews.csv
├── templates/index.html
├── static/styles.css
├── static/app.js
├── requirements.txt
├── Dockerfile
├── Procfile
├── render.yaml
└── README.md
```
