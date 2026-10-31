import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
SENTIMENT_PATH = MODEL_DIR / "sentiment_pipeline.joblib"
SALES_PATH = MODEL_DIR / "sales_model.joblib"

app = Flask(__name__)
CORS(app)
database_url = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{ROOT / 'instance' / 'app.db'}"
)

if database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql+psycopg2://",
        1
    )

elif database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://",
        "postgresql+psycopg2://",
        1
    )

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

def ensure_models():
    if SENTIMENT_PATH.exists() and SALES_PATH.exists():
        return
    data_path = ROOT / "data" / "sample_reviews.csv"
    data = pd.read_csv(data_path)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    pipeline.fit(data["review"].astype(str), data["sentiment"].astype(str))
    joblib.dump(pipeline, SENTIMENT_PATH)

    # A reproducible demo model. Replace this generated dataset with real historical sales data.
    rng = np.random.default_rng(42)
    rows = 700
    price = rng.uniform(8, 250, rows)
    discount = rng.uniform(0, 0.45, rows)
    marketing = rng.uniform(0, 1000, rows)
    rating = rng.uniform(2.5, 5, rows)
    season = rng.integers(1, 13, rows)
    units = (120 + 0.9 * marketing - 0.42 * price + 95 * discount +
             24 * rating + 18 * np.sin(2 * np.pi * season / 12) +
             rng.normal(0, 25, rows))
    sales_df = pd.DataFrame({
        "price": price, "discount": discount, "marketing_spend": marketing,
        "rating": rating, "month": season, "units_sold": np.maximum(units, 0)
    })
    sales_model = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)
    sales_model.fit(sales_df.drop(columns=["units_sold"]), sales_df["units_sold"])
    joblib.dump(sales_model, SALES_PATH)

with app.app_context():
    db.create_all()
    ensure_models()

sentiment_model = joblib.load(SENTIMENT_PATH)
sales_model = joblib.load(SALES_PATH)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ecommerce-intelligence-api"})

@app.post("/api/sentiment")
def predict_sentiment():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if len(text) < 3:
        return jsonify({"error": "Please enter a review with at least 3 characters."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Review must be 5,000 characters or fewer."}), 400
    probabilities = sentiment_model.predict_proba([text])[0]
    classes = sentiment_model.classes_
    best = int(np.argmax(probabilities))
    label = str(classes[best])
    confidence = float(probabilities[best])
    record = Review(text=text, sentiment=label, confidence=confidence)
    db.session.add(record)
    db.session.commit()
    return jsonify({
        "sentiment": label,
        "confidence": round(confidence, 4),
        "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, probabilities)},
        "id": record.id
    })

@app.post("/api/sales-prediction")
def predict_sales():
    payload = request.get_json(silent=True) or {}
    try:
        price = float(payload.get("price"))
        discount = float(payload.get("discount"))
        marketing = float(payload.get("marketing_spend"))
        rating = float(payload.get("rating"))
        month = int(payload.get("month"))
    except (TypeError, ValueError):
        return jsonify({"error": "All prediction fields must be valid numbers."}), 400
    if not (0 < price <= 100000 and 0 <= discount <= 1 and 0 <= marketing <= 10000000
            and 0 <= rating <= 5 and 1 <= month <= 12):
        return jsonify({"error": "Check ranges: price > 0, discount 0–1, marketing >= 0, rating 0–5, month 1–12."}), 400
    features = pd.DataFrame([{
        "price": price, "discount": discount, "marketing_spend": marketing,
        "rating": rating, "month": month
    }])
    prediction = max(0, float(sales_model.predict(features)[0]))
    return jsonify({"predicted_units": round(prediction), "model_note": "Demo model trained on synthetic sample data; retrain on your own historical sales data before business use."})

@app.get("/api/overview")
def overview():
    total = Review.query.count()
    rows = db.session.query(Review.sentiment, func.count(Review.id)).group_by(Review.sentiment).all()
    distribution = {label: count for label, count in rows}
    recent = Review.query.order_by(Review.created_at.desc()).limit(8).all()
    return jsonify({
        "total_reviews": total,
        "distribution": distribution,
        "recent_reviews": [{
            "id": r.id, "text": r.text, "sentiment": r.sentiment,
            "confidence": round(r.confidence, 3),
            "created_at": r.created_at.isoformat()
        } for r in recent]
    })

@app.get("/api/reviews")
def reviews():
    rows = Review.query.order_by(Review.created_at.desc()).limit(100).all()
    return jsonify([{
        "id": r.id, "text": r.text, "sentiment": r.sentiment,
        "confidence": round(r.confidence, 3), "created_at": r.created_at.isoformat()
    } for r in rows])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
