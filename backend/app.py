from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import os
import json
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app)

# Database setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_NAME = os.path.join(BASE_DIR, 'backend', 'news.db')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'ml_model', 'vectorizer.pkl')
ADVANCED_MODELS_DIR = os.path.join(BASE_DIR, 'ml_model', 'trained_models')
# Set ACTIVE_MODEL to lightgbm, gru, lstm, cnn, bert, or roberta to serve an
# advanced model. Leave unset to use the original Logistic Regression model.
ACTIVE_MODEL = os.getenv('ACTIVE_MODEL', '').lower()
advanced_predictor = None

def init_db():
    """Initialize database with articles table"""
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS articles
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      text TEXT NOT NULL,
                      prediction INTEGER,
                      confidence REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

def get_model_and_vectorizer():
    """Load trained model and vectorizer"""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

class AdvancedPredictor:
    """Adapter that gives every advanced model the same prediction interface."""
    def __init__(self, algorithm):
        model_dir = os.path.join(ADVANCED_MODELS_DIR, algorithm)
        with open(os.path.join(model_dir, 'metadata.json'), encoding='utf-8') as file:
            self.metadata = json.load(file)
        self.format = self.metadata['format']
        if self.format == 'joblib_tfidf':
            artifact = joblib.load(os.path.join(model_dir, 'model.pkl'))
            self.model, self.vectorizer = artifact['model'], artifact['vectorizer']
        elif self.format == 'keras':
            import tensorflow as tf
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.keras'))
        elif self.format == 'transformers':
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model format: {self.format}")

    def predict(self, text):
        if self.format == 'joblib_tfidf':
            probabilities = self.model.predict_proba(self.vectorizer.transform([text]))[0]
        elif self.format == 'keras':
            real_probability = float(self.model.predict(np.array([[text]]), verbose=0)[0][0])
            probabilities = np.array([1 - real_probability, real_probability])
        else:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=0).numpy()
        prediction = int(np.argmax(probabilities))
        return prediction, float(np.max(probabilities) * 100)

def get_advanced_predictor():
    global advanced_predictor
    if advanced_predictor is None:
        advanced_predictor = AdvancedPredictor(ACTIVE_MODEL)
    return advanced_predictor

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'API is running'}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict if news is fake or real"""
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if ACTIVE_MODEL:
            prediction, confidence = get_advanced_predictor().predict(text)
        else:
            model, vectorizer = get_model_and_vectorizer()
            if model is None or vectorizer is None:
                return jsonify({'error': 'Model not loaded'}), 500
            text_vectorized = vectorizer.transform([text])
            prediction = model.predict(text_vectorized)[0]
            confidence = max(model.predict_proba(text_vectorized)[0]) * 100
        
        # Save to database
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('INSERT INTO articles (text, prediction, confidence) VALUES (?, ?, ?)',
                  (text, int(prediction), float(confidence)))
        conn.commit()
        article_id = c.lastrowid
        conn.close()
        
        result = {
            'id': article_id,
            'prediction': 'FAKE' if prediction == 0 else 'REAL',
            'confidence': round(confidence, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('SELECT id, text, prediction, confidence, timestamp FROM articles ORDER BY timestamp DESC LIMIT 50')
        rows = c.fetchall()
        conn.close()
        
        history = [{
            'id': row[0],
            'text': row[1][:100] + '...' if len(row[1]) > 100 else row[1],
            'prediction': 'FAKE' if row[2] == 0 else 'REAL',
            'confidence': row[3],
            'timestamp': row[4]
        } for row in rows]
        
        return jsonify(history), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
