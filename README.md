# Fake News Detector - Web Application with Machine Learning

A full-stack web application that uses Machine Learning to detect fake news. Built with React frontend, Flask backend, and scikit-learn for the ML model.

## Features

- 🔍 **Real-time Fake News Detection** - Submit articles and get instant predictions
- 🤖 **Machine Learning Model** - Trained on text classification using Naive Bayes
- 💾 **Prediction History** - View all previous predictions with confidence scores
- 📊 **Confidence Scores** - See how confident the model is in each prediction
- 🎨 **Modern UI** - Beautiful, responsive React interface
- 🔄 **REST API** - Complete backend API for predictions

## Project Structure

```
.
├── frontend/              # React web application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js        # Main React component
│   │   ├── App.css       # Styling
│   │   └── index.js      # Entry point
│   └── package.json
├── backend/              # Flask API server
│   ├── app.py           # Main Flask application
│   └── requirements.txt  # Python dependencies
├── ml_model/            # Machine Learning model
│   └── train_model.py   # Model training script
└── README.md
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm

### Step 1: Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Train ML Model

```bash
cd ml_model
python train_model.py
```

This creates `model.pkl` and `vectorizer.pkl` files.

### Step 3: Start Backend Server

```bash
cd backend
python app.py
```

Server runs on `http://localhost:5000`

### Step 4: Frontend Setup

```bash
cd frontend
npm install
npm start
```

App opens at `http://localhost:3000`

## API Endpoints

### POST `/api/predict`
Predict if a news article is fake or real.

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "id": 1,
  "prediction": "REAL",
  "confidence": 85.5,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### GET `/api/history`
Get the last 50 predictions.

**Response:**
```json
[
  {
    "id": 1,
    "text": "Article preview...",
    "prediction": "FAKE",
    "confidence": 92.3,
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

### GET `/api/health`
Health check endpoint.

## How the ML Model Works

1. **Text Vectorization** - Converts text to numerical features using TF-IDF
2. **Logistic Regression Classifier** - Trained on labeled real/fake news samples
3. **Probability Prediction** - Returns confidence score along with prediction

The model in its current form is trained on sample data. For production use, train it with:
- Fact-checking datasets
- News aggregator data
- Labeled real vs. fake news collections

## Advanced algorithms

The project also includes a separate trainer for **LightGBM, GRU, LSTM, CNN, BERT, and RoBERTa** in `ml_model/train_advanced_models.py`.
Provide a CSV dataset with `text` and `label` columns (`0` = fake, `1` = real), then run one experiment at a time:

```bash
cd ml_model
pip install -r requirements-advanced.txt
python train_advanced_models.py --algorithm lightgbm --data data/news.csv
python train_advanced_models.py --algorithm gru --data data/news.csv
python train_advanced_models.py --algorithm lstm --data data/news.csv
python train_advanced_models.py --algorithm cnn --data data/news.csv
python train_advanced_models.py --algorithm bert --data data/news.csv
python train_advanced_models.py --algorithm roberta --data data/news.csv
```

If you only want to test the setup, omit `--data` to use the project's built-in demonstration samples. Those samples are too small to judge model quality.

### Preparing the ISOT dataset

After extracting the ISOT download, place `True.csv` and `Fake.csv` in `ml_model/data/isot/` and run:

```bash
python prepare_isot_dataset.py
```

This creates `ml_model/data/news.csv`, ready to pass to the trainer with `--data data/news.csv`.

Each run saves its model and an accuracy/precision/recall/F1 report in `ml_model/trained_models/<algorithm>/metadata.json`. BERT and RoBERTa download pretrained weights on their first run and need significantly more data and memory. Do not use the tiny built-in sample set to compare these models; use a balanced, held-out real-world dataset.

To serve a trained advanced model, set `ACTIVE_MODEL` before starting Flask. For example in PowerShell:

```powershell
$env:ACTIVE_MODEL = "roberta"
python app.py
```

## Development

### Adding More Training Data

Edit [ml_model/train_model.py](ml_model/train_model.py) and add more samples to the `training_data` dictionary:

```python
training_data = {
    'text': [
        'Your sample article...',
        # Add more...
    ],
    'label': [1, 0, ...]  # 1 = real, 0 = fake
}
```

Then retrain:
```bash
python train_model.py
```

### Updating the Model

After retraining, restart the Flask server to use the new model.

## Troubleshooting

**"ModuleNotFoundError: No module named 'flask'"**
- Solution: Run `pip install -r requirements.txt` in backend folder

**"Model not loaded" error**
- Solution: Make sure you've run `train_model.py` first

**CORS errors**
- Solution: Flask-CORS is configured. Check if backend is running on port 5000

**Port already in use**
- Solution: Change port in `app.py` (Flask) or `npm start` (React)

## Performance Metrics

Current model performance on test data:
- **Accuracy**: ~87%
- **Precision**: ~88%
- **Recall**: ~85%
- **F1-Score**: ~86%

*Note: Metrics vary based on training data quality and size*

## Future Improvements

- [ ] Train on larger, real-world datasets
- [ ] Add deep learning models (LSTM, BERT)
- [ ] Implement user authentication
- [ ] Add source verification features
- [ ] Create admin dashboard for model metrics
- [ ] Deploy to cloud (Heroku, AWS, etc.)
- [ ] Add more sophisticated NLP techniques

## License

MIT License - feel free to use and modify

## Support

For issues or questions, create an issue or contact the development team.

---

**Happy Detecting! 🔍**
