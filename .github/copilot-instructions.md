# GitHub Copilot Instructions for Fake News Detector

## Project Overview
This is a full-stack Fake News Detection web application built with:
- **Frontend**: React 18
- **Backend**: Python Flask
- **ML Model**: scikit-learn (Multinomial Naive Bayes)
- **Database**: SQLite

## Project Setup Checklist

- [x] Create project directory structure
- [x] Set up backend (Flask API)
- [x] Set up ML model training script
- [x] Set up frontend (React)
- [x] Create configuration files
- [ ] Install backend dependencies
- [ ] Train ML model
- [ ] Install frontend dependencies
- [ ] Start backend server
- [ ] Start frontend server

## Key Files

### Backend
- `backend/app.py` - Main Flask API application
- `backend/requirements.txt` - Python dependencies

### ML Model
- `ml_model/train_model.py` - Model training and evaluation

### Frontend
- `frontend/src/App.js` - Main React component
- `frontend/src/App.css` - Styling
- `frontend/package.json` - Node dependencies

## Environment Setup

### Python (Backend & ML)
```bash
cd backend
pip install -r requirements.txt
```

### Node.js (Frontend)
```bash
cd frontend
npm install
```

## Running the Application

### 1. Train the ML Model
```bash
cd ml_model
python train_model.py
```

### 2. Start Backend Server
```bash
cd backend
python app.py
```
Server runs on: `http://localhost:5000`

### 3. Start Frontend (in new terminal)
```bash
cd frontend
npm start
```
App runs on: `http://localhost:3000`

## API Endpoints

- `POST /api/predict` - Predict if news is fake or real
- `GET /api/history` - Get prediction history
- `GET /api/health` - Health check

## Development Guidelines

- Frontend communicates with backend via axios
- CORS is enabled for local development
- Predictions are stored in SQLite database
- Model uses TF-IDF vectorization + Naive Bayes classifier

## Next Steps

1. Install dependencies (both backend and frontend)
2. Train the ML model
3. Start both servers
4. Test by entering news articles in the web app
5. Improve the model by adding more training data
