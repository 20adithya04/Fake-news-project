import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './History.css';

const getCategory = (text) => {
  const lowerText = text.toLowerCase();
  if (lowerText.match(/education|school|college|university|student|teacher|study|exam/)) return 'Education';
  if (lowerText.match(/politics|government|election|president|minister|senate|congress|vote|policy|law/)) return 'Politics';
  if (lowerText.match(/health|medicine|doctor|hospital|covid|virus|disease|vaccine|care/)) return 'Health Care';
  if (lowerText.match(/technology|tech|software|hardware|ai|internet|app|phone|computer/)) return 'Technology';
  if (lowerText.match(/sports|football|basketball|soccer|tennis|match|tournament|player/)) return 'Sports';
  if (lowerText.match(/entertainment|movie|music|actor|celebrity|hollywood|film/)) return 'Entertainment';
  if (lowerText.match(/business|economy|market|stock|company|ceo|finance/)) return 'Business';
  return 'General';
};

function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const API_URL = 'http://localhost:5000/api';

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_URL}/history`);
      setHistory(response.data);
    } catch (error) {
      console.error('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  const getResultColor = (prediction) => {
    return prediction === 'REAL' ? '#10b981' : '#ef4444';
  };

  const groupedHistory = history.reduce((acc, item) => {
    const category = getCategory(item.text);
    if (!acc[category]) acc[category] = [];
    acc[category].push(item);
    return acc;
  }, {});

  // Sort categories alphabetically but keep 'General' at the end
  const categories = Object.keys(groupedHistory).sort((a, b) => {
    if (a === 'General') return 1;
    if (b === 'General') return -1;
    return a.localeCompare(b);
  });

  return (
    <div className="history-page">
      <header className="history-header">
        <Link to="/" className="back-link">← Back to Detector</Link>
        <h1>Prediction History</h1>
        <p>View all your previous news authenticity checks categorized by topic</p>
      </header>

      <main className="history-main">
        {loading ? (
          <div className="loading">Loading predictions...</div>
        ) : history.length === 0 ? (
          <div className="empty-state">
            <p>No predictions yet. Go check some news!</p>
            <Link to="/" className="btn-new-check">Start Checking</Link>
          </div>
        ) : (
          <div className="history-container">
            <div className="history-stats">
              <div className="stat">
                <span className="stat-label">Total Checks</span>
                <span className="stat-value">{history.length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Real News</span>
                <span className="stat-value">{history.filter(h => h.prediction === 'REAL').length}</span>
              </div>
              <div className="stat">
                <span className="stat-label">Fake News</span>
                <span className="stat-value">{history.filter(h => h.prediction === 'FAKE').length}</span>
              </div>
            </div>

            <div className="categories-wrapper">
              {categories.map((category) => (
                <div key={category} className="category-section">
                  <h3 className="category-title">{category} ({groupedHistory[category].length})</h3>
                  <div className="predictions-list">
                    {groupedHistory[category].map((item) => (
                      <div key={item.id} className="prediction-card">
                        <div className="card-header">
                          <span 
                            className="prediction-badge"
                            style={{ backgroundColor: getResultColor(item.prediction), color: 'white' }}
                          >
                            {item.prediction}
                          </span>
                          <span className="confidence-badge">{item.confidence.toFixed(1)}% confident</span>
                        </div>
                        <div className="card-body">
                          <p className="prediction-text">{item.text}</p>
                        </div>
                        <div className="card-footer">
                          <span className="timestamp">{new Date(item.timestamp).toLocaleString()}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default History;
