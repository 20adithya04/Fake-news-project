import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './App.css';

function App() {
  const [newsText, setNewsText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const API_URL = 'http://localhost:5000/api';

  const handlePredict = async (event) => {
    event.preventDefault();
    if (!newsText.trim()) return alert('Please enter news text to analyse.');
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/predict`, { text: newsText });
      setResult(response.data);
    } catch (error) {
      alert(error.response?.data?.error || `Prediction failed: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const isReal = result?.prediction === 'REAL';

  return (
    <div className="app-shell">
      <div className="orb orb-one" /><div className="orb orb-two" /><div className="orb orb-three" />
      <main className="glass-dashboard">
        <nav className="topbar glass-panel">
          <div className="topbar-status">Fake News Detector</div>
          <Link to="/history" className="history-link">History <span>→</span></Link>
        </nav>

        <section className="workspace">
          <form onSubmit={handlePredict} className="analysis-card glass-panel">
            <div className="card-heading"><div><p className="step">NEWS ANALYSER</p><h2>Check an article</h2></div><span className="input-icon">✦</span></div>
            <label htmlFor="newsText">ARTICLE, CLAIM OR HEADLINE</label>
            <textarea id="newsText" value={newsText} onChange={(event) => setNewsText(event.target.value)} disabled={loading}
              placeholder="Paste the news content you would like to verify…" rows="8" />
            <div className="form-footer"><span>{newsText.trim().length} characters</span><button type="submit" disabled={loading}>{loading ? 'Analysing…' : 'Check credibility'} <b>→</b></button></div>
          </form>

          <aside className={`result-card glass-panel ${result ? (isReal ? 'real' : 'fake') : ''}`}>
            <p className="step">ANALYSIS RESULT</p>
            {result ? <>
              <div className="verdict-row"><div className="verdict-icon">{isReal ? '✓' : '!'}</div><div><p className="result-label">MODEL VERDICT</p><h2>{result.prediction}</h2></div></div>
              <div className="confidence"><div><span>CONFIDENCE</span><strong>{result.confidence}%</strong></div><div className="meter"><i style={{ width: `${result.confidence}%` }} /></div></div>
              <p className="result-note">Analysed {new Date(result.timestamp).toLocaleString()}</p>
            </> : <div className="empty-result"><div className="empty-orbit">✦</div><h2>Ready to analyse</h2><p>Your prediction and confidence score will appear here.</p></div>}
          </aside>
        </section>

        <footer><span>Fake News Detector</span><span>Built for educational analysis</span></footer>
      </main>
    </div>
  );
}

export default App;
