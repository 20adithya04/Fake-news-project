import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';
import History from './History';
import './index.css';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <React.StrictMode>
    <Router basename="/Fake-news-project">
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </Router>
  </React.StrictMode>
);