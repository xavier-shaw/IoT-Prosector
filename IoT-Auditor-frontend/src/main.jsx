import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import 'bootstrap/dist/css/bootstrap.css';
import { createRoot } from "react-dom/client";

window.BACKEND_ADDRESS = process.env.NODE_ENV === 'production' ? 'https://leanprivacyreview.herokuapp.com/api' : 'http://localhost:9990/api';

const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
