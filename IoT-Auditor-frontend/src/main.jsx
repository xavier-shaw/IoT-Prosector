import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import 'bootstrap/dist/css/bootstrap.css';
import { createRoot } from "react-dom/client";

window.BACKEND_ADDRESS = 'http://localhost:9990/api';
window.HARDWARE_ADDRESS = 'http://127.0.0.1:8000'; 
const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
