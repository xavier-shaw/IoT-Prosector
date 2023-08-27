import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import 'bootstrap/dist/css/bootstrap.css';
import { createRoot } from "react-dom/client";

window.BACKEND_ADDRESS = 'http://localhost:9990/api';
window.HARDWARE_ADDRESS = 'https://localhost:8000'; 
const container = document.getElementById('root');
const root = createRoot(container);
root.render(<App />);
