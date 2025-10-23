/**
 * React Entry Point
 *
 * Mounts the Fractal VSM Observability Dashboard into the DOM
 *
 * Author: BMad
 * Date: 2025-01-20
 */

import React from 'react';
import ReactDOM from 'react-dom/client';
import './App.css';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
