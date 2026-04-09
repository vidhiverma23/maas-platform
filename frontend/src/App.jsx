import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, BrainCircuit } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Inference from './pages/Inference';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-primary)' }}>
        
        {/* Sidebar */}
        <aside style={{ 
          width: '260px', 
          borderRight: '1px solid var(--border-color)',
          background: 'var(--bg-secondary)',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ padding: '2rem 1.5rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ 
              width: '32px', height: '32px', 
              borderRadius: '8px', 
              background: 'linear-gradient(135deg, var(--accent-primary), #a855f7)',
              display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
              <BrainCircuit size={18} color="white" />
            </div>
            <h2 style={{ fontSize: '1.25rem', margin: 0, fontWeight: 700, letterSpacing: '-0.03em' }}>
              MaaS <span style={{ color: 'var(--text-secondary)', fontWeight: 400 }}>Platform</span>
            </h2>
          </div>

          <nav style={{ padding: '0 1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <NavLink 
              to="/" 
              style={({ isActive }) => ({
                display: 'flex', alignItems: 'center', gap: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-sm)',
                textDecoration: 'none',
                color: isActive ? 'white' : 'var(--text-secondary)',
                background: isActive ? 'rgba(255,255,255,0.05)' : 'transparent',
                fontWeight: isActive ? 500 : 400,
                transition: 'all 0.2s'
              })}
            >
              <LayoutDashboard size={18} /> Dashboard
            </NavLink>
            <NavLink 
              to="/inference" 
              style={({ isActive }) => ({
                display: 'flex', alignItems: 'center', gap: '0.75rem',
                padding: '0.75rem 1rem',
                borderRadius: 'var(--radius-sm)',
                textDecoration: 'none',
                color: isActive ? 'white' : 'var(--text-secondary)',
                background: isActive ? 'rgba(255,255,255,0.05)' : 'transparent',
                fontWeight: isActive ? 500 : 400,
                transition: 'all 0.2s'
              })}
            >
              <BrainCircuit size={18} /> Inference Lab
            </NavLink>
          </nav>
        </aside>

        {/* Main Content */}
        <main style={{ flex: 1, overflowY: 'auto' }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/inference" element={<Inference />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
