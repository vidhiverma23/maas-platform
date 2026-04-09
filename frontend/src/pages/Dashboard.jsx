import React, { useState, useEffect } from 'react';
import { Database, ServerCog, Activity, Clock } from 'lucide-react';
import api from '../api/client';

export default function Dashboard() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const data = await api.getModels();
      setModels(data.data || []); // Assuming paginated response with a data array
    } catch (error) {
      console.error('Failed to fetch models', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ padding: '2rem 1.5rem' }}>
      <header style={{ marginBottom: '3rem' }}>
        <h1>Model Registry</h1>
        <p>Manage and monitor your deployed machine learning models.</p>
      </header>

      <section className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))' }}>
        {/* Stat Cards */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ padding: '1rem', background: 'rgba(99, 102, 241, 0.1)', borderRadius: '12px' }}>
              <Database size={24} color="var(--accent-primary)" />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-secondary)' }}>Total Models</h3>
              <p style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-primary)' }}>{models.length}</p>
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '12px' }}>
              <ServerCog size={24} color="var(--accent-success)" />
            </div>
            <div>
              <h3 style={{ margin: 0, fontSize: '1rem', color: 'var(--text-secondary)' }}>System Status</h3>
              <p style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700, color: 'var(--text-primary)' }}>Healthy</p>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: '3rem' }}>
        <h2>Available Models</h2>
        {loading ? (
          <p>Loading registry...</p>
        ) : models.length > 0 ? (
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))' }}>
            {models.map((model) => (
              <div key={model.id} className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <h3 style={{ margin: 0 }}>{model.name}</h3>
                  <span className="badge badge-success">{model.model_type}</span>
                </div>
                <p style={{ margin: 0, fontSize: '0.9rem' }}>{model.description || 'No description provided.'}</p>
                <div style={{ display: 'flex', gap: '1rem', marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--glass-border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-tertiary)' }}>
                    <Activity size={14} /> Active
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-tertiary)' }}>
                    <Clock size={14} /> {new Date(model.created_at).toLocaleDateString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>
            <p style={{ margin: 0 }}>No models registered yet. Go upload one via the API!</p>
          </div>
        )}
      </section>
    </div>
  );
}
