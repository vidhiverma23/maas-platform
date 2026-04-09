import React, { useState, useEffect } from 'react';
import { Play, Loader2 } from 'lucide-react';
import api from '../api/client';

export default function Inference() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [inputData, setInputData] = useState('[\n  [5.1, 3.5, 1.4, 0.2]\n]');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await api.getModels();
        setModels(data.data || []);
        if (data.data?.length > 0) {
          setSelectedModel(data.data[0].name);
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchModels();
  }, []);

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      // Parse the JSON input
      const parsedData = JSON.parse(inputData);
      
      const payload = {
        model_id: selectedModel,
        input_data: parsedData,
        parameters: {}
      };

      const result = await api.predict(payload);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Error parsing JSON input');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container" style={{ padding: '2rem 1.5rem', maxWidth: '800px' }}>
      <header style={{ marginBottom: '2rem' }}>
        <h1>Interactive Inference</h1>
        <p>Test your deployed machine learning models in real-time.</p>
      </header>

      <div className="glass-panel" style={{ padding: '2rem' }}>
        <form onSubmit={handlePredict} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>Target Model</label>
            <select 
              value={selectedModel} 
              onChange={(e) => setSelectedModel(e.target.value)}
              required
            >
              <option value="" disabled>Select a model...</option>
              {models.map((m) => (
                <option key={m.id} value={m.name}>{m.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>Input Payload (JSON Array of Arrays)</label>
            <textarea 
              rows={6}
              value={inputData}
              onChange={(e) => setInputData(e.target.value)}
              placeholder="[\n  [value1, value2, value3]\n]"
              style={{ fontFamily: 'monospace' }}
              required
            />
          </div>

          <button type="submit" className="btn btn-primary" disabled={loading || !selectedModel} style={{ alignSelf: 'flex-start' }}>
            {loading ? <><Loader2 size={18} className="spin" /> Processing...</> : <><Play size={18} /> Run Prediction</>}
          </button>
        </form>
      </div>

      {/* Results Box */}
      {(prediction || error) && (
        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1rem' }}>Results</h3>
          <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: `4px solid ${error ? 'var(--accent-danger)' : 'var(--accent-success)'}` }}>
            {error ? (
              <div style={{ color: 'var(--accent-danger)' }}>
                <strong>Error: </strong> {error}
              </div>
            ) : (
              <div>
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
                  <span className="badge badge-success">Success</span>
                  <span className="badge" style={{ background: 'var(--bg-secondary)' }}>
                    Latency: {prediction.metadata?.duration_ms?.toFixed(2) || '?'}ms
                  </span>
                </div>
                <pre style={{ 
                  background: 'var(--bg-secondary)', 
                  padding: '1rem', 
                  borderRadius: 'var(--radius-sm)',
                  overflowX: 'auto',
                  fontSize: '0.85rem'
                }}>
                  {JSON.stringify(prediction.predictions, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
