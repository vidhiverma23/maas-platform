import axios from 'axios';

// Create an Axios instance with base URL pointing to the FastAPI backend
// If deployed on Vercel, this should point to the production backend URL using import.meta.env
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API Methods
export const api = {
  // System
  checkHealth: async () => {
    const response = await client.get('/health');
    return response.data;
  },

  // Models
  getModels: async (page = 1, pageSize = 10) => {
    const response = await client.get(`/api/v1/models?page=${page}&page_size=${pageSize}`);
    return response.data;
  },

  getModelById: async (id) => {
    const response = await client.get(`/api/v1/models/${id}`);
    return response.data;
  },

  registerModel: async (modelData) => {
    const response = await client.post('/api/v1/models', modelData);
    return response.data;
  },

  uploadModelVersion: async (modelId, formData) => {
    // Override Content-Type for this specific request
    const response = await client.post(`/api/v1/models/${modelId}/versions`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Inference
  predict: async (predictionRequest) => {
    const response = await client.post('/api/v1/predict', predictionRequest);
    return response.data;
  },
};

export default api;
