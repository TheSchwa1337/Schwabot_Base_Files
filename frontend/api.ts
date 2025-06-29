import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export async function analyzeEntropy(priceData: number[]) {
  const res = await axios.post(`${API_BASE}/entropy/calculate`, priceData);
  return res.data;
}

export async function generateEntropySignal(priceData: number[]) {
  const res = await axios.post(`${API_BASE}/entropy/signal`, priceData);
  return res.data;
}

export async function analyzeBit(values: number[]) {
  const res = await axios.post(`${API_BASE}/bit/analyze`, { value: values[0] });
  return res.data;
}

export async function analyzeBitSequence(values: number[]) {
  const res = await axios.post(`${API_BASE}/bit/sequence`, values);
  return res.data;
}

export async function detectBitPatterns(bitSequence: number[]) {
  const res = await axios.post(`${API_BASE}/bit/patterns`, bitSequence);
  return res.data;
}

export async function analyzePattern(priceData: number[]) {
  const res = await axios.post(`${API_BASE}/pattern/detect`, priceData);
  return res.data;
}

export async function getSummary() {
  const res = await axios.get(`${API_BASE}/summary`);
  return res.data;
} 