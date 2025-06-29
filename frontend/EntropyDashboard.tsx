import React, { useState } from 'react';
import { analyzeEntropy, generateEntropySignal, analyzePattern, getSummary } from './api';

const sampleData = Array.from({ length: 50 }, (_, i) => 100 + i * 0.2 + Math.random() * 2);

export default function EntropyDashboard() {
  const [priceData, setPriceData] = useState<number[]>(sampleData);
  const [entropy, setEntropy] = useState<any>(null);
  const [signal, setSignal] = useState<any>(null);
  const [patterns, setPatterns] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    const entropyRes = await analyzeEntropy(priceData);
    setEntropy(entropyRes);
    const signalRes = await generateEntropySignal(priceData);
    setSignal(signalRes);
    const patternRes = await analyzePattern(priceData);
    setPatterns(patternRes.patterns || []);
    const summaryRes = await getSummary();
    setSummary(summaryRes.summary || {});
    setLoading(false);
  };

  return (
    <div style={{ padding: 24, fontFamily: 'sans-serif' }}>
      <h2>Entropy Dashboard</h2>
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze Sample Data'}
      </button>
      <div style={{ marginTop: 24 }}>
        <h3>Entropy</h3>
        <pre>{JSON.stringify(entropy, null, 2)}</pre>
        <h3>Signal</h3>
        <pre>{JSON.stringify(signal, null, 2)}</pre>
        <h3>Patterns</h3>
        <pre>{JSON.stringify(patterns, null, 2)}</pre>
        <h3>System Summary</h3>
        <pre>{JSON.stringify(summary, null, 2)}</pre>
      </div>
    </div>
  );
} 