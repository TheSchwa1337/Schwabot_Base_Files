import React, { useState, useEffect } from 'react';
import { X, Key, Settings, Shield, Cpu, Zap, AlertTriangle } from 'lucide-react';

interface SettingsDrawerProps {
  open: boolean;
  onClose: () => void;
}

interface APIKeyData {
  exchange: string;
  apiKey: string;
  apiSecret: string;
  passphrase?: string;
  testnet: boolean;
}

interface SystemConfig {
  sustainment_threshold: number;
  update_interval: number;
  gpu_enabled: boolean;
}

const SettingsDrawer: React.FC<SettingsDrawerProps> = ({ open, onClose }) => {
  const [activeTab, setActiveTab] = useState('api');
  const [apiKeys, setApiKeys] = useState<APIKeyData[]>([]);
  const [newApiKey, setNewApiKey] = useState<APIKeyData>({
    exchange: 'coinbase',
    apiKey: '',
    apiSecret: '',
    passphrase: '',
    testnet: true
  });
  const [systemConfig, setSystemConfig] = useState<SystemConfig>({
    sustainment_threshold: 0.65,
    update_interval: 0.1,
    gpu_enabled: false
  });
  const [registrationStatus, setRegistrationStatus] = useState<string>('');
  const [configStatus, setConfigStatus] = useState<string>('');

  // Hash API key using Web Crypto API
  const hashApiKey = async (apiKey: string): Promise<string> => {
    const encoder = new TextEncoder();
    const data = encoder.encode(apiKey);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  };

  // Register API key with backend
  const registerApiKey = async (keyData: APIKeyData) => {
    try {
      setRegistrationStatus('Registering...');
      
      // Hash the API key before sending
      const hashedKey = await hashApiKey(keyData.apiKey);
      
      const response = await fetch('/api/register-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          api_key_hash: hashedKey,
          exchange: keyData.exchange,
          testnet: keyData.testnet
        })
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setRegistrationStatus(`✅ API key registered successfully (ID: ${result.key_id})`);
        setApiKeys([...apiKeys, { ...keyData, apiKey: '***REGISTERED***' }]);
        setNewApiKey({
          exchange: 'coinbase',
          apiKey: '',
          apiSecret: '',
          passphrase: '',
          testnet: true
        });
      } else {
        setRegistrationStatus(`❌ Registration failed: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      setRegistrationStatus(`❌ Registration error: ${error.message}`);
    }
  };

  // Update system configuration
  const updateSystemConfig = async (config: SystemConfig) => {
    try {
      setConfigStatus('Updating...');
      
      const response = await fetch('/api/configure', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setConfigStatus(`✅ Configuration updated: ${Object.keys(result.changes).join(', ')}`);
        setSystemConfig(config);
      } else {
        setConfigStatus(`❌ Configuration failed: ${result.message || 'Unknown error'}`);
      }
    } catch (error) {
      setConfigStatus(`❌ Configuration error: ${error.message}`);
    }
  };

  // Clear status messages after 5 seconds
  useEffect(() => {
    if (registrationStatus) {
      const timer = setTimeout(() => setRegistrationStatus(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [registrationStatus]);

  useEffect(() => {
    if (configStatus) {
      const timer = setTimeout(() => setConfigStatus(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [configStatus]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black bg-opacity-50" onClick={onClose} />
      
      {/* Drawer */}
      <div className="relative ml-auto h-full w-96 bg-gray-900 shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Settings</h2>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded"
          >
            <X size={20} />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-700">
          <button
            onClick={() => setActiveTab('api')}
            className={`flex-1 p-3 text-sm font-medium ${
              activeTab === 'api'
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Key size={16} className="inline mr-2" />
            API Keys
          </button>
          <button
            onClick={() => setActiveTab('system')}
            className={`flex-1 p-3 text-sm font-medium ${
              activeTab === 'system'
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            <Settings size={16} className="inline mr-2" />
            System
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'api' && (
            <div className="space-y-6">
              {/* Security Notice */}
              <div className="bg-yellow-900 border border-yellow-600 rounded-lg p-3">
                <div className="flex items-start">
                  <Shield size={16} className="text-yellow-400 mt-0.5 mr-2" />
                  <div className="text-sm text-yellow-200">
                    <p className="font-medium">Security Notice</p>
                    <p className="mt-1">API keys are hashed with SHA-256 before transmission. Only use testnet keys for initial setup.</p>
                  </div>
                </div>
              </div>

              {/* API Key Registration Form */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Add New API Key</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Exchange
                    </label>
                    <select
                      value={newApiKey.exchange}
                      onChange={(e) => setNewApiKey({ ...newApiKey, exchange: e.target.value })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                    >
                      <option value="coinbase">Coinbase Pro</option>
                      <option value="binance">Binance</option>
                      <option value="kraken">Kraken</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      API Key
                    </label>
                    <input
                      type="password"
                      value={newApiKey.apiKey}
                      onChange={(e) => setNewApiKey({ ...newApiKey, apiKey: e.target.value })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                      placeholder="Enter your API key"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      API Secret
                    </label>
                    <input
                      type="password"
                      value={newApiKey.apiSecret}
                      onChange={(e) => setNewApiKey({ ...newApiKey, apiSecret: e.target.value })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                      placeholder="Enter your API secret"
                    />
                  </div>

                  {newApiKey.exchange === 'coinbase' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Passphrase
                      </label>
                      <input
                        type="password"
                        value={newApiKey.passphrase}
                        onChange={(e) => setNewApiKey({ ...newApiKey, passphrase: e.target.value })}
                        className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                        placeholder="Enter your passphrase"
                      />
                    </div>
                  )}

                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="testnet"
                      checked={newApiKey.testnet}
                      onChange={(e) => setNewApiKey({ ...newApiKey, testnet: e.target.checked })}
                      className="rounded border-gray-600 bg-gray-700 text-blue-600"
                    />
                    <label htmlFor="testnet" className="ml-2 text-sm text-gray-300">
                      Use Testnet/Sandbox (Recommended)
                    </label>
                  </div>

                  <button
                    onClick={() => registerApiKey(newApiKey)}
                    disabled={!newApiKey.apiKey || !newApiKey.apiSecret}
                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white py-2 px-4 rounded font-medium"
                  >
                    Register API Key
                  </button>

                  {registrationStatus && (
                    <div className="text-sm p-2 rounded bg-gray-700 text-gray-300">
                      {registrationStatus}
                    </div>
                  )}
                </div>
              </div>

              {/* Registered Keys */}
              {apiKeys.length > 0 && (
                <div className="bg-gray-800 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-4">Registered API Keys</h3>
                  <div className="space-y-2">
                    {apiKeys.map((key, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-700 rounded">
                        <div>
                          <span className="text-white font-medium">{key.exchange}</span>
                          <span className="text-gray-400 ml-2">
                            {key.testnet ? '(Testnet)' : '(Live)'}
                          </span>
                        </div>
                        <span className="text-green-400 text-sm">✓ Active</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'system' && (
            <div className="space-y-6">
              {/* System Configuration */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-4">System Configuration</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Sustainment Threshold
                    </label>
                    <input
                      type="number"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={systemConfig.sustainment_threshold}
                      onChange={(e) => setSystemConfig({ 
                        ...systemConfig, 
                        sustainment_threshold: parseFloat(e.target.value) 
                      })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Minimum sustainment index required for trading (0.1 - 1.0)
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Update Interval (seconds)
                    </label>
                    <input
                      type="number"
                      min="0.01"
                      max="1.0"
                      step="0.01"
                      value={systemConfig.update_interval}
                      onChange={(e) => setSystemConfig({ 
                        ...systemConfig, 
                        update_interval: parseFloat(e.target.value) 
                      })}
                      className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      How often to update system state (0.01 - 1.0 seconds)
                    </p>
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <label className="text-sm font-medium text-gray-300">
                        GPU Acceleration
                      </label>
                      <p className="text-xs text-gray-500">
                        Enable CUDA acceleration for hash processing
                      </p>
                    </div>
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="gpu_enabled"
                        checked={systemConfig.gpu_enabled}
                        onChange={(e) => setSystemConfig({ 
                          ...systemConfig, 
                          gpu_enabled: e.target.checked 
                        })}
                        className="rounded border-gray-600 bg-gray-700 text-blue-600"
                      />
                      <label htmlFor="gpu_enabled" className="ml-2">
                        {systemConfig.gpu_enabled ? (
                          <Zap size={16} className="text-yellow-400" />
                        ) : (
                          <Cpu size={16} className="text-gray-400" />
                        )}
                      </label>
                    </div>
                  </div>

                  <button
                    onClick={() => updateSystemConfig(systemConfig)}
                    className="w-full bg-green-600 hover:bg-green-700 text-white py-2 px-4 rounded font-medium"
                  >
                    Update Configuration
                  </button>

                  {configStatus && (
                    <div className="text-sm p-2 rounded bg-gray-700 text-gray-300">
                      {configStatus}
                    </div>
                  )}
                </div>
              </div>

              {/* Performance Settings */}
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-white mb-4">Performance</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Hash Queue Management</span>
                    <span className="text-green-400">Optimized</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Memory Management</span>
                    <span className="text-green-400">Active</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">WebSocket Streaming</span>
                    <span className="text-green-400">4 FPS</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Signal Dispatch</span>
                    <span className="text-green-400">Enabled</span>
                  </div>
                </div>
              </div>

              {/* Safety Warning */}
              <div className="bg-red-900 border border-red-600 rounded-lg p-3">
                <div className="flex items-start">
                  <AlertTriangle size={16} className="text-red-400 mt-0.5 mr-2" />
                  <div className="text-sm text-red-200">
                    <p className="font-medium">Trading Safety</p>
                    <p className="mt-1">
                      Always test with paper trading or testnet before using real funds. 
                      Monitor system performance and set appropriate risk limits.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SettingsDrawer; 