import React, { useState } from 'react';
import { gnnApi, DisruptionScenario, Prediction } from '../../services/gnnApi';

export const GNNPrediction: React.FC = () => {
  const [disrupted_nodes, setDisruptedNodes] = useState<string>('1,5,10');
  const [severity, setSeverity] = useState<number>(0.8);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const nodeIds = disrupted_nodes.split(',').map(id => parseInt(id.trim()));
      
      const scenario: DisruptionScenario = {
        disrupted_nodes: nodeIds,
        disrupted_edges: [],
        disruption_severity: severity,
      };
      
      const result = await gnnApi.predict(scenario);
      setPredictions(result.predictions);
      setSummary(result.summary);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">GNN Resilience Prediction</h2>
      
      {/* Input Form */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Disrupted Nodes (comma-separated IDs):
          </label>
          <input
            type="text"
            value={disrupted_nodes}
            onChange={(e) => setDisruptedNodes(e.target.value)}
            className="w-full px-4 py-2 border rounded-lg"
            placeholder="1,5,10,15"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2">
            Disruption Severity: {severity.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={severity}
            onChange={(e) => setSeverity(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
        
        <button
          onClick={handlePredict}
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          {loading ? 'Predicting...' : 'Predict Resilience'}
        </button>
      </div>
      
      {/* Error Message */}
      {error && (
        <div className="mb-4 p-4 bg-red-100 text-red-700 rounded-lg">
          Error: {error}
        </div>
      )}
      
      {/* Summary */}
      {summary && (
        <div className="mb-6 grid grid-cols-4 gap-4">
          <div className="p-4 bg-red-100 rounded-lg">
            <div className="text-2xl font-bold text-red-700">{summary.failed}</div>
            <div className="text-sm text-red-600">Failed</div>
          </div>
          <div className="p-4 bg-yellow-100 rounded-lg">
            <div className="text-2xl font-bold text-yellow-700">{summary.degraded}</div>
            <div className="text-sm text-yellow-600">Degraded</div>
          </div>
          <div className="p-4 bg-green-100 rounded-lg">
            <div className="text-2xl font-bold text-green-700">{summary.normal}</div>
            <div className="text-sm text-green-600">Normal</div>
          </div>
          <div className="p-4 bg-blue-100 rounded-lg">
            <div className="text-2xl font-bold text-blue-700">{summary.total_nodes}</div>
            <div className="text-sm text-blue-600">Total</div>
          </div>
        </div>
      )}
      
      {/* Predictions Table */}
      {predictions.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-2 text-left">Node ID</th>
                <th className="px-4 py-2 text-left">Name</th>
                <th className="px-4 py-2 text-left">Tier</th>
                <th className="px-4 py-2 text-left">Region</th>
                <th className="px-4 py-2 text-left">Prediction</th>
                <th className="px-4 py-2 text-left">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 20).map((pred) => (
                <tr key={pred.node_id} className="border-b hover:bg-gray-50">
                  <td className="px-4 py-2">{pred.node_id}</td>
                  <td className="px-4 py-2">{pred.node_name}</td>
                  <td className="px-4 py-2">Tier {pred.tier}</td>
                  <td className="px-4 py-2">{pred.region}</td>
                  <td className="px-4 py-2">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      pred.label === 0 ? 'bg-red-100 text-red-700' :
                      pred.label === 1 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-green-100 text-green-700'
                    }`}>
                      {pred.label_name}
                    </span>
                  </td>
                  <td className="px-4 py-2">
                    {(pred.probability[pred.label] * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
