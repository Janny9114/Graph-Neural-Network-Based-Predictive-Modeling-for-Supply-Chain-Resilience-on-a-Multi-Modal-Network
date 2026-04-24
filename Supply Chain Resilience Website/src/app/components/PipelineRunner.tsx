import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Play, CheckCircle, XCircle, Loader2, RefreshCw } from 'lucide-react';
import { gnnApi, PipelineStatus } from '../../services/gnnApi';

export function PipelineRunner() {
  const [isRunning, setIsRunning] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Poll for status updates
  useEffect(() => {
    if (!taskId || !isRunning) return;

    const interval = setInterval(async () => {
      try {
        const statusData = await gnnApi.getPipelineStatus(taskId);
        setStatus(statusData);

        // Stop polling if completed or failed
        if (statusData.status === 'completed' || statusData.status === 'failed') {
          setIsRunning(false);
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error fetching pipeline status:', err);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [taskId, isRunning]);

  const handleRunPipeline = async () => {
    try {
      setError(null);
      setIsRunning(true);
      setStatus(null);

      const response = await gnnApi.runCompletePipeline({
        num_scenarios: 2000,
        use_default_data: true,
      });

      setTaskId(response.task_id);
      setStatus({
        status: 'running',
        progress: 0,
        message: response.message,
        stage: 'init',
      });
    } catch (err: any) {
      setError(err.message || 'Failed to start pipeline');
      setIsRunning(false);
    }
  };

  const getStatusColor = () => {
    if (!status) return 'text-gray-500';
    switch (status.status) {
      case 'completed':
        return 'text-green-600';
      case 'failed':
        return 'text-red-600';
      case 'running':
        return 'text-blue-600';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusIcon = () => {
    if (!status) return null;
    switch (status.status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-600" />;
      case 'running':
        return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />;
      default:
        return null;
    }
  };

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Play className="h-5 w-5" />
          Complete Training Pipeline
        </CardTitle>
        <CardDescription>
          Run the complete training pipeline to generate scenarios, train GNN models, and benchmark against ML models
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Run Button */}
        <div className="flex items-center gap-4">
          <Button
            onClick={handleRunPipeline}
            disabled={isRunning}
            className="flex items-center gap-2"
          >
            {isRunning ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Running...
              </>
            ) : (
              <>
                <Play className="h-4 w-4" />
                Run Pipeline
              </>
            )}
          </Button>

          {status && status.status === 'completed' && (
            <Button
              variant="outline"
              onClick={handleRunPipeline}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Run Again
            </Button>
          )}
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Status Display */}
        {status && (
          <div className="space-y-4">
            {/* Status Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon()}
                <span className={`font-medium ${getStatusColor()}`}>
                  {status.status.charAt(0).toUpperCase() + status.status.slice(1)}
                </span>
              </div>
              <span className="text-sm text-gray-500">
                {status.progress}% Complete
              </span>
            </div>

            {/* Progress Bar */}
            <Progress value={status.progress} className="w-full" />

            {/* Status Message */}
            <div className="text-sm text-gray-600">
              <p className="font-medium">Stage: {status.stage}</p>
              <p>{status.message}</p>
            </div>

            {/* Timestamps */}
            {status.started_at && (
              <div className="text-xs text-gray-500 space-y-1">
                <p>Started: {new Date(status.started_at).toLocaleString()}</p>
                {status.updated_at && (
                  <p>Updated: {new Date(status.updated_at).toLocaleString()}</p>
                )}
                {status.completed_at && (
                  <p>Completed: {new Date(status.completed_at).toLocaleString()}</p>
                )}
              </div>
            )}

            {/* Results Summary */}
            {status.status === 'completed' && status.results && (
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <h4 className="font-semibold text-green-900 mb-2">
                  Pipeline Completed Successfully! 🎉
                </h4>
                <p className="text-sm text-green-800">
                  Trained {status.results.length} models. Results are now available in the Model Comparison Table.
                </p>
              </div>
            )}

            {/* Error Details */}
            {status.status === 'failed' && status.error && (
              <Alert variant="destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>
                  <p className="font-medium">Pipeline Failed</p>
                  <p className="text-sm mt-1">{status.error}</p>
                  {status.traceback && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-xs">
                        Show traceback
                      </summary>
                      <pre className="mt-2 text-xs overflow-auto max-h-40 bg-red-50 p-2 rounded">
                        {status.traceback}
                      </pre>
                    </details>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>
        )}

        {/* Info Box */}
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-semibold text-blue-900 mb-2">What does this do?</h4>
          <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
            <li>Generates 1,000 disruption scenarios</li>
            <li>Trains 6 GNN models (GAT, GCN, GraphSAGE, GIN, TransformerConv, GINE)</li>
            <li>Benchmarks against 4 ML models (Random Forest, Gradient Boosting, SVM, Logistic Regression)</li>
            <li>Saves results and trained models</li>
            <li>Estimated time: 30-60 minutes</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}
