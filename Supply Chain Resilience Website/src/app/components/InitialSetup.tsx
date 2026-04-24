import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { 
  Upload, 
  FileText, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  ArrowRight,
  Download,
  Play,
  AlertCircle
} from 'lucide-react';
import { gnnApi, PipelineStatus } from '../../services/gnnApi';

interface InitialSetupProps {
  onComplete: () => void;
}

export function InitialSetup({ onComplete }: InitialSetupProps) {
  const [step, setStep] = useState<'upload' | 'training' | 'complete'>('upload');
  const [companyName, setCompanyName] = useState('');
  const [nodesFile, setNodesFile] = useState<File | null>(null);
  const [edgesFile, setEdgesFile] = useState<File | null>(null);
  const [companyId, setCompanyId] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadStats, setUploadStats] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<PipelineStatus | null>(null);
  const [trainingError, setTrainingError] = useState<string | null>(null);

  // Poll for training status
  useEffect(() => {
    if (!taskId || !isTraining) return;

    const interval = setInterval(async () => {
      try {
        const statusData = await gnnApi.getPipelineStatus(taskId);
        setTrainingStatus(statusData);

        if (statusData.status === 'completed') {
          setIsTraining(false);
          setStep('complete');
          clearInterval(interval);
        } else if (statusData.status === 'failed') {
          setIsTraining(false);
          setTrainingError(statusData.error || 'Training failed');
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error fetching training status:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId, isTraining]);

  const handleFileChange = (type: 'nodes' | 'edges', file: File | null) => {
    if (type === 'nodes') {
      setNodesFile(file);
    } else {
      setEdgesFile(file);
    }
    setUploadError(null);
  };

  const handleUpload = async () => {
    if (!nodesFile || !edgesFile || !companyName.trim()) {
      setUploadError('Please provide company name and both CSV files');
      return;
    }

    setIsUploading(true);
    setUploadError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('nodes', nodesFile);
      formData.append('edges', edgesFile);
      formData.append('company_name', companyName);

      // Simulate smooth progress for better UX (especially on localhost)
      let simulatedProgress = 0;
      const progressInterval = setInterval(() => {
        simulatedProgress += Math.random() * 15;
        if (simulatedProgress > 90) simulatedProgress = 90;
        setUploadProgress(Math.floor(simulatedProgress));
      }, 200);

      // Create XMLHttpRequest for upload progress tracking
      const xhr = new XMLHttpRequest();

      // Track upload progress (will override simulated if real progress available)
      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable) {
          clearInterval(progressInterval);
          const percentComplete = Math.round((e.loaded / e.total) * 100);
          setUploadProgress(percentComplete);
          console.log(`Upload progress: ${percentComplete}%`);
        }
      });

      // Handle completion
      const uploadPromise = new Promise<any>((resolve, reject) => {
        xhr.addEventListener('load', () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            try {
              const data = JSON.parse(xhr.responseText);
              resolve(data);
            } catch (err) {
              reject(new Error('Failed to parse response'));
            }
          } else {
            reject(new Error(`Upload failed with status ${xhr.status}`));
          }
        });

        xhr.addEventListener('error', () => {
          reject(new Error('Network error during upload'));
        });

        xhr.addEventListener('abort', () => {
          reject(new Error('Upload cancelled'));
        });
      });

      // Send request
      xhr.open('POST', 'http://localhost:5000/api/upload-graph');
      xhr.send(formData);

      // Wait for completion
      const data = await uploadPromise;
      clearInterval(progressInterval);
      setUploadProgress(100);

      // Brief delay to show 100%
      await new Promise(resolve => setTimeout(resolve, 300));

      if (data.status === 'success') {
        setCompanyId(data.company_id);
        setUploadStats(data.stats);
        // Save company_id to localStorage immediately
        localStorage.setItem('company_id', data.company_id);
        console.log('✅ Saved company_id to localStorage:', data.company_id);
        setStep('training');
        // Auto-start training
        startTraining(data.company_id);
      } else {
        // Show validation errors if any
        if (data.errors && Array.isArray(data.errors)) {
          setUploadError(data.errors.join('; '));
        } else {
          setUploadError(data.message || 'Upload failed');
        }
      }
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload files');
      setUploadProgress(0);
    } finally {
      setIsUploading(false);
    }
  };

  const startTraining = async (cId: string) => {
    try {
      setTrainingError(null);
      setIsTraining(true);

      const response = await gnnApi.runCompletePipeline({
        num_scenarios: 2000,
        use_default_data: false,
        company_id: cId,
      });

      setTaskId(response.task_id);
      setTrainingStatus({
        status: 'running',
        progress: 0,
        message: response.message,
        stage: 'init',
      });
    } catch (err: any) {
      setTrainingError(err.message || 'Failed to start training');
      setIsTraining(false);
    }
  };

  const downloadTemplate = async (type: 'nodes' | 'edges') => {
    try {
      const response = await fetch(`http://localhost:5000/api/download-template/${type}`);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${type}_template.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Failed to download template:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-600 rounded-full mb-4">
            <Upload className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Supply Chain Resilience Platform
          </h1>
          <p className="text-lg text-gray-600">
            Upload your supply chain graph and train a custom GNN model
          </p>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center justify-center mb-8">
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 ${step === 'upload' ? 'text-blue-600' : 'text-green-600'}`}>
              {step === 'upload' ? (
                <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-semibold">1</div>
              ) : (
                <CheckCircle className="w-8 h-8" />
              )}
              <span className="font-medium">Upload Graph</span>
            </div>
            <ArrowRight className="w-5 h-5 text-gray-400" />
            <div className={`flex items-center gap-2 ${step === 'training' ? 'text-blue-600' : step === 'complete' ? 'text-green-600' : 'text-gray-400'}`}>
              {step === 'complete' ? (
                <CheckCircle className="w-8 h-8" />
              ) : step === 'training' ? (
                <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-semibold">2</div>
              ) : (
                <div className="w-8 h-8 rounded-full bg-gray-300 text-white flex items-center justify-center font-semibold">2</div>
              )}
              <span className="font-medium">Train Model</span>
            </div>
            <ArrowRight className="w-5 h-5 text-gray-400" />
            <div className={`flex items-center gap-2 ${step === 'complete' ? 'text-green-600' : 'text-gray-400'}`}>
              {step === 'complete' ? (
                <CheckCircle className="w-8 h-8" />
              ) : (
                <div className="w-8 h-8 rounded-full bg-gray-300 text-white flex items-center justify-center font-semibold">3</div>
              )}
              <span className="font-medium">Complete</span>
            </div>
          </div>
        </div>

        {/* Upload Step */}
        {step === 'upload' && (
          <Card>
            <CardHeader>
              <CardTitle>Upload Supply Chain Graph</CardTitle>
              <CardDescription>
                Provide your supply chain network data in CSV format
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Company Name */}
              <div className="space-y-2">
                <Label htmlFor="company-name">Company Name</Label>
                <Input
                  id="company-name"
                  placeholder="Enter your company name"
                  value={companyName}
                  onChange={(e) => setCompanyName(e.target.value)}
                />
              </div>

              {/* File Uploads */}
              <div className="grid md:grid-cols-2 gap-4">
                {/* Nodes File */}
                <div className="space-y-2">
                  <Label htmlFor="nodes-file">Nodes CSV</Label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors">
                    <FileText className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                    <Input
                      id="nodes-file"
                      type="file"
                      accept=".csv"
                      onChange={(e) => handleFileChange('nodes', e.target.files?.[0] || null)}
                      className="hidden"
                    />
                    <label htmlFor="nodes-file" className="cursor-pointer">
                      <span className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                        {nodesFile ? nodesFile.name : 'Choose file'}
                      </span>
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Contains node attributes
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadTemplate('nodes')}
                    className="w-full"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Template
                  </Button>
                </div>

                {/* Edges File */}
                <div className="space-y-2">
                  <Label htmlFor="edges-file">Edges CSV</Label>
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors">
                    <FileText className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                    <Input
                      id="edges-file"
                      type="file"
                      accept=".csv"
                      onChange={(e) => handleFileChange('edges', e.target.files?.[0] || null)}
                      className="hidden"
                    />
                    <label htmlFor="edges-file" className="cursor-pointer">
                      <span className="text-sm text-blue-600 hover:text-blue-700 font-medium">
                        {edgesFile ? edgesFile.name : 'Choose file'}
                      </span>
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Contains edge connections
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadTemplate('edges')}
                    className="w-full"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Template
                  </Button>
                </div>
              </div>

              {/* Error Alert */}
              {uploadError && (
                <Alert variant="destructive">
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>{uploadError}</AlertDescription>
                </Alert>
              )}

              {/* Upload Button */}
              <Button
                onClick={handleUpload}
                disabled={isUploading || !nodesFile || !edgesFile || !companyName.trim()}
                className="w-full"
                size="lg"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="h-5 w-5 mr-2" />
                    Upload and Start Training
                  </>
                )}
              </Button>

              {/* Upload Loading Animation */}
              {isUploading && (
                <div className="space-y-4 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-300 rounded-lg">
                  <div className="flex flex-col items-center justify-center">
                    {/* Animated Spinner */}
                    <div className="relative w-20 h-20 mb-4">
                      <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
                      <div className="absolute inset-0 border-4 border-blue-600 rounded-full border-t-transparent animate-spin"></div>
                      <div className="absolute inset-2 border-4 border-indigo-400 rounded-full border-t-transparent animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1s' }}></div>
                    </div>
                    
                    {/* Loading Text */}
                    <div className="text-center space-y-2">
                      <h3 className="text-lg font-semibold text-blue-900">
                        {uploadProgress < 100 ? 'Uploading Your Files...' : 'Processing Graph Data...'}
                      </h3>
                      <p className="text-sm text-blue-700">
                        {uploadProgress < 30 && 'Reading CSV files...'}
                        {uploadProgress >= 30 && uploadProgress < 60 && 'Validating graph structure...'}
                        {uploadProgress >= 60 && uploadProgress < 90 && 'Checking data integrity...'}
                        {uploadProgress >= 90 && 'Finalizing upload...'}
                      </p>
                      
                      {/* Animated Dots */}
                      <div className="flex items-center justify-center gap-1 mt-2">
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                        <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Info Box */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-2">Required CSV Format:</h4>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li><strong>Nodes:</strong> node_id, tier (0-3), capacity (positive), cost_factor, risk_level (0-1), reliability (0-1), x/y or latitude/longitude, region (optional)</li>
                  <li><strong>Edges:</strong> source, target, capacity_share (optional), lead_time (optional), cost (optional)</li>
                  <li><strong>Minimum:</strong> 20 nodes required, 50+ recommended for better accuracy</li>
                </ul>
              </div>

              {/* Validation Info */}
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-xs">
                  <p className="font-semibold mb-1">Validation checks:</p>
                  <ul className="list-disc list-inside space-y-0.5">
                    <li>All required columns present</li>
                    <li>Capacity values are positive</li>
                    <li>Risk level and reliability between 0-1</li>
                    <li>Tier values are 0, 1, 2, or 3</li>
                    <li>Minimum graph size requirements met</li>
                  </ul>
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        )}

        {/* Training Step */}
        {step === 'training' && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                Training GNN Model
              </CardTitle>
              <CardDescription>
                Training your custom model on {uploadStats?.num_nodes} nodes and {uploadStats?.num_edges} edges
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Upload Success */}
              <Alert>
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription>
                  Graph uploaded successfully! Training has started automatically.
                </AlertDescription>
              </Alert>

              {/* Training Status with Animation */}
              {trainingStatus && (
                <div className="space-y-4 p-6 bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-300 rounded-lg">
                  <div className="flex flex-col items-center justify-center">
                    {/* Animated Training Spinner */}
                    <div className="relative w-24 h-24 mb-4">
                      {/* Outer rotating ring */}
                      <div className="absolute inset-0 border-4 border-purple-200 rounded-full"></div>
                      <div className="absolute inset-0 border-4 border-purple-600 rounded-full border-t-transparent animate-spin"></div>
                      {/* Middle rotating ring */}
                      <div className="absolute inset-3 border-4 border-pink-400 rounded-full border-t-transparent animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                      {/* Inner pulsing circle */}
                      <div className="absolute inset-6 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full animate-pulse"></div>
                    </div>
                    
                    {/* Training Status Text */}
                    <div className="text-center space-y-2">
                      <h3 className="text-lg font-semibold text-purple-900">
                        Training GNN Models...
                      </h3>
                      <p className="text-sm text-purple-700 font-medium">
                        Stage: {trainingStatus.stage}
                      </p>
                      <p className="text-sm text-purple-600">
                        {trainingStatus.message}
                      </p>
                      
                      {/* Animated Progress Indicator */}
                      <div className="flex items-center justify-center gap-2 mt-3">
                        <div className="flex gap-1">
                          {[...Array(5)].map((_, i) => (
                            <div
                              key={i}
                              className="w-2 h-8 bg-purple-600 rounded-full animate-pulse"
                              style={{ 
                                animationDelay: `${i * 150}ms`,
                                animationDuration: '1s'
                              }}
                            ></div>
                          ))}
                        </div>
                      </div>
                      
                      {/* Time Info */}
                      {trainingStatus.started_at && (
                        <div className="text-xs text-purple-500 mt-3 space-y-1">
                          <p>Started: {new Date(trainingStatus.started_at).toLocaleTimeString()}</p>
                          {trainingStatus.updated_at && (
                            <p>Last update: {new Date(trainingStatus.updated_at).toLocaleTimeString()}</p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Training Error */}
              {trainingError && (
                <Alert variant="destructive">
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>{trainingError}</AlertDescription>
                </Alert>
              )}

              {/* Info */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-2">What's happening?</h4>
                <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
                  <li>Generating 1,000 disruption scenarios</li>
                  <li>Training 6 GNN models (GAT, GCN, GraphSAGE, GIN, TransformerConv, GINE)</li>
                  <li>Benchmarking against ML models</li>
                  <li>Estimated time: 30-60 minutes</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Complete Step */}
        {step === 'complete' && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-green-600">
                <CheckCircle className="h-6 w-6" />
                Setup Complete!
              </CardTitle>
              <CardDescription>
                Your custom GNN model has been trained successfully
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
                <CheckCircle className="h-16 w-16 text-green-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-green-900 mb-2">
                  Training Completed Successfully!
                </h3>
                <p className="text-green-800 mb-4">
                  Your supply chain resilience model is ready to use
                </p>
                {trainingStatus?.results && (
                  <p className="text-sm text-green-700">
                    Trained {trainingStatus.results.length} models
                  </p>
                )}
              </div>

              <Button
                onClick={onComplete}
                className="w-full"
                size="lg"
              >
                <ArrowRight className="h-5 w-5 mr-2" />
                Go to Dashboard
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
