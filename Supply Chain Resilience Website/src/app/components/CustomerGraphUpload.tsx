// src/app/components/CustomGraphUpload.tsx

import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { useState } from 'react';
import { Upload, Download, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

export function CustomGraphUpload() {
  const [nodesFile, setNodesFile] = useState<File | null>(null);
  const [edgesFile, setEdgesFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string>('');
  const [companyId, setCompanyId] = useState<string>('');

  const handleUpload = async () => {
    if (!nodesFile || !edgesFile) {
      setStatus('Please upload both nodes and edges CSV files');
      return;
    }

    if (!companyId.trim()) {
      setStatus('Please enter a company name');
      return;
    }

    setUploading(true);
    setStatus('Uploading files...');
    setProgress(5);

    const formData = new FormData();
    formData.append('nodes', nodesFile);
    formData.append('edges', edgesFile);
    formData.append('company_name', companyId);

    try {
      // Upload files
      const response = await fetch('http://localhost:5000/api/upload-graph', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        setStatus(`Files uploaded! Found ${data.stats.num_nodes} nodes, ${data.stats.num_edges} edges`);
        setProgress(20);
        
        // Start scenario generation
        await generateScenarios(data.company_id);
      } else {
        setStatus(`Error: ${data.message}`);
        setUploading(false);
      }
    } catch (error: any) {
      setStatus(`Error: ${error.message}`);
      setUploading(false);
    }
  };

  const generateScenarios = async (companyId: string) => {
    setStatus('Starting scenario generation and model training...');
    setProgress(30);

    try {
      const response = await fetch('http://localhost:5000/api/generate-scenarios', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_id: companyId, num_scenarios: 10000 })
      });

      const data = await response.json();
      
      if (data.status === 'started') {
        // Poll for progress
        pollProgress(data.task_id, companyId);
      } else {
        setStatus(`Error: ${data.message}`);
        setUploading(false);
      }
    } catch (error: any) {
      setStatus(`Error: ${error.message}`);
      setUploading(false);
    }
  };

  const pollProgress = async (taskId: string, companyId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/task-status/${taskId}`);
        const data = await response.json();

        setProgress(data.progress || 0);
        setStatus(data.message || 'Processing...');

        if (data.status === 'completed') {
          clearInterval(interval);
          setProgress(100);
          setStatus('Training complete! Your custom model is ready.');
          setUploading(false);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          setStatus(`Error: ${data.error || 'Training failed'}`);
          setUploading(false);
        }
      } catch (error: any) {
        clearInterval(interval);
        setStatus(`Error: ${error.message}`);
        setUploading(false);
      }
    }, 3000);
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
    } catch (error: any) {
      console.error('Download failed:', error);
    }
  };

  const handleFileChange = (type: 'nodes' | 'edges', event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (type === 'nodes') {
        setNodesFile(file);
      } else {
        setEdgesFile(file);
      }
    }
  };

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5 text-blue-600" />
          Upload Custom Supply Chain Graph
        </CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          Train a custom GNN model on your company's supply chain data
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Instructions */}
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <p className="font-semibold mb-1">Requirements:</p>
            <ul className="text-xs space-y-1 list-disc list-inside">
              <li>Minimum 50 nodes recommended (100+ ideal)</li>
              <li>CSV format with required columns</li>
              <li>Training takes 2-4 hours</li>
            </ul>
          </AlertDescription>
        </Alert>

        {/* Company Name */}
        <div className="space-y-2">
          <Label htmlFor="company-name">Company Name</Label>
          <Input
            id="company-name"
            type="text"
            value={companyId}
            onChange={(e) => setCompanyId(e.target.value)}
            placeholder="e.g., TechCorp Electronics"
            disabled={uploading}
          />
        </div>

        {/* File Uploads */}
        <div className="grid grid-cols-2 gap-4">
          {/* Nodes File */}
          <div className="space-y-2">
            <Label htmlFor="nodes-file">Nodes CSV</Label>
            <div className="flex flex-col gap-2">
              <Input
                id="nodes-file"
                type="file"
                accept=".csv"
                onChange={(e) => handleFileChange('nodes', e)}
                disabled={uploading}
              />
              {nodesFile && (
                <div className="flex items-center gap-2 text-sm text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  {nodesFile.name}
                </div>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Required: node_id, tier, capacity, cost_factor, risk_level, reliability, latitude, longitude
            </p>
          </div>

          {/* Edges File */}
          <div className="space-y-2">
            <Label htmlFor="edges-file">Edges CSV</Label>
            <div className="flex flex-col gap-2">
              <Input
                id="edges-file"
                type="file"
                accept=".csv"
                onChange={(e) => handleFileChange('edges', e)}
                disabled={uploading}
              />
              {edgesFile && (
                <div className="flex items-center gap-2 text-sm text-green-600">
                  <CheckCircle className="h-4 w-4" />
                  {edgesFile.name}
                </div>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Required: source, target, capacity_share
            </p>
          </div>
        </div>

        {/* Upload Button */}
        <Button
          onClick={handleUpload}
          disabled={!nodesFile || !edgesFile || !companyId.trim() || uploading}
          className="w-full"
          size="lg"
        >
          {uploading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="h-4 w-4 mr-2" />
              Upload & Train Custom Model
            </>
          )}
        </Button>

        {/* Progress */}
        {uploading && (
          <div className="space-y-3 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">Training Progress</span>
              <span className="text-blue-600">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
            <p className="text-sm text-center text-blue-800">{status}</p>
            <Alert>
              <AlertDescription className="text-xs">
                ⏱️ This process takes 2-4 hours. You can close this page - we'll save your progress.
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Status Message */}
        {status && !uploading && (
          <Alert variant={status.includes('Error') ? 'destructive' : 'default'}>
            <AlertDescription>{status}</AlertDescription>
          </Alert>
        )}

        {/* Template Download */}
        <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
          <h3 className="font-semibold text-sm mb-3 flex items-center gap-2">
            <Download className="h-4 w-4" />
            Need a template?
          </h3>
          <div className="flex gap-2">
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => downloadTemplate('nodes')}
            >
              <Download className="h-3 w-3 mr-1" />
              Nodes Template
            </Button>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => downloadTemplate('edges')}
            >
              <Download className="h-3 w-3 mr-1" />
              Edges Template
            </Button>
          </div>
        </div>

        {/* Info Box */}
        <div className="p-4 bg-gray-50 rounded-lg border">
          <h4 className="font-semibold text-sm mb-2">What happens next?</h4>
          <ol className="text-xs space-y-1 list-decimal list-inside text-gray-700">
            <li>Files are validated (10 seconds)</li>
            <li>10,000 disruption scenarios generated (30 minutes)</li>
            <li>GNN model trained on your graph (2-3 hours)</li>
            <li>Model ready for predictions!</li>
          </ol>
        </div>
      </CardContent>
    </Card>
  );
}
