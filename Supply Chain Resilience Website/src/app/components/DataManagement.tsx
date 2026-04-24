// src/app/components/DataManagement.tsx

import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Alert, AlertDescription } from './ui/alert';
import { useState, useEffect } from 'react';
import { Trash2, Database, AlertCircle, CheckCircle } from 'lucide-react';

export function DataManagement() {
  const [companies, setCompanies] = useState<string[]>([]);
  const [selectedCompany, setSelectedCompany] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string>('');
  const [messageType, setMessageType] = useState<'success' | 'error' | ''>('');

  useEffect(() => {
    fetchCompanies();
  }, []);

  const fetchCompanies = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/list-companies');
      const data = await response.json();
      
      if (data.status === 'success') {
        setCompanies(data.companies);
      }
    } catch (error: any) {
      console.error('Error fetching companies:', error);
    }
  };

  const handleDelete = async () => {
    if (!selectedCompany) {
      setMessage('Please select a company to delete');
      setMessageType('error');
      return;
    }

    const confirmed = window.confirm(
      `Are you sure you want to delete all data for "${selectedCompany}"?\n\n` +
      'This will permanently delete:\n' +
      '• Uploaded graph files (nodes.csv, edges.csv)\n' +
      '• Trained models\n' +
      '• Training results\n' +
      '• All generated scenarios\n\n' +
      'This action cannot be undone!'
    );

    if (!confirmed) return;

    setLoading(true);
    setMessage('');

    try {
      const response = await fetch(`http://localhost:5000/api/delete-company/${selectedCompany}`, {
        method: 'DELETE'
      });

      const data = await response.json();

      if (data.status === 'success') {
        setMessage(`Successfully deleted all data for "${selectedCompany}"`);
        setMessageType('success');
        
        // Clear localStorage if deleting current company
        const currentCompanyId = localStorage.getItem('company_id');
        if (currentCompanyId === selectedCompany) {
          localStorage.removeItem('setupComplete');
          localStorage.removeItem('company_id');
          
          // Reload page to show upload screen
          setTimeout(() => {
            window.location.reload();
          }, 2000);
        }
        
        // Refresh company list
        fetchCompanies();
        setSelectedCompany('');
      } else {
        setMessage(`Error: ${data.message}`);
        setMessageType('error');
      }
    } catch (error: any) {
      setMessage(`Error: ${error.message}`);
      setMessageType('error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Database className="h-5 w-5 text-red-600" />
          Data Management
        </CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          Manage uploaded company data and trained models
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Warning Alert */}
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <p className="font-semibold mb-1">⚠️ Danger Zone</p>
            <p className="text-xs">
              Deleting company data is permanent and cannot be undone. All uploaded files, 
              trained models, and results will be permanently removed.
            </p>
          </AlertDescription>
        </Alert>

        {/* Company Selector */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Select Company to Delete</label>
          <select
            value={selectedCompany}
            onChange={(e) => setSelectedCompany(e.target.value)}
            className="w-full p-2 border rounded-md bg-white"
            disabled={loading}
          >
            <option value="">-- Select a company --</option>
            {companies.map((company) => (
              <option key={company} value={company}>
                {company}
              </option>
            ))}
          </select>
          <p className="text-xs text-muted-foreground">
            {companies.length} company data folder(s) found in uploads/
          </p>
        </div>

        {/* Delete Button */}
        <Button
          onClick={handleDelete}
          disabled={!selectedCompany || loading}
          variant="destructive"
          className="w-full"
          size="lg"
        >
          {loading ? (
            <>
              <Trash2 className="h-4 w-4 mr-2 animate-pulse" />
              Deleting...
            </>
          ) : (
            <>
              <Trash2 className="h-4 w-4 mr-2" />
              Delete Company Data
            </>
          )}
        </Button>

        {/* Status Message */}
        {message && (
          <Alert variant={messageType === 'error' ? 'destructive' : 'default'}>
            {messageType === 'success' ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            <AlertDescription>{message}</AlertDescription>
          </Alert>
        )}

        {/* Info Box */}
        <div className="p-4 bg-gray-50 rounded-lg border">
          <h4 className="font-semibold text-sm mb-2">What gets deleted?</h4>
          <ul className="text-xs space-y-1 list-disc list-inside text-gray-700">
            <li>Uploaded graph files (nodes.csv, edges.csv)</li>
            <li>All trained GNN models (.pt files)</li>
            <li>Training results and metadata</li>
            <li>Generated disruption scenarios</li>
            <li>Model comparison data</li>
          </ul>
        </div>

        {/* Current Company Info */}
        <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="font-semibold text-sm mb-2">Current Session</h4>
          <p className="text-xs text-gray-700">
            Company ID: <span className="font-mono font-semibold">
              {localStorage.getItem('company_id') || 'None (using default data)'}
            </span>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            If you delete the current company, you'll be returned to the upload page.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
