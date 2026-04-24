import { useState, useEffect } from "react";
import { RiskMetricsCard } from "./components/RiskMetricsCard";
import { ResilienceChart } from "./components/ResilienceChart";
import { SupplierRiskTable } from "./components/SupplierRiskTable";
import { RiskDistributionChart } from "./components/RiskDistributionChart";
import { AlertsPanel } from "./components/AlertsPanel";
import { ScenarioAnalysis } from "./components/ScenarioAnalysis";
import { WorldMapNetwork } from "./components/WorldMapNetwork";
import { WhatIfSimulation } from "./components/WhatIfSimulation";
import { MitigationPlanning } from "./components/MitigationPlanning";
import { AIChatbot } from "./components/AIChatbot";
import { CustomGraphUpload } from "./components/CustomerGraphUpload";
import { ModelComparisonTable } from "./components/ModelComparisonTable";
import { VulnerableNodesAnalysis } from "./components/VulnerableNodesAnalysis";
import { NetworkTopologyMetrics } from "./components/NetworkTopologyMetrics";
import { CascadingFailureHeatmap } from "./components/CascadingFailureHeatmap";
import { PipelineRunner } from "./components/PipelineRunner";
import { InitialSetup } from "./components/InitialSetup";
import { DataManagement } from "./components/DataManagement";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Button } from "./components/ui/button";
import { 
  TrendingUp, 
  Shield, 
  AlertTriangle, 
  Package, 
  Download,
  RefreshCw,
  Upload
} from "lucide-react";

export default function App() {
  // Check localStorage for setup completion status on initial load
  const [setupComplete, setSetupComplete] = useState(() => {
    const saved = localStorage.getItem('setupComplete');
    const companyId = localStorage.getItem('company_id');
    // Show InitialSetup if either setupComplete is not set OR company_id is missing
    // This ensures "Fresh Start" properly returns to upload page
    return saved === 'true' && companyId !== null;
  });
  const [lastUpdated, setLastUpdated] = useState(new Date().toLocaleString());
  
  // Simplified beforeunload - only show save prompt
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      const companyId = localStorage.getItem('company_id');
      const setupComplete = localStorage.getItem('setupComplete');
      
      // Only show prompt if user has data
      if (companyId || setupComplete === 'true') {
        e.preventDefault();
        e.returnValue = ''; // Browser shows default "Leave site?" message
        return '';
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);
  const [nodeMetrics, setNodeMetrics] = useState({
    total: 49,
    high: 0,
    medium: 13,
    safe: 36
  });
  const [overviewMetrics, setOverviewMetrics] = useState({
    avgResilience: 85,
    avgRisk: 35,
    avgLeadTime: 5,
    networkDensity: 1.8
  });

  useEffect(() => {
    // Only fetch if setup is complete
    if (setupComplete) {
      fetchNodeMetrics();
      fetchOverviewMetrics();
    }
  }, [setupComplete]);

  const fetchNodeMetrics = async () => {
    try {
      const companyId = localStorage.getItem('company_id');
      const url = companyId 
        ? `http://localhost:5000/api/graph?company_id=${companyId}`
        : 'http://localhost:5000/api/graph';
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.status === 'success') {
        const nodes = data.nodes;
        const total = nodes.length;
        
        // Calculate risk categories
        let high = 0, medium = 0, safe = 0;
        nodes.forEach((node: any) => {
          const riskLevel = node.risk_level || 0.5;
          if (riskLevel >= 0.9) high++;
          else if (riskLevel >= 0.6) high++;
          else if (riskLevel >= 0.3) medium++;
          else safe++;
        });
        
        setNodeMetrics({ total, high, medium, safe });
      }
    } catch (error) {
      console.error('Error fetching node metrics:', error);
    }
  };

  const fetchOverviewMetrics = async () => {
    try {
      const companyId = localStorage.getItem('company_id');
      const url = companyId 
        ? `http://localhost:5000/api/overview-metrics?company_id=${companyId}`
        : 'http://localhost:5000/api/overview-metrics';
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.status === 'success') {
        setOverviewMetrics({
          avgResilience: data.avg_resilience,
          avgRisk: data.avg_risk,
          avgLeadTime: data.avg_lead_time,
          networkDensity: data.network_density
        });
      }
    } catch (error) {
      console.error('Error fetching overview metrics:', error);
    }
  };

  const handleRefresh = () => {
    setLastUpdated(new Date().toLocaleString());
    fetchNodeMetrics();
    fetchOverviewMetrics();
  };

  const handleSetupComplete = () => {
    setSetupComplete(true);
    // Save to localStorage so it persists across page refreshes
    localStorage.setItem('setupComplete', 'true');
    // Refresh data after setup
    fetchNodeMetrics();
    fetchOverviewMetrics();
  };

  // Show initial setup page if not completed
  if (!setupComplete) {
    return <InitialSetup onComplete={handleSetupComplete} />;
  }

  // Show main dashboard after setup
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b shadow-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Supply Chain Resilience</h1>
                <p className="text-sm text-gray-500">Real-time analysis and monitoring</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500">Last updated: {lastUpdated}</span>
              <Button variant="outline" size="sm" onClick={handleRefresh}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => {
                  if (confirm('Reset setup? This will clear all data and return to the upload page.')) {
                    localStorage.removeItem('setupComplete');
                    localStorage.removeItem('company_id');
                    setSetupComplete(false);
                  }
                }}
              >
                <Upload className="h-4 w-4 mr-2" />
                Reset Setup
              </Button>
              <Button size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export Report
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 lg:w-auto lg:inline-grid">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="suppliers" className="flex items-center gap-2">
              <Package className="h-4 w-4" />
              Nodes
            </TabsTrigger>
            <TabsTrigger value="risks" className="flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Risk Analysis
            </TabsTrigger>
            <TabsTrigger value="scenarios" className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Scenarios
            </TabsTrigger>
            <TabsTrigger value="custom" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Custom Graph
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Pipeline Runner */}
            <div className="grid gap-6 md:grid-cols-4">
              <PipelineRunner />
            </div>

            {/* Key Metrics */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <RiskMetricsCard
                title="Avg Resilience Score"
                value={overviewMetrics.avgResilience}
                change={0}
                trend="up"
              />
              <RiskMetricsCard
                title="Avg Risk Level"
                value={overviewMetrics.avgRisk}
                change={0}
                trend="down"
              />
              <RiskMetricsCard
                title="Avg Lead Time"
                value={overviewMetrics.avgLeadTime}
                change={0}
                unit=" days"
                trend="neutral"
              />
              <RiskMetricsCard
                title="Network Density"
                value={overviewMetrics.networkDensity}
                change={0}
                unit=" edges/node"
                trend="up"
              />
            </div>

            {/* World Map Network */}
            <div className="grid gap-6 md:grid-cols-4">
              <WorldMapNetwork />
            </div>

            {/* Model Comparison Table */}
            <div className="grid gap-6 md:grid-cols-4">
              <ModelComparisonTable />
            </div>

          </TabsContent>

          {/* Nodes Tab */}
          <TabsContent value="suppliers" className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <RiskMetricsCard
                title="Total Nodes"
                value={nodeMetrics.total}
                change={0}
                unit=""
                trend="neutral"
              />
              <RiskMetricsCard
                title="High Risk Nodes"
                value={nodeMetrics.high}
                change={0}
                unit=""
                trend="up"
              />
              <RiskMetricsCard
                title="Medium Risk Nodes"
                value={nodeMetrics.medium}
                change={0}
                unit=""
                trend="neutral"
              />
              <RiskMetricsCard
                title="Safe Nodes"
                value={nodeMetrics.safe}
                change={0}
                unit=""
                trend="up"
              />
            </div>

            <div className="grid gap-6 md:grid-cols-4">
              <SupplierRiskTable />
            </div>
          </TabsContent>

          {/* Risk Analysis Tab */}
          <TabsContent value="risks" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <RiskDistributionChart />
              <NetworkTopologyMetrics />
            </div>

            {/* Vulnerable Node Analysis — Betweenness, Eigenvector, Articulation Points */}
            <div className="grid gap-6 md:grid-cols-4">
              <VulnerableNodesAnalysis />
            </div>

            {/* Cascading Failure Simulation */}
            <div className="grid gap-6 md:grid-cols-4">
              <CascadingFailureHeatmap />
            </div>
          </TabsContent>

          {/* Scenarios Tab */}
          <TabsContent value="scenarios" className="space-y-6">
            {/* GNN-Based Disruption Simulation - Main Feature with Real Predictions */}
            <div className="grid gap-6 md:grid-cols-4">
              <WhatIfSimulation />
            </div>

            {/* Risk Scenario Analysis */}
            <div className="grid gap-6 md:grid-cols-4">
              <ScenarioAnalysis />
            </div>

            {/* AI Chatbot Assistant */}
            <div className="grid gap-6 md:grid-cols-4">
              <AIChatbot />
            </div>
          </TabsContent>

          {/* Custom Graph Upload Tab */}
          <TabsContent value="custom" className="space-y-6">
            <div className="grid gap-6 md:grid-cols-4">
              <CustomGraphUpload />
            </div>
            
            {/* Data Management Section */}
            <div className="grid gap-6 md:grid-cols-4">
              <DataManagement />
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <p>© 2026 Supply Chain Resilience Platform</p>
            <p>Powered by real-time analytics and AI-driven insights</p>
          </div>
        </div>
      </footer>
    </div>
  );
}