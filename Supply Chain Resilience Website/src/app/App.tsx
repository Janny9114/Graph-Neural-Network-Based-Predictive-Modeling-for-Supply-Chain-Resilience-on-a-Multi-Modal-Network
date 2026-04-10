import { useState } from "react";
import { RiskMetricsCard } from "./components/RiskMetricsCard";
import { ResilienceChart } from "./components/ResilienceChart";
import { SupplierRiskTable } from "./components/SupplierRiskTable";
import { RiskDistributionChart } from "./components/RiskDistributionChart";
import { PerformanceMetrics } from "./components/PerformanceMetrics";
import { AlertsPanel } from "./components/AlertsPanel";
import { ScenarioAnalysis } from "./components/ScenarioAnalysis";
import { WorldMapNetwork } from "./components/WorldMapNetwork";
import { WhatIfSimulation } from "./components/WhatIfSimulation";
import { MitigationPlanning } from "./components/MitigationPlanning";
import { AIChatbot } from "./components/AIChatbot";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Button } from "./components/ui/button";
import { 
  TrendingUp, 
  Shield, 
  AlertTriangle, 
  Package, 
  Download,
  RefreshCw
} from "lucide-react";

export default function App() {
  const [lastUpdated, setLastUpdated] = useState(new Date().toLocaleString());

  const handleRefresh = () => {
    setLastUpdated(new Date().toLocaleString());
  };

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
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid">
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
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <RiskMetricsCard
                title="Overall Resilience Score"
                value={88}
                change={3.2}
                trend="up"
              />
              <RiskMetricsCard
                title="Supply Chain Risk"
                value={25}
                change={-5.1}
                trend="up"
              />
              <RiskMetricsCard
                title="Supplier Performance"
                value={90}
                change={2.8}
                trend="up"
              />
              <RiskMetricsCard
                title="Active Alerts"
                value={12}
                change={-15.0}
                unit=""
                trend="up"
              />
            </div>

            {/* World Map Network */}
            <div className="grid gap-6 md:grid-cols-4">
              <WorldMapNetwork />
            </div>

            {/* Charts */}
            <div className="grid gap-6 md:grid-cols-4">
              <ResilienceChart />
            </div>

            {/* Secondary Charts */}
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <RiskDistributionChart />
              <PerformanceMetrics />
            </div>

            {/* Alerts */}
            <div className="grid gap-6 md:grid-cols-4">
              <AlertsPanel />
            </div>
          </TabsContent>

          {/* Suppliers Tab */}
          <TabsContent value="suppliers" className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <RiskMetricsCard
                title="Total Suppliers"
                value={124}
                change={8.3}
                unit=""
                trend="up"
              />
              <RiskMetricsCard
                title="High Risk Suppliers"
                value={12}
                change={-16.7}
                unit=""
                trend="up"
              />
              <RiskMetricsCard
                title="Avg On-Time Delivery"
                value={92.5}
                change={1.2}
                trend="up"
              />
              <RiskMetricsCard
                title="Avg Quality Score"
                value={90.8}
                change={2.5}
                trend="up"
              />
            </div>

            <div className="grid gap-6 md:grid-cols-4">
              <SupplierRiskTable />
            </div>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <PerformanceMetrics />
              <RiskDistributionChart />
            </div>
          </TabsContent>

          {/* Risk Analysis Tab */}
          <TabsContent value="risks" className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <RiskMetricsCard
                title="Critical Risks"
                value={3}
                change={-25.0}
                unit=""
                trend="up"
              />
              <RiskMetricsCard
                title="High Risks"
                value={8}
                change={-11.1}
                unit=""
                trend="up"
              />
              <RiskMetricsCard
                title="Medium Risks"
                value={15}
                change={7.1}
                unit=""
                trend="down"
              />
              <RiskMetricsCard
                title="Risk Mitigation Rate"
                value={78}
                change={5.4}
                trend="up"
              />
            </div>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <RiskDistributionChart />
              <PerformanceMetrics />
            </div>

            <div className="grid gap-6 md:grid-cols-4">
              <AlertsPanel />
            </div>
          </TabsContent>

          {/* Scenarios Tab */}
          <TabsContent value="scenarios" className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <RiskMetricsCard
                title="Scenarios Analyzed"
                value={24}
                change={9.1}
                unit=""
                trend="up"
              />
              <RiskMetricsCard
                title="Avg Probability"
                value={43}
                change={-3.4}
                trend="up"
              />
              <RiskMetricsCard
                title="Mitigation Coverage"
                value={76.6}
                change={12.8}
                trend="up"
              />
              <RiskMetricsCard
                title="Readiness Score"
                value={82}
                change={6.5}
                trend="up"
              />
            </div>

            {/* GNN-Based Disruption Simulation - Main Feature */}
            <div className="grid gap-6 md:grid-cols-4">
              <WhatIfSimulation />
            </div>

            {/* Risk Scenario Analysis */}
            <div className="grid gap-6 md:grid-cols-4">
              <ScenarioAnalysis />
            </div>

            {/* Mitigation Planning */}
            <div className="grid gap-6 md:grid-cols-4">
              <MitigationPlanning />
            </div>

            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
              <ResilienceChart />
            </div>

            {/* AI Chatbot Assistant */}
            <div className="grid gap-6 md:grid-cols-4">
              <AIChatbot />
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