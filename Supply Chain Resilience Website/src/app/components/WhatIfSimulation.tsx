import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Badge } from "./ui/badge";
import { PlayCircle, RotateCcw, AlertTriangle, TrendingDown, Clock, DollarSign, Network, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import Papa from "papaparse";
import { ComposableMap, Geographies, Geography, Marker, ZoomableGroup } from "react-simple-maps";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { gnnApi, type Prediction } from "../../services/gnnApi";

interface DisruptionData {
  Disruption_Event: string;
  Origin_City: string;
  Destination_City: string;
  Route_Type: string;
  Product_Category: string;
  Delay_Days: number;
  Geopolitical_Risk_Index: number;
  Weather_Severity_Index: number;
  Inflation_Rate_Pct: number;
  Shipping_Cost_USD: number;
}

export function WhatIfSimulation() {
  const [disruptedNodeIds, setDisruptedNodeIds] = useState("0");
  const [selectedDisruption, setSelectedDisruption] = useState("port-congestion");
  const [severity, setSeverity] = useState([70]);
  const [bufferCapacity, setBufferCapacity] = useState([50]);
  const [simulated, setSimulated] = useState(false);
  const [historicalData, setHistoricalData] = useState<DisruptionData[]>([]);
  const [allNodes, setAllNodes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [predictionSummary, setPredictionSummary] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Trade restriction specific states
  const [tradeCountry1, setTradeCountry1] = useState("China 1");
  const [tradeCountry2, setTradeCountry2] = useState("United States 1");
  const [tariffRate, setTariffRate] = useState([45]); // 20-70% range, default 45%
  
  // Random supplier failure states
  const [numSuppliers, setNumSuppliers] = useState([5]); // 1-10 range, default 5
  const [supplierSeverity, setSupplierSeverity] = useState([60]); // 10-100% range, default 60%
  
  // Transportation route disruption states
  const [numRoutes, setNumRoutes] = useState([5]); // 1-15 range, default 5
  const [routeSeverityType, setRouteSeverityType] = useState("moderate"); // minor, moderate, severe

  const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

  // Extract unique countries from loaded nodes
  const [availableCountries, setAvailableCountries] = useState<string[]>([]);

  useEffect(() => {
    // Load ALL nodes from synthetic_nodes.csv
    Papa.parse('/synthetic_nodes.csv', {
      download: true,
      header: true,
      complete: (results: any) => {
        const nodeData = results.data;
        const loadedNodes = nodeData
          .filter((node: any) => node.node_id !== undefined && node.x && node.y)
          .map((node: any) => ({
            nodeId: parseInt(node.node_id),
            label: `${node.region} Node ${node.node_id}`,
            region: node.region,
            tier: parseInt(node.tier),
            coordinates: [parseFloat(node.x), parseFloat(node.y)] as [number, number],
            risk_level: parseFloat(node.risk_level),
            reliability: parseFloat(node.reliability),
            capacity: parseFloat(node.capacity)
          }));
        
        setAllNodes(loadedNodes);
        
        // Extract unique countries from nodes
        const uniqueCountries = Array.from(new Set(loadedNodes.map((node: any) => node.region as string)))
          .filter((region): region is string => typeof region === 'string' && region.trim() !== '')
          .sort();
        setAvailableCountries(uniqueCountries);
        
        // Set default countries if available
        if (uniqueCountries.length >= 2) {
          setTradeCountry1(uniqueCountries[0] as string);
          setTradeCountry2(uniqueCountries[1] as string);
        }
        
        // Load historical disruption data
        Papa.parse('/external_disruption_data/global_supply_chain_disruption_v1.csv', {
          download: true,
          header: true,
          complete: (results: any) => {
            const data = results.data as DisruptionData[];
            setHistoricalData(data.filter(d => d.Disruption_Event && d.Disruption_Event !== 'None'));
            setLoading(false);
          }
        });
      }
    });
  }, []);

  const disruptionTypes = [
    // Node-Only Disruptions (5 types)
    { 
      value: "regional_failure_variable", 
      label: "Regional Failure (Variable Radius)",
      icon: AlertTriangle,
      description: "Geographic region disruption affecting multiple nodes",
      category: "Node-Only"
    },
    { 
      value: "distributor_hub_failure", 
      label: "Distributor Hub Failure",
      icon: Network,
      description: "Critical distribution center failure (Tier 2 nodes)",
      category: "Node-Only"
    },
    { 
      value: "random_supplier_failure", 
      label: "Random Supplier Failure",
      icon: DollarSign,
      description: "Random supplier node disruption (5 suppliers)",
      category: "Node-Only"
    },
    { 
      value: "central", 
      label: "High-Centrality Node Failure",
      icon: TrendingDown,
      description: "High betweenness centrality node failure (5 nodes from top 10%)",
      category: "Node-Only"
    },
    { 
      value: "random_multi_node", 
      label: "Random Multi-Node Failure",
      icon: AlertTriangle,
      description: "Random failure of 1-6 nodes simultaneously",
      category: "Node-Only"
    },
    // Edge-Only Disruptions (4 types)
    { 
      value: "transportation_route_disruption", 
      label: "Transportation Route Disruption",
      icon: Clock,
      description: "Weather/accidents affecting 5-15 transportation routes",
      category: "Edge-Only"
    },
    { 
      value: "trade_restriction", 
      label: "Trade Restriction",
      icon: AlertTriangle,
      description: "Geopolitical conflicts blocking cross-border routes",
      category: "Edge-Only"
    },
    { 
      value: "cyber_attack_logistics", 
      label: "Cyber Attack on Logistics",
      icon: Network,
      description: "Digital infrastructure attack on logistics network (10-30% edges)",
      category: "Edge-Only"
    },
    { 
      value: "major_supplier_failure", 
      label: "Major Supplier Failure",
      icon: DollarSign,
      description: "High-capacity supplier failure (1-3 suppliers, all outgoing edges)",
      category: "Edge-Only"
    },
    // Hybrid Disruptions (2 types)
    { 
      value: "infrastructure_failure", 
      label: "Infrastructure Failure",
      icon: AlertTriangle,
      description: "Regional infrastructure collapse (nodes + edges within radius)",
      category: "Hybrid"
    },
    { 
      value: "hybrid_node_edge", 
      label: "Hybrid Node + Edge Disruption",
      icon: TrendingDown,
      description: "Combined regional node failure + transportation route failures",
      category: "Hybrid"
    },
  ];

  const handleReset = () => {
    setSimulated(false);
    setDisruptedNodeIds("0");
    setSeverity([70]);
    setPredictions([]);
    setPredictionSummary(null);
    setError(null);
  };

  const handleSimulate = async () => {
    setPredicting(true);
    setError(null);
    
    try {
      // Parse disrupted node IDs from comma-separated string
      const nodeIds = disruptedNodeIds
        .split(',')
        .map(id => parseInt(id.trim()))
        .filter(id => !isNaN(id) && id >= 0 && id < 200);
      
      if (nodeIds.length === 0) {
        throw new Error('Please enter valid node IDs (0-199)');
      }
      
      // Call GNN API with buffer capacity
      const response = await gnnApi.predict({
        disrupted_nodes: nodeIds,
        disrupted_edges: [],
        disruption_severity: severity[0] / 100, // Normalize to 0-1
        buffer_capacity: bufferCapacity[0] / 100 // Normalize to 0-1
      });
      
      // Store predictions
      setPredictions(response.predictions);
      setPredictionSummary(response.summary);
      setSimulated(true);
      
    } catch (err: any) {
      setError(err.message || 'Failed to get predictions from GNN model');
      console.error('GNN Prediction Error:', err);
    } finally {
      setPredicting(false);
    }
  };

  const getHistoricalInsights = () => {
    if (historicalData.length === 0) return null;
    
    const relevantDisruptions = historicalData.filter(d => 
      d.Disruption_Event.toLowerCase().includes(selectedDisruption.split('-')[0])
    );
    
    if (relevantDisruptions.length === 0) return null;
    
    const avgDelay = relevantDisruptions.reduce((sum, d) => sum + (Number(d.Delay_Days) || 0), 0) / relevantDisruptions.length;
    const avgCost = relevantDisruptions.reduce((sum, d) => sum + (Number(d.Shipping_Cost_USD) || 0), 0) / relevantDisruptions.length;
    const avgRisk = relevantDisruptions.reduce((sum, d) => sum + (Number(d.Geopolitical_Risk_Index) || 0), 0) / relevantDisruptions.length;
    
    return {
      count: relevantDisruptions.length,
      avgDelay: avgDelay.toFixed(1),
      avgCost: avgCost.toFixed(0),
      avgRisk: (avgRisk * 100).toFixed(1)
    };
  };

  // Get node status from GNN predictions
  const getNodeStatus = (nodeId: number) => {
    if (!simulated || predictions.length === 0) return { 
      color: "#94a3b8", 
      status: "Unknown", 
      risk: "Unknown",
      level: -1,
      size: 8,
      confidence: 0
    };
    
    const prediction = predictions.find(p => p.node_id === nodeId);
    
    if (!prediction) return { 
      color: "#94a3b8", 
      status: "Unknown", 
      risk: "Unknown",
      level: -1,
      size: 8,
      confidence: 0
    };
    
    // Map GNN labels: 0=Failed, 1=Degraded, 2=Normal
    const colors = {
      0: "#dc2626", // Failed - Red
      1: "#f59e0b", // Degraded - Orange
      2: "#10b981"  // Normal - Green
    };
    
    const statuses = {
      0: "FAILED",
      1: "DEGRADED",
      2: "NORMAL"
    };
    
    const risks = {
      0: "Critical",
      1: "High",
      2: "Low"
    };
    
    const confidence = Math.max(...prediction.probability) * 100;
    
    return { 
      color: colors[prediction.label as keyof typeof colors],
      status: statuses[prediction.label as keyof typeof statuses],
      risk: risks[prediction.label as keyof typeof risks],
      level: prediction.label,
      size: prediction.label === 0 ? 12 : prediction.label === 1 ? 10 : 8,
      confidence: confidence
    };
  };

  const historical = getHistoricalInsights();
  
  // Parse disrupted node IDs
  const disruptedIds = disruptedNodeIds.split(',').map(id => parseInt(id.trim())).filter(id => !isNaN(id));

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PlayCircle className="h-5 w-5 text-blue-600" />
          GNN-Based Disruption Impact Simulation
        </CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          Predict supply chain resilience using trained GINE model (76.6% accuracy)
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Step 1: Disruption Type Selection (Always Visible) */}
        <div className="space-y-2">
          <Label htmlFor="disruption">Step 1: Select Disruption Type</Label>
          <Select value={selectedDisruption} onValueChange={setSelectedDisruption} disabled={predicting}>
            <SelectTrigger id="disruption">
              <SelectValue placeholder="Choose disruption type" />
            </SelectTrigger>
            <SelectContent>
              {disruptionTypes.map((type) => (
                <SelectItem key={type.value} value={type.value}>
                  {type.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-xs font-semibold text-blue-900 mb-1">
            {disruptionTypes.find(d => d.value === selectedDisruption)?.label}
          </p>
          <p className="text-xs text-blue-800">
            {disruptionTypes.find(d => d.value === selectedDisruption)?.description}
          </p>
        </div>

        {/* Step 2: Disruption-Specific Configuration */}
        {selectedDisruption === "trade_restriction" && (
          <div className="p-4 bg-amber-50 rounded-lg border-2 border-amber-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-amber-600" />
              <h4 className="font-semibold text-sm text-amber-900">Step 2: Trade Restriction Configuration</h4>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="country1">Country 1</Label>
                <Select value={tradeCountry1} onValueChange={setTradeCountry1} disabled={predicting}>
                  <SelectTrigger id="country1">
                    <SelectValue placeholder="Select first country" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableCountries.length > 0 ? (
                      availableCountries.map((country) => (
                        <SelectItem key={country} value={country}>
                          {country}
                        </SelectItem>
                      ))
                    ) : (
                      <SelectItem value="loading" disabled>Loading countries...</SelectItem>
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="country2">Country 2</Label>
                <Select value={tradeCountry2} onValueChange={setTradeCountry2} disabled={predicting}>
                  <SelectTrigger id="country2">
                    <SelectValue placeholder="Select second country" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableCountries.length > 0 ? (
                      availableCountries.map((country) => (
                        <SelectItem key={country} value={country}>
                          {country}
                        </SelectItem>
                      ))
                    ) : (
                      <SelectItem value="loading" disabled>Loading countries...</SelectItem>
                    )}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="tariff">Tariff Rate (Capacity Impact)</Label>
                <span className="text-sm font-medium text-amber-900">{tariffRate[0]}%</span>
              </div>
              <Slider
                id="tariff"
                value={tariffRate}
                onValueChange={setTariffRate}
                min={20}
                max={70}
                step={5}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-amber-700">
                Tariff impact on cross-border capacity between {tradeCountry1} and {tradeCountry2} (20-70%)
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-amber-200">
              <p className="text-xs text-amber-800">
                <strong>Trade Policy:</strong> {tariffRate[0]}% tariff between {tradeCountry1} ↔ {tradeCountry2}
              </p>
              <p className="text-xs text-amber-700 mt-1">
                This will reduce capacity on all cross-border routes between these countries by approximately {tariffRate[0]}% (±10% variation for implementation differences)
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "random_supplier_failure" && (
          <div className="p-4 bg-purple-50 rounded-lg border-2 border-purple-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="h-4 w-4 text-purple-600" />
              <h4 className="font-semibold text-sm text-purple-900">Step 2: Random Supplier Failure Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="numSuppliers">Number of Suppliers</Label>
                <span className="text-sm font-medium text-purple-900">{numSuppliers[0]}</span>
              </div>
              <Slider
                id="numSuppliers"
                value={numSuppliers}
                onValueChange={setNumSuppliers}
                min={1}
                max={10}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-purple-700">
                Number of random supplier nodes that will fail (1-10)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="supplierSeverity">Impact Severity</Label>
                <span className="text-sm font-medium text-purple-900">{supplierSeverity[0]}%</span>
              </div>
              <Slider
                id="supplierSeverity"
                value={supplierSeverity}
                onValueChange={setSupplierSeverity}
                min={10}
                max={100}
                step={5}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-purple-700">
                Production impact percentage on failed suppliers (10-100%)
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-purple-200">
              <p className="text-xs text-purple-800">
                <strong>Scenario:</strong> {numSuppliers[0]} random supplier{numSuppliers[0] > 1 ? 's' : ''} will fail with {supplierSeverity[0]}% production impact
              </p>
              <p className="text-xs text-purple-700 mt-1">
                This simulates unexpected supplier failures due to operational issues, quality problems, or financial difficulties
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "transportation_route_disruption" && (
          <div className="p-4 bg-blue-50 rounded-lg border-2 border-blue-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Clock className="h-4 w-4 text-blue-600" />
              <h4 className="font-semibold text-sm text-blue-900">Step 2: Transportation Route Disruption Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="numRoutes">Number of Routes</Label>
                <span className="text-sm font-medium text-blue-900">{numRoutes[0]}</span>
              </div>
              <Slider
                id="numRoutes"
                value={numRoutes}
                onValueChange={setNumRoutes}
                min={1}
                max={15}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-blue-700">
                Number of transportation routes/edges that will be disrupted (1-15)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="routeSeverity">Severity Type</Label>
              <Select value={routeSeverityType} onValueChange={setRouteSeverityType} disabled={predicting}>
                <SelectTrigger id="routeSeverity">
                  <SelectValue placeholder="Select severity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="minor">Minor (20-40% capacity loss)</SelectItem>
                  <SelectItem value="moderate">Moderate (40-70% capacity loss)</SelectItem>
                  <SelectItem value="severe">Severe (70-100% capacity loss)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-blue-700">
                Severity of disruption caused by weather, accidents, or infrastructure issues
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-blue-200">
              <p className="text-xs text-blue-800">
                <strong>Scenario:</strong> {numRoutes[0]} transportation route{numRoutes[0] > 1 ? 's' : ''} will experience {routeSeverityType} disruption
              </p>
              <p className="text-xs text-blue-700 mt-1">
                {routeSeverityType === 'minor' && 'Minor delays due to weather or minor accidents (20-40% capacity reduction)'}
                {routeSeverityType === 'moderate' && 'Moderate disruptions from severe weather or road closures (40-70% capacity reduction)'}
                {routeSeverityType === 'severe' && 'Severe disruptions from natural disasters or major infrastructure failures (70-100% capacity reduction)'}
              </p>
            </div>
          </div>
        )}

        <div className="flex gap-2">
          <Button onClick={handleSimulate} className="flex-1" disabled={predicting || loading}>
            {predicting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running GNN Prediction...
              </>
            ) : (
              <>
                <PlayCircle className="h-4 w-4 mr-2" />
                Run GNN Simulation
              </>
            )}
          </Button>
          <Button variant="outline" onClick={handleReset} disabled={predicting}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        {error && (
          <div className="p-4 bg-red-50 rounded-lg border border-red-200">
            <p className="text-sm font-semibold text-red-900">Error</p>
            <p className="text-xs text-red-700 mt-1">{error}</p>
          </div>
        )}

        {simulated && predictionSummary && (
          <div className="mt-6 space-y-4">
            {/* GNN Prediction Summary */}
            <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border-2 border-blue-200">
              <h4 className="font-semibold text-sm flex items-center gap-2 mb-3">
                <Network className="h-4 w-4 text-blue-600" />
                GNN Resilience Prediction (GINE Model - 76.6% Accuracy)
              </h4>
              <div className="grid grid-cols-4 gap-4">
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-xs text-muted-foreground mb-1">Failed Nodes</p>
                  <p className="text-2xl font-bold text-red-600">{predictionSummary.failed}</p>
                  <p className="text-xs text-muted-foreground mt-1">Critical failures</p>
                </div>
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-xs text-muted-foreground mb-1">Degraded Nodes</p>
                  <p className="text-2xl font-bold text-orange-600">{predictionSummary.degraded}</p>
                  <p className="text-xs text-muted-foreground mt-1">Partial impact</p>
                </div>
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-xs text-muted-foreground mb-1">Normal Nodes</p>
                  <p className="text-2xl font-bold text-green-600">{predictionSummary.normal}</p>
                  <p className="text-xs text-muted-foreground mt-1">Operational</p>
                </div>
                <div className="p-3 bg-white rounded-lg border">
                  <p className="text-xs text-muted-foreground mb-1">Total Nodes</p>
                  <p className="text-2xl font-bold text-blue-600">{predictionSummary.total_nodes}</p>
                  <p className="text-xs text-muted-foreground mt-1">In network</p>
                </div>
              </div>
            </div>

            {/* Network Visualization Map */}
            <div className="grid grid-cols-2 gap-4">
              {/* Map Side */}
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle className="text-sm">Network Impact Visualization (All 200 Nodes)</CardTitle>
                  <p className="text-xs text-muted-foreground mt-1">
                    GNN predictions • Zoom to see details
                  </p>
                </CardHeader>
                <CardContent className="h-[500px] flex flex-col">
                  <div className="flex-1 relative">
                    <TooltipProvider>
                      <ComposableMap
                        projection="geoMercator"
                        className="w-full h-full"
                        style={{ width: "100%", height: "100%" }}
                      >
                        <ZoomableGroup
                          center={[20, 20]}
                          zoom={1}
                          minZoom={1}
                          maxZoom={8}
                        >
                          <Geographies geography={geoUrl}>
                            {({ geographies }: any) =>
                              geographies.map((geo: any) => (
                                <Geography
                                  key={geo.rsmKey}
                                  geography={geo}
                                  fill="#E5E7EB"
                                  stroke="#9CA3AF"
                                  strokeWidth={0.5}
                                  style={{
                                    default: { outline: "none" },
                                    hover: { fill: "#D1D5DB", outline: "none" },
                                    pressed: { outline: "none" },
                                  }}
                                />
                              ))
                            }
                          </Geographies>

                          {/* Draw ALL nodes */}
                          {allNodes.map((node) => {
                            const status = getNodeStatus(node.nodeId);
                            const isDisrupted = disruptedIds.includes(node.nodeId);
                            
                            return (
                              <Marker key={node.nodeId} coordinates={node.coordinates}>
                                <Tooltip>
                                  <TooltipTrigger asChild>
                                    <g>
                                      {/* Pulse effect for disrupted nodes */}
                                      {isDisrupted && (
                                        <circle
                                          r={15}
                                          fill={status.color}
                                          fillOpacity={0.3}
                                          className="animate-ping"
                                        />
                                      )}
                                      {/* Main node */}
                                      <circle
                                        r={status.size}
                                        fill={status.color}
                                        stroke="#fff"
                                        strokeWidth={1.5}
                                        style={{ cursor: "pointer" }}
                                      />
                                      {/* Warning icon for disrupted */}
                                      {isDisrupted && (
                                        <text
                                          textAnchor="middle"
                                          y={3}
                                          style={{ fontSize: "8px", fill: "#fff", fontWeight: "bold" }}
                                        >
                                          !
                                        </text>
                                      )}
                                    </g>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    <div className="text-sm">
                                      <p className="font-semibold">{node.label}</p>
                                      <p className="text-xs text-muted-foreground">Tier {node.tier}</p>
                                      <p className="text-xs">Status: <span className={
                                        status.level === 2 ? "text-green-600" :
                                        status.level === 1 ? "text-orange-600" :
                                        status.level === 0 ? "text-red-600" :
                                        "text-gray-600"
                                      }>{status.status}</span></p>
                                      {status.confidence > 0 && (
                                        <p className="text-xs">Confidence: {status.confidence.toFixed(1)}%</p>
                                      )}
                                      {isDisrupted && <p className="text-xs text-red-600 font-bold">⚠️ DISRUPTED</p>}
                                    </div>
                                  </TooltipContent>
                                </Tooltip>
                              </Marker>
                            );
                          })}
                        </ZoomableGroup>
                      </ComposableMap>
                    </TooltipProvider>
                  </div>

                  {/* Map Legend */}
                  <div className="mt-3 pt-3 border-t flex flex-wrap gap-3 text-xs">
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full bg-[#dc2626]"></div>
                      <span>Failed ({predictionSummary.failed})</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full bg-[#f59e0b]"></div>
                      <span>Degraded ({predictionSummary.degraded})</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full bg-[#10b981]"></div>
                      <span>Normal ({predictionSummary.normal})</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Node Status List Side */}
              <Card className="col-span-1">
                <CardHeader>
                  <CardTitle className="text-sm">Critical & Degraded Nodes</CardTitle>
                </CardHeader>
                <CardContent className="h-[500px] overflow-y-auto">
                  <div className="space-y-2">
                    {predictions
                      .filter(pred => pred.label === 0 || pred.label === 1)
                      .slice(0, 30)
                      .map((pred) => {
                        const node = allNodes.find(n => n.nodeId === pred.node_id);
                        if (!node) return null;
                        
                        const status = getNodeStatus(pred.node_id);
                        const bgColor = pred.label === 0 ? 'bg-red-50 border-red-300' : 'bg-orange-50 border-orange-300';
                        const isDisrupted = disruptedIds.includes(pred.node_id);
                        
                        return (
                          <div 
                            key={pred.node_id} 
                            className={`p-2 rounded-lg border ${bgColor} ${isDisrupted ? 'ring-2 ring-red-500' : ''}`}
                          >
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <div className="flex items-center gap-2">
                                  <div 
                                    className="w-3 h-3 rounded-full" 
                                    style={{ backgroundColor: status.color }}
                                  ></div>
                                  <p className="font-semibold text-xs">{node.label}</p>
                                  {isDisrupted && <span className="text-xs text-red-600">⚠️</span>}
                                </div>
                                <p className="text-xs text-muted-foreground">
                                  {pred.region} • Tier {pred.tier}
                                </p>
                              </div>
                              <Badge className={
                                pred.label === 0 ? "bg-red-500 text-white" : "bg-orange-500 text-white"
                              }>
                                {pred.label_name}
                              </Badge>
                            </div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              Confidence: {status.confidence.toFixed(1)}%
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Historical Data Insights */}
            {historical && (
              <div className="p-4 bg-amber-50 rounded-lg border border-amber-200">
                <h4 className="font-semibold text-sm text-amber-900 mb-2">
                  📊 Historical Data Insights
                </h4>
                <div className="grid grid-cols-4 gap-3 text-xs">
                  <div>
                    <p className="text-amber-700">Similar Events</p>
                    <p className="font-bold text-amber-900">{historical.count}</p>
                  </div>
                  <div>
                    <p className="text-amber-700">Avg Delay</p>
                    <p className="font-bold text-amber-900">{historical.avgDelay} days</p>
                  </div>
                  <div>
                    <p className="text-amber-700">Avg Cost</p>
                    <p className="font-bold text-amber-900">${Number(historical.avgCost).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="text-amber-700">Risk Index</p>
                    <p className="font-bold text-amber-900">{historical.avgRisk}%</p>
                  </div>
                </div>
              </div>
            )}

            {/* Mitigation Recommendations */}
            <div className="p-3 bg-green-50 rounded-lg border border-green-200">
              <p className="text-xs font-semibold text-green-900 mb-2">🎯 GNN-Recommended Mitigation Actions</p>
              <ul className="text-xs text-green-800 space-y-1 list-disc list-inside">
                <li>Monitor {predictionSummary.degraded} degraded nodes for potential failures</li>
                <li>Increase safety stock by {Math.floor(severity[0] * 0.5)}% for critical components</li>
                {predictionSummary.failed > 10 && <li className="text-red-600 font-semibold">⚠️ CRITICAL: {predictionSummary.failed} nodes predicted to fail - initiate emergency protocols</li>}
                {predictionSummary.failed > 0 && <li className="text-orange-600 font-semibold">⚠️ HIGH PRIORITY: Strengthen resilience of {predictionSummary.failed} critical nodes</li>}
                <li>Activate backup suppliers in affected regions</li>
              </ul>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
