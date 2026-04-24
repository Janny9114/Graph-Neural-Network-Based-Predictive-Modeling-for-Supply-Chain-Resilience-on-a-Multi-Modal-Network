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
  const [selectedDisruption, setSelectedDisruption] = useState("regional_failure_variable");
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
  
  // Regional failure (variable radius) states
  const [centerNodeId, setCenterNodeId] = useState<number>(0); // Epicenter node
  const [radius, setRadius] = useState([500]); // 100-1000 km, default 500
  const [regionalSeverity, setRegionalSeverity] = useState([70]); // 30-100%, default 70%
  
  // Distributor hub failure states
  const [selectedTier, setSelectedTier] = useState(2); // Default: Tier 2 (Distributors)
  const [numHubs, setNumHubs] = useState([3]); // 1-5 hubs, default 3
  const [failureSeverity, setFailureSeverity] = useState([75]); // 50-100%, default 75%
  
  // High-centrality node failure states
  const [centralityType, setCentralityType] = useState("betweenness"); // betweenness, eigenvector, degree
  const [numCentralNodes, setNumCentralNodes] = useState([5]); // 1-10 nodes, default 5
  const [centralFailureType, setCentralFailureType] = useState("complete"); // complete or partial
  
  // Random multi-node failure states
  const [minNodes, setMinNodes] = useState([1]); // 1-6, default 1
  const [maxNodes, setMaxNodes] = useState([6]); // 1-6, default 6
  const [severityRange, setSeverityRange] = useState([30, 100]); // 30-100%, default range
  const [randomSeed, setRandomSeed] = useState<string>(""); // Optional random seed
  
  // Cyber attack on logistics states
  const [attackScope, setAttackScope] = useState("targeted"); // targeted or widespread
  const [affectedPercentage, setAffectedPercentage] = useState([20]); // 10-30%, default 20%
  const [recoveryTime, setRecoveryTime] = useState([7]); // 1-14 days, default 7
  const [targetRegion, setTargetRegion] = useState<string>(""); // Optional target region
  
  // Major supplier failure states
  const [supplierCapacityThreshold, setSupplierCapacityThreshold] = useState("10"); // top 10%, 20%, 30%
  const [numMajorSuppliers, setNumMajorSuppliers] = useState([2]); // 1-3, default 2
  const [cascadeEffect, setCascadeEffect] = useState(true); // boolean
  
  // Infrastructure failure states
  const [infraEpicenterNodeId, setInfraEpicenterNodeId] = useState<number>(0);
  const [infraRadius, setInfraRadius] = useState([250]); // 50-500 km, default 250
  const [infrastructureType, setInfrastructureType] = useState("all"); // port, airport, warehouse, all
  const [infraDuration, setInfraDuration] = useState([7]); // 1-30 days, default 7
  
  // Hybrid node + edge disruption states
  const [nodeDisruptionType, setNodeDisruptionType] = useState("regional_failure_variable");
  const [edgeDisruptionType, setEdgeDisruptionType] = useState("transportation_route_disruption");

  const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

  // Extract unique countries from loaded nodes
  const [availableCountries, setAvailableCountries] = useState<string[]>([]);

  useEffect(() => {
    // Load nodes from API (company-specific or default)
    const companyId = localStorage.getItem('company_id');
    const apiUrl = companyId 
      ? `http://localhost:5000/api/graph?company_id=${companyId}`
      : 'http://localhost:5000/api/graph';
    
    fetch(apiUrl)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success') {
          const loadedNodes = data.nodes.map((node: any) => ({
            nodeId: node.id,  // Use row index from backend
            label: `${node.region} Node ${node.id}`,
            region: node.region,
            tier: node.tier,
            coordinates: [node.x, node.y] as [number, number],
            risk_level: node.risk_level,
            reliability: node.reliability,
            capacity: node.capacity
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
        } else {
          console.error('Failed to load graph data:', data);
          setLoading(false);
        }
      })
      .catch(err => {
        console.error('Error loading graph:', err);
        setLoading(false);
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
      let nodeIds: number[] = [];
      let disruptionSeverity = severity[0] / 100;
      
      // Convert disruption type parameters to node IDs
      switch (selectedDisruption) {
        case "regional_failure_variable":
          // Find all nodes within radius of epicenter
          const epicenter = allNodes.find(n => n.nodeId === centerNodeId);
          if (!epicenter) {
            throw new Error(`Epicenter node ${centerNodeId} not found`);
          }
          nodeIds = allNodes
            .filter(node => {
              const distance = calculateDistance(epicenter.coordinates, node.coordinates);
              return distance <= radius[0];
            })
            .map(n => n.nodeId);
          disruptionSeverity = regionalSeverity[0] / 100;
          break;
          
        case "distributor_hub_failure":
          // Select random Tier 2 nodes
          const tier2Nodes = allNodes.filter(n => n.tier === selectedTier);
          nodeIds = tier2Nodes
            .sort(() => Math.random() - 0.5)
            .slice(0, numHubs[0])
            .map(n => n.nodeId);
          disruptionSeverity = failureSeverity[0] / 100;
          break;
          
        case "random_supplier_failure":
          // Select random Tier 0 (supplier) nodes
          const supplierNodes = allNodes.filter(n => n.tier === 0);
          nodeIds = supplierNodes
            .sort(() => Math.random() - 0.5)
            .slice(0, numSuppliers[0])
            .map(n => n.nodeId);
          disruptionSeverity = supplierSeverity[0] / 100;
          break;
          
        case "central":
          // Select high-centrality nodes (simplified - using capacity as proxy)
          const sortedByCapacity = [...allNodes].sort((a, b) => b.capacity - a.capacity);
          nodeIds = sortedByCapacity
            .slice(0, numCentralNodes[0])
            .map(n => n.nodeId);
          disruptionSeverity = centralFailureType === 'complete' ? 1.0 : 0.65;
          break;
          
        case "random_multi_node":
          // Random selection of nodes
          const numToFail = Math.floor(Math.random() * (maxNodes[0] - minNodes[0] + 1)) + minNodes[0];
          nodeIds = allNodes
            .sort(() => Math.random() - 0.5)
            .slice(0, numToFail)
            .map(n => n.nodeId);
          disruptionSeverity = (severityRange[0] + severityRange[1]) / 200; // Average of range
          break;
          
        case "trade_restriction":
          // Find nodes in the two countries
          const country1Nodes = allNodes.filter(n => n.region === tradeCountry1);
          const country2Nodes = allNodes.filter(n => n.region === tradeCountry2);
          // Select border nodes (simplified - select some from each country)
          nodeIds = [
            ...country1Nodes.slice(0, 2).map(n => n.nodeId),
            ...country2Nodes.slice(0, 2).map(n => n.nodeId)
          ];
          disruptionSeverity = tariffRate[0] / 100;
          break;
          
        case "transportation_route_disruption":
          // Select random nodes to simulate route disruptions
          nodeIds = allNodes
            .sort(() => Math.random() - 0.5)
            .slice(0, numRoutes[0])
            .map(n => n.nodeId);
          disruptionSeverity = routeSeverityType === 'severe' ? 0.85 : 
                              routeSeverityType === 'moderate' ? 0.55 : 0.30;
          break;
          
        case "cyber_attack_logistics":
          // Select percentage of nodes
          const numAffected = Math.floor(allNodes.length * (affectedPercentage[0] / 100));
          if (targetRegion) {
            const regionNodes = allNodes.filter(n => n.region === targetRegion);
            nodeIds = regionNodes
              .sort(() => Math.random() - 0.5)
              .slice(0, Math.min(numAffected, regionNodes.length))
              .map(n => n.nodeId);
          } else {
            nodeIds = allNodes
              .sort(() => Math.random() - 0.5)
              .slice(0, numAffected)
              .map(n => n.nodeId);
          }
          disruptionSeverity = 0.7;
          break;
          
        case "major_supplier_failure":
          // Select top capacity suppliers
          const threshold = parseInt(supplierCapacityThreshold);
          const topSuppliers = [...allNodes]
            .filter(n => n.tier === 0)
            .sort((a, b) => b.capacity - a.capacity)
            .slice(0, Math.floor(allNodes.filter(n => n.tier === 0).length * (threshold / 100)));
          nodeIds = topSuppliers
            .slice(0, numMajorSuppliers[0])
            .map(n => n.nodeId);
          disruptionSeverity = cascadeEffect ? 0.9 : 0.7;
          break;
          
        case "infrastructure_failure":
          // Similar to regional failure but with infrastructure type consideration
          const infraEpicenter = allNodes.find(n => n.nodeId === infraEpicenterNodeId);
          if (!infraEpicenter) {
            throw new Error(`Infrastructure epicenter node ${infraEpicenterNodeId} not found`);
          }
          nodeIds = allNodes
            .filter(node => {
              const distance = calculateDistance(infraEpicenter.coordinates, node.coordinates);
              return distance <= infraRadius[0];
            })
            .map(n => n.nodeId);
          disruptionSeverity = 0.8;
          break;
          
        case "hybrid_node_edge":
          // Combine node and edge disruptions (simplified - use more nodes)
          nodeIds = allNodes
            .sort(() => Math.random() - 0.5)
            .slice(0, 10)
            .map(n => n.nodeId);
          disruptionSeverity = 0.75;
          break;
          
        default:
          throw new Error('Unknown disruption type');
      }
      
      if (nodeIds.length === 0) {
        throw new Error('No nodes selected for disruption. Please check your parameters.');
      }
      
      // Update the displayed node IDs
      setDisruptedNodeIds(nodeIds.join(', '));
      
      // Call GNN API with buffer capacity and company_id
      const response = await gnnApi.predict({
        company_id: localStorage.getItem('company_id') || undefined,
        disrupted_nodes: nodeIds,
        disrupted_edges: [],
        disruption_severity: disruptionSeverity,
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
  
  // Helper function to calculate distance between two coordinates (Haversine formula)
  const calculateDistance = (coord1: [number, number], coord2: [number, number]): number => {
    const R = 6371; // Earth's radius in km
    const dLat = (coord2[1] - coord1[1]) * Math.PI / 180;
    const dLon = (coord2[0] - coord1[0]) * Math.PI / 180;
    const a = 
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos(coord1[1] * Math.PI / 180) * Math.cos(coord2[1] * Math.PI / 180) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
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
          Predict supply chain resilience using trained GINE model (77.8% accuracy - Best performing)
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

        {selectedDisruption === "regional_failure_variable" && (
          <div className="p-4 bg-indigo-50 rounded-lg border-2 border-indigo-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-indigo-600" />
              <h4 className="font-semibold text-sm text-indigo-900">Step 2: Regional Failure Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="centerNode">Epicenter Node ID</Label>
              <Input
                id="centerNode"
                type="number"
                value={centerNodeId}
                onChange={(e) => setCenterNodeId(Number(e.target.value))}
                min={0}
                max={199}
                placeholder="Enter node ID (0-199)"
                disabled={predicting}
                className="w-full"
              />
              <p className="text-xs text-indigo-700">
                Enter the node ID for the center of the disruption (0-199)
                {allNodes.find(n => n.nodeId === centerNodeId) && (
                  <span className="block mt-1 font-semibold">
                    Selected: {allNodes.find(n => n.nodeId === centerNodeId)?.region} (Tier {allNodes.find(n => n.nodeId === centerNodeId)?.tier})
                  </span>
                )}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="radius">Disruption Radius (km)</Label>
                <span className="text-sm font-medium text-indigo-900">{radius[0]} km</span>
              </div>
              <Slider
                id="radius"
                value={radius}
                onValueChange={setRadius}
                min={100}
                max={1000}
                step={50}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-indigo-700">
                Geographic radius of disruption from epicenter (100-1000 km)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="regionalSeverity">Impact Severity</Label>
                <span className="text-sm font-medium text-indigo-900">{regionalSeverity[0]}%</span>
              </div>
              <Slider
                id="regionalSeverity"
                value={regionalSeverity}
                onValueChange={setRegionalSeverity}
                min={30}
                max={100}
                step={5}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-indigo-700">
                Severity of impact on nodes within radius (30-100%)
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-indigo-200">
              <p className="text-xs text-indigo-800">
                <strong>Scenario:</strong> {radius[0]}km radius disruption centered on Node {centerNodeId} with {regionalSeverity[0]}% severity
              </p>
              <p className="text-xs text-indigo-700 mt-1">
                This simulates regional disasters like earthquakes, floods, or political instability affecting all nodes within the geographic area
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "distributor_hub_failure" && (
          <div className="p-4 bg-teal-50 rounded-lg border-2 border-teal-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Network className="h-4 w-4 text-teal-600" />
              <h4 className="font-semibold text-sm text-teal-900">Step 2: Distributor Hub Failure Configuration (Tier 2 Only)</h4>
            </div>
            
            <div className="p-2 bg-teal-100 rounded border border-teal-300 mb-3">
              <p className="text-xs text-teal-900 font-semibold">
                🏢 Target: Tier 2 - Distribution Centers
              </p>
              <p className="text-xs text-teal-700 mt-1">
                This disruption specifically targets distribution hubs (Tier 2 nodes) which are critical for product flow
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="numHubs">Number of Hubs</Label>
                <span className="text-sm font-medium text-teal-900">{numHubs[0]}</span>
              </div>
              <Slider
                id="numHubs"
                value={numHubs}
                onValueChange={setNumHubs}
                min={1}
                max={5}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-teal-700">
                Number of hub nodes that will fail (1-5)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="hubSeverity">Failure Severity</Label>
                <span className="text-sm font-medium text-teal-900">{failureSeverity[0]}%</span>
              </div>
              <Slider
                id="hubSeverity"
                value={failureSeverity}
                onValueChange={setFailureSeverity}
                min={50}
                max={100}
                step={5}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-teal-700">
                Operational capacity loss at failed hubs (50-100%)
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-teal-200">
              <p className="text-xs text-teal-800">
                <strong>Scenario:</strong> {numHubs[0]} Tier {selectedTier} hub{numHubs[0] > 1 ? 's' : ''} will fail with {failureSeverity[0]}% capacity loss
              </p>
              <p className="text-xs text-teal-700 mt-1">
                This simulates critical distribution center failures affecting downstream supply chain operations
              </p>
              <p className="text-xs text-teal-600 mt-1">
                <strong>Preview:</strong> {allNodes.filter(n => n.tier === selectedTier).length} Tier {selectedTier} nodes available in network
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

        {selectedDisruption === "random_multi_node" && (
          <div className="p-4 bg-slate-50 rounded-lg border-2 border-slate-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-slate-600" />
              <h4 className="font-semibold text-sm text-slate-900">Step 2: Random Multi-Node Failure Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="minNodes">Minimum Nodes</Label>
                <span className="text-sm font-medium text-slate-900">{minNodes[0]}</span>
              </div>
              <Slider
                id="minNodes"
                value={minNodes}
                onValueChange={setMinNodes}
                min={1}
                max={6}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-slate-700">
                Minimum number of nodes that will fail simultaneously (1-6)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="maxNodes">Maximum Nodes</Label>
                <span className="text-sm font-medium text-slate-900">{maxNodes[0]}</span>
              </div>
              <Slider
                id="maxNodes"
                value={maxNodes}
                onValueChange={setMaxNodes}
                min={minNodes[0]}
                max={6}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-slate-700">
                Maximum number of nodes that will fail simultaneously ({minNodes[0]}-6)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="severityRange">Severity Range</Label>
                <span className="text-sm font-medium text-slate-900">{severityRange[0]}% - {severityRange[1]}%</span>
              </div>
              <Slider
                id="severityRange"
                value={severityRange}
                onValueChange={setSeverityRange}
                min={30}
                max={100}
                step={5}
                className="w-full"
                disabled={predicting}
                minStepsBetweenThumbs={1}
              />
              <p className="text-xs text-slate-700">
                Range of impact severity for failed nodes (30-100%)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="randomSeed">Random Seed (Optional)</Label>
              <Input
                id="randomSeed"
                type="text"
                value={randomSeed}
                onChange={(e) => setRandomSeed(e.target.value)}
                placeholder="Leave empty for random selection"
                disabled={predicting}
                className="w-full"
              />
              <p className="text-xs text-slate-700">
                Enter a seed for reproducible random selection (optional)
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-slate-200">
              <p className="text-xs text-slate-800">
                <strong>Scenario:</strong> {minNodes[0]}-{maxNodes[0]} random nodes will fail with {severityRange[0]}-{severityRange[1]}% severity
              </p>
              <p className="text-xs text-slate-700 mt-1">
                This simulates unpredictable multi-node failures across the network, testing resilience to random disruptions
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "cyber_attack_logistics" && (
          <div className="p-4 bg-red-50 rounded-lg border-2 border-red-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <Network className="h-4 w-4 text-red-600" />
              <h4 className="font-semibold text-sm text-red-900">Step 2: Cyber Attack on Logistics Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="attackScope">Attack Scope</Label>
              <Select value={attackScope} onValueChange={setAttackScope} disabled={predicting}>
                <SelectTrigger id="attackScope">
                  <SelectValue placeholder="Select attack scope" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="targeted">Targeted Attack</SelectItem>
                  <SelectItem value="widespread">Widespread Attack</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-red-700">
                {attackScope === 'targeted' ? 'Focused attack on specific high-value logistics routes' : 'Broad attack affecting multiple logistics systems'}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="affectedPercentage">Affected Percentage</Label>
                <span className="text-sm font-medium text-red-900">{affectedPercentage[0]}%</span>
              </div>
              <Slider
                id="affectedPercentage"
                value={affectedPercentage}
                onValueChange={setAffectedPercentage}
                min={10}
                max={30}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-red-700">
                Percentage of logistics edges affected by the cyber attack (10-30%)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="recoveryTime">Recovery Time (Days)</Label>
                <span className="text-sm font-medium text-red-900">{recoveryTime[0]} days</span>
              </div>
              <Slider
                id="recoveryTime"
                value={recoveryTime}
                onValueChange={setRecoveryTime}
                min={1}
                max={14}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-red-700">
                Estimated time to restore affected logistics systems (1-14 days)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="targetRegion">Target Region (Optional)</Label>
              <Select value={targetRegion} onValueChange={setTargetRegion} disabled={predicting}>
                <SelectTrigger id="targetRegion">
                  <SelectValue placeholder="All regions (leave empty for global)" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Regions</SelectItem>
                  {availableCountries.map((country) => (
                    <SelectItem key={country} value={country}>
                      {country}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-red-700">
                Optionally target a specific region for the cyber attack
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-red-200">
              <p className="text-xs text-red-800">
                <strong>Scenario:</strong> {attackScope === 'targeted' ? 'Targeted' : 'Widespread'} cyber attack affecting {affectedPercentage[0]}% of logistics routes
                {targetRegion && ` in ${targetRegion}`}
              </p>
              <p className="text-xs text-red-700 mt-1">
                Recovery time: {recoveryTime[0]} day{recoveryTime[0] > 1 ? 's' : ''}. This simulates ransomware, DDoS, or system compromise affecting digital logistics infrastructure
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "major_supplier_failure" && (
          <div className="p-4 bg-orange-50 rounded-lg border-2 border-orange-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <DollarSign className="h-4 w-4 text-orange-600" />
              <h4 className="font-semibold text-sm text-orange-900">Step 2: Major Supplier Failure Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="capacityThreshold">Capacity Threshold</Label>
              <Select value={supplierCapacityThreshold} onValueChange={setSupplierCapacityThreshold} disabled={predicting}>
                <SelectTrigger id="capacityThreshold">
                  <SelectValue placeholder="Select capacity threshold" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">Top 10% (Highest Capacity)</SelectItem>
                  <SelectItem value="20">Top 20%</SelectItem>
                  <SelectItem value="30">Top 30%</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-orange-700">
                Target suppliers in the top {supplierCapacityThreshold}% by capacity
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="numMajorSuppliers">Number of Suppliers</Label>
                <span className="text-sm font-medium text-orange-900">{numMajorSuppliers[0]}</span>
              </div>
              <Slider
                id="numMajorSuppliers"
                value={numMajorSuppliers}
                onValueChange={setNumMajorSuppliers}
                min={1}
                max={3}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-orange-700">
                Number of high-capacity suppliers that will fail (1-3)
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="cascadeEffect">Cascade Effect</Label>
                <Button
                  variant={cascadeEffect ? "default" : "outline"}
                  size="sm"
                  onClick={() => setCascadeEffect(!cascadeEffect)}
                  disabled={predicting}
                >
                  {cascadeEffect ? "Enabled" : "Disabled"}
                </Button>
              </div>
              <p className="text-xs text-orange-700">
                {cascadeEffect ? 'Failure will affect all outgoing edges (downstream impact)' : 'Only the supplier nodes will fail'}
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-orange-200">
              <p className="text-xs text-orange-800">
                <strong>Scenario:</strong> {numMajorSuppliers[0]} supplier{numMajorSuppliers[0] > 1 ? 's' : ''} from top {supplierCapacityThreshold}% by capacity will fail
                {cascadeEffect && ' with cascading edge failures'}
              </p>
              <p className="text-xs text-orange-700 mt-1">
                This simulates critical supplier failures that disrupt major supply flows, potentially causing widespread downstream impacts
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "infrastructure_failure" && (
          <div className="p-4 bg-gray-50 rounded-lg border-2 border-gray-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-gray-600" />
              <h4 className="font-semibold text-sm text-gray-900">Step 2: Infrastructure Failure Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="infraEpicenter">Epicenter Node ID</Label>
              <Input
                id="infraEpicenter"
                type="number"
                value={infraEpicenterNodeId}
                onChange={(e) => setInfraEpicenterNodeId(Number(e.target.value))}
                min={0}
                max={199}
                placeholder="Enter node ID (0-199)"
                disabled={predicting}
                className="w-full"
              />
              <p className="text-xs text-gray-700">
                Center of infrastructure collapse
                {allNodes.find(n => n.nodeId === infraEpicenterNodeId) && (
                  <span className="block mt-1 font-semibold">
                    Selected: {allNodes.find(n => n.nodeId === infraEpicenterNodeId)?.region} (Tier {allNodes.find(n => n.nodeId === infraEpicenterNodeId)?.tier})
                  </span>
                )}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="infraRadius">Radius (km)</Label>
                <span className="text-sm font-medium text-gray-900">{infraRadius[0]} km</span>
              </div>
              <Slider
                id="infraRadius"
                value={infraRadius}
                onValueChange={setInfraRadius}
                min={50}
                max={500}
                step={25}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-gray-700">
                Radius of infrastructure failure (50-500 km)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="infraType">Infrastructure Type</Label>
              <Select value={infrastructureType} onValueChange={setInfrastructureType} disabled={predicting}>
                <SelectTrigger id="infraType">
                  <SelectValue placeholder="Select infrastructure type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Infrastructure</SelectItem>
                  <SelectItem value="port">Port Facilities</SelectItem>
                  <SelectItem value="airport">Airport Facilities</SelectItem>
                  <SelectItem value="warehouse">Warehouse Facilities</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-gray-700">
                Type of infrastructure affected by the failure
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="infraDuration">Duration (Days)</Label>
                <span className="text-sm font-medium text-gray-900">{infraDuration[0]} days</span>
              </div>
              <Slider
                id="infraDuration"
                value={infraDuration}
                onValueChange={setInfraDuration}
                min={1}
                max={30}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-gray-700">
                Expected duration of infrastructure outage (1-30 days)
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-gray-200">
              <p className="text-xs text-gray-800">
                <strong>Scenario:</strong> {infrastructureType === 'all' ? 'All' : infrastructureType.charAt(0).toUpperCase() + infrastructureType.slice(1)} infrastructure within {infraRadius[0]}km of Node {infraEpicenterNodeId} fails for {infraDuration[0]} day{infraDuration[0] > 1 ? 's' : ''}
              </p>
              <p className="text-xs text-gray-700 mt-1">
                This simulates regional infrastructure collapse affecting both nodes and connecting edges (hybrid disruption)
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "hybrid_node_edge" && (
          <div className="p-4 bg-violet-50 rounded-lg border-2 border-violet-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="h-4 w-4 text-violet-600" />
              <h4 className="font-semibold text-sm text-violet-900">Step 2: Hybrid Node + Edge Disruption Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="nodeDisruption">Node Disruption Type</Label>
              <Select value={nodeDisruptionType} onValueChange={setNodeDisruptionType} disabled={predicting}>
                <SelectTrigger id="nodeDisruption">
                  <SelectValue placeholder="Select node disruption" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="regional_failure_variable">Regional Failure</SelectItem>
                  <SelectItem value="distributor_hub_failure">Distributor Hub Failure</SelectItem>
                  <SelectItem value="random_supplier_failure">Random Supplier Failure</SelectItem>
                  <SelectItem value="central">High-Centrality Node Failure</SelectItem>
                  <SelectItem value="random_multi_node">Random Multi-Node Failure</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-violet-700">
                Select the type of node disruption to combine
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="edgeDisruption">Edge Disruption Type</Label>
              <Select value={edgeDisruptionType} onValueChange={setEdgeDisruptionType} disabled={predicting}>
                <SelectTrigger id="edgeDisruption">
                  <SelectValue placeholder="Select edge disruption" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="transportation_route_disruption">Transportation Route Disruption</SelectItem>
                  <SelectItem value="trade_restriction">Trade Restriction</SelectItem>
                  <SelectItem value="cyber_attack_logistics">Cyber Attack on Logistics</SelectItem>
                  <SelectItem value="major_supplier_failure">Major Supplier Failure</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-violet-700">
                Select the type of edge disruption to combine
              </p>
            </div>

            <div className="p-2 bg-violet-100 rounded border border-violet-300">
              <p className="text-xs text-violet-900 font-semibold mb-2">
                ⚠️ Combined Impact Preview
              </p>
              <p className="text-xs text-violet-800">
                <strong>Node:</strong> {disruptionTypes.find(d => d.value === nodeDisruptionType)?.label}
              </p>
              <p className="text-xs text-violet-800">
                <strong>Edge:</strong> {disruptionTypes.find(d => d.value === edgeDisruptionType)?.label}
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-violet-200">
              <p className="text-xs text-violet-800">
                <strong>Scenario:</strong> Simultaneous {disruptionTypes.find(d => d.value === nodeDisruptionType)?.label} and {disruptionTypes.find(d => d.value === edgeDisruptionType)?.label}
              </p>
              <p className="text-xs text-violet-700 mt-1">
                This simulates compound disruptions where both nodes and edges are affected simultaneously, testing network resilience under extreme conditions
              </p>
              <p className="text-xs text-violet-600 mt-1">
                <strong>Note:</strong> Configure individual disruption parameters using their respective panels before running simulation
              </p>
            </div>
          </div>
        )}

        {selectedDisruption === "central" && (
          <div className="p-4 bg-rose-50 rounded-lg border-2 border-rose-300 space-y-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="h-4 w-4 text-rose-600" />
              <h4 className="font-semibold text-sm text-rose-900">Step 2: High-Centrality Node Failure Configuration</h4>
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="centralityMetric">Centrality Metric</Label>
              <Select value={centralityType} onValueChange={setCentralityType} disabled={predicting}>
                <SelectTrigger id="centralityMetric">
                  <SelectValue placeholder="Select centrality metric" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="betweenness">Betweenness Centrality</SelectItem>
                  <SelectItem value="eigenvector">Eigenvector Centrality</SelectItem>
                  <SelectItem value="degree">Degree Centrality</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-rose-700">
                {centralityType === 'betweenness' && 'Nodes that act as bridges between other nodes (critical for flow)'}
                {centralityType === 'eigenvector' && 'Nodes connected to other important nodes (influence measure)'}
                {centralityType === 'degree' && 'Nodes with most connections (highly connected hubs)'}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <Label htmlFor="numCentralNodes">Number of Nodes</Label>
                <span className="text-sm font-medium text-rose-900">{numCentralNodes[0]}</span>
              </div>
              <Slider
                id="numCentralNodes"
                value={numCentralNodes}
                onValueChange={setNumCentralNodes}
                min={1}
                max={10}
                step={1}
                className="w-full"
                disabled={predicting}
              />
              <p className="text-xs text-rose-700">
                Number of high-centrality nodes to fail (1-10)
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="centralFailure">Failure Type</Label>
              <Select value={centralFailureType} onValueChange={setCentralFailureType} disabled={predicting}>
                <SelectTrigger id="centralFailure">
                  <SelectValue placeholder="Select failure type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="complete">Complete Failure (100% capacity loss)</SelectItem>
                  <SelectItem value="partial">Partial Failure (50-80% capacity loss)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-rose-700">
                {centralFailureType === 'complete' ? 'Node completely stops operating' : 'Node operates at reduced capacity'}
              </p>
            </div>

            <div className="p-2 bg-white rounded border border-rose-200">
              <p className="text-xs text-rose-800">
                <strong>Scenario:</strong> {numCentralNodes[0]} node{numCentralNodes[0] > 1 ? 's' : ''} with highest {centralityType} centrality will experience {centralFailureType} failure
              </p>
              <p className="text-xs text-rose-700 mt-1">
                This targets the most critical nodes in the network based on {centralityType} centrality, simulating strategic disruptions or targeted attacks
              </p>
              <p className="text-xs text-rose-600 mt-1">
                <strong>Impact:</strong> High-centrality node failures typically cause cascading effects throughout the network
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
