import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ComposableMap, Geographies, Geography, Marker, Line } from "react-simple-maps";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { useEffect, useState } from "react";
import Papa from "papaparse";

const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

interface Location {
  id: string;
  name: string;
  type: "supplier" | "manufacturer" | "distribution";
  coordinates: [number, number];
  riskLevel: "Low" | "Medium" | "High" | "Critical";
  status: string;
  tier?: number;
  region?: string;
  capacity?: number;
  cost_factor?: number;
  base_risk?: number;
  risk_level?: number;
}

interface NodeData {
  node_id: string;
  tier: number;
  region: string;
  x: number;
  y: number;
  capacity: number;
  cost_factor: number;
  base_risk: number;
  risk_level: number;
}

interface EdgeData {
  source: string;
  target: string;
  lead_time: number;
  transport_cost: number;
  capacity_share: number;
  disruption_probability: number;
}

const locations: Location[] = [
  // Manufacturers (your facilities)
  { id: "M001", name: "Main Factory (USA)", type: "manufacturer", coordinates: [-95.7129, 37.0902], riskLevel: "Low", status: "Operational" },
  { id: "M002", name: "EU Manufacturing", type: "manufacturer", coordinates: [10.4515, 51.1657], riskLevel: "Low", status: "Operational" },
  
  // Suppliers
  { id: "S001", name: "TechParts Inc.", type: "supplier", coordinates: [121.5654, 25.0330], riskLevel: "Low", status: "Active" },
  { id: "S002", name: "Global Components", type: "supplier", coordinates: [116.4074, 39.9042], riskLevel: "Medium", status: "Delayed" },
  { id: "S003", name: "FastShip Logistics", type: "supplier", coordinates: [103.8198, 1.3521], riskLevel: "Low", status: "Active" },
  { id: "S004", name: "Reliable Materials", type: "supplier", coordinates: [77.1025, 28.7041], riskLevel: "High", status: "Risk Alert" },
  { id: "S005", name: "Prime Suppliers Co.", type: "supplier", coordinates: [105.8342, 21.0285], riskLevel: "Medium", status: "Active" },
  { id: "S006", name: "Quality First Ltd.", type: "supplier", coordinates: [127.0276, 37.5665], riskLevel: "Low", status: "Active" },
  { id: "S007", name: "EuroParts GmbH", type: "supplier", coordinates: [13.4050, 52.5200], riskLevel: "Low", status: "Active" },
  { id: "S008", name: "Pacific Materials", type: "supplier", coordinates: [139.6503, 35.6762], riskLevel: "Medium", status: "Active" },
  
  // Distribution Centers
  { id: "D001", name: "East Coast Hub", type: "distribution", coordinates: [-74.0060, 40.7128], riskLevel: "Low", status: "Active" },
  { id: "D002", name: "Rotterdam Port", type: "distribution", coordinates: [4.4777, 51.9225], riskLevel: "Low", status: "Active" },
];

// Define connections (edges) from suppliers to manufacturers
const connections = [
  // To Main Factory USA
  { from: "S001", to: "M001" }, // Taiwan to USA
  { from: "S002", to: "M001" }, // China to USA
  { from: "S003", to: "M001" }, // Singapore to USA
  { from: "S004", to: "M001" }, // India to USA
  { from: "S005", to: "M001" }, // Vietnam to USA
  { from: "D001", to: "M001" }, // East Coast Hub to Main Factory
  
  // To EU Manufacturing
  { from: "S006", to: "M002" }, // South Korea to EU
  { from: "S007", to: "M002" }, // Germany to EU Manufacturing
  { from: "S008", to: "M002" }, // Japan to EU
  { from: "D002", to: "M002" }, // Rotterdam to EU Manufacturing
];

export function WorldMapNetwork() {
  const [dynamicLocations, setDynamicLocations] = useState<Location[]>([]);
  const [dynamicConnections, setDynamicConnections] = useState<{from: string, to: string}[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load nodes data
    Papa.parse('/synthetic_nodes.csv', {
      download: true,
      header: true,
      complete: (results: any) => {
        console.log('CSV loaded:', results);
        const nodes = results.data as NodeData[];
        
        // Convert nodes to Location format
        const loadedLocations: Location[] = nodes
          .filter(node => node.node_id && node.x && node.y) // Filter out invalid rows
          .map(node => {
            const tierType = 
              Number(node.tier) === 0 ? "supplier" :
              Number(node.tier) === 1 ? "supplier" :
              Number(node.tier) === 2 ? "distribution" :
              "manufacturer";
            
            const riskLevel = 
              Number(node.risk_level) < 0.3 ? "Low" :
              Number(node.risk_level) < 0.6 ? "Medium" :
              Number(node.risk_level) < 0.9 ? "High" :
              "Critical";
            
            return {
              id: String(node.node_id),
              name: `${node.region} Node ${node.node_id}`,
              type: tierType,
              coordinates: [Number(node.x), Number(node.y)] as [number, number],
              riskLevel: riskLevel,
              status: riskLevel === "Critical" ? "Risk Alert" : riskLevel === "High" ? "Delayed" : "Active",
              tier: Number(node.tier),
              region: String(node.region),
              capacity: Number(node.capacity),
              cost_factor: Number(node.cost_factor),
              base_risk: Number(node.base_risk),
              risk_level: Number(node.risk_level)
            };
          });
        
        console.log('Loaded locations:', loadedLocations.length);
        setDynamicLocations(loadedLocations);
        
        // Load edges data
        Papa.parse('/synthetic_edges.csv', {
          download: true,
          header: true,
          complete: (edgeResults: any) => {
            const edges = edgeResults.data as EdgeData[];
            
            // Convert edges to connections format
            const loadedConnections = edges
              .filter(edge => edge.source && edge.target)
              .map(edge => ({
                from: edge.source,
                to: edge.target
              }));
            
            console.log('Loaded connections:', loadedConnections.length);
            setDynamicConnections(loadedConnections);
            setLoading(false);
          }
        });
      }
    });
  }, []);

  // Use dynamic data if loaded, otherwise fall back to static data
  const displayLocations = dynamicLocations.length > 0 ? dynamicLocations : locations;
  const displayConnections = dynamicConnections.length > 0 ? dynamicConnections : connections;

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "Low":
        return "#10b981"; // green
      case "Medium":
        return "#f59e0b"; // yellow/orange
      case "High":
        return "#ef4444"; // red
      case "Critical":
        return "#7f1d1d"; // dark red
      default:
        return "#6b7280"; // gray
    }
  };

  const getNodeSize = (type: string) => {
    switch (type) {
      case "manufacturer":
        return 12;
      case "supplier":
        return 8;
      case "distribution":
        return 6;
      default:
        return 6;
    }
  };

  const getConnectionColor = (fromId: string) => {
    const location = locations.find(loc => loc.id === fromId);
    if (!location) return "#94a3b8";
    
    switch (location.riskLevel) {
      case "Low":
        return "#10b98150";
      case "Medium":
        return "#f59e0b50";
      case "High":
        return "#ef444450";
      case "Critical":
        return "#7f1d1d50";
      default:
        return "#94a3b850";
    }
  };

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Global Supply Chain Network</CardTitle>
      </CardHeader>
      <CardContent className="h-[600px]">
        <TooltipProvider>
          <ComposableMap
            projection="geoMercator"
            projectionConfig={{
              scale: 147,
              center: [20, 30]
            }}
            className="w-full h-full"
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

            {/* Draw connections first (so they appear behind nodes) */}
            {displayConnections.map((connection, i) => {
              const fromLocation = displayLocations.find(loc => loc.id === connection.from);
              const toLocation = displayLocations.find(loc => loc.id === connection.to);
              
              if (!fromLocation || !toLocation) return null;

              return (
                <Line
                  key={`connection-${i}`}
                  from={fromLocation.coordinates}
                  to={toLocation.coordinates}
                  stroke={getConnectionColor(connection.from)}
                  strokeWidth={2}
                  strokeLinecap="round"
                  style={{
                    default: { outline: "none" },
                    hover: { outline: "none" },
                    pressed: { outline: "none" },
                  }}
                />
              );
            })}

            {/* Draw nodes */}
            {displayLocations.map((location) => (
              <Marker key={location.id} coordinates={location.coordinates}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <g>
                      {/* Outer glow for manufacturers */}
                      {location.type === "manufacturer" && (
                        <circle
                          r={getNodeSize(location.type) + 4}
                          fill={getRiskColor(location.riskLevel)}
                          fillOpacity={0.3}
                        />
                      )}
                      {/* Main node */}
                      <circle
                        r={getNodeSize(location.type)}
                        fill={getRiskColor(location.riskLevel)}
                        stroke="#fff"
                        strokeWidth={2}
                        style={{
                          cursor: "pointer",
                        }}
                      />
                      {/* Icon overlay for manufacturers */}
                      {location.type === "manufacturer" && (
                        <circle
                          r={4}
                          fill="#fff"
                        />
                      )}
                    </g>
                  </TooltipTrigger>
                  <TooltipContent>
                    <div className="text-sm">
                      <p className="font-semibold">{location.name}</p>
                      <p className="text-xs text-muted-foreground capitalize">
                        {location.type} | Tier {location.tier}
                      </p>
                      <p className="text-xs">Region: {location.region}</p>
                      <p className="text-xs">Risk: <span className={
                        location.riskLevel === "Low" ? "text-green-600" :
                        location.riskLevel === "Medium" ? "text-yellow-600" :
                        location.riskLevel === "High" ? "text-orange-600" :
                        "text-red-600"
                      }>{location.riskLevel} ({(location.risk_level! * 100).toFixed(1)}%)</span></p>
                      <p className="text-xs">Capacity: {location.capacity?.toFixed(0)}</p>
                      <p className="text-xs">Status: {location.status}</p>
                    </div>
                  </TooltipContent>
                </Tooltip>
              </Marker>
            ))}
          </ComposableMap>
        </TooltipProvider>

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#10b981]"></div>
            <span>Low Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#f59e0b]"></div>
            <span>Medium Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#ef4444]"></div>
            <span>High Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#7f1d1d]"></div>
            <span>Critical Risk</span>
          </div>
          <div className="ml-auto flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-600 border-2 border-white"></div>
              <span>Manufacturer</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gray-600"></div>
              <span>Supplier</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-gray-600"></div>
              <span>Distribution</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
