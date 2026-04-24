import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ComposableMap, Geographies, Geography, Marker, Line, ZoomableGroup } from "react-simple-maps";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "./ui/tooltip";
import { Button } from "./ui/button";
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

const geoUrl = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

interface Location {
  id: string;
  name: string;
  type: "supplier" | "manufacturer" | "distribution" | "retailer";
  coordinates: [number, number];
  riskLevel: "Low" | "Medium" | "High" | "Critical";
  status: string;
  tier?: number;
  region?: string;
  capacity?: number;
  cost_factor?: number;
  risk_level?: number;
}

const locations: Location[] = [
  { id: "M001", name: "Main Factory (USA)", type: "manufacturer", coordinates: [-95.7129, 37.0902], riskLevel: "Low", status: "Operational" },
  { id: "M002", name: "EU Manufacturing", type: "manufacturer", coordinates: [10.4515, 51.1657], riskLevel: "Low", status: "Operational" },
  { id: "S001", name: "TechParts Inc.", type: "supplier", coordinates: [121.5654, 25.0330], riskLevel: "Low", status: "Active" },
  { id: "S002", name: "Global Components", type: "supplier", coordinates: [116.4074, 39.9042], riskLevel: "Medium", status: "Delayed" },
  { id: "S003", name: "FastShip Logistics", type: "supplier", coordinates: [103.8198, 1.3521], riskLevel: "Low", status: "Active" },
  { id: "S004", name: "Reliable Materials", type: "supplier", coordinates: [77.1025, 28.7041], riskLevel: "High", status: "Risk Alert" },
  { id: "S005", name: "Prime Suppliers Co.", type: "supplier", coordinates: [105.8342, 21.0285], riskLevel: "Medium", status: "Active" },
  { id: "S006", name: "Quality First Ltd.", type: "supplier", coordinates: [127.0276, 37.5665], riskLevel: "Low", status: "Active" },
  { id: "S007", name: "EuroParts GmbH", type: "supplier", coordinates: [13.4050, 52.5200], riskLevel: "Low", status: "Active" },
  { id: "S008", name: "Pacific Materials", type: "supplier", coordinates: [139.6503, 35.6762], riskLevel: "Medium", status: "Active" },
  { id: "D001", name: "East Coast Hub", type: "distribution", coordinates: [-74.0060, 40.7128], riskLevel: "Low", status: "Active" },
  { id: "D002", name: "Rotterdam Port", type: "distribution", coordinates: [4.4777, 51.9225], riskLevel: "Low", status: "Active" },
];

const connections = [
  { from: "S001", to: "M001" },
  { from: "S002", to: "M001" },
  { from: "S003", to: "M001" },
  { from: "S004", to: "M001" },
  { from: "S005", to: "M001" },
  { from: "D001", to: "M001" },
  { from: "S006", to: "M002" },
  { from: "S007", to: "M002" },
  { from: "S008", to: "M002" },
  { from: "D002", to: "M002" },
];

export function WorldMapNetwork() {
  const [dynamicLocations, setDynamicLocations] = useState<Location[]>([]);
  const [dynamicConnections, setDynamicConnections] = useState<{from: string, to: string}[]>([]);
  const [loading, setLoading] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [center, setCenter] = useState<[number, number]>([20, 30]);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    const fetchGraphData = async () => {
      try {
        const companyId = localStorage.getItem('company_id');
        const url = companyId 
          ? `http://localhost:5000/api/graph?company_id=${companyId}`
          : 'http://localhost:5000/api/graph';
        
        console.log('WorldMap fetching from:', url);
        const response = await fetch(url);
        const data = await response.json();
        
        if (data.status === 'success') {
          const nodes = data.nodes;
          const edges = data.edges;
          console.log('WorldMap loaded:', nodes.length, 'nodes,', edges.length, 'edges');
        
          const loadedLocations: Location[] = nodes
            .filter((node: any) => node.id !== undefined && node.x && node.y)
            .map((node: any) => {
              const tierType = 
                Number(node.tier) === 0 ? "supplier" :
                Number(node.tier) === 1 ? "manufacturer" :
                Number(node.tier) === 2 ? "distribution" :
                "retailer";
              
              const riskLevelValue = Number(node.risk_level);
              const riskLevel = 
                riskLevelValue < 0.3 ? "Low" :
                riskLevelValue < 0.6 ? "Medium" :
                riskLevelValue < 0.9 ? "High" :
                "Critical";
              
              return {
                id: String(node.id),
                name: node.name || `${node.region} Node ${node.id}`,
                type: tierType,
                coordinates: [Number(node.x), Number(node.y)] as [number, number],
                riskLevel: riskLevel,
                status: riskLevel === "Critical" ? "Risk Alert" : riskLevel === "High" ? "Delayed" : "Active",
                tier: Number(node.tier),
                region: String(node.region),
                capacity: Number(node.capacity),
                cost_factor: Number(node.cost_factor),
                risk_level: riskLevelValue
              };
            });
          
          const loadedConnections = edges
            .filter((edge: any) => edge.source !== undefined && edge.target !== undefined)
            .map((edge: any) => ({
              from: String(edge.source),
              to: String(edge.target)
            }));
          
          setDynamicLocations(loadedLocations);
          setDynamicConnections(loadedConnections);
          setLoading(false);
        }
      } catch (error) {
        console.error('Error fetching graph data:', error);
        setLoading(false);
      }
    };
    
    fetchGraphData();
  }, []);

  const displayLocations = dynamicLocations.length > 0 ? dynamicLocations : locations;
  const displayConnections = dynamicConnections.length > 0 ? dynamicConnections : connections;

  const getTierColor = (tier: number) => {
    switch (tier) {
      case 0: return "#9333ea";
      case 1: return "#3b82f6";
      case 2: return "#10b981";
      case 3: return "#f59e0b";
      default: return "#94a3b8";
    }
  };

  const getTierColorWithOpacity = (tier: number) => {
    switch (tier) {
      case 0: return "#9333ea50";
      case 1: return "#3b82f650";
      case 2: return "#10b98150";
      case 3: return "#f59e0b50";
      default: return "#94a3b850";
    }
  };

  const getConnectionColor = (fromId: string) => {
    const location = displayLocations.find(loc => loc.id === fromId);
    if (!location) return "#94a3b8";
    
    switch (location.riskLevel) {
      case "Low": return "#10b98150";
      case "Medium": return "#f59e0b50";
      case "High": return "#ef444450";
      case "Critical": return "#7f1d1d50";
      default: return "#94a3b850";
    }
  };

  const handleZoomIn = () => setZoom(prev => Math.min(prev * 1.5, 8));
  const handleZoomOut = () => setZoom(prev => Math.max(prev / 1.5, 1));
  const handleReset = () => {
    setZoom(1);
    setCenter([20, 30]);
  };

  const getNodeRadius = (baseRadius: number) => {
    const scaleFactor = Math.max(0.4, 1 / Math.sqrt(zoom));
    return baseRadius * scaleFactor;
  };

  return (
    <Card className="col-span-4">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle>Global Supply Chain Network</CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Drag to pan • Scroll to zoom • Click buttons for quick zoom
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleZoomIn}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={handleZoomOut}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="h-[600px]">
        <TooltipProvider>
          <ComposableMap projection="geoMercator" className="w-full h-full">
            <ZoomableGroup
              center={center}
              zoom={zoom}
              minZoom={1}
              maxZoom={8}
              onMoveEnd={({ coordinates, zoom: newZoom }) => {
                setCenter(coordinates as [number, number]);
                setZoom(newZoom);
              }}
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

              {displayConnections.map((connection, i) => {
                const fromLocation = displayLocations.find(loc => loc.id === connection.from);
                const toLocation = displayLocations.find(loc => loc.id === connection.to);
                
                if (!fromLocation || !toLocation) return null;

                const isConnected = hoveredNode && (connection.from === hoveredNode || connection.to === hoveredNode);

                return (
                  <Line
                    key={`connection-${i}`}
                    from={fromLocation.coordinates}
                    to={toLocation.coordinates}
                    stroke={isConnected ? "#3b82f6" : getConnectionColor(connection.from)}
                    strokeWidth={isConnected ? 2 : 0.5}
                    strokeLinecap="round"
                  />
                );
              })}

              {displayLocations.map((location) => (
                <Marker key={location.id} coordinates={location.coordinates}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <g
                        onMouseEnter={() => setHoveredNode(location.id)}
                        onMouseLeave={() => setHoveredNode(null)}
                      >
                        {location.type === "manufacturer" && (
                          <circle
                            r={getNodeRadius(12)}
                            fill={getTierColorWithOpacity(location.tier || 0)}
                            fillOpacity={0.3}
                          />
                        )}
                        <circle
                          r={getNodeRadius(location.type === "manufacturer" ? 8 : location.type === "distribution" ? 6 : 5)}
                          fill={getTierColor(location.tier || 0)}
                          stroke="#fff"
                          strokeWidth={getNodeRadius(2)}
                          style={{ cursor: "pointer" }}
                        />
                        {location.type === "manufacturer" && (
                          <circle r={getNodeRadius(4)} fill="#fff" />
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
                        <p className="text-xs">Risk Level: <span className={
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
            </ZoomableGroup>
          </ComposableMap>
        </TooltipProvider>

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
              <div className="w-[10px] h-[10px] rounded-full bg-purple-600 border-2 border-white"></div>
              <span>Supplier (Tier 0)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-600 border-2 border-white"></div>
              <span>Manufacturer (Tier 1)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-600"></div>
              <span>Distributor (Tier 2)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-[6px] h-[6px] rounded-full bg-orange-500"></div>
              <span>Retailer (Tier 3)</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
