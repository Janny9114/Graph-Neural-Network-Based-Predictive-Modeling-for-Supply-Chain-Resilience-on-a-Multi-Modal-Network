import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./ui/table";
import { useEffect, useState } from "react";
import Papa from "papaparse";

interface Supplier {
  id: string;
  name: string;
  location: string;
  tier: number;
  nodeType: string;
  riskLevel: "Low" | "Medium" | "High" | "Critical";
  riskScore: number;
  resilience: number;
  capacity: number;
}

export function SupplierRiskTable() {
  const [suppliers, setSuppliers] = useState<Supplier[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load node data from synthetic_nodes.csv
    Papa.parse('/synthetic_nodes.csv', {
      download: true,
      header: true,
      complete: (results: any) => {
        const nodes = results.data;
        
        // Show all nodes (all tiers)
        const supplierNodes = nodes
          .filter((node: any) => node.node_id)
          .slice(0, 100) // Limit to first 100 nodes for display
          .map((node: any, index: number) => {
            const riskLevel = parseFloat(node.risk_level);
            const reliability = parseFloat(node.reliability);
            const tierNum = parseInt(node.tier);
            
            // Determine node type based on tier
            let nodeType = "";
            if (tierNum === 0) {
              nodeType = "Supplier";
            } else if (tierNum === 1) {
              nodeType = "Manufacturer";
            } else if (tierNum === 2) {
              nodeType = "Distributor";
            } else {
              nodeType = "Retailer";
            }
            
            // Determine risk category based on risk_level
            let riskCategory: "Low" | "Medium" | "High" | "Critical";
            if (riskLevel < 0.3) {
              riskCategory = "Low";
            } else if (riskLevel < 0.6) {
              riskCategory = "Medium";
            } else if (riskLevel < 0.9) {
              riskCategory = "High";
            } else {
              riskCategory = "Critical";
            }
            
            return {
              id: `N${String(index + 1).padStart(3, '0')}`,
              name: `${node.region} Node ${node.node_id}`,
              location: node.region,
              tier: tierNum,
              nodeType: nodeType,
              riskLevel: riskCategory,
              riskScore: Math.round(riskLevel * 100),
              resilience: Math.round(reliability * 100),
              capacity: Math.round(parseFloat(node.capacity))
            };
          });
        
        setSuppliers(supplierNodes);
        setLoading(false);
      },
      error: (error: any) => {
        console.error('Error loading node data:', error);
        setLoading(false);
      }
    });
  }, []);
  
  const getRiskBadgeVariant = (risk: string) => {
    switch (risk) {
      case "Low":
        return "default";
      case "Medium":
        return "secondary";
      case "High":
        return "destructive";
      case "Critical":
        return "destructive";
      default:
        return "default";
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Low":
        return "bg-green-100 text-green-800 hover:bg-green-100";
      case "Medium":
        return "bg-yellow-100 text-yellow-800 hover:bg-yellow-100";
      case "High":
        return "bg-orange-100 text-orange-800 hover:bg-orange-100";
      case "Critical":
        return "bg-red-100 text-red-800 hover:bg-red-100";
      default:
        return "";
    }
  };

  const getNodeTypeColor = (type: string) => {
    switch (type) {
      case "Supplier":
        return "bg-purple-100 text-purple-800";
      case "Manufacturer":
        return "bg-blue-100 text-blue-800";
      case "Distributor":
        return "bg-green-100 text-green-800";
      case "Retailer":
        return "bg-orange-100 text-orange-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  if (loading) {
    return (
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle>Node Risk Assessment</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <p className="text-muted-foreground">Loading node data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Node Risk Assessment</CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          Data loaded from synthetic_nodes.csv • Showing all supply chain nodes
        </p>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Node ID</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Region</TableHead>
              <TableHead>Tier</TableHead>
              <TableHead>Node Type</TableHead>
              <TableHead>Risk Level</TableHead>
              <TableHead className="text-right">Risk Score</TableHead>
              <TableHead className="text-right">Resilience</TableHead>
              <TableHead className="text-right">Capacity</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {suppliers.map((supplier) => (
              <TableRow key={supplier.id}>
                <TableCell className="font-medium">{supplier.id}</TableCell>
                <TableCell>{supplier.name}</TableCell>
                <TableCell>{supplier.location}</TableCell>
                <TableCell>
                  <span className="text-xs px-2 py-1 rounded bg-slate-100">
                    Tier {supplier.tier}
                  </span>
                </TableCell>
                <TableCell>
                  <Badge className={getNodeTypeColor(supplier.nodeType)}>
                    {supplier.nodeType}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge className={getRiskColor(supplier.riskLevel)}>
                    {supplier.riskLevel}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  <span className={
                    supplier.riskScore < 30 ? "text-green-600 font-semibold" :
                    supplier.riskScore < 60 ? "text-yellow-600 font-semibold" :
                    supplier.riskScore < 90 ? "text-orange-600 font-semibold" :
                    "text-red-600 font-semibold"
                  }>
                    {supplier.riskScore}%
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  <span className={
                    supplier.resilience >= 90 ? "text-green-600 font-semibold" :
                    supplier.resilience >= 80 ? "text-blue-600 font-semibold" :
                    supplier.resilience >= 70 ? "text-yellow-600 font-semibold" :
                    "text-orange-600 font-semibold"
                  }>
                    {supplier.resilience}%
                  </span>
                </TableCell>
                <TableCell className="text-right">{supplier.capacity.toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
