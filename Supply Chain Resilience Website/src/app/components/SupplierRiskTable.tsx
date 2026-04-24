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
import { Button } from "./ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

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
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 100;

  useEffect(() => {
    // Load node data from API (uses uploaded CSV or default)
    fetchNodeData();
  }, []);

  const fetchNodeData = async () => {
    try {
      setLoading(true);
      const companyId = localStorage.getItem('company_id');
      const url = companyId 
        ? `http://localhost:5000/api/graph?company_id=${companyId}`
        : 'http://localhost:5000/api/graph';
      const response = await fetch(url);
      const data = await response.json();
      
      if (data.status === 'success') {
        const nodes = data.nodes;
        
        // Show all nodes (all tiers) - don't slice here, we'll paginate later
        const supplierNodes = nodes
          .map((node: any, index: number) => {
            const riskLevel = node.risk_level || 0.5;
            const reliability = node.reliability || 0.8;
            const tierNum = node.tier || 0;
            
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
              name: node.name || `Node ${index}`,
              location: node.region || 'Unknown',
              tier: tierNum,
              nodeType: nodeType,
              riskLevel: riskCategory,
              riskScore: Math.round(riskLevel * 100),
              resilience: Math.round(reliability * 100),
              capacity: Math.round(node.capacity || 1000)
            };
          });
        
        setSuppliers(supplierNodes);
      }
    } catch (error) {
      console.error('Error loading node data:', error);
    } finally {
      setLoading(false);
    }
  };
  
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

  // Pagination logic
  const totalPages = Math.ceil(suppliers.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentSuppliers = suppliers.slice(startIndex, endIndex);

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
            {currentSuppliers.map((supplier) => (
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

        {/* Pagination Controls */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-4">
            <div className="text-sm text-muted-foreground">
              Showing {startIndex + 1}-{Math.min(endIndex, suppliers.length)} of {suppliers.length} nodes
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
              >
                <ChevronLeft className="h-4 w-4 mr-1" />
                Previous
              </Button>
              <div className="text-sm">
                Page {currentPage} of {totalPages}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
              >
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
