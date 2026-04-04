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

interface Supplier {
  id: string;
  name: string;
  location: string;
  riskLevel: "Low" | "Medium" | "High" | "Critical";
  onTimeDelivery: number;
  qualityScore: number;
}

const suppliers: Supplier[] = [
  { id: "S001", name: "TechParts Inc.", location: "Taiwan", riskLevel: "Low", onTimeDelivery: 98, qualityScore: 95 },
  { id: "S002", name: "Global Components", location: "China", riskLevel: "Medium", onTimeDelivery: 92, qualityScore: 90 },
  { id: "S003", name: "FastShip Logistics", location: "Singapore", riskLevel: "Low", onTimeDelivery: 96, qualityScore: 93 },
  { id: "S004", name: "Reliable Materials", location: "India", riskLevel: "High", onTimeDelivery: 78, qualityScore: 82 },
  { id: "S005", name: "Prime Suppliers Co.", location: "Vietnam", riskLevel: "Medium", onTimeDelivery: 88, qualityScore: 87 },
  { id: "S006", name: "Quality First Ltd.", location: "South Korea", riskLevel: "Low", onTimeDelivery: 97, qualityScore: 96 },
];

export function SupplierRiskTable() {
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

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Supplier Risk Assessment</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Supplier ID</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Location</TableHead>
              <TableHead>Risk Level</TableHead>
              <TableHead className="text-right">On-Time Delivery</TableHead>
              <TableHead className="text-right">Quality Score</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {suppliers.map((supplier) => (
              <TableRow key={supplier.id}>
                <TableCell className="font-medium">{supplier.id}</TableCell>
                <TableCell>{supplier.name}</TableCell>
                <TableCell>{supplier.location}</TableCell>
                <TableCell>
                  <Badge className={getRiskColor(supplier.riskLevel)}>
                    {supplier.riskLevel}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">{supplier.onTimeDelivery}%</TableCell>
                <TableCell className="text-right">{supplier.qualityScore}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
