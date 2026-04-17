import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts";

const performanceData = [
  { category: "On-Time Delivery", current: 92, target: 95 },
  { category: "Quality Rate", current: 88, target: 90 },
  { category: "Lead Time", current: 78, target: 85 },
  { category: "Cost Efficiency", current: 85, target: 80 },
  { category: "Flexibility", current: 82, target: 85 },
];

export function PerformanceMetrics() {
  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle>Performance vs Target</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={performanceData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={[0, 100]} />
            <YAxis dataKey="category" type="category" width={120} />
            <Tooltip />
            <Legend />
            <Bar dataKey="current" fill="#3b82f6" name="Current" />
            <Bar dataKey="target" fill="#10b981" name="Target" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
