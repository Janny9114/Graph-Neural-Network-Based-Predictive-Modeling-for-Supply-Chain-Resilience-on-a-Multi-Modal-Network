import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  Legend, 
  Tooltip 
} from "recharts";

const riskData = [
  { name: "Low Risk", value: 35, color: "#10b981" },
  { name: "Medium Risk", value: 40, color: "#f59e0b" },
  { name: "High Risk", value: 20, color: "#ef4444" },
  { name: "Critical Risk", value: 5, color: "#7f1d1d" },
];

export function RiskDistributionChart() {
  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle>Risk Distribution</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={riskData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {riskData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
