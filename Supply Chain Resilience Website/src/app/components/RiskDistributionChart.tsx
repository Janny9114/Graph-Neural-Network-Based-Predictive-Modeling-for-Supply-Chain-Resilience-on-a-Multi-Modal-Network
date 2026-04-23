import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer, 
  Legend, 
  Tooltip 
} from "recharts";
import { useEffect, useState } from "react";

export function RiskDistributionChart() {
  const [riskData, setRiskData] = useState([
    { name: "Low Risk", value: 35, color: "#10b981" },
    { name: "Medium Risk", value: 40, color: "#f59e0b" },
    { name: "High Risk", value: 20, color: "#ef4444" },
    { name: "Critical Risk", value: 5, color: "#7f1d1d" },
  ]);

  useEffect(() => {
    fetchRiskDistribution();
  }, []);

  const fetchRiskDistribution = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/graph');
      const data = await response.json();
      
      if (data.status === 'success') {
        const nodes = data.nodes;
        
        // Calculate risk distribution
        let low = 0, medium = 0, high = 0, critical = 0;
        nodes.forEach((node: any) => {
          const riskLevel = node.risk_level || 0.5;
          if (riskLevel < 0.3) low++;
          else if (riskLevel < 0.6) medium++;
          else if (riskLevel < 0.9) high++;
          else critical++;
        });
        
        // Only include categories with non-zero values
        const distribution = [
          { name: "Low Risk", value: low, color: "#10b981" },
          { name: "Medium Risk", value: medium, color: "#f59e0b" },
          { name: "High Risk", value: high, color: "#ef4444" },
          { name: "Critical Risk", value: critical, color: "#7f1d1d" },
        ].filter(item => item.value > 0);
        
        setRiskData(distribution);
      }
    } catch (error) {
      console.error('Error fetching risk distribution:', error);
    }
  };

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
