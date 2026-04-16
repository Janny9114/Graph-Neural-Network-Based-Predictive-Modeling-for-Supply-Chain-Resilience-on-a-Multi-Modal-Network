import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts";

const resilienceData = [
  { month: "Jan", score: 78, risk: 35, performance: 82 },
  { month: "Feb", score: 75, risk: 42, performance: 78 },
  { month: "Mar", score: 82, risk: 30, performance: 85 },
  { month: "Apr", score: 79, risk: 38, performance: 80 },
  { month: "May", score: 85, risk: 28, performance: 88 },
  { month: "Jun", score: 88, risk: 25, performance: 90 },
];

export function ResilienceChart() {
  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Supply Chain Resilience Trends</CardTitle>
      </CardHeader>
      <CardContent className="pl-2">
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={resilienceData}>
            <defs>
              <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="colorPerformance" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="score" 
              stroke="#3b82f6" 
              fillOpacity={1} 
              fill="url(#colorScore)" 
              name="Resilience Score"
            />
            <Area 
              type="monotone" 
              dataKey="risk" 
              stroke="#ef4444" 
              fillOpacity={1} 
              fill="url(#colorRisk)" 
              name="Risk Level"
            />
            <Area 
              type="monotone" 
              dataKey="performance" 
              stroke="#10b981" 
              fillOpacity={1} 
              fill="url(#colorPerformance)" 
              name="Performance"
            />
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
