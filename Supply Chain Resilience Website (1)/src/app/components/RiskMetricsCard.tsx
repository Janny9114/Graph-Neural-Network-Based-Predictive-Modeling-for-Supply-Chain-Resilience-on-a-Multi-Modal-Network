import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface RiskMetricsCardProps {
  title: string;
  value: number;
  change: number;
  unit?: string;
  trend?: "up" | "down" | "neutral";
}

export function RiskMetricsCard({ 
  title, 
  value, 
  change, 
  unit = "%",
  trend = "neutral" 
}: RiskMetricsCardProps) {
  const getTrendIcon = () => {
    if (trend === "up") return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (trend === "down") return <TrendingDown className="h-4 w-4 text-red-600" />;
    return <Minus className="h-4 w-4 text-gray-600" />;
  };

  const getTrendColor = () => {
    if (trend === "up") return "text-green-600";
    if (trend === "down") return "text-red-600";
    return "text-gray-600";
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {getTrendIcon()}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          {value}{unit}
        </div>
        <p className={`text-xs ${getTrendColor()} flex items-center mt-1`}>
          {change > 0 ? "+" : ""}{change}% from last month
        </p>
      </CardContent>
    </Card>
  );
}
