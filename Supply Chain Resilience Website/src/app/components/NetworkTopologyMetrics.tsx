import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Network, AlertCircle } from "lucide-react";

interface TopologyMetric {
  value: number;
  ideal?: number;
  percentage?: number;
  description: string;
}

interface TopologyData {
  metrics: {
    density: TopologyMetric;
    avg_path_length: TopologyMetric;
    clustering: TopologyMetric;
    diameter: TopologyMetric;
    assortativity: TopologyMetric;
  };
  summary: {
    num_nodes: number;
    num_edges: number;
    is_connected: boolean;
    num_components: number;
  };
}

export function NetworkTopologyMetrics() {
  const [data, setData] = useState<TopologyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTopology();
  }, []);

  const fetchTopology = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch("http://localhost:5000/api/network-topology");
      const json = await res.json();
      if (json.status === "success") {
        setData(json as TopologyData);
      } else {
        setError(json.message ?? "Unknown error");
      }
    } catch (e: any) {
      setError(e.message ?? "Failed to fetch topology data");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5 text-blue-600" />
            Network Topology Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            Computing topology metrics…
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className="col-span-2">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5 text-blue-600" />
            Network Topology Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-red-600 py-4">
            <AlertCircle className="h-4 w-4" />
            <span>{error ?? "No data available"}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { metrics } = data;

  // Prepare chart data
  const chartData = [
    {
      name: "Density",
      value: metrics.density.value,
      ideal: metrics.density.ideal,
      percentage: metrics.density.percentage,
      display: (metrics.density.value * 100).toFixed(1),
    },
    {
      name: "Clustering",
      value: metrics.clustering.value,
      ideal: metrics.clustering.ideal,
      percentage: metrics.clustering.percentage,
      display: (metrics.clustering.value * 100).toFixed(1),
    },
    {
      name: "Path Length",
      value: metrics.avg_path_length.value,
      ideal: null,
      percentage: null,
      display: metrics.avg_path_length.value.toFixed(2),
    },
    {
      name: "Diameter",
      value: metrics.diameter.value,
      ideal: null,
      percentage: null,
      display: metrics.diameter.value.toString(),
    },
    {
      name: "Assortativity",
      value: Math.abs(metrics.assortativity.value),
      ideal: null,
      percentage: null,
      display: metrics.assortativity.value.toFixed(3),
    },
  ];

  // Color based on health (green = good, yellow = ok, red = poor)
  const getBarColor = (item: typeof chartData[0]) => {
    if (item.percentage !== null && item.percentage !== undefined) {
      if (item.percentage >= 80 && item.percentage <= 120) return "#10b981"; // green
      if (item.percentage >= 60 && item.percentage <= 140) return "#f59e0b"; // amber
      return "#ef4444"; // red
    }
    // For metrics without ideal values, use blue
    return "#3b82f6";
  };

  return (
    <Card className="col-span-2">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Network className="h-5 w-5 text-blue-600" />
              Network Topology Metrics
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Structural resilience indicators for the supply chain network
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Bar chart */}
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={chartData} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 11 }} />
            <YAxis
              type="category"
              dataKey="name"
              width={90}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(value: number, name: string, props: any) => [
                props.payload.display,
                name,
              ]}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={getBarColor(entry)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Metric descriptions */}
        <div className="grid grid-cols-1 gap-2 text-xs">
          <div className="flex items-start gap-2 p-2 rounded bg-slate-50">
            <span className="font-semibold text-slate-700 min-w-[90px]">
              Density:
            </span>
            <span className="text-slate-600">
              {metrics.density.description} (
              <strong>{(metrics.density.value * 100).toFixed(1)}%</strong>)
            </span>
          </div>

          <div className="flex items-start gap-2 p-2 rounded bg-slate-50">
            <span className="font-semibold text-slate-700 min-w-[90px]">
              Clustering:
            </span>
            <span className="text-slate-600">
              {metrics.clustering.description} (
              <strong>{(metrics.clustering.value * 100).toFixed(1)}%</strong>)
            </span>
          </div>

          <div className="flex items-start gap-2 p-2 rounded bg-slate-50">
            <span className="font-semibold text-slate-700 min-w-[90px]">
              Path Length:
            </span>
            <span className="text-slate-600">
              {metrics.avg_path_length.description} (
              <strong>{metrics.avg_path_length.value.toFixed(2)} hops</strong>)
            </span>
          </div>

          <div className="flex items-start gap-2 p-2 rounded bg-slate-50">
            <span className="font-semibold text-slate-700 min-w-[90px]">
              Diameter:
            </span>
            <span className="text-slate-600">
              {metrics.diameter.description} (
              <strong>{metrics.diameter.value} hops</strong>)
            </span>
          </div>

          <div className="flex items-start gap-2 p-2 rounded bg-slate-50">
            <span className="font-semibold text-slate-700 min-w-[90px]">
              Assortativity:
            </span>
            <span className="text-slate-600">
              {metrics.assortativity.description} (
              <strong>{metrics.assortativity.value.toFixed(3)}</strong>)
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
