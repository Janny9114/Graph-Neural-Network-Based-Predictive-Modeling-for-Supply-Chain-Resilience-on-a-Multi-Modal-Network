import { useEffect, useState } from "react";
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
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  Legend,
} from "recharts";
import { AlertTriangle, TrendingDown } from "lucide-react";

interface CascadeStep {
  node_id: number;
  node_name: string;
  tier: number;
  region: string;
  betweenness: number;
  nodes_disconnected: number;
  edges_lost: number;
  remaining_nodes: number;
  num_components: number;
  is_connected: boolean;
  largest_component_size: number;
  fragmentation_ratio: number;
}

interface CascadeData {
  initial_state: {
    num_nodes: number;
    num_edges: number;
    is_connected: boolean;
  };
  cascade_sequence: CascadeStep[];
}

const TIER_LABELS: Record<number, string> = {
  0: "Supplier",
  1: "Manufacturer",
  2: "Distributor",
  3: "Retailer",
};

const TIER_COLORS: Record<number, string> = {
  0: "bg-purple-100 text-purple-800",
  1: "bg-blue-100 text-blue-800",
  2: "bg-green-100 text-green-800",
  3: "bg-orange-100 text-orange-800",
};

export function CascadingFailureHeatmap() {
  const [data, setData] = useState<CascadeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCascade();
  }, []);

  const fetchCascade = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(
        "http://localhost:5000/api/cascading-failure?top_n=10"
      );
      const json = await res.json();
      if (json.status === "success") {
        setData(json as CascadeData);
      } else {
        setError(json.message ?? "Unknown error");
      }
    } catch (e: any) {
      setError(e.message ?? "Failed to fetch cascade data");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingDown className="h-5 w-5 text-red-600" />
            Cascading Failure Simulation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            Simulating cascading failures…
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error || !data) {
    return (
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingDown className="h-5 w-5 text-red-600" />
            Cascading Failure Simulation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-red-600 py-4">
            <AlertTriangle className="h-4 w-4" />
            <span>{error ?? "No data available"}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { initial_state, cascade_sequence } = data;

  // Prepare chart data showing fragmentation over time
  const chartData = cascade_sequence.map((step, idx) => ({
    step: `Step ${idx + 1}`,
    node_name: step.node_name.length > 12 ? step.node_name.slice(0, 11) + "…" : step.node_name,
    fragmentation: (step.fragmentation_ratio * 100).toFixed(1),
    remaining: step.remaining_nodes,
    components: step.num_components,
  }));

  // Color gradient from yellow to red based on fragmentation
  const getBarColor = (fragmentation: number) => {
    if (fragmentation < 10) return "#f59e0b"; // amber
    if (fragmentation < 25) return "#fb923c"; // orange
    if (fragmentation < 50) return "#f87171"; // light red
    return "#dc2626"; // red
  };

  return (
    <Card className="col-span-4">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <TrendingDown className="h-5 w-5 text-red-600" />
              Cascading Failure Simulation
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Sequential removal of the top 10 most critical nodes (by betweenness
              centrality) — shows the domino effect on network connectivity
            </p>
          </div>
          <Badge variant="outline" className="text-xs shrink-0 mt-1">
            Initial: {initial_state.num_nodes} nodes · {initial_state.num_edges}{" "}
            edges
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Fragmentation chart */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Network Fragmentation Over Sequential Removals
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="node_name"
                tick={{ fontSize: 10 }}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis
                label={{
                  value: "Fragmentation %",
                  angle: -90,
                  position: "insideLeft",
                  style: { fontSize: 12 },
                }}
                tick={{ fontSize: 11 }}
              />
              <Tooltip
                formatter={(value: any, name: string) => {
                  if (name === "fragmentation") return [`${value}%`, "Fragmentation"];
                  if (name === "remaining") return [value, "Remaining Nodes"];
                  if (name === "components") return [value, "Components"];
                  return [value, name];
                }}
              />
              <Legend />
              <Bar dataKey="fragmentation" name="Fragmentation %" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={getBarColor(parseFloat(entry.fragmentation))} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Detailed table */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">
            Cascade Sequence Details
          </h3>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Step</TableHead>
                <TableHead>Removed Node</TableHead>
                <TableHead>Tier</TableHead>
                <TableHead>Region</TableHead>
                <TableHead className="text-right">Betweenness</TableHead>
                <TableHead className="text-right">Edges Lost</TableHead>
                <TableHead className="text-right">Remaining Nodes</TableHead>
                <TableHead className="text-right">Components</TableHead>
                <TableHead className="text-right">Fragmentation</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {cascade_sequence.map((step, idx) => (
                <TableRow
                  key={step.node_id}
                  className={
                    step.fragmentation_ratio > 0.5
                      ? "bg-red-50/60"
                      : step.fragmentation_ratio > 0.25
                      ? "bg-orange-50/60"
                      : ""
                  }
                >
                  <TableCell className="font-medium">{idx + 1}</TableCell>
                  <TableCell className="font-medium">{step.node_name}</TableCell>
                  <TableCell>
                    <Badge
                      className={
                        TIER_COLORS[step.tier] ?? "bg-gray-100 text-gray-800"
                      }
                    >
                      {TIER_LABELS[step.tier] ?? `Tier ${step.tier}`}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {step.region}
                  </TableCell>
                  <TableCell className="text-right font-mono text-sm">
                    {(step.betweenness * 100).toFixed(3)}%
                  </TableCell>
                  <TableCell className="text-right">
                    <span className="text-red-600 font-semibold">
                      -{step.edges_lost}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    {step.remaining_nodes}
                  </TableCell>
                  <TableCell className="text-right">
                    <span
                      className={
                        step.num_components > 1
                          ? "text-red-600 font-semibold"
                          : "text-green-600"
                      }
                    >
                      {step.num_components}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <span
                      className={
                        step.fragmentation_ratio > 0.5
                          ? "text-red-600 font-bold"
                          : step.fragmentation_ratio > 0.25
                          ? "text-orange-600 font-semibold"
                          : "text-yellow-600 font-semibold"
                      }
                    >
                      {(step.fragmentation_ratio * 100).toFixed(1)}%
                    </span>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Summary warning */}
        <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
            <div className="text-red-800">
              <p className="font-semibold">Critical Vulnerability Detected</p>
              <p className="mt-1">
                Removing just{" "}
                <strong>
                  {cascade_sequence.length} nodes (
                  {((cascade_sequence.length / initial_state.num_nodes) * 100).toFixed(1)}
                  %)
                </strong>{" "}
                causes{" "}
                <strong>
                  {(
                    cascade_sequence[cascade_sequence.length - 1]
                      .fragmentation_ratio * 100
                  ).toFixed(1)}
                  % fragmentation
                </strong>
                . Prioritize adding redundant paths and backup suppliers for these
                nodes.
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
