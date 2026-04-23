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
} from "recharts";
import { AlertTriangle, Network, Zap, Scissors } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface VulnerableNode {
  node_id: number;
  name: string;
  tier: number;
  region: string;
  risk_level: number;
  reliability: number;
  capacity: number;
  degree: number;
  betweenness_centrality?: number;
  eigenvector_score?: number;
  is_articulation_point: boolean;
}

interface VulnerabilitySummary {
  num_nodes: number;
  num_edges: number;
  num_articulation_points: number;
  articulation_point_fraction: number;
  avg_betweenness: number;
  max_betweenness: number;
  avg_eigenvector: number;
  max_eigenvector: number;
  centrality_method: string;
  is_connected: boolean;
  num_connected_components: number;
}

interface VulnerabilityData {
  summary: VulnerabilitySummary;
  betweenness_centrality: VulnerableNode[];
  eigenvector_centrality: VulnerableNode[];
  articulation_points: VulnerableNode[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

function riskColor(r: number) {
  if (r < 0.3) return "text-green-600 font-semibold";
  if (r < 0.6) return "text-yellow-600 font-semibold";
  if (r < 0.9) return "text-orange-600 font-semibold";
  return "text-red-600 font-semibold";
}

function barColor(score: number, max: number) {
  const ratio = max > 0 ? score / max : 0;
  if (ratio > 0.7) return "#ef4444";
  if (ratio > 0.4) return "#f59e0b";
  return "#3b82f6";
}

// ─── Sub-components ───────────────────────────────────────────────────────────

/** Horizontal bar chart for top-N nodes */
function CentralityBarChart({
  data,
  scoreKey,
  label,
}: {
  data: VulnerableNode[];
  scoreKey: "betweenness_centrality" | "eigenvector_score";
  label: string;
}) {
  const chartData = data.map((n) => ({
    name: n.name.length > 14 ? n.name.slice(0, 13) + "…" : n.name,
    score: Number(((n[scoreKey] ?? 0) * 100).toFixed(3)),
    raw: n[scoreKey] ?? 0,
  }));

  const maxScore = Math.max(...chartData.map((d) => d.raw));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={chartData}
        layout="vertical"
        margin={{ top: 4, right: 24, left: 8, bottom: 4 }}
      >
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          tickFormatter={(v) => `${v.toFixed(2)}%`}
          tick={{ fontSize: 11 }}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={110}
          tick={{ fontSize: 11 }}
        />
        <Tooltip
          formatter={(value: number) => [`${value.toFixed(4)}%`, label]}
        />
        <Bar dataKey="score" radius={[0, 4, 4, 0]}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={barColor(entry.raw, maxScore)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Compact node table */
function NodeTable({
  nodes,
  scoreKey,
  scoreLabel,
}: {
  nodes: VulnerableNode[];
  scoreKey: "betweenness_centrality" | "eigenvector_score";
  scoreLabel: string;
}) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Node</TableHead>
          <TableHead>Region</TableHead>
          <TableHead>Tier</TableHead>
          <TableHead className="text-right">Degree</TableHead>
          <TableHead className="text-right">{scoreLabel}</TableHead>
          <TableHead className="text-right">Risk</TableHead>
          <TableHead className="text-center">Cut Vertex</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {nodes.map((n) => (
          <TableRow key={n.node_id}>
            <TableCell className="font-medium">{n.name}</TableCell>
            <TableCell className="text-sm text-muted-foreground">
              {n.region}
            </TableCell>
            <TableCell>
              <Badge className={TIER_COLORS[n.tier] ?? "bg-gray-100 text-gray-800"}>
                {TIER_LABELS[n.tier] ?? `Tier ${n.tier}`}
              </Badge>
            </TableCell>
            <TableCell className="text-right">{n.degree}</TableCell>
            <TableCell className="text-right font-mono text-sm">
              {((n[scoreKey] ?? 0) * 100).toFixed(3)}%
            </TableCell>
            <TableCell className="text-right">
              <span className={riskColor(n.risk_level)}>
                {(n.risk_level * 100).toFixed(1)}%
              </span>
            </TableCell>
            <TableCell className="text-center">
              {n.is_articulation_point ? (
                <span className="inline-flex items-center gap-1 text-red-600 font-semibold text-xs">
                  <Scissors className="h-3 w-3" /> Yes
                </span>
              ) : (
                <span className="text-gray-400 text-xs">—</span>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export function VulnerableNodesAnalysis() {
  const [data, setData] = useState<VulnerabilityData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<
    "betweenness" | "eigenvector" | "articulation"
  >("betweenness");

  useEffect(() => {
    fetchVulnerability();
  }, []);

  const fetchVulnerability = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(
        "http://localhost:5000/api/network-vulnerability?top_n=10"
      );
      const json = await res.json();
      if (json.status === "success") {
        setData(json as VulnerabilityData);
      } else {
        setError(json.message ?? "Unknown error");
      }
    } catch (e: any) {
      setError(e.message ?? "Failed to fetch vulnerability data");
    } finally {
      setLoading(false);
    }
  };

  // ── Loading / Error states ─────────────────────────────────────────────────
  if (loading) {
    return (
      <Card className="col-span-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5 text-blue-600" />
            Vulnerable Node Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-12 text-muted-foreground">
            Computing network centrality metrics…
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
            <Network className="h-5 w-5 text-blue-600" />
            Vulnerable Node Analysis
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

  const { summary, betweenness_centrality, eigenvector_centrality, articulation_points } = data;

  // ── Summary stat cards ─────────────────────────────────────────────────────
  const statCards = [
    {
      icon: <Scissors className="h-5 w-5 text-red-500" />,
      label: "Articulation Points",
      value: summary.num_articulation_points,
      sub: `${(summary.articulation_point_fraction * 100).toFixed(1)}% of nodes`,
      bg: "bg-red-50 border-red-200",
    },
    {
      icon: <Network className="h-5 w-5 text-orange-500" />,
      label: "Max Betweenness",
      value: `${(summary.max_betweenness * 100).toFixed(2)}%`,
      sub: `Avg ${(summary.avg_betweenness * 100).toFixed(3)}%`,
      bg: "bg-orange-50 border-orange-200",
    },
    {
      icon: <Zap className="h-5 w-5 text-blue-500" />,
      label: summary.centrality_method,
      value: `${(summary.max_eigenvector * 100).toFixed(2)}%`,
      sub: `Avg ${(summary.avg_eigenvector * 100).toFixed(3)}%`,
      bg: "bg-blue-50 border-blue-200",
    },
    {
      icon: <AlertTriangle className="h-5 w-5 text-yellow-500" />,
      label: "Network Connected",
      value: summary.is_connected ? "Yes" : "No",
      sub: `${summary.num_connected_components} component${summary.num_connected_components !== 1 ? "s" : ""}`,
      bg: summary.is_connected
        ? "bg-green-50 border-green-200"
        : "bg-red-50 border-red-200",
    },
  ];

  // ── Tab config ─────────────────────────────────────────────────────────────
  const tabs: {
    key: typeof activeTab;
    label: string;
    icon: React.ReactNode;
    description: string;
  }[] = [
    {
      key: "betweenness",
      label: "Betweenness Centrality",
      icon: <Network className="h-4 w-4" />,
      description:
        "Nodes that lie on the most shortest paths — removing them fragments the network into isolated clusters.",
    },
    {
      key: "eigenvector",
      label: summary.centrality_method,
      icon: <Zap className="h-4 w-4" />,
      description:
        "Nodes whose influence is amplified by being connected to other highly-connected nodes — the true hubs of the supply chain.",
    },
    {
      key: "articulation",
      label: "Articulation Points",
      icon: <Scissors className="h-4 w-4" />,
      description:
        "Cut vertices: removing any one of these nodes instantly disconnects the graph into two or more isolated subgraphs.",
    },
  ];

  const activeTabConfig = tabs.find((t) => t.key === activeTab)!;

  return (
    <Card className="col-span-4">
      <CardHeader>
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Network className="h-5 w-5 text-blue-600" />
              Vulnerable Node Analysis
            </CardTitle>
            <p className="text-sm text-muted-foreground mt-1">
              Graph-theoretic metrics that identify the "glue" nodes holding the
              supply chain together — the highest-priority targets for resilience
              investment.
            </p>
          </div>
          <Badge variant="outline" className="text-xs shrink-0 mt-1">
            {summary.num_nodes} nodes · {summary.num_edges} edges
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* ── Summary stat cards ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {statCards.map((s, i) => (
            <div
              key={i}
              className={`rounded-lg border p-3 ${s.bg} flex flex-col gap-1`}
            >
              <div className="flex items-center gap-2">
                {s.icon}
                <span className="text-xs font-medium text-gray-600">
                  {s.label}
                </span>
              </div>
              <span className="text-xl font-bold text-gray-900">{s.value}</span>
              <span className="text-xs text-gray-500">{s.sub}</span>
            </div>
          ))}
        </div>

        {/* ── Tab selector ── */}
        <div className="flex gap-1 rounded-lg bg-slate-100 p-1 w-full">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              className={`flex-1 flex items-center justify-center gap-1.5 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                activeTab === t.key
                  ? "bg-white shadow text-gray-900"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {t.icon}
              <span className="hidden sm:inline">{t.label}</span>
            </button>
          ))}
        </div>

        {/* ── Tab description ── */}
        <div className="rounded-lg bg-slate-50 border border-slate-200 px-4 py-3 text-sm text-slate-700 flex items-start gap-2">
          {activeTabConfig.icon}
          <span>{activeTabConfig.description}</span>
        </div>

        {/* ── Betweenness tab ── */}
        {activeTab === "betweenness" && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-700">
              Top 10 Nodes by Betweenness Centrality
            </h3>
            <CentralityBarChart
              data={betweenness_centrality}
              scoreKey="betweenness_centrality"
              label="Betweenness"
            />
            <NodeTable
              nodes={betweenness_centrality}
              scoreKey="betweenness_centrality"
              scoreLabel="Betweenness"
            />
          </div>
        )}

        {/* ── Eigenvector / PageRank tab ── */}
        {activeTab === "eigenvector" && (
          <div className="space-y-4">
            <h3 className="text-sm font-semibold text-gray-700">
              Top 10 Nodes by {summary.centrality_method}
            </h3>
            <CentralityBarChart
              data={eigenvector_centrality}
              scoreKey="eigenvector_score"
              label={summary.centrality_method}
            />
            <NodeTable
              nodes={eigenvector_centrality}
              scoreKey="eigenvector_score"
              scoreLabel={summary.centrality_method}
            />
          </div>
        )}

        {/* ── Articulation Points tab ── */}
        {activeTab === "articulation" && (
          <div className="space-y-4">
            {articulation_points.length === 0 ? (
              <div className="rounded-lg bg-green-50 border border-green-200 px-4 py-6 text-center text-green-700">
                <p className="font-semibold">No articulation points found 🎉</p>
                <p className="text-sm mt-1">
                  The network has no single point of failure — every node can be
                  removed without disconnecting the graph.
                </p>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-gray-700">
                    {articulation_points.length} Cut{" "}
                    {articulation_points.length === 1 ? "Vertex" : "Vertices"}{" "}
                    Detected
                  </h3>
                  <Badge className="bg-red-100 text-red-800 text-xs">
                    High Priority
                  </Badge>
                </div>

                {/* Mini bar chart for articulation points (by betweenness) */}
                {articulation_points.length > 0 && (
                  <CentralityBarChart
                    data={articulation_points.slice(0, 10)}
                    scoreKey="betweenness_centrality"
                    label="Betweenness"
                  />
                )}

                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Node</TableHead>
                      <TableHead>Region</TableHead>
                      <TableHead>Tier</TableHead>
                      <TableHead className="text-right">Degree</TableHead>
                      <TableHead className="text-right">Betweenness</TableHead>
                      <TableHead className="text-right">
                        {summary.centrality_method}
                      </TableHead>
                      <TableHead className="text-right">Risk</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {articulation_points.map((n) => (
                      <TableRow
                        key={n.node_id}
                        className="bg-red-50/40 hover:bg-red-50"
                      >
                        <TableCell className="font-medium">
                          <span className="flex items-center gap-1">
                            <Scissors className="h-3 w-3 text-red-500 shrink-0" />
                            {n.name}
                          </span>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {n.region}
                        </TableCell>
                        <TableCell>
                          <Badge
                            className={
                              TIER_COLORS[n.tier] ?? "bg-gray-100 text-gray-800"
                            }
                          >
                            {TIER_LABELS[n.tier] ?? `Tier ${n.tier}`}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">{n.degree}</TableCell>
                        <TableCell className="text-right font-mono text-sm">
                          {((n.betweenness_centrality ?? 0) * 100).toFixed(3)}%
                        </TableCell>
                        <TableCell className="text-right font-mono text-sm">
                          {((n.eigenvector_score ?? 0) * 100).toFixed(3)}%
                        </TableCell>
                        <TableCell className="text-right">
                          <span className={riskColor(n.risk_level)}>
                            {(n.risk_level * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                <p className="text-xs text-muted-foreground">
                  ⚠️ Each row above is a single point of failure. Prioritise
                  adding redundant paths or backup suppliers for these nodes.
                </p>
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
