import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Label } from "./ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Badge } from "./ui/badge";
import { PlayCircle, RotateCcw } from "lucide-react";
import { useState } from "react";

export function WhatIfSimulation() {
  const [selectedScenario, setSelectedScenario] = useState("port-congestion");
  const [severity, setSeverity] = useState([50]);
  const [duration, setDuration] = useState([7]);
  const [simulated, setSimulated] = useState(false);

  const scenarios = [
    { value: "port-congestion", label: "Port Congestion" },
    { value: "supplier-bankruptcy", label: "Supplier Bankruptcy" },
    { value: "natural-disaster", label: "Natural Disaster" },
    { value: "cyber-attack", label: "Cyber Attack" },
    { value: "trade-restrictions", label: "Trade Restrictions" },
  ];

  const handleSimulate = () => {
    setSimulated(true);
  };

  const handleReset = () => {
    setSimulated(false);
    setSeverity([50]);
    setDuration([7]);
  };

  const getImpactLevel = () => {
    const score = (severity[0] * duration[0]) / 100;
    if (score < 150) return { level: "Low", color: "bg-green-100 text-green-800" };
    if (score < 300) return { level: "Medium", color: "bg-yellow-100 text-yellow-800" };
    if (score < 500) return { level: "High", color: "bg-orange-100 text-orange-800" };
    return { level: "Critical", color: "bg-red-100 text-red-800" };
  };

  const impact = getImpactLevel();

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <PlayCircle className="h-5 w-5 text-blue-600" />
          What-If Simulation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="scenario">Select Scenario</Label>
          <Select value={selectedScenario} onValueChange={setSelectedScenario}>
            <SelectTrigger id="scenario">
              <SelectValue placeholder="Choose a scenario" />
            </SelectTrigger>
            <SelectContent>
              {scenarios.map((scenario) => (
                <SelectItem key={scenario.value} value={scenario.value}>
                  {scenario.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="severity">Severity</Label>
            <span className="text-sm font-medium">{severity[0]}%</span>
          </div>
          <Slider
            id="severity"
            value={severity}
            onValueChange={setSeverity}
            max={100}
            step={5}
            className="w-full"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <Label htmlFor="duration">Duration</Label>
            <span className="text-sm font-medium">{duration[0]} days</span>
          </div>
          <Slider
            id="duration"
            value={duration}
            onValueChange={setDuration}
            max={30}
            step={1}
            className="w-full"
          />
        </div>

        <div className="flex gap-2">
          <Button onClick={handleSimulate} className="flex-1">
            <PlayCircle className="h-4 w-4 mr-2" />
            Run Simulation
          </Button>
          <Button variant="outline" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        {simulated && (
          <div className="mt-6 p-4 bg-slate-50 rounded-lg space-y-3">
            <h4 className="font-semibold text-sm">Simulation Results</h4>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-xs text-muted-foreground">Predicted Impact</p>
                <Badge className={impact.color}>{impact.level}</Badge>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Supply Chain Delay</p>
                <p className="text-sm font-semibold">{Math.floor(duration[0] * 1.5)} days</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Cost Increase</p>
                <p className="text-sm font-semibold">{Math.floor(severity[0] * 0.8)}%</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Recovery Time</p>
                <p className="text-sm font-semibold">{Math.floor(duration[0] * 2)} days</p>
              </div>
            </div>
            <div className="pt-2">
              <p className="text-xs text-muted-foreground">Affected Suppliers</p>
              <p className="text-sm font-semibold">{Math.floor((severity[0] / 100) * 8)} out of 8 suppliers</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
