import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";

interface Scenario {
  id: string;
  name: string;
  description: string;
  probability: number;
  impact: "Low" | "Medium" | "High" | "Critical";
  mitigationScore: number;
}

const scenarios: Scenario[] = [
  {
    id: "1",
    name: "Port Congestion",
    description: "Major shipping delays due to port congestion in Asia",
    probability: 65,
    impact: "High",
    mitigationScore: 72
  },
  {
    id: "2",
    name: "Supplier Bankruptcy",
    description: "Key supplier faces financial difficulties",
    probability: 25,
    impact: "Critical",
    mitigationScore: 85
  },
  {
    id: "3",
    name: "Natural Disaster",
    description: "Typhoon or earthquake affects production facilities",
    probability: 40,
    impact: "High",
    mitigationScore: 68
  },
  {
    id: "4",
    name: "Cyber Attack",
    description: "Supply chain systems targeted by ransomware",
    probability: 30,
    impact: "Medium",
    mitigationScore: 78
  },
  {
    id: "5",
    name: "Trade Restrictions",
    description: "New tariffs or export controls implemented",
    probability: 55,
    impact: "Medium",
    mitigationScore: 80
  },
];

export function ScenarioAnalysis() {
  const getImpactColor = (impact: string) => {
    switch (impact) {
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
        <CardTitle>Risk Scenario Analysis</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {scenarios.map((scenario) => (
            <div key={scenario.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <h4 className="font-semibold">{scenario.name}</h4>
                    <Badge className={getImpactColor(scenario.impact)}>
                      {scenario.impact} Impact
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{scenario.description}</p>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mt-2">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">Probability</span>
                    <span className="font-medium">{scenario.probability}%</span>
                  </div>
                  <Progress value={scenario.probability} className="h-2" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-muted-foreground">Mitigation Readiness</span>
                    <span className="font-medium">{scenario.mitigationScore}%</span>
                  </div>
                  <Progress value={scenario.mitigationScore} className="h-2" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
