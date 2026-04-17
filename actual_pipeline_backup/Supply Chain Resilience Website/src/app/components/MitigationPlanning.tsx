import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { CheckCircle2, Circle, Shield, TrendingUp } from "lucide-react";

interface MitigationAction {
  id: string;
  title: string;
  description: string;
  status: "completed" | "in-progress" | "planned";
  priority: "High" | "Medium" | "Low";
  effectiveness: number;
}

const mitigationActions: MitigationAction[] = [
  {
    id: "1",
    title: "Diversify Supplier Base",
    description: "Add 3 alternative suppliers in different regions",
    status: "in-progress",
    priority: "High",
    effectiveness: 85
  },
  {
    id: "2",
    title: "Increase Safety Stock",
    description: "Boost inventory levels for critical components by 30%",
    status: "completed",
    priority: "High",
    effectiveness: 92
  },
  {
    id: "3",
    title: "Multi-Modal Transportation",
    description: "Establish air freight alternatives for urgent shipments",
    status: "in-progress",
    priority: "Medium",
    effectiveness: 78
  },
  {
    id: "4",
    title: "Supplier Development Program",
    description: "Implement quality and reliability improvement initiatives",
    status: "planned",
    priority: "Medium",
    effectiveness: 70
  },
  {
    id: "5",
    title: "Regional Warehousing",
    description: "Set up distribution centers in key geographic locations",
    status: "planned",
    priority: "Low",
    effectiveness: 65
  },
];

export function MitigationPlanning() {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-600" />;
      case "in-progress":
        return <TrendingUp className="h-4 w-4 text-blue-600" />;
      case "planned":
        return <Circle className="h-4 w-4 text-gray-400" />;
      default:
        return <Circle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Badge className="bg-green-100 text-green-800 hover:bg-green-100">Completed</Badge>;
      case "in-progress":
        return <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">In Progress</Badge>;
      case "planned":
        return <Badge className="bg-gray-100 text-gray-800 hover:bg-gray-100">Planned</Badge>;
      default:
        return <Badge>Unknown</Badge>;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "High":
        return "bg-red-100 text-red-800 hover:bg-red-100";
      case "Medium":
        return "bg-yellow-100 text-yellow-800 hover:bg-yellow-100";
      case "Low":
        return "bg-green-100 text-green-800 hover:bg-green-100";
      default:
        return "";
    }
  };

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5 text-green-600" />
          Mitigation Planning
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {mitigationActions.map((action) => (
          <div key={action.id} className="p-4 bg-slate-50 rounded-lg space-y-3">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-3 flex-1">
                {getStatusIcon(action.status)}
                <div className="flex-1">
                  <h4 className="font-semibold text-sm">{action.title}</h4>
                  <p className="text-xs text-muted-foreground mt-1">{action.description}</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Badge className={getPriorityColor(action.priority)}>
                  {action.priority}
                </Badge>
                {getStatusBadge(action.status)}
              </div>
            </div>
            
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">Effectiveness</span>
                <span className="font-medium">{action.effectiveness}%</span>
              </div>
              <Progress value={action.effectiveness} className="h-2" />
            </div>
          </div>
        ))}

        <div className="pt-4 border-t">
          <Button className="w-full" variant="outline">
            <Shield className="h-4 w-4 mr-2" />
            Generate New Mitigation Plan
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
