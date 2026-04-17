import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { AlertTriangle, AlertCircle, Info } from "lucide-react";

interface AlertItem {
  id: string;
  type: "warning" | "error" | "info";
  title: string;
  description: string;
  timestamp: string;
}

const alerts: AlertItem[] = [
  {
    id: "1",
    type: "error",
    title: "Critical Delay Detected",
    description: "Supplier S004 has reported a 3-day delay in shipment for critical components.",
    timestamp: "2 hours ago"
  },
  {
    id: "2",
    type: "warning",
    title: "Geopolitical Risk",
    description: "Trade tensions may affect shipments from Southeast Asia region.",
    timestamp: "5 hours ago"
  },
  {
    id: "3",
    type: "info",
    title: "New Alternative Supplier",
    description: "Quality First Ltd. has been approved as a backup supplier for Category A components.",
    timestamp: "1 day ago"
  },
  {
    id: "4",
    type: "warning",
    title: "Inventory Level Low",
    description: "Component XYZ-123 inventory below safety threshold. Reorder recommended.",
    timestamp: "1 day ago"
  },
];

export function AlertsPanel() {
  const getAlertIcon = (type: string) => {
    switch (type) {
      case "error":
        return <AlertCircle className="h-4 w-4" />;
      case "warning":
        return <AlertTriangle className="h-4 w-4" />;
      case "info":
        return <Info className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  const getAlertClass = (type: string) => {
    switch (type) {
      case "error":
        return "border-red-500 bg-red-50";
      case "warning":
        return "border-yellow-500 bg-yellow-50";
      case "info":
        return "border-blue-500 bg-blue-50";
      default:
        return "";
    }
  };

  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Recent Alerts & Notifications</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {alerts.map((alert) => (
          <Alert key={alert.id} className={getAlertClass(alert.type)}>
            {getAlertIcon(alert.type)}
            <AlertTitle className="flex items-center justify-between">
              {alert.title}
              <span className="text-xs text-muted-foreground font-normal">
                {alert.timestamp}
              </span>
            </AlertTitle>
            <AlertDescription>{alert.description}</AlertDescription>
          </Alert>
        ))}
      </CardContent>
    </Card>
  );
}
