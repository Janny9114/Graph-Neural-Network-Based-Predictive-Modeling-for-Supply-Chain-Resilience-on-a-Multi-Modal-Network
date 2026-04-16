import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { ScrollArea } from "./ui/scroll-area";
import { Badge } from "./ui/badge";
import { Bot, Send, Sparkles } from "lucide-react";
import { useState } from "react";

interface Message {
  id: string;
  text: string;
  sender: "user" | "ai";
  timestamp: Date;
}

const initialMessages: Message[] = [
  {
    id: "1",
    text: "Hello! I'm your AI Supply Chain Assistant. I can help you with risk analysis, scenario planning, and mitigation strategies. How can I assist you today?",
    sender: "ai",
    timestamp: new Date(),
  },
];

const suggestedQuestions = [
  "What's the biggest risk to our supply chain?",
  "How can we mitigate port congestion?",
  "Analyze supplier S004 performance",
  "What if China closes borders?",
];

export function AIChatbot() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = (text?: string) => {
    const messageText = text || inputValue;
    if (!messageText.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      text: messageText,
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = getAIResponse(messageText);
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: aiResponse,
        sender: "ai",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
      setIsTyping(false);
    }, 1500);
  };

  const getAIResponse = (question: string): string => {
    const lowerQ = question.toLowerCase();
    
    if (lowerQ.includes("biggest risk") || lowerQ.includes("main risk")) {
      return "Based on current analysis, port congestion in Asia poses the highest risk with a 65% probability. This affects 5 key suppliers and could delay shipments by 7-14 days. I recommend activating our multi-modal transportation plan and increasing safety stock for critical components.";
    }
    
    if (lowerQ.includes("port congestion") || lowerQ.includes("mitigate")) {
      return "To mitigate port congestion risks, I suggest: 1) Diversify shipping routes using alternative ports, 2) Establish air freight contracts for critical components, 3) Increase safety stock by 30%, and 4) Develop relationships with freight forwarders in multiple regions. These actions could reduce impact by 40%.";
    }
    
    if (lowerQ.includes("s004") || lowerQ.includes("supplier")) {
      return "Supplier S004 (Reliable Materials) currently has a HIGH risk rating. Key concerns: 78% on-time delivery (below 95% target), located in a geopolitically sensitive region, and single-source for 3 critical components. Recommendation: Qualify 2 alternative suppliers within 60 days and reduce dependency to 50%.";
    }
    
    if (lowerQ.includes("china") || lowerQ.includes("border")) {
      return "Scenario Analysis: If China closes borders, we'd face CRITICAL impact affecting 3 major suppliers (S002, S006, S008) representing 35% of our supply base. Estimated disruption: 21-45 days. Immediate actions: Activate European backup suppliers, expedite inventory build for affected components, and prepare to shift 40% of production to EU facility.";
    }
    
    return "I've analyzed your question. Based on our current data and AI models, I recommend reviewing the risk scenario dashboard for detailed insights. Would you like me to run a specific simulation or provide mitigation recommendations for a particular scenario?";
  };

  const handleSuggestionClick = (question: string) => {
    handleSend(question);
  };

  return (
    <Card className="col-span-4">
      <CardHeader className="bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-t-lg">
        <CardTitle className="flex items-center gap-2">
          <div className="relative">
            <Bot className="h-6 w-6" />
            <Sparkles className="h-3 w-3 absolute -top-1 -right-1 text-yellow-300" />
          </div>
          AI Supply Chain Assistant
          <Badge className="ml-2 bg-white/20 text-white hover:bg-white/30">Beta</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ScrollArea className="h-96 p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.sender === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-slate-100 text-slate-900"
                  }`}
                >
                  {message.sender === "ai" && (
                    <div className="flex items-center gap-2 mb-1">
                      <Bot className="h-4 w-4 text-blue-600" />
                      <span className="text-xs font-semibold text-blue-600">AI Assistant</span>
                    </div>
                  )}
                  <p className="text-sm">{message.text}</p>
                  <p className={`text-xs mt-1 ${message.sender === "user" ? "text-blue-100" : "text-slate-500"}`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="flex justify-start">
                <div className="bg-slate-100 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    <Bot className="h-4 w-4 text-blue-600" />
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></span>
                      <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0.1s" }}></span>
                      <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0.2s" }}></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </ScrollArea>

        {messages.length === 1 && (
          <div className="px-4 pb-4">
            <p className="text-xs text-muted-foreground mb-2">Suggested questions:</p>
            <div className="flex flex-wrap gap-2">
              {suggestedQuestions.map((question, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => handleSuggestionClick(question)}
                  className="text-xs"
                >
                  {question}
                </Button>
              ))}
            </div>
          </div>
        )}

        <div className="p-4 border-t bg-slate-50">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask about risks, scenarios, or mitigation strategies..."
              className="flex-1"
            />
            <Button onClick={() => handleSend()} disabled={!inputValue.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
