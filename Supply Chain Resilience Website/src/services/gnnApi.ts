const API_BASE_URL = 'http://localhost:5000/api';

export interface DisruptionScenario {
  disrupted_nodes: number[];
  disrupted_edges: number[][];
  disruption_severity: number;
  buffer_capacity?: number;
}

export interface Prediction {
  node_id: number;
  node_name: string;
  label: number;
  label_name: 'Failed' | 'Degraded' | 'Normal';
  probability: number[];
  tier: number;
  region: string;
}

export interface PredictionResponse {
  predictions: Prediction[];
  summary: {
    failed: number;
    degraded: number;
    normal: number;
    total_nodes: number;
  };
  status: string;
}

export interface GraphData {
  nodes: Array<{
    id: number;
    name: string;
    tier: number;
    region: string;
    capacity: number;
    x: number;
    y: number;
  }>;
  edges: Array<{
    source: number;
    target: number;
    weight: number;
  }>;
}

export interface PipelineRequest {
  num_scenarios?: number;
  use_default_data?: boolean;
  company_id?: string;
}

export interface PipelineResponse {
  status: string;
  task_id: string;
  message: string;
}

export interface PipelineStatus {
  status: string;
  progress: number;
  message: string;
  stage: string;
  started_at?: string;
  updated_at?: string;
  completed_at?: string;
  results?: any[];
  error?: string;
  traceback?: string;
}

export const gnnApi = {
  async predict(scenario: DisruptionScenario): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(scenario),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async getGraph(): Promise<GraphData> {
    const response = await fetch(`${API_BASE_URL}/graph`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },
  
  async healthCheck(): Promise<{ status: string; model: string; device: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  async runCompletePipeline(request: PipelineRequest): Promise<PipelineResponse> {
    const response = await fetch(`${API_BASE_URL}/run-complete-pipeline`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },

  async getPipelineStatus(taskId: string): Promise<PipelineStatus> {
    const response = await fetch(`${API_BASE_URL}/pipeline-status/${taskId}`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },

  async getTrainingResults(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/training-results`);
    
    if (!response.ok) {
      throw new Error(`API error: ${response.statusText}`);
    }
    
    return response.json();
  },
};
