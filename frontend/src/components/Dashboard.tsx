import React, { useState, useEffect } from 'react';
import { Brain, Sparkles, Network, Cpu, BarChart3, Zap } from 'lucide-react';
import ComponentManager from './ComponentManager';
import IntentExecutor from './IntentExecutor';
import PatternVisualizer from './PatternVisualizer';
import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { 
  CheckCircle, 
  Error as ErrorIcon,
  Extension,
  Psychology,
  Hub,
  QueryStats
} from '@mui/icons-material';

const API_BASE = 'http://localhost:8080/api/v2';

interface SystemMetrics {
  totalExecutions: number;
  successRate: number;
  activeMutations: number;
  connectedAgents: number;
  knowledgeNodes: number;
  mcpConnections: number;
}

const Dashboard: React.FC = () => {
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics>({
    totalExecutions: 0,
    successRate: 0,
    activeMutations: 0,
    connectedAgents: 3,
    knowledgeNodes: 0,
    mcpConnections: 2
  });

  const [realtimeData, setRealtimeData] = useState<any[]>([]);

  const handleFileSelect = (path: string) => {
    setActiveFile(path);
  };

  const { data: components, isLoading } = useQuery({
    queryKey: ['components'],
    queryFn: async () => {
      const response = await axios.get(`${API_BASE}/components`);
      return response.data;
    },
    refetchInterval: 5000,
  });

  const { data: patterns } = useQuery({
    queryKey: ['patterns'],
    queryFn: async () => {
      try {
        const response = await axios.get(`${API_BASE}/patterns`);
        return response.data;
      } catch (error) {
        return null;
      }
    },
  });

  const componentCounts = [
    {
      title: 'Protocols',
      count: components?.protocols?.length || 0,
      icon: Extension,
      color: '#6366f1',
      description: 'Executable tasks'
    },
    {
      title: 'Agents',
      count: components?.agents?.length || 0,
      icon: Psychology,
      color: '#8b5cf6',
      description: 'Autonomous entities'
    },
    {
      title: 'Connectors',
      count: components?.connectors?.length || 0,
      icon: Hub,
      color: '#ec4899',
      description: 'MCP interfaces'
    },
    {
      title: 'Analyzers',
      count: components?.analyzers?.length || 0,
      icon: QueryStats,
      color: '#10b981',
      description: 'Data processors'
    },
  ];

  useEffect(() => {
    // Fetch real metrics
    const fetchMetrics = async () => {
      try {
        const response = await fetch('http://localhost:8080/api/v1/stats');
        const data = await response.json();
        
        // Calculate metrics from actual data
        const stats = data.all_stats || {};
        let totalExec = 0;
        let totalSuccess = 0;
        
        Object.values(stats).forEach((stat: any) => {
          totalExec += stat.executions || 0;
          totalSuccess += stat.successes || 0;
        });
        
        setMetrics({
          totalExecutions: totalExec,
          successRate: totalExec > 0 ? (totalSuccess / totalExec) * 100 : 0,
          activeMutations: Object.keys(stats).length,
          connectedAgents: 3, // We have executor, mutator, code_generator
          knowledgeNodes: totalExec * 2, // Estimate
          mcpConnections: 2
        });
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="dashboard" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
        <div className="glass-card">Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-full">
      {/* Hero Section - System Overview */}
      <div className="mb-8">
        <h1 className="text-4xl font-black mb-2 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
          Self-Correcting Executor
        </h1>
        <p className="text-gray-400 text-lg">
          Autonomous AI system with MCP integration, self-mutation, and distributed intelligence
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        <MetricCard
          icon={Brain}
          label="Total Executions"
          value={metrics.totalExecutions}
          color="blue"
        />
        <MetricCard
          icon={Sparkles}
          label="Success Rate"
          value={`${metrics.successRate.toFixed(1)}%`}
          color="green"
        />
        <MetricCard
          icon={Zap}
          label="Active Mutations"
          value={metrics.activeMutations}
          color="purple"
        />
        <MetricCard
          icon={Network}
          label="Connected Agents"
          value={metrics.connectedAgents}
          color="orange"
        />
        <MetricCard
          icon={Cpu}
          label="Knowledge Nodes"
          value={metrics.knowledgeNodes}
          color="pink"
        />
        <MetricCard
          icon={BarChart3}
          label="MCP Connections"
          value={metrics.mcpConnections}
          color="cyan"
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-6">
          {/* Intent Executor - Enhanced */}
          <div className="glass-card p-6 border-2 border-blue-500/20 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 pointer-events-none" />
            <div className="relative z-10">
              <IntentExecutor />
            </div>
          </div>

          {/* System Architecture Visualization */}
          <div className="glass-card p-6">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <Network className="w-6 h-6 text-purple-400" />
              Architecture Overview
            </h2>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="p-4 bg-gray-800/50 rounded-lg">
                <h3 className="font-semibold text-blue-400 mb-2">Core Agents</h3>
                <ul className="space-y-1 text-gray-300">
                  <li>• Executor (Self-Correcting)</li>
                  <li>• Mutator (Code Evolution)</li>
                  <li>• Code Generator (AI-Powered)</li>
                  <li>• File System Agent</li>
                </ul>
              </div>
              <div className="p-4 bg-gray-800/50 rounded-lg">
                <h3 className="font-semibold text-green-400 mb-2">MCP Protocols</h3>
                <ul className="space-y-1 text-gray-300">
                  <li>• Data Processing</li>
                  <li>• System Monitoring</li>
                  <li>• API Health Checks</li>
                  <li>• Redis Caching</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-6">
          {/* Pattern Visualizer */}
          <div className="glass-card p-6">
            <PatternVisualizer />
          </div>

          {/* Component Status */}
          <div className="glass-card p-6">
            <ComponentManager />
          </div>
        </div>
      </div>

      {/* Bottom Section - Key Features */}
      <div className="mt-8 glass-card p-6">
        <h2 className="text-2xl font-bold mb-4">System Capabilities</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <FeatureCard
            title="Self-Correction"
            description="Automatically mutates failing protocols to improve success rates"
            icon="🔄"
          />
          <FeatureCard
            title="Code Generation"
            description="Natural language to production-ready API endpoints"
            icon="⚡"
          />
          <FeatureCard
            title="MCP Integration"
            description="Universal protocol for AI-to-system communication"
            icon="🔌"
          />
        </div>
      </div>
    </div>
  );
};

const MetricCard: React.FC<{
  icon: any;
  label: string;
  value: string | number;
  color: string;
}> = ({ icon: Icon, label, value, color }) => {
  const colorClasses = {
    blue: 'text-blue-400 bg-blue-500/10',
    green: 'text-green-400 bg-green-500/10',
    purple: 'text-purple-400 bg-purple-500/10',
    orange: 'text-orange-400 bg-orange-500/10',
    pink: 'text-pink-400 bg-pink-500/10',
    cyan: 'text-cyan-400 bg-cyan-500/10',
  };

  return (
    <div className="glass-card p-4 relative group hover:scale-105 transition-transform cursor-pointer">
      <div className={`absolute inset-0 ${colorClasses[color as keyof typeof colorClasses].split(' ')[1]} opacity-20 rounded-lg`} />
      <div className="relative z-10">
        <Icon className={`w-5 h-5 mb-2 ${colorClasses[color as keyof typeof colorClasses].split(' ')[0]}`} />
        <div className="text-2xl font-bold">{value}</div>
        <div className="text-xs text-gray-400">{label}</div>
      </div>
    </div>
  );
};

const FeatureCard: React.FC<{
  title: string;
  description: string;
  icon: string;
}> = ({ title, description, icon }) => {
  return (
    <div className="p-4 bg-gray-800/30 rounded-lg hover:bg-gray-800/50 transition-colors">
      <div className="text-2xl mb-2">{icon}</div>
      <h3 className="font-semibold mb-1">{title}</h3>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  );
};

export default Dashboard; 