import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  BarChart3,
  Brain,
  Clock,
  Cpu,
  Database,
  Globe,
  HardDrive,
  Lock,
  MemoryStick,
  MessageSquare,
  Monitor,
  Network,
  PieChart,
  Server,
  Shield,
  TrendingUp,
  Users,
  Zap,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Wifi,
  RefreshCw
} from 'lucide-react';

// Types for monitoring data
interface SystemMetrics {
  cpu: {
    usage: number;
    cores: number;
    temperature: number;
    load: number[];
  };
  memory: {
    used: number;
    total: number;
    available: number;
    swapUsage: number;
  };
  disk: {
    usage: number;
    readSpeed: number;
    writeSpeed: number;
    totalSpace: number;
  };
  network: {
    inbound: number;
    outbound: number;
    latency: number;
    packetsPerSecond: number;
  };
}

interface AIMetrics {
  conversations: {
    active: number;
    total: number;
    avgDuration: number;
    successRate: number;
  };
  models: {
    gpt4o: { usage: number; responseTime: number; };
    gpt4oMini: { usage: number; responseTime: number; };
    gpt35Turbo: { usage: number; responseTime: number; };
  };
  tools: {
    quantum: { calls: number; successRate: number; avgTime: number; };
    mcp: { calls: number; successRate: number; avgTime: number; };
    security: { calls: number; successRate: number; avgTime: number; };
    analytics: { calls: number; successRate: number; avgTime: number; };
  };
  tokens: {
    input: number;
    output: number;
    total: number;
    cost: number;
  };
}

interface SecurityMetrics {
  threats: {
    blocked: number;
    attempted: number;
    lastThreat: string;
  };
  authentication: {
    successful: number;
    failed: number;
    rate: number;
  };
  compliance: {
    score: number;
    issues: number;
    lastAudit: string;
  };
}

interface PerformanceMetrics {
  responseTime: {
    p50: number;
    p95: number;
    p99: number;
  };
  availability: number;
  errorRate: number;
  throughput: number;
}

interface AlertItem {
  id: string;
  type: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: Date;
  component: string;
  resolved: boolean;
}

/**
 * Comprehensive Monitoring and Analytics Dashboard
 * Real-time system monitoring with AI SDK 5 Beta integration
 */
export const MonitoringDashboard: React.FC = () => {
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [aiMetrics, setAIMetrics] = useState<AIMetrics | null>(null);
  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetrics | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [isLive, setIsLive] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [activeTab, setActiveTab] = useState<'overview' | 'system' | 'ai' | 'security' | 'performance'>('overview');

  // Fetch monitoring data
  const fetchMetrics = useCallback(async () => {
    try {
      // Simulate API calls - replace with actual API endpoints
      const systemData = await fetchSystemMetrics();
      const aiData = await fetchAIMetrics();
      const securityData = await fetchSecurityMetrics();
      const performanceData = await fetchPerformanceMetrics();
      const alertsData = await fetchAlerts();

      setSystemMetrics(systemData);
      setAIMetrics(aiData);
      setSecurityMetrics(securityData);
      setPerformanceMetrics(performanceData);
      setAlerts(alertsData);
    } catch (error) {
      console.error('Failed to fetch monitoring metrics:', error);
    }
  }, [selectedTimeframe]);

  // Auto-refresh functionality
  useEffect(() => {
    fetchMetrics();
    
    if (isLive) {
      const interval = setInterval(fetchMetrics, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchMetrics, isLive, refreshInterval]);

  // Mock data generators (replace with actual API calls)
  const fetchSystemMetrics = async (): Promise<SystemMetrics> => {
    return {
      cpu: {
        usage: Math.random() * 80 + 10,
        cores: 8,
        temperature: Math.random() * 20 + 45,
        load: [Math.random(), Math.random(), Math.random()]
      },
      memory: {
        used: Math.random() * 70 + 20,
        total: 16,
        available: Math.random() * 50 + 30,
        swapUsage: Math.random() * 10
      },
      disk: {
        usage: Math.random() * 60 + 30,
        readSpeed: Math.random() * 100 + 50,
        writeSpeed: Math.random() * 80 + 40,
        totalSpace: 1000
      },
      network: {
        inbound: Math.random() * 100 + 10,
        outbound: Math.random() * 50 + 5,
        latency: Math.random() * 20 + 5,
        packetsPerSecond: Math.random() * 1000 + 500
      }
    };
  };

  const fetchAIMetrics = async (): Promise<AIMetrics> => {
    return {
      conversations: {
        active: Math.floor(Math.random() * 50 + 10),
        total: Math.floor(Math.random() * 1000 + 500),
        avgDuration: Math.random() * 300 + 120,
        successRate: 0.95 + Math.random() * 0.04
      },
      models: {
        gpt4o: { usage: Math.random() * 60 + 20, responseTime: Math.random() * 1000 + 500 },
        gpt4oMini: { usage: Math.random() * 40 + 10, responseTime: Math.random() * 500 + 200 },
        gpt35Turbo: { usage: Math.random() * 30 + 5, responseTime: Math.random() * 300 + 150 }
      },
      tools: {
        quantum: { calls: Math.floor(Math.random() * 100 + 20), successRate: 0.9 + Math.random() * 0.08, avgTime: Math.random() * 2000 + 1000 },
        mcp: { calls: Math.floor(Math.random() * 200 + 50), successRate: 0.95 + Math.random() * 0.04, avgTime: Math.random() * 500 + 200 },
        security: { calls: Math.floor(Math.random() * 80 + 10), successRate: 0.98 + Math.random() * 0.01, avgTime: Math.random() * 1000 + 300 },
        analytics: { calls: Math.floor(Math.random() * 150 + 30), successRate: 0.92 + Math.random() * 0.06, avgTime: Math.random() * 800 + 400 }
      },
      tokens: {
        input: Math.floor(Math.random() * 100000 + 50000),
        output: Math.floor(Math.random() * 80000 + 40000),
        total: Math.floor(Math.random() * 180000 + 90000),
        cost: Math.random() * 50 + 20
      }
    };
  };

  const fetchSecurityMetrics = async (): Promise<SecurityMetrics> => {
    return {
      threats: {
        blocked: Math.floor(Math.random() * 20 + 5),
        attempted: Math.floor(Math.random() * 50 + 10),
        lastThreat: new Date(Date.now() - Math.random() * 3600000).toISOString()
      },
      authentication: {
        successful: Math.floor(Math.random() * 1000 + 500),
        failed: Math.floor(Math.random() * 50 + 10),
        rate: 0.95 + Math.random() * 0.04
      },
      compliance: {
        score: 85 + Math.random() * 10,
        issues: Math.floor(Math.random() * 5),
        lastAudit: new Date(Date.now() - Math.random() * 86400000 * 7).toISOString()
      }
    };
  };

  const fetchPerformanceMetrics = async (): Promise<PerformanceMetrics> => {
    return {
      responseTime: {
        p50: Math.random() * 200 + 100,
        p95: Math.random() * 500 + 300,
        p99: Math.random() * 1000 + 800
      },
      availability: 0.995 + Math.random() * 0.004,
      errorRate: Math.random() * 0.05,
      throughput: Math.random() * 1000 + 500
    };
  };

  const fetchAlerts = async (): Promise<AlertItem[]> => {
    const alerts: AlertItem[] = [];
    const alertTypes: AlertItem['type'][] = ['info', 'warning', 'error', 'critical'];
    const components = ['API', 'Database', 'AI Models', 'Security', 'Network', 'Storage'];
    
    for (let i = 0; i < Math.floor(Math.random() * 10 + 2); i++) {
      alerts.push({
        id: `alert_${i}`,
        type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
        message: `Alert from ${components[Math.floor(Math.random() * components.length)]}`,
        timestamp: new Date(Date.now() - Math.random() * 3600000),
        component: components[Math.floor(Math.random() * components.length)],
        resolved: Math.random() > 0.3
      });
    }
    
    return alerts.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  };

  // Utility functions
  const getStatusColor = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return 'text-green-600';
    if (value <= thresholds.warning) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getStatusIcon = (value: number, thresholds: { good: number; warning: number }) => {
    if (value <= thresholds.good) return <CheckCircle className="w-4 h-4 text-green-600" />;
    if (value <= thresholds.warning) return <AlertTriangle className="w-4 h-4 text-yellow-600" />;
    return <XCircle className="w-4 h-4 text-red-600" />;
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  };

  // Component renders
  const renderOverviewTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
      {/* System Health */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">System Health</h3>
          <Monitor className="w-6 h-6 text-blue-600" />
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">CPU</span>
            <div className="flex items-center space-x-2">
              <span className={`text-sm font-medium ${getStatusColor(systemMetrics?.cpu.usage || 0, { good: 70, warning: 85 })}`}>
                {systemMetrics?.cpu.usage.toFixed(1)}%
              </span>
              {getStatusIcon(systemMetrics?.cpu.usage || 0, { good: 70, warning: 85 })}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Memory</span>
            <div className="flex items-center space-x-2">
              <span className={`text-sm font-medium ${getStatusColor(systemMetrics?.memory.used || 0, { good: 70, warning: 85 })}`}>
                {systemMetrics?.memory.used.toFixed(1)}%
              </span>
              {getStatusIcon(systemMetrics?.memory.used || 0, { good: 70, warning: 85 })}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Disk</span>
            <div className="flex items-center space-x-2">
              <span className={`text-sm font-medium ${getStatusColor(systemMetrics?.disk.usage || 0, { good: 80, warning: 90 })}`}>
                {systemMetrics?.disk.usage.toFixed(1)}%
              </span>
              {getStatusIcon(systemMetrics?.disk.usage || 0, { good: 80, warning: 90 })}
            </div>
          </div>
        </div>
      </div>

      {/* AI Performance */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">AI Performance</h3>
          <Brain className="w-6 h-6 text-purple-600" />
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Active Conversations</span>
            <span className="text-sm font-medium text-gray-800">
              {aiMetrics?.conversations.active}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Success Rate</span>
            <span className="text-sm font-medium text-green-600">
              {((aiMetrics?.conversations.successRate || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Avg Response Time</span>
            <span className="text-sm font-medium text-gray-800">
              {aiMetrics?.models.gpt4o.responseTime.toFixed(0)}ms
            </span>
          </div>
        </div>
      </div>

      {/* Security Status */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Security Status</h3>
          <Shield className="w-6 h-6 text-green-600" />
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Threats Blocked</span>
            <span className="text-sm font-medium text-red-600">
              {securityMetrics?.threats.blocked}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Auth Success Rate</span>
            <span className="text-sm font-medium text-green-600">
              {((securityMetrics?.authentication.rate || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Compliance Score</span>
            <span className="text-sm font-medium text-blue-600">
              {securityMetrics?.compliance.score.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Performance</h3>
          <TrendingUp className="w-6 h-6 text-orange-600" />
        </div>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Availability</span>
            <span className="text-sm font-medium text-green-600">
              {((performanceMetrics?.availability || 0) * 100).toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Error Rate</span>
            <span className="text-sm font-medium text-red-600">
              {((performanceMetrics?.errorRate || 0) * 100).toFixed(2)}%
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-600">Throughput</span>
            <span className="text-sm font-medium text-gray-800">
              {performanceMetrics?.throughput.toFixed(0)} req/min
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSystemTab = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* CPU Metrics */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">CPU Metrics</h3>
          <Cpu className="w-6 h-6 text-blue-600" />
        </div>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Usage</span>
              <span className="text-sm font-medium">{systemMetrics?.cpu.usage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.cpu.usage || 0}%` }}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-gray-600">Cores</span>
              <p className="text-lg font-semibold">{systemMetrics?.cpu.cores}</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Temperature</span>
              <p className="text-lg font-semibold">{systemMetrics?.cpu.temperature.toFixed(1)}°C</p>
            </div>
          </div>
        </div>
      </div>

      {/* Memory Metrics */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Memory Metrics</h3>
          <MemoryStick className="w-6 h-6 text-green-600" />
        </div>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Usage</span>
              <span className="text-sm font-medium">{systemMetrics?.memory.used.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-green-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.memory.used || 0}%` }}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-gray-600">Total</span>
              <p className="text-lg font-semibold">{systemMetrics?.memory.total}GB</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Available</span>
              <p className="text-lg font-semibold">{systemMetrics?.memory.available.toFixed(1)}GB</p>
            </div>
          </div>
        </div>
      </div>

      {/* Disk Metrics */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Disk Metrics</h3>
          <HardDrive className="w-6 h-6 text-purple-600" />
        </div>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Usage</span>
              <span className="text-sm font-medium">{systemMetrics?.disk.usage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${systemMetrics?.disk.usage || 0}%` }}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-gray-600">Read Speed</span>
              <p className="text-lg font-semibold">{systemMetrics?.disk.readSpeed.toFixed(1)}MB/s</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Write Speed</span>
              <p className="text-lg font-semibold">{systemMetrics?.disk.writeSpeed.toFixed(1)}MB/s</p>
            </div>
          </div>
        </div>
      </div>

      {/* Network Metrics */}
      <div className="bg-white rounded-lg shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Network Metrics</h3>
          <Network className="w-6 h-6 text-orange-600" />
        </div>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-gray-600">Inbound</span>
              <p className="text-lg font-semibold">{systemMetrics?.network.inbound.toFixed(1)}MB/s</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Outbound</span>
              <p className="text-lg font-semibold">{systemMetrics?.network.outbound.toFixed(1)}MB/s</p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-sm text-gray-600">Latency</span>
              <p className="text-lg font-semibold">{systemMetrics?.network.latency.toFixed(1)}ms</p>
            </div>
            <div>
              <span className="text-sm text-gray-600">Packets/sec</span>
              <p className="text-lg font-semibold">{systemMetrics?.network.packetsPerSecond.toFixed(0)}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAlertsPanel = () => (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Recent Alerts</h3>
      </div>
      <div className="max-h-96 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            <CheckCircle className="w-8 h-8 mx-auto mb-2 text-green-600" />
            <p>No active alerts</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {alerts.map(alert => (
              <div key={alert.id} className="p-4 hover:bg-gray-50">
                <div className="flex items-start space-x-3">
                  <div className={`flex-shrink-0 w-2 h-2 rounded-full mt-2 ${
                    alert.type === 'critical' ? 'bg-red-600' :
                    alert.type === 'error' ? 'bg-red-500' :
                    alert.type === 'warning' ? 'bg-yellow-500' :
                    'bg-blue-500'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium text-gray-800">{alert.message}</p>
                      <span className="text-xs text-gray-500">
                        {alert.timestamp.toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">{alert.component}</p>
                    {alert.resolved && (
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800 mt-2">
                        Resolved
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  return (
    <div className="monitoring-dashboard max-w-7xl mx-auto p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">Monitoring Dashboard</h1>
            <p className="text-gray-600 mt-1">Real-time system and AI performance monitoring</p>
          </div>
          <div className="flex items-center space-x-4">
            {/* Live Toggle */}
            <button
              onClick={() => setIsLive(!isLive)}
              className={`flex items-center px-4 py-2 rounded-lg transition-colors ${
                isLive ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
              }`}
            >
              <Wifi className={`w-4 h-4 mr-2 ${isLive ? 'animate-pulse' : ''}`} />
              {isLive ? 'Live' : 'Paused'}
            </button>

            {/* Refresh Button */}
            <button
              onClick={fetchMetrics}
              className="flex items-center px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
            >
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </button>

            {/* Timeframe Selector */}
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value as any)}
              className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="mt-6 border-b border-gray-200">
          <nav className="flex space-x-8">
            {[
              { id: 'overview', name: 'Overview', icon: BarChart3 },
              { id: 'system', name: 'System', icon: Server },
              { id: 'ai', name: 'AI Metrics', icon: Brain },
              { id: 'security', name: 'Security', icon: Shield },
              { id: 'performance', name: 'Performance', icon: TrendingUp }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4 mr-2" />
                {tab.name}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Main Content */}
        <div className="xl:col-span-3">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {activeTab === 'overview' && renderOverviewTab()}
              {activeTab === 'system' && renderSystemTab()}
              {activeTab === 'ai' && (
                <div className="text-center py-12">
                  <Brain className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">AI metrics implementation coming soon</p>
                </div>
              )}
              {activeTab === 'security' && (
                <div className="text-center py-12">
                  <Shield className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">Security metrics implementation coming soon</p>
                </div>
              )}
              {activeTab === 'performance' && (
                <div className="text-center py-12">
                  <TrendingUp className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                  <p className="text-gray-600">Performance metrics implementation coming soon</p>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Sidebar - Alerts */}
        <div className="xl:col-span-1">
          {renderAlertsPanel()}
        </div>
      </div>
    </div>
  );
};

export default MonitoringDashboard;