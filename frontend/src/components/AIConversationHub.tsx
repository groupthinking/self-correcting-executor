import React, { useState, useEffect, useRef } from 'react';
import { useChat } from '@ai-sdk/react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, 
  User, 
  Settings, 
  Download, 
  Upload, 
  Zap, 
  Brain, 
  Shield, 
  Activity,
  MessageSquare,
  Code,
  Database,
  Cpu
} from 'lucide-react';
import { toast } from 'react-hot-toast';

interface ConversationMetrics {
  totalMessages: number;
  avgResponseTime: number;
  tokensUsed: number;
  toolCalls: number;
  quantumOperations: number;
  mcpConnections: number;
}

interface ConversationConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  enableTools: boolean;
  enableQuantum: boolean;
  enableMCP: boolean;
  systemPrompt: string;
}

/**
 * Advanced AI Conversation Hub with AI SDK 5 Beta
 * Demonstrates comprehensive frontend integration
 */
export const AIConversationHub: React.FC = () => {
  const [config, setConfig] = useState<ConversationConfig>({
    model: 'gpt-4o',
    temperature: 0.7,
    maxTokens: 2000,
    enableTools: true,
    enableQuantum: true,
    enableMCP: true,
    systemPrompt: 'You are an advanced AI assistant with access to quantum computing and MCP protocols.'
  });

  const [metrics, setMetrics] = useState<ConversationMetrics>({
    totalMessages: 0,
    avgResponseTime: 0,
    tokensUsed: 0,
    toolCalls: 0,
    quantumOperations: 0,
    mcpConnections: 0
  });

  const [conversationId, setConversationId] = useState<string>('');
  const [isRecording, setIsRecording] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const startTimeRef = useRef<number>(0);

  const { 
    messages, 
    input, 
    handleInputChange, 
    handleSubmit, 
    isLoading, 
    stop, 
    reload,
    setMessages 
  } = useChat({
    api: '/api/ai-conversation',
    headers: {
      'Authorization': `Bearer ${process.env.REACT_APP_AI_TOKEN}`,
      'X-Conversation-ID': conversationId,
      'X-Config': JSON.stringify(config),
    },
    onResponse: (response) => {
      const responseTime = Date.now() - startTimeRef.current;
      setMetrics(prev => ({
        ...prev,
        avgResponseTime: (prev.avgResponseTime + responseTime) / 2,
        totalMessages: prev.totalMessages + 1
      }));
    },
    onToolCall: (toolCall) => {
      setMetrics(prev => ({
        ...prev,
        toolCalls: prev.toolCalls + 1,
        quantumOperations: toolCall.toolName.includes('quantum') ? prev.quantumOperations + 1 : prev.quantumOperations,
        mcpConnections: toolCall.toolName.includes('mcp') ? prev.mcpConnections + 1 : prev.mcpConnections
      }));
    },
    onFinish: (message) => {
      setMetrics(prev => ({
        ...prev,
        tokensUsed: prev.tokensUsed + (message.content?.length || 0) / 4 // Rough token estimation
      }));
    },
    onError: (error) => {
      toast.error(`AI Error: ${error.message}`);
    }
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    setConversationId(`conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  }, []);

  const handleAdvancedSubmit = (e: React.FormEvent) => {
    startTimeRef.current = Date.now();
    handleSubmit(e);
  };

  const exportConversation = () => {
    const conversationData = {
      id: conversationId,
      messages,
      config,
      metrics,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(conversationData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${conversationId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Conversation exported successfully!');
  };

  const importConversation = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = JSON.parse(e.target?.result as string);
          setMessages(data.messages);
          setConfig(data.config);
          setMetrics(data.metrics);
          setConversationId(data.id);
          toast.success('Conversation imported successfully!');
        } catch (error) {
          toast.error('Invalid conversation file');
        }
      };
      reader.readAsText(file);
    }
  };

  const getMessageIcon = (role: string) => {
    switch (role) {
      case 'user':
        return <User className="w-5 h-5" />;
      case 'assistant':
        return <Bot className="w-5 h-5" />;
      case 'system':
        return <Settings className="w-5 h-5" />;
      default:
        return <MessageSquare className="w-5 h-5" />;
    }
  };

  const getToolIcon = (toolName: string) => {
    if (toolName.includes('quantum')) return <Zap className="w-4 h-4" />;
    if (toolName.includes('mcp')) return <Activity className="w-4 h-4" />;
    if (toolName.includes('code')) return <Code className="w-4 h-4" />;
    if (toolName.includes('database')) return <Database className="w-4 h-4" />;
    return <Cpu className="w-4 h-4" />;
  };

  return (
    <div className="ai-conversation-hub max-w-7xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Main Conversation Area */}
      <div className="lg:col-span-3">
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 h-[600px] flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-purple-50">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Brain className="w-6 h-6 text-blue-600" />
                <div>
                  <h2 className="text-xl font-bold text-gray-800">AI Conversation Hub</h2>
                  <p className="text-sm text-gray-600">AI SDK 5 Beta • {config.model}</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={exportConversation}
                  className="p-2 rounded-lg bg-blue-100 hover:bg-blue-200 transition-colors"
                  title="Export Conversation"
                >
                  <Download className="w-4 h-4 text-blue-600" />
                </button>
                <label className="p-2 rounded-lg bg-green-100 hover:bg-green-200 transition-colors cursor-pointer">
                  <Upload className="w-4 h-4 text-green-600" />
                  <input
                    type="file"
                    accept=".json"
                    onChange={importConversation}
                    className="hidden"
                  />
                </label>
                <button
                  onClick={() => setIsRecording(!isRecording)}
                  className={`p-2 rounded-lg transition-colors ${
                    isRecording ? 'bg-red-100 hover:bg-red-200' : 'bg-gray-100 hover:bg-gray-200'
                  }`}
                >
                  <Activity className={`w-4 h-4 ${isRecording ? 'text-red-600' : 'text-gray-600'}`} />
                </button>
              </div>
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={message.id || index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : message.role === 'system'
                        ? 'bg-gray-100 text-gray-800'
                        : 'bg-gray-50 text-gray-800 border border-gray-200'
                    }`}
                  >
                    <div className="flex items-center space-x-2 mb-2">
                      {getMessageIcon(message.role)}
                      <span className="font-medium capitalize">{message.role}</span>
                      {message.createdAt && (
                        <span className="text-xs opacity-70">
                          {new Date(message.createdAt).toLocaleTimeString()}
                        </span>
                      )}
                    </div>
                    
                    <div className="message-content">
                      {message.content}
                    </div>

                    {/* Tool Invocations */}
                    {message.toolInvocations && message.toolInvocations.length > 0 && (
                      <div className="mt-3 space-y-2">
                        <div className="text-sm font-medium">Tool Calls:</div>
                        {message.toolInvocations.map((tool, toolIndex) => (
                          <div
                            key={toolIndex}
                            className="bg-black/10 rounded-lg p-3 text-sm"
                          >
                            <div className="flex items-center space-x-2 mb-1">
                              {getToolIcon(tool.toolName)}
                              <span className="font-medium">{tool.toolName}</span>
                            </div>
                            <div className="text-xs opacity-80">
                              Args: {JSON.stringify(tool.args, null, 2)}
                            </div>
                            {tool.result && (
                              <div className="text-xs opacity-80 mt-1">
                                Result: {JSON.stringify(tool.result, null, 2)}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <div className="flex items-center space-x-2">
                    <Bot className="w-5 h-5" />
                    <span className="font-medium">Assistant</span>
                  </div>
                  <div className="mt-2 flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </motion.div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <form onSubmit={handleAdvancedSubmit} className="flex space-x-2">
              <input
                value={input}
                onChange={handleInputChange}
                placeholder="Ask about quantum computing, MCP protocols, or anything else..."
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? 'Thinking...' : 'Send'}
              </button>
              {isLoading && (
                <button
                  type="button"
                  onClick={stop}
                  className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  Stop
                </button>
              )}
            </form>
          </div>
        </div>
      </div>

      {/* Sidebar - Configuration & Metrics */}
      <div className="space-y-6">
        {/* Configuration Panel */}
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Configuration
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-1">Model</label>
              <select
                value={config.model}
                onChange={(e) => setConfig(prev => ({ ...prev, model: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Temperature: {config.temperature}</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={config.temperature}
                onChange={(e) => setConfig(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">Max Tokens</label>
              <input
                type="number"
                min="100"
                max="4000"
                value={config.maxTokens}
                onChange={(e) => setConfig(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.enableTools}
                  onChange={(e) => setConfig(prev => ({ ...prev, enableTools: e.target.checked }))}
                  className="rounded"
                />
                <span className="text-sm">Enable Tools</span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.enableQuantum}
                  onChange={(e) => setConfig(prev => ({ ...prev, enableQuantum: e.target.checked }))}
                  className="rounded"
                />
                <span className="text-sm">Quantum Computing</span>
              </label>
              
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.enableMCP}
                  onChange={(e) => setConfig(prev => ({ ...prev, enableMCP: e.target.checked }))}
                  className="rounded"
                />
                <span className="text-sm">MCP Protocols</span>
              </label>
            </div>

            <div>
              <label className="block text-sm font-medium mb-1">System Prompt</label>
              <textarea
                value={config.systemPrompt}
                onChange={(e) => setConfig(prev => ({ ...prev, systemPrompt: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows={3}
              />
            </div>
          </div>
        </div>

        {/* Metrics Panel */}
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Metrics
          </h3>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Total Messages</span>
              <span className="font-medium">{metrics.totalMessages}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Avg Response Time</span>
              <span className="font-medium">{Math.round(metrics.avgResponseTime)}ms</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Tokens Used</span>
              <span className="font-medium">{metrics.tokensUsed}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Tool Calls</span>
              <span className="font-medium">{metrics.toolCalls}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">Quantum Ops</span>
              <span className="font-medium">{metrics.quantumOperations}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-600">MCP Connections</span>
              <span className="font-medium">{metrics.mcpConnections}</span>
            </div>
          </div>
        </div>

        {/* Status Panel */}
        <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-4">
          <h3 className="text-lg font-bold mb-4 flex items-center">
            <Shield className="w-5 h-5 mr-2" />
            Status
          </h3>
          
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span className="text-sm">AI SDK 5 Beta</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span className="text-sm">Quantum Ready</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span className="text-sm">MCP Connected</span>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full" />
              <span className="text-sm">Security Active</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIConversationHub;