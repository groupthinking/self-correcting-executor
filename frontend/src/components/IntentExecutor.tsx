import React, { useState } from 'react';
import { Terminal, Send, Loader, CheckCircle, XCircle, Code, Sparkles } from 'lucide-react';

interface ExecutionResult {
  workflow_id: string;
  status: 'completed' | 'failed';
  steps_completed: Array<{
    step: string;
    status: string;
    output?: any;
    error?: string;
  }>;
  outputs?: any;
}

const IntentExecutor: React.FC = () => {
  const [intent, setIntent] = useState('');
  const [sources, setSources] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExecutionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const exampleIntents = [
    { text: "generate CRUD API for products", icon: "⚡" },
    { text: "analyze system performance", icon: "📊" },
    { text: "create authentication endpoints", icon: "🔐" },
    { text: "optimize database queries", icon: "🚀" },
  ];

  const handleExecute = async () => {
    if (!intent.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8080/api/v2/intent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          intent,
          sources,
          options: {}
        }),
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        setResult(data.result);
      } else {
        setError(data.message || 'Execution failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection error');
    } finally {
      setLoading(false);
    }
  };

  const renderGeneratedCode = (output: any) => {
    if (output?.generated_code) {
      return (
        <div className="mt-4">
          <h4 className="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
            <Code className="w-4 h-4" />
            Generated Code
          </h4>
          <pre className="bg-gray-900 border border-gray-700 rounded p-4 overflow-x-auto">
            <code className="text-sm text-gray-300 font-mono">
              {output.generated_code}
            </code>
          </pre>
          {output.instructions && (
            <p className="mt-2 text-sm text-blue-400 italic">
              💡 {output.instructions}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
        <Terminal className="w-8 h-8 text-blue-400" />
        AI Command Center
        <Sparkles className="w-5 h-5 text-yellow-400" />
      </h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Natural Language Command
          </label>
          <div className="relative">
            <input
              type="text"
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleExecute()}
              actual="Tell the AI what to build or analyze..."
              className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600 rounded-lg text-white actual-gray-400 focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-400/20"
              disabled={loading}
            />
            <button
              onClick={handleExecute}
              disabled={loading || !intent.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 disabled:from-gray-600 disabled:to-gray-600 rounded-lg transition-all duration-200 transform hover:scale-105"
            >
              {loading ? (
                <Loader className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-2 gap-2">
          {exampleIntents.map((example, index) => (
            <button
              key={index}
              onClick={() => setIntent(example.text)}
              className="p-2 text-left text-sm bg-gray-800/30 hover:bg-gray-800/50 border border-gray-700 hover:border-gray-600 rounded transition-all duration-200 flex items-center gap-2"
            >
              <span>{example.icon}</span>
              <span>{example.text}</span>
            </button>
          ))}
        </div>

        {error && (
          <div className="bg-red-900/20 border border-red-500 rounded-lg p-4 flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-red-300">{error}</div>
          </div>
        )}

        {result && (
          <div className="space-y-4">
            <div className="bg-green-900/20 border border-green-500 rounded-lg p-4">
              <h3 className="font-semibold text-green-300 mb-2 flex items-center gap-2">
                <CheckCircle className="w-5 h-5" />
                Execution Complete
              </h3>
              <div className="space-y-2 text-sm">
                <p>
                  <span className="text-gray-400">Workflow:</span>{' '}
                  <span className="font-mono text-green-300">{result.workflow_id}</span>
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <h3 className="font-semibold text-gray-300">AI Processing Steps:</h3>
              {result.steps_completed.map((step, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border ${
                    step.status === 'success'
                      ? 'bg-gray-800/50 border-gray-600'
                      : 'bg-red-900/20 border-red-500'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-200 capitalize">
                      {step.step.replace('_', ' ')}
                    </span>
                    {step.status === 'success' ? (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-400" />
                    )}
                  </div>
                  
                  {/* Show generated code if present */}
                  {step.output && renderGeneratedCode(step.output)}
                  
                  {/* Show other output data */}
                  {step.output && !step.output.generated_code && (
                    <pre className="text-xs text-gray-400 bg-gray-900 rounded p-2 overflow-x-auto">
                      {JSON.stringify(step.output, null, 2)}
                    </pre>
                  )}
                  
                  {step.error && (
                    <p className="text-sm text-red-300 mt-2">{step.error}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default IntentExecutor; 