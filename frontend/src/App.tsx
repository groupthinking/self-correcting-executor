import { useState } from 'react'
import { AISDKIntegration, AdvancedAISDKExample } from './ai-sdk-integration'
import './App.css'

type View = 'dashboard' | 'ai-sdk' | 'advanced-ai' | 'monitoring'

function App() {
  const [currentView, setCurrentView] = useState<View>('dashboard')

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <div style={{ padding: '2rem', color: 'white' }}>
            <h1>AI SDK 5 Beta Dashboard</h1>
            <p>Welcome to your comprehensive AI development platform!</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem', marginTop: '2rem' }}>
              <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem', borderRadius: '8px' }}>
                <h3>🤖 AI SDK 5 Beta</h3>
                <p>Experience the latest AI SDK with LanguageModelV2 architecture</p>
                <button onClick={() => setCurrentView('ai-sdk')} style={{ padding: '0.5rem 1rem', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                  Try AI SDK
                </button>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem', borderRadius: '8px' }}>
                <h3>⚡ Advanced Features</h3>
                <p>Tool calling, quantum computing, and MCP integration</p>
                <button onClick={() => setCurrentView('advanced-ai')} style={{ padding: '0.5rem 1rem', background: '#8b5cf6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                  Advanced AI
                </button>
              </div>
              <div style={{ background: 'rgba(255,255,255,0.1)', padding: '1rem', borderRadius: '8px' }}>
                <h3>📊 Monitoring</h3>
                <p>Real-time system metrics and performance monitoring</p>
                <button onClick={() => setCurrentView('monitoring')} style={{ padding: '0.5rem 1rem', background: '#10b981', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>
                  View Metrics
                </button>
              </div>
            </div>
          </div>
        )
      case 'ai-sdk':
        return <AISDKIntegration />
      case 'advanced-ai':
        return <AdvancedAISDKExample />
      case 'monitoring':
        return (
          <div style={{ padding: '2rem', color: 'white' }}>
            <h1>📊 Monitoring Dashboard</h1>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginTop: '2rem' }}>
              <div style={{ background: 'rgba(34, 197, 94, 0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                <h3 style={{ color: '#22c55e' }}>✅ System Health</h3>
                <p>Status: <span style={{ color: '#22c55e', fontWeight: 'bold' }}>Healthy</span></p>
                <p>CPU: 23% | Memory: 67% | Disk: 45%</p>
              </div>
              <div style={{ background: 'rgba(59, 130, 246, 0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.3)' }}>
                <h3 style={{ color: '#3b82f6' }}>🤖 AI Performance</h3>
                <p>Active Conversations: 12</p>
                <p>Avg Response Time: 1.2s</p>
                <p>Success Rate: 99.2%</p>
              </div>
              <div style={{ background: 'rgba(139, 92, 246, 0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                <h3 style={{ color: '#8b5cf6' }}>⚡ Quantum Tools</h3>
                <p>Quantum Operations: 47</p>
                <p>MCP Connections: 3</p>
                <p>Tool Success Rate: 94.8%</p>
              </div>
              <div style={{ background: 'rgba(245, 158, 11, 0.2)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(245, 158, 11, 0.3)' }}>
                <h3 style={{ color: '#f59e0b' }}>🔒 Security</h3>
                <p>Threats Blocked: 15</p>
                <p>Auth Success: 99.8%</p>
                <p>Compliance Score: 97%</p>
              </div>
            </div>
          </div>
        )
      default:
        return (
          <div style={{ padding: '2rem', color: 'white' }}>
            <h1>AI SDK 5 Beta Platform</h1>
            <p>Loading...</p>
          </div>
        )
    }
  }

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", sans-serif'
    }}>
      {/* Simple Navigation */}
      <nav style={{ 
        background: 'rgba(0,0,0,0.2)', 
        padding: '1rem', 
        display: 'flex', 
        gap: '1rem',
        borderBottom: '1px solid rgba(255,255,255,0.1)'
      }}>
        <button 
          onClick={() => setCurrentView('dashboard')}
          style={{ 
            padding: '0.5rem 1rem', 
            background: currentView === 'dashboard' ? 'rgba(255,255,255,0.3)' : 'transparent',
            color: 'white', 
            border: '1px solid rgba(255,255,255,0.3)', 
            borderRadius: '4px', 
            cursor: 'pointer' 
          }}
        >
          🏠 Dashboard
        </button>
        <button 
          onClick={() => setCurrentView('ai-sdk')}
          style={{ 
            padding: '0.5rem 1rem', 
            background: currentView === 'ai-sdk' ? 'rgba(255,255,255,0.3)' : 'transparent',
            color: 'white', 
            border: '1px solid rgba(255,255,255,0.3)', 
            borderRadius: '4px', 
            cursor: 'pointer' 
          }}
        >
          🤖 AI SDK 5
        </button>
        <button 
          onClick={() => setCurrentView('advanced-ai')}
          style={{ 
            padding: '0.5rem 1rem', 
            background: currentView === 'advanced-ai' ? 'rgba(255,255,255,0.3)' : 'transparent',
            color: 'white', 
            border: '1px solid rgba(255,255,255,0.3)', 
            borderRadius: '4px', 
            cursor: 'pointer' 
          }}
        >
          ⚡ Advanced AI
        </button>
        <button 
          onClick={() => setCurrentView('monitoring')}
          style={{ 
            padding: '0.5rem 1rem', 
            background: currentView === 'monitoring' ? 'rgba(255,255,255,0.3)' : 'transparent',
            color: 'white', 
            border: '1px solid rgba(255,255,255,0.3)', 
            borderRadius: '4px', 
            cursor: 'pointer' 
          }}
        >
          📊 Monitoring
        </button>
      </nav>

      {/* Main Content */}
      <main style={{ minHeight: 'calc(100vh - 80px)' }}>
        {renderView()}
      </main>
    </div>
  )
}

export default App