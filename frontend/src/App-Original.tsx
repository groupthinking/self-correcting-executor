import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'
import Dashboard from './components/Dashboard'
import IntentExecutor from './components/IntentExecutor'
import ComponentManager from './components/ComponentManager'
import PatternVisualizer from './components/PatternVisualizer'
import Navigation from './components/Navigation'
import BackgroundAnimation from './components/BackgroundAnimation'
import { AISDKIntegration, AdvancedAISDKExample } from './ai-sdk-integration'
import './App.css'

const queryClient = new QueryClient()

type View = 'dashboard' | 'intent' | 'components' | 'patterns' | 'ai-sdk' | 'advanced-ai'

function App() {
  const [currentView, setCurrentView] = useState<View>('dashboard')

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />
      case 'intent':
        return <IntentExecutor />
      case 'components':
        return <ComponentManager />
      case 'patterns':
        return <PatternVisualizer />
      case 'ai-sdk':
        return <AISDKIntegration />
      case 'advanced-ai':
        return <AdvancedAISDKExample />
      default:
        return <Dashboard />
    }
  }

  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <BackgroundAnimation />
        <Navigation currentView={currentView} setCurrentView={setCurrentView} />
        <main className="main-content">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentView}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderView()}
            </motion.div>
          </AnimatePresence>
        </main>
        <Toaster position="top-right" />
      </div>
    </QueryClientProvider>
  )
}

export default App
