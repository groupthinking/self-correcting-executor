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
import './App.css'

const queryClient = new QueryClient()

type View = 'dashboard' | 'intent' | 'components' | 'patterns'

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
      default:
        return <Dashboard />
    }
  }

  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <BackgroundAnimation />
        <Navigation />
        <main className="main-content">
          <Dashboard />
        </main>
      </div>
    </QueryClientProvider>
  )
}

export default App
