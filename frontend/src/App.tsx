import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Dashboard from './components/Dashboard'
import Navigation from './components/Navigation'
import BackgroundAnimation from './components/BackgroundAnimation'
import './App.css'

const queryClient = new QueryClient()

function App() {
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
