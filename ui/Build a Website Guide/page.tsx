'use client'

import { useState } from 'react'
import Link from 'next/link'

const subjects = [
  { 
    id: 'technology', 
    name: 'Technology', 
    icon: '💻', 
    count: 24,
    description: 'Explore cutting-edge tech innovations and ideas',
    color: 'bg-blue-500'
  },
  { 
    id: 'business', 
    name: 'Business', 
    icon: '📊', 
    count: 18,
    description: 'Business models, strategies, and entrepreneurship',
    color: 'bg-purple-500'
  },
  { 
    id: 'science', 
    name: 'Science', 
    icon: '🔬', 
    count: 15,
    description: 'Scientific research, discoveries, and theories',
    color: 'bg-green-500'
  },
  { 
    id: 'arts', 
    name: 'Arts', 
    icon: '🎨', 
    count: 12,
    description: 'Creative expressions across various mediums',
    color: 'bg-pink-500'
  },
  { 
    id: 'health', 
    name: 'Health', 
    icon: '🏥', 
    count: 20,
    description: 'Healthcare innovations and wellness concepts',
    color: 'bg-red-500'
  },
  { 
    id: 'finance', 
    name: 'Finance', 
    icon: '💰', 
    count: 16,
    description: 'Financial systems, investments, and economics',
    color: 'bg-yellow-500'
  },
  { 
    id: 'education', 
    name: 'Education', 
    icon: '📚', 
    count: 14,
    description: 'Learning methodologies and educational tools',
    color: 'bg-indigo-500'
  },
  { 
    id: 'environment', 
    name: 'Environment', 
    icon: '🌱', 
    count: 17,
    description: 'Sustainability solutions and ecological concepts',
    color: 'bg-teal-500'
  },
  { 
    id: 'sales', 
    name: 'Sales', 
    icon: '🤝', 
    count: 13,
    description: 'Sales strategies, tactics, and playbooks',
    color: 'bg-orange-500'
  },
  { 
    id: 'marketing', 
    name: 'Marketing', 
    icon: '📣', 
    count: 19,
    description: 'Marketing strategies and campaign concepts',
    color: 'bg-cyan-500'
  },
  { 
    id: 'leadership', 
    name: 'Leadership', 
    icon: '👑', 
    count: 11,
    description: 'Leadership principles and management strategies',
    color: 'bg-lime-500'
  },
  { 
    id: 'legal', 
    name: 'Legal', 
    icon: '⚖️', 
    count: 9,
    description: 'Legal frameworks and compliance strategies',
    color: 'bg-gray-500'
  }
]

const categories = [
  { id: 'all', name: 'All Subjects' },
  { id: 'tech-science', name: 'Tech & Science' },
  { id: 'business-finance', name: 'Business & Finance' },
  { id: 'creative-education', name: 'Creative & Education' },
  { id: 'health-environment', name: 'Health & Environment' },
  { id: 'leadership-legal', name: 'Leadership & Legal' }
]

const agents = [
  {
    title: 'Collaboration Agent',
    description: 'Connect with team members and share ideas seamlessly',
    icon: '👥',
    color: 'bg-blue-500'
  },
  {
    title: 'Research Agent',
    description: 'Automatically gather information and insights for your ideas',
    icon: '🔍',
    color: 'bg-purple-500'
  },
  {
    title: 'Development Agent',
    description: 'Turn your ideas into prototypes and working solutions',
    icon: '⚙️',
    color: 'bg-green-500'
  }
]

const recentIdeas = [
  {
    id: 1,
    title: 'AI-Powered Personal Assistant for Elderly Care',
    subject: 'Technology',
    date: '2025-04-15',
    excerpt: 'A smart home system that uses AI to monitor elderly individuals, predict potential health issues, and alert caregivers.',
    tags: ['AIHealthcareIoT']
  },
  {
    id: 2,
    title: 'Decentralized Knowledge Management System',
    subject: 'Technology',
    date: '2025-04-12',
    excerpt: 'A blockchain-based platform for organizations to store, share, and verify knowledge assets across departments.',
    tags: ['BlockchainEnterpriseKnowledge Management']
  },
  {
    id: 3,
    title: 'Neural Interface for Creative Collaboration',
    subject: 'Technology',
    date: '2025-04-08',
    excerpt: 'A brain-computer interface that allows multiple creatives to collaborate in a shared mental workspace.',
    tags: ['NeurotechCreativityCollaboration']
  }
]

export default function HomePage() {
  const [activeCategory, setActiveCategory] = useState('all')
  const [searchQuery, setSearchQuery] = useState('')
  
  // Filter subjects based on active category
  const getFilteredSubjects = () => {
    if (activeCategory === 'all') return subjects
    
    const categoryMap = {
      'tech-science': ['technology', 'science'],
      'business-finance': ['business', 'finance', 'sales', 'marketing'],
      'creative-education': ['arts', 'education'],
      'health-environment': ['health', 'environment'],
      'leadership-legal': ['leadership', 'legal']
    }
    
    return subjects.filter(subject => 
      categoryMap[activeCategory as keyof typeof categoryMap]?.includes(subject.id)
    )
  }
  
  const filteredSubjects = getFilteredSubjects()
  
  return (
    <div className="container mx-auto px-4 py-8">
      <section className="mb-12">
        <div className="max-w-3xl mx-auto text-center mb-10">
          <h1 className="text-4xl font-bold mb-4 text-gray-900">Portfolio of Ideas</h1>
          <p className="text-xl text-gray-600">
            Organize, consolidate, and discover actionable insights from your ideas across various subjects
          </p>
          
          {/* Search Bar */}
          <div className="mt-8 relative max-w-2xl mx-auto">
            <input
              type="text"
              actual="Search ideas or ask AI for insights..."
              className="w-full px-6 py-3 pr-12 border border-gray-300 rounded-full shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <button className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-blue-600 text-white px-4 py-1 rounded-full text-sm">
              Ask AI
            </button>
          </div>
        </div>
        
        {/* Call to Action */}
        <div className="max-w-md mx-auto mb-12">
          <Link 
            href="/insights" 
            className="flex items-center justify-center w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl shadow-lg hover:shadow-xl transition-all"
          >
            <svg className="h-6 w-6 mr-3" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M12 16v-4"></path>
              <path d="M12 8h.01"></path>
            </svg>
            Generate Insights for Your Ideas
          </Link>
        </div>
        
        {/* Category Filters */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setActiveCategory(category.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                activeCategory === category.id
                  ? 'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {category.name}
            </button>
          ))}
          
          <button className="px-4 py-2 rounded-full text-sm font-medium bg-gray-100 text-gray-700 hover:bg-gray-200 flex items-center">
            <svg className="h-4 w-4 mr-1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19"></line>
              <line x1="5" y1="12" x2="19" y2="12"></line>
            </svg>
            Subjects
          </button>
        </div>
        
        {/* Subject Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredSubjects.map((subject) => (
            <Link href={`/${subject.id}`} key={subject.id} className="subject-tile">
              <div className={`h-2 ${subject.color} rounded-t-xl`}></div>
              <div className="subject-tile-header">
                <div className="subject-tile-icon">{subject.icon}</div>
                <h3 className="subject-tile-title">{subject.name}</h3>
                <p className="text-sm text-gray-600 mb-2">{subject.description}</p>
                <span className="subject-tile-count">{subject.count} ideas</span>
              </div>
              <div className="subject-tile-footer">
                <span className="text-sm text-blue-600 font-medium">View ideas →</span>
              </div>
            </Link>
          ))}
        </div>
      </section>
      
      {/* Personal Agents Section */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-6 text-gray-900">Personal Agents</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {agents.map((agent, index) => (
            <div key={index} className="apple-card p-6">
              <div className={`w-12 h-12 rounded-full ${agent.color} flex items-center justify-center text-white text-2xl mb-4`}>
                {agent.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2">{agent.title}</h3>
              <p className="text-gray-600">{agent.description}</p>
              <button className="mt-4 px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium hover:bg-gray-50">
                Deploy Agent
              </button>
            </div>
          ))}
        </div>
      </section>
      
      {/* Recent Ideas Section */}
      <section>
        <h2 className="text-2xl font-bold mb-6 text-gray-900">Recent Ideas</h2>
        <div className="space-y-6">
          {recentIdeas.map((idea) => (
            <div key={idea.id} className="apple-card p-6">
              <div className="flex justify-between items-start mb-2">
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                  {idea.subject}
                </span>
                <span className="text-sm text-gray-500">{idea.date}</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">{idea.title}</h3>
              <p className="text-gray-600 mb-4">{idea.excerpt}</p>
              <div className="flex justify-between items-center">
                <div className="flex space-x-2">
                  {idea.tags.map((tag, index) => (
                    <span key={index} className="text-xs text-gray-500">#{tag}</span>
                  ))}
                </div>
                <button className="text-blue-600 text-sm font-medium">
                  View Details →
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
