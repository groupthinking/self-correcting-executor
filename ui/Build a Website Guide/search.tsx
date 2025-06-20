'use client'
import * as React from "react"
import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Search, X } from "lucide-react"

interface SearchResult {
  id: number
  title: string
  description: string
  subject: string
  tags: string[]
  relevance: number // 0-100
}

export interface SearchProps {
  isOpen: boolean
  onClose: () => void
}

// Mock data for search results
const mockSearchResults = (query: string): SearchResult[] => {
  if (!query.trim()) return []
  
  const results: SearchResult[] = [
    {
      id: 1,
      title: 'AI-Powered Personal Assistant for Elderly Care',
      description: 'A smart home system that uses AI to monitor elderly individuals, predict potential health issues, and provide companionship.',
      subject: 'Technology',
      tags: ['AI', 'Healthcare', 'IoT'],
      relevance: 95
    },
    {
      id: 2,
      title: 'Decentralized Knowledge Management System',
      description: 'A blockchain-based platform for organizations to store, share, and verify knowledge assets across departments.',
      subject: 'Technology',
      tags: ['Blockchain', 'Enterprise', 'Knowledge Management'],
      relevance: 88
    },
    {
      id: 3,
      title: 'Subscription-Based Sustainable Fashion Platform',
      description: 'A platform that offers sustainable clothing on a subscription basis, reducing waste and promoting ethical fashion.',
      subject: 'Business',
      tags: ['Sustainability', 'Fashion', 'Subscription Economy'],
      relevance: 75
    },
    {
      id: 4,
      title: 'Neural Interface for Creative Collaboration',
      description: 'A brain-computer interface that allows multiple users to collaborate on creative projects by sharing neural patterns.',
      subject: 'Technology',
      tags: ['Neurotechnology', 'Creativity', 'Collaboration'],
      relevance: 70
    },
    {
      id: 5,
      title: 'Micro-Investment Platform for Local Businesses',
      description: 'A platform enabling community members to make small investments in local businesses, fostering economic development.',
      subject: 'Business',
      tags: ['Fintech', 'Community Development', 'Small Business'],
      relevance: 65
    }
  ]
  
  // Filter based on query (in a real app, this would be done by the backend)
  return results.filter(result => 
    result.title.toLowerCase().includes(query.toLowerCase()) || 
    result.description.toLowerCase().includes(query.toLowerCase()) ||
    result.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
  )
}

export function Search({ isOpen, onClose }: SearchProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  
  useEffect(() => {
    if (query.trim()) {
      setIsSearching(true)
      
      // Simulate API call delay
      const timer = setTimeout(() => {
        setResults(mockSearchResults(query))
        setIsSearching(false)
      }, 500)
      
      return () => clearTimeout(timer)
    } else {
      setResults([])
    }
  }, [query])
  
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20">
      <div className="fixed inset-0 bg-black/80 backdrop-blur-sm" onClick={onClose} />
      
      <Card className="relative w-full max-w-2xl max-h-[80vh] flex flex-col rounded-lg border border-gray-800 bg-gray-900 shadow-xl">
        {/* Search header */}
        <div className="flex items-center border-b border-gray-800 p-4">
          <Search className="h-5 w-5 text-gray-400 mr-2" />
          <Input
            placeholder="Search ideas, tags, or subjects..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 border-0 focus-visible:ring-0 focus-visible:ring-offset-0"
            autoFocus
          />
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        
        {/* Search results */}
        <div className="flex-1 overflow-y-auto p-4">
          {isSearching ? (
            <div className="flex justify-center items-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            </div>
          ) : results.length > 0 ? (
            <div className="space-y-4">
              {results.map((result) => (
                <div 
                  key={result.id}
                  className="p-4 rounded-lg border border-gray-800 hover:bg-gray-800/50 transition-colors cursor-pointer"
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium">{result.title}</h3>
                    <span className="text-xs px-2 py-1 rounded-full bg-blue-900/50 text-blue-300">
                      {result.subject}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 mb-3">{result.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {result.tags.map((tag, index) => (
                      <span 
                        key={index}
                        className="px-2 py-1 bg-gray-800 rounded-full text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : query.trim() ? (
            <div className="text-center py-8 text-gray-400">
              No results found for "{query}"
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              Start typing to search for ideas
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}
