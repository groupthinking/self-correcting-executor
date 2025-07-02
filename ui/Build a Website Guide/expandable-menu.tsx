'use client'

import { useState } from 'react'
import { ChevronDown, ChevronRight, Plus, Edit, Check } from 'lucide-react'

interface Subject {
  id: string
  title: string
  icon: string
}

interface ExpandableMenuProps {
  defaultSubjects: Subject[]
  onSelectSubject: (subject: string) => void
  onAddSubject?: (subject: Subject) => void
  onEditSubject?: (id: string, newTitle: string) => void
  className?: string
}

export function ExpandableMenu({
  defaultSubjects,
  onSelectSubject,
  onAddSubject,
  onEditSubject,
  className = ''
}: ExpandableMenuProps) {
  const [expanded, setExpanded] = useState(false)
  const [subjects, setSubjects] = useState<Subject[]>(defaultSubjects)
  const [newSubject, setNewSubject] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [showAddForm, setShowAddForm] = useState(false)

  const toggleExpanded = () => {
    setExpanded(!expanded)
  }

  const handleSelectSubject = (id: string) => {
    onSelectSubject(id)
    setExpanded(false)
  }

  const handleAddSubject = () => {
    if (newSubject.trim() === '') return
    
    const newSubjectObj = {
      id: newSubject.toLowerCase().replace(/\s+/g, '-'),
      title: newSubject,
      icon: '📄' // Default icon
    }
    
    setSubjects([...subjects, newSubjectObj])
    if (onAddSubject) {
      onAddSubject(newSubjectObj)
    }
    setNewSubject('')
    setShowAddForm(false)
  }

  const startEditing = (id: string, currentTitle: string) => {
    setEditingId(id)
    setEditValue(currentTitle)
  }

  const handleEditSubject = () => {
    if (editingId && editValue.trim() !== '') {
      const updatedSubjects = subjects.map(subject => 
        subject.id === editingId 
          ? { ...subject, title: editValue } 
          : subject
      )
      setSubjects(updatedSubjects)
      
      if (onEditSubject) {
        onEditSubject(editingId, editValue)
      }
      
      setEditingId(null)
      setEditValue('')
    }
  }

  return (
    <div className={`relative ${className}`}>
      <button 
        onClick={toggleExpanded}
        className="flex items-center justify-between w-full px-4 py-2 text-left bg-white rounded-lg border border-gray-200 shadow-sm hover:bg-gray-50 transition-colors"
      >
        <span className="font-medium text-gray-700">Subjects</span>
        {expanded ? (
          <ChevronDown className="h-4 w-4 text-gray-500" />
        ) : (
          <ChevronRight className="h-4 w-4 text-gray-500" />
        )}
      </button>
      
      {expanded && (
        <div className="absolute z-50 mt-2 w-full bg-white rounded-lg border border-gray-200 shadow-lg">
          <div className="max-h-80 overflow-y-auto p-2">
            {subjects.map((subject) => (
              <div key={subject.id} className="flex items-center justify-between group">
                <button
                  onClick={() => handleSelectSubject(subject.id)}
                  className="flex items-center w-full px-3 py-2 text-left text-gray-700 hover:bg-gray-50 rounded-md"
                >
                  <span className="mr-2">{subject.icon}</span>
                  {editingId === subject.id ? (
                    <input
                      type="text"
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      className="flex-grow border border-gray-300 rounded px-2 py-1 text-sm"
                      onClick={(e) => e.stopPropagation()}
                    />
                  ) : (
                    <span>{subject.title}</span>
                  )}
                </button>
                
                {onEditSubject && (
                  <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                    {editingId === subject.id ? (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleEditSubject()
                        }}
                        className="p-1 text-green-500 hover:text-green-600"
                      >
                        <Check className="h-4 w-4" />
                      </button>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          startEditing(subject.id, subject.title)
                        }}
                        className="p-1 text-gray-400 hover:text-gray-600"
                      >
                        <Edit className="h-4 w-4" />
                      </button>
                    )}
                  </div>
                )}
              </div>
            ))}
            
            {onAddSubject && (
              <div className="mt-2 pt-2 border-t border-gray-100">
                {showAddForm ? (
                  <div className="flex items-center px-3 py-2">
                    <input
                      type="text"
                      actual="New subject name..."
                      value={newSubject}
                      onChange={(e) => setNewSubject(e.target.value)}
                      className="flex-grow border border-gray-300 rounded-l px-3 py-1 text-sm"
                    />
                    <button
                      onClick={handleAddSubject}
                      className="bg-blue-500 text-white px-3 py-1 rounded-r text-sm hover:bg-blue-600"
                    >
                      Add
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={() => setShowAddForm(true)}
                    className="flex items-center w-full px-3 py-2 text-left text-blue-500 hover:bg-gray-50 rounded-md"
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    <span>Add new subject</span>
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
