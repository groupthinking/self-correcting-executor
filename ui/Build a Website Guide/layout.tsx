'use client'

import Header from '@/components/header'
import { ReactNode } from 'react'

export default function RootLayout({
  children,
}: {
  children: ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <title>Portfolio of Ideas</title>
        <meta name="description" content="Organize, consolidate, and discover actionable insights from your ideas across various subjects" />
      </head>
      <body className="min-h-screen bg-white">
        <Header />
        <main>
          {children}
        </main>
        <footer className="py-8 text-center text-gray-500 text-sm">
          © 2025 Portfolio of Ideas
        </footer>
      </body>
    </html>
  )
}
