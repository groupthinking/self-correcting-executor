import React from 'react';
import { LayoutDashboard, Terminal, Folder, Settings, Bot, Brain, Sparkles } from 'lucide-react';

type View = 'dashboard' | 'intent' | 'components' | 'patterns' | 'ai-sdk' | 'advanced-ai';

interface NavigationProps {
  currentView: View;
  setCurrentView: (view: View) => void;
}

const Navigation: React.FC<NavigationProps> = ({ currentView, setCurrentView }) => {
  const navItems = [
    { icon: Bot, label: 'Logo', isLogo: true },
    { icon: LayoutDashboard, label: 'Dashboard', view: 'dashboard' as View },
    { icon: Folder, label: 'Components', view: 'components' as View },
    { icon: Terminal, label: 'Executor', view: 'intent' as View },
    { icon: Brain, label: 'Patterns', view: 'patterns' as View },
    { icon: Sparkles, label: 'AI SDK 5', view: 'ai-sdk' as View },
    { icon: Bot, label: 'Advanced AI', view: 'advanced-ai' as View },
    { icon: Settings, label: 'Settings', isBottom: true },
  ];

  const handleNavClick = (view: View) => {
    setCurrentView(view);
  };

  return (
    <nav className="h-screen w-20 flex flex-col items-center py-4 bg-[var(--glass-bg)] border-r border-[var(--border-color)]">
      <div className="flex flex-col items-center flex-grow">
        {navItems.filter(item => !item.isBottom).map((item, index) => (
          <div 
            key={index} 
            className={`w-full flex justify-center p-4 my-2 cursor-pointer group relative ${
              item.view && currentView === item.view ? 'bg-blue-500/20' : ''
            }`}
            onClick={() => item.view && handleNavClick(item.view)}
          >
            <item.icon
              className={`h-8 w-8 transition-all duration-300 group-hover:scale-110 ${
                item.isLogo 
                  ? 'text-blue-400' 
                  : item.view && currentView === item.view
                    ? 'text-blue-400'
                    : 'text-gray-400 group-hover:text-white'
              }`}
            />
            <span className="absolute left-full ml-4 px-2 py-1 bg-gray-900 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
              {item.label}
            </span>
          </div>
        ))}
      </div>
      <div className="flex flex-col items-center">
        {navItems.filter(item => item.isBottom).map((item, index) => (
           <div key={index} className="w-full flex justify-center p-4 my-2 cursor-pointer group relative">
            <item.icon className="h-8 w-8 text-gray-400 group-hover:text-white transition-all duration-300 group-hover:scale-110" />
            <span className="absolute left-full ml-4 px-2 py-1 bg-gray-900 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </nav>
  );
};

export default Navigation; 