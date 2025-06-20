import React from 'react';
import { LayoutDashboard, Terminal, Folder, Settings, Bot } from 'lucide-react';

interface NavigationProps {
  // Props to select active view can be added here
}

const Navigation: React.FC<NavigationProps> = () => {
  const navItems = [
    { icon: Bot, label: 'Logo', isLogo: true },
    { icon: LayoutDashboard, label: 'Dashboard' },
    { icon: Folder, label: 'Files' },
    { icon: Terminal, label: 'Executor' },
    { icon: Settings, label: 'Settings', isBottom: true },
  ];

  return (
    <nav className="h-screen w-20 flex flex-col items-center py-4 bg-[var(--glass-bg)] border-r border-[var(--border-color)]">
      <div className="flex flex-col items-center flex-grow">
        {navItems.filter(item => !item.isBottom).map((item, index) => (
          <div key={index} className="w-full flex justify-center p-4 my-2 cursor-pointer group relative">
            <item.icon
              className={`h-8 w-8 text-gray-400 group-hover:text-white transition-all duration-300 group-hover:scale-110 ${item.isLogo ? 'text-blue-400' : ''}`}
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