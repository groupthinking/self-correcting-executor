import { ComponentType, SVGProps } from 'react';

declare module 'lucide-react' {
  export interface IconProps extends SVGProps<SVGSVGElement> {
    color?: string;
    size?: string | number;
  }

  export type Icon = ComponentType<IconProps>;
  
  export const Brain: Icon;
  export const Sparkles: Icon;
  export const Network: Icon;
  export const Cpu: Icon;
  export const BarChart3: Icon;
  export const Zap: Icon;
  export const Terminal: Icon;
  export const Send: Icon;
  export const Loader: Icon;
  export const CheckCircle: Icon;
  export const XCircle: Icon;
  export const Code: Icon;
  export const LayoutDashboard: Icon;
  export const Folder: Icon;
  export const Settings: Icon;
  export const Bot: Icon;
}