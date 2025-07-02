import { Code, PlusCircle } from 'lucide-react'

// This file contains the resources data from work-bench.com playbooks
// organized by subject categories

// Define the resource type
export interface Resource {
  id: string;
  title: string;
  description: string;
  category: string;
  url?: string;
  fileUrl?: string;
}

// Define the subject data type
export interface SubjectData {
  subject: {
    id: string;
    title: string;
    description: string;
    color: string;
    icon: string;
    overview: string;
    ideas: Array<{
      id: number;
      title: string;
      description: string;
      tags: string[];
      date: string;
      connections: string[];
      insights: string;
    }>;
    resources: Array<{
      title: string;
      url: string;
    }>;
  };
  resources: Resource[];
}

// Real data for resources based on work-bench.com playbooks
const resourcesData: Record<string, Resource[]> = {
  sales: [
    {
      id: 'sales-1',
      title: 'How to Master Outbound Sales without a BDR',
      description: 'A comprehensive guide for founders to handle outbound sales before hiring dedicated BDRs.',
      category: 'Early-Day and Founder-Led',
      url: 'https://www.work-bench.com/playbooks/sales/early-day-and-founder-led/how-to-master-outbound-sales-without-a-bdr-yet'
    },
    {
      id: 'sales-2',
      title: 'Onboarding & Outbound Execution',
      description: 'Step-by-step guide for building a BDR organization and revving up outbound go-to-market engine.',
      category: 'Early-Day and Founder-Led',
      url: 'https://www.work-bench.com/playbooks/sales/early-day-and-founder-led/onboarding-outbound-execution'
    },
    {
      id: 'sales-3',
      title: 'Hiring Strategy for your First Sales Hire',
      description: 'Define a successful hiring strategy beyond founder-led sales with best practices for compensation, sourcing, and candidate assessments.',
      category: 'Hiring',
      url: 'https://www.work-bench.com/playbooks/sales/hiring/hiring-strategy-for-your-first-sales-hire'
    },
    {
      id: 'sales-4',
      title: 'How to price your product at seed',
      description: 'Proven frameworks and tested methods for approaching strategic product pricing with real-world examples.',
      category: 'General Tactics',
      url: 'https://www.work-bench.com/playbooks/sales/general-tactics/how-to-price-your-product-at-seed'
    },
    {
      id: 'sales-5',
      title: 'Technical Founders Selling Technical Products',
      description: 'Guide for technical founders to avoid over-indexing on product details and focus on business impact and value.',
      category: 'General Tactics',
      url: 'https://www.work-bench.com/playbooks/sales/general-tactics/technical-founders-selling-technical-products'
    }
  ],
  marketing: [
    {
      id: 'marketing-1',
      title: 'Startup Marketing from $0 to $100M in ARR',
      description: 'Comprehensive marketing strategy guide for startups at different growth stages.',
      category: 'General',
      url: 'https://www.work-bench.com/playbooks/marketing/general/startup-marketing-from-0-to-100m-in-arr-with-snyk'
    },
    {
      id: 'marketing-2',
      title: 'Business Value Calculator',
      description: 'Tool to help sellers demonstrate ROI of their product versus home-grown tools by creating a business case.',
      category: 'General',
      url: 'https://www.work-bench.com/playbooks/marketing/general/business-value-calculator'
    },
    {
      id: 'marketing-3',
      title: 'Newsletter 101: Writing Content People Want to Read',
      description: 'Guide to creating engaging newsletter content that resonates with your audience.',
      category: 'Content Creation',
      url: 'https://www.work-bench.com/playbooks/marketing/content-creation/newsletter-101-writing-content-people-want-to-read'
    },
    {
      id: 'marketing-4',
      title: '10 quick wins to grow pipeline',
      description: 'Practical strategies for quickly growing your sales pipeline and accelerating growth.',
      category: 'Demand Generation',
      url: 'https://www.work-bench.com/playbooks/marketing/demand-generation/10-quick-wins-to-grow-pipeline'
    }
  ],
  'finance-metrics': [
    {
      id: 'finance-1',
      title: 'Enterprise Sales Pipeline Review',
      description: 'Template providing qualitative and quantitative data points to paint a complete picture of sales traction.',
      category: 'Finance & Metrics',
      url: 'https://www.work-bench.com/playbooks/finance-metrics/enterprise-sales-pipeline-review'
    },
    {
      id: 'finance-2',
      title: 'How to build a value-based calculator',
      description: 'Create a structured process to calculate ROI and pricing for confident sales team articulation of value.',
      category: 'Finance & Metrics',
      url: 'https://www.work-bench.com/playbooks/finance-metrics/how-to-build-a-value-based-calculator'
    }
  ],
  fundraising: [
    {
      id: 'fundraising-1',
      title: 'Series A Fundraising Deck Template',
      description: 'Comprehensive template for creating an effective Series A fundraising presentation.',
      category: 'Fundraising',
      url: 'https://www.work-bench.com/playbooks/fundraising/series-a-fundraising-deck-template'
    },
    {
      id: 'fundraising-2',
      title: 'Seed Fundraising Template: ICP Definition & GTM Sequencing',
      description: 'Template to help founders build a narrative around go-to-market vision for Seed fundraising.',
      category: 'Fundraising',
      url: 'https://www.work-bench.com/playbooks/fundraising/seed-fundraising-template-icp-definition-gtm-sequencing'
    }
  ],
  'developer-relations': [
    {
      id: 'devrel-1',
      title: 'Community Strategies to Grow Open-Source Businesses',
      description: 'Three proven community strategies to grow an open-source business by cultivating developer ecosystems.',
      category: 'Developer Relations',
      url: 'https://www.work-bench.com/playbooks/developer-relations/community-strategies-to-grow-open-source-businesses'
    },
    {
      id: 'devrel-2',
      title: 'Experiments To Find a Winning Developer Acquisition Strategy',
      description: 'Practical experiments and approaches to identify effective developer acquisition strategies.',
      category: 'Developer Relations',
      url: 'https://www.work-bench.com/playbooks/developer-relations/experiments-to-find-a-winning-developer-acquisition-strategy'
    }
  ],
  legal: [
    {
      id: 'legal-1',
      title: 'Legal Contract Negotiation & Review Tactics',
      description: 'Best practices for enterprise startup leaders to navigate legal challenges when closing deals.',
      category: 'Legal & Contracts',
      url: 'https://www.work-bench.com/playbooks/legal-contracts/legal-contract-negotiation-review-tactics'
    },
    {
      id: 'legal-2',
      title: 'SOC 2 Compliance',
      description: 'Best practices to help enterprise startup leaders conquer SOC 2 compliance requirements.',
      category: 'Legal & Contracts',
      url: 'https://www.work-bench.com/playbooks/legal-contracts/soc-2-compliance'
    },
    {
      id: 'legal-3',
      title: 'How to Structure a POC & POC Agreement Template',
      description: 'Step-by-step framework and template for structuring Proof of Concept agreements.',
      category: 'Legal & Contracts',
      url: 'https://www.work-bench.com/playbooks/legal-contracts/how-to-structure-a-poc-poc-agreement-template'
    }
  ],
  leadership: [
    {
      id: 'leadership-1',
      title: 'Founder <> VC User Manual',
      description: 'Q&A document designed to help teams learn and appreciate individual quirks to accelerate collaboration.',
      category: 'Leadership',
      url: 'https://www.work-bench.com/playbooks/leadership/founder-vc-user-manual'
    },
    {
      id: 'leadership-2',
      title: 'Tactics for a Great Distributed Culture',
      description: 'Strategies and approaches for building and maintaining an effective remote work culture.',
      category: 'Leadership',
      url: 'https://www.work-bench.com/playbooks/leadership/tactics-for-a-great-distributed-culture'
    }
  ]
};

// Default subject data for fallback
const defaultSubjectData = {
  id: 'subject-not-found',
  title: 'Subject Not Found',
  description: '',
  color: 'from-gray-500 to-gray-400',
  icon: '❓',
  overview: 'This subject does not exist.',
  ideas: [],
  resources: []
};

// Real data for subjects
const subjectsData = {
  technology: {
    id: 'technology',
    title: 'Technology',
    description: 'Explore cutting-edge tech innovations and ideas',
    color: 'bg-tech-gradient',
    icon: '💻',
    overview: 'Technology is rapidly evolving, creating new opportunities for innovation across industries. This section explores emerging technologies, software concepts, hardware innovations, and digital transformation ideas.',
    ideas: [
      { 
        id: 1, 
        title: 'AI-Powered Personal Assistant for Elderly Care', 
        description: 'A smart home system that uses AI to monitor elderly individuals, predict potential health issues, and provide companionship.',
        tags: ['AI', 'Healthcare', 'IoT'],
        date: '2025-04-15',
        connections: ['Health', 'Business'],
        insights: 'Market analysis shows growing demand for elderly care solutions as global population ages. Integration with healthcare systems could provide real-time monitoring and emergency response.'
      },
      { 
        id: 2, 
        title: 'Decentralized Knowledge Management System', 
        description: 'A blockchain-based platform for organizations to store, share, and verify knowledge assets across departments.',
        tags: ['Blockchain', 'Enterprise', 'Knowledge Management'],
        date: '2025-04-12',
        connections: ['Business', 'Education'],
        insights: 'Organizations struggle with knowledge silos. This system could reduce redundancy and improve collaboration while maintaining security and attribution.'
      },
      { 
        id: 3, 
        title: 'Neural Interface for Creative Collaboration', 
        description: 'A brain-computer interface that allows multiple users to collaborate on creative projects by sharing neural patterns.',
        tags: ['Neurotechnology', 'Creativity', 'Collaboration'],
        date: '2025-04-08',
        connections: ['Arts', 'Health'],
        insights: 'Early research in neural interfaces shows promise for direct brain-to-brain communication. Creative applications could revolutionize collaborative art, music, and design.'
      },
    ],
    resources: [
      { title: 'AI Research Papers Database', url: 'https://example.com/ai-research' },
      { title: 'Technology Innovation Grants', url: 'https://example.com/tech-grants' },
      { title: 'Hardware Prototyping Community', url: 'https://example.com/hardware-community' },
    ]
  },
  business: {
    id: 'business',
    title: 'Business',
    description: 'Business models, startups, and entrepreneurship concepts',
    color: 'bg-business-gradient',
    icon: '💼',
    overview: 'Business innovation drives economic growth and creates new market opportunities. This section explores novel business models, startup concepts, market strategies, and entrepreneurial ideas.',
    ideas: [
      { 
        id: 1, 
        title: 'Subscription-Based Sustainable Fashion Platform', 
        description: 'A platform that offers sustainable clothing on a subscription basis, reducing waste and promoting ethical fashion.',
        tags: ['Sustainability', 'Fashion', 'Subscription Economy'],
        date: '2025-04-14',
        connections: ['Environment', 'Technology'],
        insights: 'Consumer interest in sustainable fashion is growing rapidly. A subscription model could reduce clothing waste while providing consistent revenue.'
      },
      { 
        id: 2, 
        title: 'Micro-Investment Platform for Local Businesses', 
        description: 'A platform enabling community members to make small investments in local businesses, fostering economic development.',
        tags: ['Fintech', 'Community Development', 'Small Business'],
        date: '2025-04-10',
        connections: ['Finance', 'Technology'],
        insights: 'Local businesses often struggle to secure traditional financing. This platform could democratize investment while strengthening community ties.'
      },
    ],
    resources: [
      { title: 'Business Model Canvas Templates', url: 'https://example.com/business-canvas' },
      { title: 'Startup Funding Directory', url: 'https://example.com/startup-funding' },
    ]
  },
  sales: {
    id: 'sales',
    title: 'Sales',
    description: 'Sales strategies, tactics, and playbooks',
    color: 'bg-blue-gradient',
    icon: '📊',
    overview: 'Effective sales strategies are crucial for business growth and customer acquisition. This section explores various sales methodologies, outbound techniques, hiring strategies, and customer engagement approaches.',
    ideas: [
      { 
        id: 1, 
        title: 'AI-Powered Sales Conversation Analyzer', 
        description: 'A tool that uses AI to analyze sales calls and provide real-time coaching and insights to improve conversion rates.',
        tags: ['AI', 'Sales Enablement', 'Coaching'],
        date: '2025-04-13',
        connections: ['Technology', 'Business'],
        insights: 'Sales teams often struggle with consistent messaging and objection handling. Real-time AI analysis could significantly improve performance and reduce ramp-up time for new hires.'
      },
      { 
        id: 2, 
        title: 'Value-Based Pricing Calculator', 
        description: 'An interactive tool that helps sales teams quantify and communicate the value of their solutions to prospects.',
        tags: ['Pricing', 'ROI', 'Sales Tools'],
        date: '2025-04-09',
        connections: ['Finance', 'Marketing'],
        insights: 'Value-based pricing can increase deal sizes by 15-30% compared to cost-plus or competitive pricing models. This tool could help sales teams justify premium pricing.'
      },
    ],
    resources: [
      { title: 'Enterprise Sales Frameworks', url: 'https://example.com/sales-frameworks' },
      { title: 'Objection Handling Scripts', url: 'https://example.com/objection-handling' },
    ]
  },
  marketing: {
    id: 'marketing',
    title: 'Marketing',
    description: 'Content creation, demand generation, and PR',
    color: 'bg-purple-gradient',
    icon: '📣',
    overview: 'Effective marketing strategies help businesses reach their target audience and communicate their value proposition. This section explores content creation, demand generation, public relations, and marketing analytics.',
    ideas: [
      { 
        id: 1, 
        title: 'Interactive Content Experience Platform', 
        description: 'A platform for creating interactive content experiences that engage prospects and capture valuable data.',
        tags: ['Content Marketing', 'Interactive', 'Lead Generation'],
        date: '2025-04-11',
        connections: ['Technology', 'Sales'],
        insights: 'Interactive content generates 2x more conversions than passive content. This platform could help marketers create engaging experiences without technical expertise.'
      },
      { 
        id: 2, 
        title: 'AI-Powered Content Personalization Engine', 
        description: 'An engine that uses AI to dynamically personalize website content based on visitor behavior and attributes.',
        tags: ['Personalization', 'AI', 'Conversion Optimization'],
        date: '2025-04-07',
        connections: ['Technology', 'Business'],
        insights: 'Personalized content can increase engagement by 70% and conversion rates by 20%. This engine could help marketers scale personalization efforts efficiently.'
      },
    ],
    resources: [
      { title: 'B2B Content Marketing Playbook', url: 'https://example.com/b2b-content' },
      { title: 'Demand G
(Content truncated due to size limit. Use line ranges to read in chunks)