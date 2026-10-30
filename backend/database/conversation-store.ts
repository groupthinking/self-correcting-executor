import { z } from 'zod';
import { EventEmitter } from 'events';
import crypto from 'crypto';

// Database schemas
const ConversationSchema = z.object({
  id: z.string(),
  userId: z.string(),
  title: z.string().max(200),
  model: z.string(),
  systemPrompt: z.string().max(2000),
  config: z.object({
    temperature: z.number().min(0).max(1),
    maxTokens: z.number().min(100).max(4000),
    enableTools: z.boolean(),
    enableQuantum: z.boolean(),
    enableMCP: z.boolean()
  }),
  metadata: z.object({
    totalMessages: z.number(),
    totalTokens: z.number(),
    avgResponseTime: z.number(),
    toolCalls: z.number(),
    quantumOperations: z.number(),
    mcpConnections: z.number(),
    tags: z.array(z.string()).optional(),
    isStarred: z.boolean().default(false),
    isArchived: z.boolean().default(false)
  }),
  createdAt: z.date(),
  updatedAt: z.date(),
  lastMessageAt: z.date().optional()
});

const MessageSchema = z.object({
  id: z.string(),
  conversationId: z.string(),
  role: z.enum(['user', 'assistant', 'system', 'tool']),
  content: z.string().max(50000),
  metadata: z.object({
    model: z.string().optional(),
    tokens: z.number().optional(),
    responseTime: z.number().optional(),
    toolCalls: z.array(z.object({
      toolName: z.string(),
      args: z.any(),
      result: z.any(),
      duration: z.number().optional()
    })).optional(),
    parentMessageId: z.string().optional(),
    isEdited: z.boolean().default(false),
    editHistory: z.array(z.object({
      content: z.string(),
      editedAt: z.date()
    })).optional()
  }),
  createdAt: z.date(),
  updatedAt: z.date()
});

const UserPreferencesSchema = z.object({
  userId: z.string(),
  preferences: z.object({
    defaultModel: z.string(),
    defaultTemperature: z.number(),
    enableAutoSave: z.boolean(),
    enableNotifications: z.boolean(),
    theme: z.enum(['light', 'dark', 'auto']),
    language: z.string(),
    conversationRetention: z.number(), // days
    exportFormat: z.enum(['json', 'markdown', 'pdf'])
  }),
  apiUsage: z.object({
    totalMessages: z.number(),
    totalTokens: z.number(),
    monthlyLimit: z.number(),
    currentMonthUsage: z.number(),
    lastResetDate: z.date()
  }),
  createdAt: z.date(),
  updatedAt: z.date()
});

// TypeScript interfaces
type Conversation = z.infer<typeof ConversationSchema>;
type Message = z.infer<typeof MessageSchema>;
type UserPreferences = z.infer<typeof UserPreferencesSchema>;

interface ConversationFilter {
  userId: string;
  search?: string;
  tags?: string[];
  isStarred?: boolean;
  isArchived?: boolean;
  model?: string;
  dateFrom?: Date;
  dateTo?: Date;
  limit?: number;
  offset?: number;
}

interface ConversationStats {
  totalConversations: number;
  totalMessages: number;
  totalTokens: number;
  avgConversationLength: number;
  mostUsedModel: string;
  toolUsageStats: Record<string, number>;
  dailyActivity: Array<{ date: string; count: number }>;
}

/**
 * Comprehensive Database Service for AI Conversations
 * Supports conversation persistence, analytics, and user preferences
 */
export class ConversationStore extends EventEmitter {
  private conversations: Map<string, Conversation> = new Map();
  private messages: Map<string, Message[]> = new Map();
  private userPreferences: Map<string, UserPreferences> = new Map();
  private conversationIndex: Map<string, Set<string>> = new Map(); // userId -> conversationIds
  private readonly encryptionKey: string;

  constructor() {
    super();
    this.encryptionKey = process.env.ENCRYPTION_KEY || 'default-key-change-in-production';
    
    // Initialize with some sample data for demonstration
    this.initializeSampleData();
    
    console.log('📊 ConversationStore initialized');
  }

  // Create a new conversation
  async createConversation(data: Omit<Conversation, 'id' | 'createdAt' | 'updatedAt'>): Promise<Conversation> {
    try {
      const conversationId = this.generateId('conv');
      const now = new Date();
      
      const conversation: Conversation = {
        ...data,
        id: conversationId,
        createdAt: now,
        updatedAt: now
      };

      // Validate data
      const validationResult = ConversationSchema.safeParse(conversation);
      if (!validationResult.success) {
        throw new Error(`Invalid conversation data: ${validationResult.error.message}`);
      }

      // Store conversation
      this.conversations.set(conversationId, conversation);
      
      // Update user index
      const userConversations = this.conversationIndex.get(data.userId) || new Set();
      userConversations.add(conversationId);
      this.conversationIndex.set(data.userId, userConversations);

      // Initialize empty message array
      this.messages.set(conversationId, []);

      // Emit event
      this.emit('conversation_created', conversation);

      console.log(`✅ Created conversation: ${conversationId} for user: ${data.userId}`);
      return conversation;

    } catch (error) {
      console.error('Error creating conversation:', error);
      throw error;
    }
  }

  // Get conversation by ID
  async getConversation(conversationId: string, userId: string): Promise<Conversation | null> {
    try {
      const conversation = this.conversations.get(conversationId);
      
      if (!conversation) {
        return null;
      }

      // Security check - ensure user owns the conversation
      if (conversation.userId !== userId) {
        throw new Error('Unauthorized access to conversation');
      }

      return conversation;

    } catch (error) {
      console.error('Error getting conversation:', error);
      throw error;
    }
  }

  // Get conversations with filtering and pagination
  async getConversations(filter: ConversationFilter): Promise<{
    conversations: Conversation[];
    total: number;
    hasMore: boolean;
  }> {
    try {
      const userConversationIds = this.conversationIndex.get(filter.userId) || new Set();
      let filteredConversations: Conversation[] = [];

      // Get all user conversations
      for (const conversationId of userConversationIds) {
        const conversation = this.conversations.get(conversationId);
        if (conversation) {
          filteredConversations.push(conversation);
        }
      }

      // Apply filters
      if (filter.search) {
        const searchLower = filter.search.toLowerCase();
        filteredConversations = filteredConversations.filter(conv =>
          conv.title.toLowerCase().includes(searchLower) ||
          (conv.metadata.tags?.some(tag => tag.toLowerCase().includes(searchLower)))
        );
      }

      if (filter.tags && filter.tags.length > 0) {
        filteredConversations = filteredConversations.filter(conv =>
          conv.metadata.tags?.some(tag => filter.tags!.includes(tag))
        );
      }

      if (filter.isStarred !== undefined) {
        filteredConversations = filteredConversations.filter(conv =>
          conv.metadata.isStarred === filter.isStarred
        );
      }

      if (filter.isArchived !== undefined) {
        filteredConversations = filteredConversations.filter(conv =>
          conv.metadata.isArchived === filter.isArchived
        );
      }

      if (filter.model) {
        filteredConversations = filteredConversations.filter(conv =>
          conv.model === filter.model
        );
      }

      if (filter.dateFrom) {
        filteredConversations = filteredConversations.filter(conv =>
          conv.createdAt >= filter.dateFrom!
        );
      }

      if (filter.dateTo) {
        filteredConversations = filteredConversations.filter(conv =>
          conv.createdAt <= filter.dateTo!
        );
      }

      // Sort by last message date or created date
      filteredConversations.sort((a, b) => {
        const aDate = a.lastMessageAt || a.createdAt;
        const bDate = b.lastMessageAt || b.createdAt;
        return bDate.getTime() - aDate.getTime();
      });

      const total = filteredConversations.length;
      const offset = filter.offset || 0;
      const limit = filter.limit || 50;
      
      const paginatedConversations = filteredConversations.slice(offset, offset + limit);
      const hasMore = offset + limit < total;

      return {
        conversations: paginatedConversations,
        total,
        hasMore
      };

    } catch (error) {
      console.error('Error getting conversations:', error);
      throw error;
    }
  }

  // Add message to conversation
  async addMessage(conversationId: string, data: Omit<Message, 'id' | 'conversationId' | 'createdAt' | 'updatedAt'>): Promise<Message> {
    try {
      const conversation = this.conversations.get(conversationId);
      if (!conversation) {
        throw new Error('Conversation not found');
      }

      const messageId = this.generateId('msg');
      const now = new Date();

      const message: Message = {
        ...data,
        id: messageId,
        conversationId,
        createdAt: now,
        updatedAt: now
      };

      // Validate message
      const validationResult = MessageSchema.safeParse(message);
      if (!validationResult.success) {
        throw new Error(`Invalid message data: ${validationResult.error.message}`);
      }

      // Add to messages
      const conversationMessages = this.messages.get(conversationId) || [];
      conversationMessages.push(message);
      this.messages.set(conversationId, conversationMessages);

      // Update conversation metadata
      conversation.metadata.totalMessages++;
      conversation.lastMessageAt = now;
      conversation.updatedAt = now;

      if (message.metadata.tokens) {
        conversation.metadata.totalTokens += message.metadata.tokens;
      }

      if (message.metadata.responseTime) {
        const avgResponseTime = conversation.metadata.avgResponseTime;
        const totalMessages = conversation.metadata.totalMessages;
        conversation.metadata.avgResponseTime = 
          (avgResponseTime * (totalMessages - 1) + message.metadata.responseTime) / totalMessages;
      }

      if (message.metadata.toolCalls) {
        conversation.metadata.toolCalls += message.metadata.toolCalls.length;
        
        message.metadata.toolCalls.forEach(toolCall => {
          if (toolCall.toolName.includes('quantum')) {
            conversation.metadata.quantumOperations++;
          }
          if (toolCall.toolName.includes('mcp')) {
            conversation.metadata.mcpConnections++;
          }
        });
      }

      this.conversations.set(conversationId, conversation);

      // Emit events
      this.emit('message_added', { conversationId, message });
      this.emit('conversation_updated', conversation);

      return message;

    } catch (error) {
      console.error('Error adding message:', error);
      throw error;
    }
  }

  // Get messages for a conversation
  async getMessages(conversationId: string, userId: string, options?: {
    limit?: number;
    offset?: number;
    beforeMessageId?: string;
    afterMessageId?: string;
  }): Promise<{
    messages: Message[];
    total: number;
    hasMore: boolean;
  }> {
    try {
      // Verify user access
      const conversation = await this.getConversation(conversationId, userId);
      if (!conversation) {
        throw new Error('Conversation not found or unauthorized');
      }

      let messages = this.messages.get(conversationId) || [];
      
      // Apply filtering
      if (options?.beforeMessageId) {
        const beforeIndex = messages.findIndex(m => m.id === options.beforeMessageId);
        if (beforeIndex > -1) {
          messages = messages.slice(0, beforeIndex);
        }
      }

      if (options?.afterMessageId) {
        const afterIndex = messages.findIndex(m => m.id === options.afterMessageId);
        if (afterIndex > -1) {
          messages = messages.slice(afterIndex + 1);
        }
      }

      const total = messages.length;
      const offset = options?.offset || 0;
      const limit = options?.limit || 100;
      
      const paginatedMessages = messages.slice(offset, offset + limit);
      const hasMore = offset + limit < total;

      return {
        messages: paginatedMessages,
        total,
        hasMore
      };

    } catch (error) {
      console.error('Error getting messages:', error);
      throw error;
    }
  }

  // Update conversation
  async updateConversation(conversationId: string, userId: string, updates: Partial<Conversation>): Promise<Conversation> {
    try {
      const conversation = await this.getConversation(conversationId, userId);
      if (!conversation) {
        throw new Error('Conversation not found or unauthorized');
      }

      const updatedConversation = {
        ...conversation,
        ...updates,
        id: conversationId, // Prevent ID changes
        userId: conversation.userId, // Prevent user changes
        updatedAt: new Date()
      };

      // Validate updated conversation
      const validationResult = ConversationSchema.safeParse(updatedConversation);
      if (!validationResult.success) {
        throw new Error(`Invalid conversation update: ${validationResult.error.message}`);
      }

      this.conversations.set(conversationId, updatedConversation);

      this.emit('conversation_updated', updatedConversation);
      return updatedConversation;

    } catch (error) {
      console.error('Error updating conversation:', error);
      throw error;
    }
  }

  // Delete conversation
  async deleteConversation(conversationId: string, userId: string): Promise<void> {
    try {
      const conversation = await this.getConversation(conversationId, userId);
      if (!conversation) {
        throw new Error('Conversation not found or unauthorized');
      }

      // Remove from maps
      this.conversations.delete(conversationId);
      this.messages.delete(conversationId);

      // Update user index
      const userConversations = this.conversationIndex.get(userId);
      if (userConversations) {
        userConversations.delete(conversationId);
      }

      this.emit('conversation_deleted', { conversationId, userId });
      console.log(`🗑️ Deleted conversation: ${conversationId}`);

    } catch (error) {
      console.error('Error deleting conversation:', error);
      throw error;
    }
  }

  // Get user preferences
  async getUserPreferences(userId: string): Promise<UserPreferences> {
    try {
      let preferences = this.userPreferences.get(userId);
      
      if (!preferences) {
        // Create default preferences
        preferences = {
          userId,
          preferences: {
            defaultModel: 'gpt-4o',
            defaultTemperature: 0.7,
            enableAutoSave: true,
            enableNotifications: true,
            theme: 'auto',
            language: 'en',
            conversationRetention: 90,
            exportFormat: 'json'
          },
          apiUsage: {
            totalMessages: 0,
            totalTokens: 0,
            monthlyLimit: 10000,
            currentMonthUsage: 0,
            lastResetDate: new Date()
          },
          createdAt: new Date(),
          updatedAt: new Date()
        };
        
        this.userPreferences.set(userId, preferences);
      }

      return preferences;

    } catch (error) {
      console.error('Error getting user preferences:', error);
      throw error;
    }
  }

  // Update user preferences
  async updateUserPreferences(userId: string, updates: Partial<UserPreferences['preferences']>): Promise<UserPreferences> {
    try {
      const currentPreferences = await this.getUserPreferences(userId);
      
      const updatedPreferences = {
        ...currentPreferences,
        preferences: {
          ...currentPreferences.preferences,
          ...updates
        },
        updatedAt: new Date()
      };

      this.userPreferences.set(userId, updatedPreferences);
      this.emit('preferences_updated', updatedPreferences);

      return updatedPreferences;

    } catch (error) {
      console.error('Error updating user preferences:', error);
      throw error;
    }
  }

  // Get conversation statistics
  async getConversationStats(userId: string): Promise<ConversationStats> {
    try {
      const userConversationIds = this.conversationIndex.get(userId) || new Set();
      const conversations: Conversation[] = [];
      const allMessages: Message[] = [];

      // Collect data
      for (const conversationId of userConversationIds) {
        const conversation = this.conversations.get(conversationId);
        if (conversation) {
          conversations.push(conversation);
          const messages = this.messages.get(conversationId) || [];
          allMessages.push(...messages);
        }
      }

      // Calculate statistics
      const totalConversations = conversations.length;
      const totalMessages = allMessages.length;
      const totalTokens = conversations.reduce((sum, conv) => sum + conv.metadata.totalTokens, 0);
      const avgConversationLength = totalConversations > 0 ? totalMessages / totalConversations : 0;

      // Most used model
      const modelCounts: Record<string, number> = {};
      conversations.forEach(conv => {
        modelCounts[conv.model] = (modelCounts[conv.model] || 0) + 1;
      });
      const mostUsedModel = Object.entries(modelCounts)
        .sort(([,a], [,b]) => b - a)[0]?.[0] || 'none';

      // Tool usage statistics
      const toolUsageStats: Record<string, number> = {};
      allMessages.forEach(message => {
        message.metadata.toolCalls?.forEach(toolCall => {
          toolUsageStats[toolCall.toolName] = (toolUsageStats[toolCall.toolName] || 0) + 1;
        });
      });

      // Daily activity (last 30 days)
      const dailyActivity: Array<{ date: string; count: number }> = [];
      const thirtyDaysAgo = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000);
      
      for (let i = 0; i < 30; i++) {
        const date = new Date(thirtyDaysAgo.getTime() + i * 24 * 60 * 60 * 1000);
        const dateStr = date.toISOString().split('T')[0];
        const count = allMessages.filter(message => 
          message.createdAt.toISOString().split('T')[0] === dateStr
        ).length;
        
        dailyActivity.push({ date: dateStr, count });
      }

      return {
        totalConversations,
        totalMessages,
        totalTokens,
        avgConversationLength,
        mostUsedModel,
        toolUsageStats,
        dailyActivity
      };

    } catch (error) {
      console.error('Error getting conversation stats:', error);
      throw error;
    }
  }

  // Export conversation data
  async exportConversation(conversationId: string, userId: string, format: 'json' | 'markdown' = 'json'): Promise<string> {
    try {
      const conversation = await this.getConversation(conversationId, userId);
      if (!conversation) {
        throw new Error('Conversation not found or unauthorized');
      }

      const messagesResult = await this.getMessages(conversationId, userId);
      const messages = messagesResult.messages;

      if (format === 'json') {
        return JSON.stringify({
          conversation,
          messages,
          exportedAt: new Date().toISOString(),
          version: '1.0'
        }, null, 2);
      } else {
        // Markdown format
        let markdown = `# ${conversation.title}\n\n`;
        markdown += `**Model:** ${conversation.model}\n`;
        markdown += `**Created:** ${conversation.createdAt.toISOString()}\n`;
        markdown += `**Total Messages:** ${conversation.metadata.totalMessages}\n\n`;
        markdown += `---\n\n`;

        messages.forEach(message => {
          markdown += `## ${message.role.charAt(0).toUpperCase() + message.role.slice(1)}\n\n`;
          markdown += `${message.content}\n\n`;
          
          if (message.metadata.toolCalls && message.metadata.toolCalls.length > 0) {
            markdown += `**Tool Calls:**\n`;
            message.metadata.toolCalls.forEach(toolCall => {
              markdown += `- ${toolCall.toolName}: ${JSON.stringify(toolCall.args)}\n`;
            });
            markdown += `\n`;
          }
          
          markdown += `*${message.createdAt.toLocaleString()}*\n\n`;
          markdown += `---\n\n`;
        });

        return markdown;
      }

    } catch (error) {
      console.error('Error exporting conversation:', error);
      throw error;
    }
  }

  // Search across conversations
  async searchConversations(userId: string, query: string, options?: {
    includeMessages?: boolean;
    limit?: number;
  }): Promise<{
    conversations: Array<Conversation & { relevantMessages?: Message[] }>;
    total: number;
  }> {
    try {
      const searchLower = query.toLowerCase();
      const userConversationIds = this.conversationIndex.get(userId) || new Set();
      const results: Array<Conversation & { relevantMessages?: Message[] }> = [];

      for (const conversationId of userConversationIds) {
        const conversation = this.conversations.get(conversationId);
        if (!conversation) continue;

        let isRelevant = false;
        let relevantMessages: Message[] = [];

        // Check conversation title and tags
        if (conversation.title.toLowerCase().includes(searchLower) ||
            conversation.metadata.tags?.some(tag => tag.toLowerCase().includes(searchLower))) {
          isRelevant = true;
        }

        // Check messages if requested
        if (options?.includeMessages) {
          const messages = this.messages.get(conversationId) || [];
          relevantMessages = messages.filter(message =>
            message.content.toLowerCase().includes(searchLower)
          );
          
          if (relevantMessages.length > 0) {
            isRelevant = true;
          }
        }

        if (isRelevant) {
          results.push({
            ...conversation,
            relevantMessages: options?.includeMessages ? relevantMessages : undefined
          });
        }
      }

      // Sort by relevance (conversations with matching titles first)
      results.sort((a, b) => {
        const aTitle = a.title.toLowerCase().includes(searchLower);
        const bTitle = b.title.toLowerCase().includes(searchLower);
        if (aTitle && !bTitle) return -1;
        if (!aTitle && bTitle) return 1;
        return (b.lastMessageAt || b.createdAt).getTime() - (a.lastMessageAt || a.createdAt).getTime();
      });

      const limit = options?.limit || 50;
      const limitedResults = results.slice(0, limit);

      return {
        conversations: limitedResults,
        total: results.length
      };

    } catch (error) {
      console.error('Error searching conversations:', error);
      throw error;
    }
  }

  // Utility methods
  private generateId(prefix: string): string {
    return `${prefix}_${Date.now()}_${crypto.randomBytes(8).toString('hex')}`;
  }

  private encrypt(text: string): string {
    const cipher = crypto.createCipher('aes-256-cbc', this.encryptionKey);
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    return encrypted;
  }

  private decrypt(encryptedText: string): string {
    const decipher = crypto.createDecipher('aes-256-cbc', this.encryptionKey);
    let decrypted = decipher.update(encryptedText, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    return decrypted;
  }

  // Initialize sample data for demonstration
  private initializeSampleData(): void {
    const sampleUserId = 'user_123';
    
    // Create sample conversation
    const conversationId = this.generateId('conv');
    const now = new Date();
    
    const sampleConversation: Conversation = {
      id: conversationId,
      userId: sampleUserId,
      title: 'Quantum Computing Discussion',
      model: 'gpt-4o',
      systemPrompt: 'You are an expert in quantum computing and AI.',
      config: {
        temperature: 0.7,
        maxTokens: 2000,
        enableTools: true,
        enableQuantum: true,
        enableMCP: true
      },
      metadata: {
        totalMessages: 2,
        totalTokens: 150,
        avgResponseTime: 1200,
        toolCalls: 1,
        quantumOperations: 1,
        mcpConnections: 0,
        tags: ['quantum', 'computing', 'science'],
        isStarred: true,
        isArchived: false
      },
      createdAt: now,
      updatedAt: now,
      lastMessageAt: now
    };

    this.conversations.set(conversationId, sampleConversation);
    
    const userConversations = new Set([conversationId]);
    this.conversationIndex.set(sampleUserId, userConversations);

    // Sample messages
    const sampleMessages: Message[] = [
      {
        id: this.generateId('msg'),
        conversationId,
        role: 'user',
        content: 'Can you explain quantum superposition?',
        metadata: {},
        createdAt: now,
        updatedAt: now
      },
      {
        id: this.generateId('msg'),
        conversationId,
        role: 'assistant',
        content: 'Quantum superposition is a fundamental principle where a quantum system exists in multiple states simultaneously until measured.',
        metadata: {
          model: 'gpt-4o',
          tokens: 75,
          responseTime: 1200,
          toolCalls: [
            {
              toolName: 'quantum_analyzer',
              args: { concept: 'superposition' },
              result: { explanation: 'detailed analysis', confidence: 0.95 },
              duration: 150
            }
          ]
        },
        createdAt: new Date(now.getTime() + 5000),
        updatedAt: new Date(now.getTime() + 5000)
      }
    ];

    this.messages.set(conversationId, sampleMessages);

    console.log('📝 Sample conversation data initialized');
  }
}

// Export singleton instance
export const conversationStore = new ConversationStore();