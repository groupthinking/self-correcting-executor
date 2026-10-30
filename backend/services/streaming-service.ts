import { WebSocket, WebSocketServer } from 'ws';
import { EventEmitter } from 'events';
import { z } from 'zod';
import { IncomingMessage } from 'http';
import { URL } from 'url';
import jwt from 'jsonwebtoken';
import { openai } from '@ai-sdk/openai';
import { streamText, generateObject } from 'ai';
import crypto from 'crypto';

// WebSocket Message Schemas
const WSMessageSchema = z.object({
  id: z.string(),
  type: z.enum(['chat', 'tool_call', 'system_update', 'analytics', 'heartbeat']),
  payload: z.any(),
  timestamp: z.string().optional(),
  conversationId: z.string().optional()
});

const ChatMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string().max(10000),
  metadata: z.object({
    model: z.string().optional(),
    temperature: z.number().optional(),
    enableTools: z.boolean().optional(),
    enableQuantum: z.boolean().optional(),
    enableMCP: z.boolean().optional()
  }).optional()
});

// Connection Types
interface AuthenticatedConnection {
  ws: WebSocket;
  userId: string;
  connectionId: string;
  userRole: string;
  connectedAt: Date;
  lastActivity: Date;
  subscriptions: Set<string>;
  rateLimitData: {
    messageCount: number;
    windowStart: number;
  };
}

interface StreamingSession {
  id: string;
  connectionId: string;
  conversationId: string;
  model: string;
  startTime: Date;
  isActive: boolean;
  messageCount: number;
  tokenCount: number;
}

/**
 * Advanced Streaming Service with WebSocket Integration
 * Supports real-time AI conversations, tool calling, and system updates
 */
export class StreamingService extends EventEmitter {
  private wss: WebSocketServer;
  private connections: Map<string, AuthenticatedConnection> = new Map();
  private sessions: Map<string, StreamingSession> = new Map();
  private readonly JWT_SECRET = process.env.JWT_SECRET || 'change-in-production';
  private readonly RATE_LIMIT_WINDOW = 60000; // 1 minute
  private readonly RATE_LIMIT_MAX = 60; // 60 messages per minute
  private readonly HEARTBEAT_INTERVAL = 30000; // 30 seconds
  private heartbeatTimer?: NodeJS.Timeout;

  constructor(port: number = 8080) {
    super();
    this.wss = new WebSocketServer({ 
      port,
      verifyClient: this.verifyClient.bind(this)
    });
    
    this.setupWebSocketServer();
    this.startHeartbeat();
    
    console.log(`🚀 Streaming Service started on port ${port}`);
  }

  // Client verification with JWT authentication
  private verifyClient(info: { origin: string; secure: boolean; req: IncomingMessage }): boolean {
    try {
      const url = new URL(info.req.url || '', `http://${info.req.headers.host}`);
      const token = url.searchParams.get('token');
      
      if (!token) {
        console.warn('WebSocket connection rejected: No token provided');
        return false;
      }

      const decoded = jwt.verify(token, this.JWT_SECRET) as any;
      
      // Store auth info for later use
      (info.req as any).authData = {
        userId: decoded.userId,
        userRole: decoded.role || 'user',
        email: decoded.email
      };

      return true;
    } catch (error) {
      console.warn('WebSocket connection rejected: Invalid token', error);
      return false;
    }
  }

  // Setup WebSocket server handlers
  private setupWebSocketServer(): void {
    this.wss.on('connection', (ws: WebSocket, req: IncomingMessage) => {
      const authData = (req as any).authData;
      const connectionId = this.generateConnectionId();
      
      const connection: AuthenticatedConnection = {
        ws,
        userId: authData.userId,
        connectionId,
        userRole: authData.userRole,
        connectedAt: new Date(),
        lastActivity: new Date(),
        subscriptions: new Set(),
        rateLimitData: {
          messageCount: 0,
          windowStart: Date.now()
        }
      };

      this.connections.set(connectionId, connection);
      
      console.log(`✅ WebSocket connected: ${connectionId} (User: ${authData.userId})`);
      
      // Send welcome message
      this.sendMessage(connectionId, {
        id: this.generateMessageId(),
        type: 'system_update',
        payload: {
          status: 'connected',
          connectionId,
          features: ['ai_chat', 'tool_calling', 'real_time_updates', 'quantum_computing', 'mcp_integration'],
          serverTime: new Date().toISOString()
        }
      });

      // Setup message handlers
      ws.on('message', (data: Buffer) => {
        this.handleMessage(connectionId, data);
      });

      ws.on('close', (code: number, reason: Buffer) => {
        this.handleDisconnection(connectionId, code, reason);
      });

      ws.on('error', (error: Error) => {
        console.error(`WebSocket error for ${connectionId}:`, error);
        this.handleDisconnection(connectionId, 1011, Buffer.from('Internal error'));
      });

      // Emit connection event
      this.emit('connection', { connectionId, userId: authData.userId });
    });
  }

  // Handle incoming WebSocket messages
  private async handleMessage(connectionId: string, data: Buffer): Promise<void> {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    try {
      // Rate limiting
      if (!this.checkRateLimit(connection)) {
        this.sendError(connectionId, 'Rate limit exceeded', 'RATE_LIMIT');
        return;
      }

      const rawMessage = data.toString();
      const messageData = JSON.parse(rawMessage);
      
      // Validate message structure
      const validationResult = WSMessageSchema.safeParse(messageData);
      if (!validationResult.success) {
        this.sendError(connectionId, 'Invalid message format', 'VALIDATION_ERROR');
        return;
      }

      const message = validationResult.data;
      connection.lastActivity = new Date();

      // Handle different message types
      switch (message.type) {
        case 'chat':
          await this.handleChatMessage(connectionId, message);
          break;
        case 'tool_call':
          await this.handleToolCall(connectionId, message);
          break;
        case 'heartbeat':
          this.handleHeartbeat(connectionId, message);
          break;
        default:
          this.sendError(connectionId, 'Unsupported message type', 'UNSUPPORTED_TYPE');
      }

    } catch (error) {
      console.error(`Error handling message from ${connectionId}:`, error);
      this.sendError(connectionId, 'Message processing failed', 'PROCESSING_ERROR');
    }
  }

  // Handle chat messages with AI SDK 5 Beta streaming
  private async handleChatMessage(connectionId: string, message: any): Promise<void> {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    try {
      const chatValidation = ChatMessageSchema.safeParse(message.payload);
      if (!chatValidation.success) {
        this.sendError(connectionId, 'Invalid chat message format', 'CHAT_VALIDATION_ERROR');
        return;
      }

      const chatMessage = chatValidation.data;
      const conversationId = message.conversationId || this.generateConversationId();
      
      // Create or update streaming session
      const sessionId = `${connectionId}_${conversationId}`;
      let session = this.sessions.get(sessionId);
      
      if (!session) {
        session = {
          id: sessionId,
          connectionId,
          conversationId,
          model: chatMessage.metadata?.model || 'gpt-4o',
          startTime: new Date(),
          isActive: true,
          messageCount: 0,
          tokenCount: 0
        };
        this.sessions.set(sessionId, session);
      }

      session.messageCount++;
      session.isActive = true;

      // Notify client that streaming is starting
      this.sendMessage(connectionId, {
        id: this.generateMessageId(),
        type: 'system_update',
        payload: {
          status: 'streaming_started',
          conversationId,
          sessionId: session.id
        }
      });

      // AI SDK 5 Beta streaming implementation
      const result = await streamText({
        model: openai(session.model),
        messages: [
          {
            role: 'system',
            content: 'You are an advanced AI assistant with access to quantum computing and MCP protocols. Provide helpful, accurate, and secure responses.'
          },
          chatMessage
        ],
        temperature: chatMessage.metadata?.temperature || 0.7,
        maxTokens: 2000,
        
        // Enhanced streaming with real-time updates
        experimental_streamingTimeout: 30000,
        experimental_telemetry: {
          isEnabled: true,
          recordInputs: false,
          recordOutputs: false
        },

        // Real-time streaming callback
        onChunk: (chunk) => {
          if (chunk.type === 'text-delta') {
            this.sendMessage(connectionId, {
              id: this.generateMessageId(),
              type: 'chat',
              payload: {
                type: 'text_delta',
                content: chunk.textDelta,
                conversationId,
                sessionId: session!.id
              }
            });
          } else if (chunk.type === 'tool-call') {
            this.sendMessage(connectionId, {
              id: this.generateMessageId(),
              type: 'tool_call',
              payload: {
                toolName: chunk.toolName,
                args: chunk.args,
                conversationId,
                sessionId: session!.id
              }
            });
          }
        },

        onFinish: (result) => {
          session!.tokenCount += result.usage?.totalTokens || 0;
          session!.isActive = false;
          
          this.sendMessage(connectionId, {
            id: this.generateMessageId(),
            type: 'system_update',
            payload: {
              status: 'streaming_completed',
              conversationId,
              sessionId: session!.id,
              usage: result.usage,
              finishReason: result.finishReason
            }
          });
        },

        onError: (error) => {
          console.error('Streaming error:', error);
          session!.isActive = false;
          
          this.sendError(connectionId, 'AI streaming failed', 'STREAMING_ERROR', {
            conversationId,
            sessionId: session!.id
          });
        }
      });

    } catch (error) {
      console.error('Chat message handling error:', error);
      this.sendError(connectionId, 'Chat processing failed', 'CHAT_ERROR');
    }
  }

  // Handle tool calls
  private async handleToolCall(connectionId: string, message: any): Promise<void> {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    try {
      const { toolName, args, conversationId } = message.payload;
      
      // Emit tool call event for processing
      this.emit('tool_call', {
        connectionId,
        toolName,
        args,
        conversationId
      });

      // Send acknowledgment
      this.sendMessage(connectionId, {
        id: this.generateMessageId(),
        type: 'system_update',
        payload: {
          status: 'tool_call_received',
          toolName,
          conversationId
        }
      });

    } catch (error) {
      console.error('Tool call handling error:', error);
      this.sendError(connectionId, 'Tool call failed', 'TOOL_ERROR');
    }
  }

  // Handle heartbeat messages
  private handleHeartbeat(connectionId: string, message: any): void {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    connection.lastActivity = new Date();
    
    this.sendMessage(connectionId, {
      id: this.generateMessageId(),
      type: 'heartbeat',
      payload: {
        status: 'alive',
        serverTime: new Date().toISOString()
      }
    });
  }

  // Send message to specific connection
  private sendMessage(connectionId: string, message: any): void {
    const connection = this.connections.get(connectionId);
    if (!connection || connection.ws.readyState !== WebSocket.OPEN) return;

    const messageWithTimestamp = {
      ...message,
      timestamp: new Date().toISOString()
    };

    connection.ws.send(JSON.stringify(messageWithTimestamp));
  }

  // Send error message
  private sendError(connectionId: string, message: string, code: string, details?: any): void {
    this.sendMessage(connectionId, {
      id: this.generateMessageId(),
      type: 'system_update',
      payload: {
        status: 'error',
        error: {
          message,
          code,
          details
        }
      }
    });
  }

  // Broadcast message to all connections
  public broadcast(message: any, filter?: (connection: AuthenticatedConnection) => boolean): void {
    this.connections.forEach((connection, connectionId) => {
      if (!filter || filter(connection)) {
        this.sendMessage(connectionId, message);
      }
    });
  }

  // Send to specific user
  public sendToUser(userId: string, message: any): void {
    this.connections.forEach((connection, connectionId) => {
      if (connection.userId === userId) {
        this.sendMessage(connectionId, message);
      }
    });
  }

  // Rate limiting check
  private checkRateLimit(connection: AuthenticatedConnection): boolean {
    const now = Date.now();
    const window = connection.rateLimitData;
    
    // Reset window if expired
    if (now - window.windowStart >= this.RATE_LIMIT_WINDOW) {
      window.messageCount = 0;
      window.windowStart = now;
    }
    
    window.messageCount++;
    return window.messageCount <= this.RATE_LIMIT_MAX;
  }

  // Handle connection disconnection
  private handleDisconnection(connectionId: string, code: number, reason: Buffer): void {
    const connection = this.connections.get(connectionId);
    if (!connection) return;

    console.log(`🔌 WebSocket disconnected: ${connectionId} (Code: ${code}, Reason: ${reason.toString()})`);
    
    // Clean up sessions
    this.sessions.forEach((session, sessionId) => {
      if (session.connectionId === connectionId) {
        session.isActive = false;
        // Keep session for a while for potential reconnection
        setTimeout(() => {
          this.sessions.delete(sessionId);
        }, 300000); // 5 minutes
      }
    });

    this.connections.delete(connectionId);
    
    // Emit disconnection event
    this.emit('disconnection', { connectionId, userId: connection.userId });
  }

  // Start heartbeat mechanism
  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      const now = new Date();
      const staleThreshold = new Date(now.getTime() - this.HEARTBEAT_INTERVAL * 2);
      
      this.connections.forEach((connection, connectionId) => {
        if (connection.lastActivity < staleThreshold) {
          console.warn(`Stale connection detected: ${connectionId}`);
          connection.ws.close(1001, 'Connection stale');
        }
      });
    }, this.HEARTBEAT_INTERVAL);
  }

  // Utility methods
  private generateSecureRandomString(lengthBytes: number = 16): string {
    return crypto.randomBytes(lengthBytes).toString('hex');
  }

  private generateConnectionId(): string {
    return `conn_${Date.now()}_${this.generateSecureRandomString()}`;
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${this.generateSecureRandomString()}`;
  }

  private generateConversationId(): string {
    return `conv_${Date.now()}_${this.generateSecureRandomString()}`;
  }

  // Analytics and monitoring
  public getConnectionStats(): any {
    const activeConnections = this.connections.size;
    const activeSessions = Array.from(this.sessions.values()).filter(s => s.isActive).length;
    const totalSessions = this.sessions.size;
    
    return {
      activeConnections,
      activeSessions,
      totalSessions,
      connections: Array.from(this.connections.values()).map(conn => ({
        connectionId: conn.connectionId,
        userId: conn.userId,
        connectedAt: conn.connectedAt,
        lastActivity: conn.lastActivity,
        subscriptions: Array.from(conn.subscriptions)
      }))
    };
  }

  // Graceful shutdown
  public async shutdown(): Promise<void> {
    console.log('🛑 Shutting down Streaming Service...');
    
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
    }

    // Close all connections gracefully
    this.connections.forEach((connection, connectionId) => {
      this.sendMessage(connectionId, {
        id: this.generateMessageId(),
        type: 'system_update',
        payload: {
          status: 'server_shutdown',
          message: 'Server is shutting down'
        }
      });
      connection.ws.close(1001, 'Server shutdown');
    });

    // Close WebSocket server
    return new Promise((resolve) => {
      this.wss.close(() => {
        console.log('✅ Streaming Service shutdown complete');
        resolve();
      });
    });
  }
}

// Export singleton instance
export const streamingService = new StreamingService();