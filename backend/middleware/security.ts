import { NextRequest, NextResponse } from 'next/server';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { z } from 'zod';
import crypto from 'crypto';

// Security Configuration
const JWT_SECRET = process.env.JWT_SECRET || 'change-in-production';
const JWT_EXPIRATION = process.env.JWT_EXPIRATION || '24h';
const BCRYPT_ROUNDS = 12;
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_TIME = 15 * 60 * 1000; // 15 minutes

// Validation Schemas
const LoginSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(8).max(128),
  rememberMe: z.boolean().optional().default(false)
});

const RegisterSchema = z.object({
  email: z.string().email().max(255),
  password: z.string().min(8).max(128)
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/, 
      'Password must contain uppercase, lowercase, number, and special character'),
  confirmPassword: z.string(),
  firstName: z.string().min(1).max(50),
  lastName: z.string().min(1).max(50)
}).refine(data => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"]
});

// User Interface
interface User {
  id: string;
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  role: 'user' | 'admin' | 'developer';
  isActive: boolean;
  lastLogin?: Date;
  loginAttempts: number;
  lockoutUntil?: Date;
  createdAt: Date;
  updatedAt: Date;
}

// Mock user database (replace with real database)
const users: Map<string, User> = new Map();

// Security utilities
export class SecurityService {
  private static instance: SecurityService;
  private readonly pepper: string;

  constructor() {
    this.pepper = process.env.PASSWORD_PEPPER || 'default-pepper-change-in-production';
  }

  static getInstance(): SecurityService {
    if (!SecurityService.instance) {
      SecurityService.instance = new SecurityService();
    }
    return SecurityService.instance;
  }

  // Password hashing with pepper
  async hashPassword(password: string): Promise<string> {
    const passwordWithPepper = password + this.pepper;
    return bcrypt.hash(passwordWithPepper, BCRYPT_ROUNDS);
  }

  // Password verification
  async verifyPassword(password: string, hash: string): Promise<boolean> {
    const passwordWithPepper = password + this.pepper;
    return bcrypt.compare(passwordWithPepper, hash);
  }

  // Generate secure JWT token
  generateToken(payload: object, expiresIn: string = JWT_EXPIRATION): string {
    return jwt.sign(
      {
        ...payload,
        iat: Math.floor(Date.now() / 1000),
        jti: crypto.randomUUID() // JWT ID for token tracking
      },
      JWT_SECRET,
      { expiresIn }
    );
  }

  // Verify JWT token
  verifyToken(token: string): { valid: boolean; payload?: any; error?: string } {
    try {
      const payload = jwt.verify(token, JWT_SECRET);
      return { valid: true, payload };
    } catch (error) {
      if (error instanceof jwt.TokenExpiredError) {
        return { valid: false, error: 'Token expired' };
      } else if (error instanceof jwt.JsonWebTokenError) {
        return { valid: false, error: 'Invalid token' };
      }
      return { valid: false, error: 'Token verification failed' };
    }
  }

  // Generate secure random string
  generateSecureRandom(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  // Hash sensitive data for logging
  hashForLogging(data: string): string {
    return crypto.createHash('sha256').update(data).digest('hex').substring(0, 8);
  }

  // Check if user is locked out
  isLockedOut(user: User): boolean {
    return !!(user.lockoutUntil && user.lockoutUntil > new Date());
  }

  // Increment login attempts
  async incrementLoginAttempts(userId: string): Promise<void> {
    const user = users.get(userId);
    if (!user) return;

    user.loginAttempts += 1;
    
    if (user.loginAttempts >= MAX_LOGIN_ATTEMPTS) {
      user.lockoutUntil = new Date(Date.now() + LOCKOUT_TIME);
    }
    
    user.updatedAt = new Date();
    users.set(userId, user);
  }

  // Reset login attempts
  async resetLoginAttempts(userId: string): Promise<void> {
    const user = users.get(userId);
    if (!user) return;

    user.loginAttempts = 0;
    user.lockoutUntil = undefined;
    user.lastLogin = new Date();
    user.updatedAt = new Date();
    users.set(userId, user);
  }

  // Sanitize input to prevent XSS
  sanitizeInput(input: string): string {
    return input
      .replace(/[<>]/g, '') // Remove potential HTML tags
      .replace(/javascript:/gi, '') // Remove javascript: protocol
      .replace(/data:/gi, '') // Remove data: protocol
      .replace(/vbscript:/gi, '') // Remove vbscript: protocol
      .replace(/on\w+=/gi, '') // Remove event handlers
      .trim();
  }

  // Validate IP address
  isValidIP(ip: string): boolean {
    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$|^([0-9a-f]{1,4}:){7}[0-9a-f]{1,4}$/i;
    return ipRegex.test(ip);
  }

  // Rate limiting key generator
  getRateLimitKey(request: NextRequest, identifier?: string): string {
    const ip = this.getClientIP(request);
    const userAgent = request.headers.get('user-agent') || '';
    const hashedUA = crypto.createHash('md5').update(userAgent).digest('hex').substring(0, 8);
    
    return identifier ? `${identifier}:${ip}:${hashedUA}` : `${ip}:${hashedUA}`;
  }

  // Get client IP address
  getClientIP(request: NextRequest): string {
    const forwarded = request.headers.get('x-forwarded-for');
    const realIP = request.headers.get('x-real-ip');
    const remoteAddr = request.headers.get('remote-addr');
    
    if (forwarded) {
      return forwarded.split(',')[0].trim();
    }
    
    return realIP || remoteAddr || 'unknown';
  }
}

// Authentication middleware
export class AuthenticationMiddleware {
  private security: SecurityService;

  constructor() {
    this.security = SecurityService.getInstance();
  }

  // Registration endpoint
  async register(request: NextRequest): Promise<NextResponse> {
    try {
      const body = await request.json();
      const validationResult = RegisterSchema.safeParse(body);

      if (!validationResult.success) {
        return NextResponse.json(
          { 
            error: 'Invalid registration data',
            details: validationResult.error.issues
          },
          { status: 400 }
        );
      }

      const { email, password, firstName, lastName } = validationResult.data;

      // Check if user already exists
      const existingUser = Array.from(users.values()).find(u => u.email === email);
      if (existingUser) {
        return NextResponse.json(
          { error: 'User already exists' },
          { status: 409 }
        );
      }

      // Create new user
      const userId = crypto.randomUUID();
      const hashedPassword = await this.security.hashPassword(password);
      
      const newUser: User = {
        id: userId,
        email: this.security.sanitizeInput(email),
        password: hashedPassword,
        firstName: this.security.sanitizeInput(firstName),
        lastName: this.security.sanitizeInput(lastName),
        role: 'user',
        isActive: true,
        loginAttempts: 0,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      users.set(userId, newUser);

      // Generate token
      const token = this.security.generateToken({
        userId,
        email,
        role: newUser.role
      });

      // Log registration (with hashed email for privacy)
      console.log('User registration:', {
        userId,
        email: this.security.hashForLogging(email),
        timestamp: new Date().toISOString(),
        ip: this.security.getClientIP(request)
      });

      return NextResponse.json(
        {
          success: true,
          user: {
            id: userId,
            email,
            firstName,
            lastName,
            role: newUser.role
          },
          token
        },
        { status: 201 }
      );

    } catch (error) {
      console.error('Registration error:', error);
      return NextResponse.json(
        { error: 'Registration failed' },
        { status: 500 }
      );
    }
  }

  // Login endpoint
  async login(request: NextRequest): Promise<NextResponse> {
    try {
      const body = await request.json();
      const validationResult = LoginSchema.safeParse(body);

      if (!validationResult.success) {
        return NextResponse.json(
          { 
            error: 'Invalid login data',
            details: validationResult.error.issues
          },
          { status: 400 }
        );
      }

      const { email, password, rememberMe } = validationResult.data;

      // Find user
      const user = Array.from(users.values()).find(u => u.email === email);
      if (!user) {
        return NextResponse.json(
          { error: 'Invalid credentials' },
          { status: 401 }
        );
      }

      // Check if account is locked
      if (this.security.isLockedOut(user)) {
        return NextResponse.json(
          { 
            error: 'Account temporarily locked due to multiple failed attempts',
            lockoutUntil: user.lockoutUntil
          },
          { status: 423 }
        );
      }

      // Check if account is active
      if (!user.isActive) {
        return NextResponse.json(
          { error: 'Account is disabled' },
          { status: 403 }
        );
      }

      // Verify password
      const isValidPassword = await this.security.verifyPassword(password, user.password);
      if (!isValidPassword) {
        await this.security.incrementLoginAttempts(user.id);
        
        return NextResponse.json(
          { error: 'Invalid credentials' },
          { status: 401 }
        );
      }

      // Reset login attempts on successful login
      await this.security.resetLoginAttempts(user.id);

      // Generate token
      const expiresIn = rememberMe ? '30d' : JWT_EXPIRATION;
      const token = this.security.generateToken({
        userId: user.id,
        email: user.email,
        role: user.role
      }, expiresIn);

      // Log successful login
      console.log('Successful login:', {
        userId: user.id,
        email: this.security.hashForLogging(email),
        timestamp: new Date().toISOString(),
        ip: this.security.getClientIP(request),
        rememberMe
      });

      return NextResponse.json({
        success: true,
        user: {
          id: user.id,
          email: user.email,
          firstName: user.firstName,
          lastName: user.lastName,
          role: user.role
        },
        token
      });

    } catch (error) {
      console.error('Login error:', error);
      return NextResponse.json(
        { error: 'Login failed' },
        { status: 500 }
      );
    }
  }

  // Token verification middleware
  async verifyAuthToken(request: NextRequest): Promise<{ authorized: boolean; user?: any; error?: string }> {
    try {
      const authHeader = request.headers.get('Authorization');
      if (!authHeader?.startsWith('Bearer ')) {
        return { authorized: false, error: 'Missing or invalid authorization header' };
      }

      const token = authHeader.substring(7);
      const { valid, payload, error } = this.security.verifyToken(token);

      if (!valid) {
        return { authorized: false, error };
      }

      // Check if user still exists and is active
      const user = users.get(payload.userId);
      if (!user || !user.isActive) {
        return { authorized: false, error: 'User not found or inactive' };
      }

      return { 
        authorized: true, 
        user: {
          id: user.id,
          email: user.email,
          role: user.role
        }
      };

    } catch (error) {
      return { authorized: false, error: 'Token verification failed' };
    }
  }

  // Logout endpoint (for token blacklisting)
  async logout(request: NextRequest): Promise<NextResponse> {
    try {
      const { authorized, user } = await this.verifyAuthToken(request);
      
      if (!authorized) {
        return NextResponse.json(
          { error: 'Not authenticated' },
          { status: 401 }
        );
      }

      // In a real implementation, add token to blacklist
      // For now, just log the logout
      console.log('User logout:', {
        userId: user?.id,
        timestamp: new Date().toISOString(),
        ip: this.security.getClientIP(request)
      });

      return NextResponse.json({ success: true });

    } catch (error) {
      console.error('Logout error:', error);
      return NextResponse.json(
        { error: 'Logout failed' },
        { status: 500 }
      );
    }
  }
}

// Security headers middleware
export function applySecurityHeaders(response: NextResponse): NextResponse {
  const securityHeaders = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
    'X-Permitted-Cross-Domain-Policies': 'none'
  };

  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  return response;
}

// Rate limiting configuration
export const createRateLimiter = (options: {
  windowMs: number;
  max: number;
  message?: string;
}) => {
  return rateLimit({
    windowMs: options.windowMs,
    max: options.max,
    message: options.message || 'Too many requests',
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req: any) => {
      const security = SecurityService.getInstance();
      return security.getRateLimitKey(req);
    }
  });
};

// Export instances
export const securityService = SecurityService.getInstance();
export const authMiddleware = new AuthenticationMiddleware();