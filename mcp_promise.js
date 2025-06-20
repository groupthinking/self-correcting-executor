
import fs from 'fs/promises';

let config = null;

async function loadConfig() {
  try {
    const data = await fs.readFile('./mcp.config.json', 'utf8');
    config = JSON.parse(data);
    console.log('[MCP] Configuration loaded successfully');
    return config;
  } catch (error) {
    console.error('[MCP] Error loading config:', error.message);
    return { 
      enabled: false, 
      defaultKeys: ["user", "task", "intent", "env", "code_state"],
      required: false 
    };
  }
}

// Initialize config
const configPromise = loadConfig();

export function getContext(req = {}) {
  // Allow context overrides from headers or query params
  const context = {
    user: req.headers?.['x-mcp-user'] || req.query?.user || req.user || 'anon',
    task: req.headers?.['x-mcp-task'] || req.query?.task || req.task || 'unspecified',
    intent: req.headers?.['x-mcp-intent'] || req.query?.intent || req.intent || 'unknown',
    env: process.env.NODE_ENV || 'dev',
    code_state: req.headers?.['x-mcp-code-state'] || req.query?.code_state || 'clean'
  };
  return context;
}

export function validateContext(ctx) {
  if (!config) {
    console.warn('[MCP] Config not loaded yet');
    return true; // Allow operations to continue if config isn't loaded
  }
  
  // Use a Set for faster lookups
  const requiredKeys = new Set(config.defaultKeys);
  const missingKeys = [];
  
  for (const key of requiredKeys) {
    if (!(key in ctx)) {
      missingKeys.push(key);
    }
  }
  
  if (missingKeys.length > 0) {
    console.warn(`[MCP] Missing context keys: ${missingKeys.join(', ')}`);
    return false;
  }
  
  return true;
}

export async function attachContext(req, res, next) {
  // Ensure config is loaded
  if (!config) {
    await configPromise;
  }
  
  if (!config.enabled) {
    next();
    return;
  }
  
  req.context = getContext(req);
  
  if (config.required && !validateContext(req.context)) {
    return res.status(400).json({ 
      error: 'Invalid MCP context', 
      missing: config.defaultKeys.filter(key => !(key in req.context))
    });
  }
  
  next();
}

export function withMCPContext(prompt, ctx) {
  if (!validateContext(ctx)) {
    throw new Error('Invalid MCP context');
  }
  return `Context:\n${JSON.stringify(ctx, null, 2)}\n\nPrompt:\n${prompt}`;
}
