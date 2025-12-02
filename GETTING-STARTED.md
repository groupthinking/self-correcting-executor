# How to View and Interact with Your AI SDK 5 Beta Application

## 🚀 Quick Start Guide

### **Option 1: Frontend Development Server (Recommended)**

1. **Navigate to the frontend directory:**
```bash
cd /Users/garvey/self-correcting-executor-local/frontend
```

2. **Install dependencies (if not already done):**
```bash
npm install
```

3. **Start the development server:**
```bash
npm run dev
```

4. **Open your browser:**
   - Go to: `http://localhost:5173`
   - You'll see the enhanced navigation with new AI SDK 5 tabs

### **What You'll See:**
- **Dashboard**: Main system overview
- **AI SDK 5 Beta**: Basic AI chat interface
- **Advanced AI Features**: Full conversation hub with metrics
- **Components**: Component management
- **Patterns**: Pattern visualization

---

## 🎯 Available Components to Interact With

### **1. AI Conversation Hub**
**Location**: Click "Advanced AI Features" in navigation

**Features to Try:**
- **Real-time chat** with AI models
- **Model selection** (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- **Temperature control** and configuration
- **Tool calling** with quantum and MCP tools
- **Export conversations** to JSON
- **Live metrics** tracking (response time, tokens, tool calls)
- **Recording mode** for session analysis

### **2. Monitoring Dashboard**
**Location**: Will be added to navigation or accessible directly

**Real-time Features:**
- **System Health**: CPU, Memory, Disk, Network metrics
- **AI Performance**: Model usage and response times
- **Security Status**: Threat monitoring and compliance
- **Live Alerts**: Real-time system notifications
- **Performance Graphs**: Interactive charts and visualizations

### **3. AI SDK 5 Beta Basic Interface**
**Location**: Click "AI SDK 5" in navigation

**Features:**
- **Simple chat interface** with AI SDK 5 Beta
- **Model configuration** panel
- **Basic tool calling** demonstration
- **Streaming responses** with real-time updates

---

## 🔧 Backend Services Setup

### **1. Start the WebSocket Streaming Service**

```bash
cd /Users/garvey/self-correcting-executor-local/backend
npm install ws jsonwebtoken bcryptjs helmet express-rate-limit
node -r ts-node/register services/streaming-service.ts
```

**What it provides:**
- **Real-time chat streaming**
- **WebSocket connections** on port 8080
- **Tool calling** broadcast capabilities
- **Multi-user session** support

### **2. Test the Tool Registry**

```bash
cd /Users/garvey/self-correcting-executor-local/backend
node -r ts-node/register tools/tool-registry.ts
```

**Available Tools:**
- **quantum_analyzer**: Quantum computing problem analysis
- **mcp_connector**: MCP server management
- **system_diagnostics**: Health checks and monitoring
- **secure_code_generator**: AI-powered code generation
- **security_audit**: Vulnerability scanning
- **data_analyzer**: Pattern recognition and analytics

### **3. Start Database Services**

```bash
cd /Users/garvey/self-correcting-executor-local/backend
node -r ts-node/register database/conversation-store.ts
```

**Features:**
- **Conversation persistence**
- **Message storage** with metadata
- **Search functionality**
- **Analytics and reporting**
- **Export capabilities**

---

## 🌐 Full Application Setup

### **Complete Development Environment:**

1. **Frontend Server:**
```bash
cd frontend && npm run dev
# Access: http://localhost:5173
```

2. **Backend API Server:**
```bash
cd backend && npm run dev
# API: http://localhost:3001
```

3. **WebSocket Service:**
```bash
cd backend && node services/streaming-service.ts
# WebSocket: ws://localhost:8080
```

4. **Database Service:**
```bash
cd backend && node database/conversation-store.ts
# Database: In-memory with persistence
```

---

## 📱 Mobile/Responsive Access

**The application is fully responsive and works on:**
- **Desktop browsers** (Chrome, Firefox, Safari, Edge)
- **Mobile devices** (iOS Safari, Android Chrome)
- **Tablets** (iPad, Android tablets)
- **Different screen sizes** with adaptive layouts

---

## 🔑 Authentication Setup

### **For Full Feature Access:**

1. **Create environment file:**
```bash
cd backend
echo "JWT_SECRET=your-super-secure-secret-key" > .env
echo "ENCRYPTION_KEY=your-encryption-key" >> .env
```

2. **Test authentication:**
```bash
# Register a test user
curl -X POST http://localhost:3001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123!","firstName":"Test","lastName":"User"}'
```

3. **Login and get token:**
```bash
# Login to get JWT token
curl -X POST http://localhost:3001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"SecurePass123!"}'
```

---

## 🧪 Testing Individual Components

### **1. Test AI SDK 5 Beta Integration:**
```typescript
// In browser console on localhost:5173
// Open Developer Tools > Console and run:
fetch('/api/ai-conversation', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_JWT_TOKEN'
  },
  body: JSON.stringify({
    messages: [
      { role: 'user', content: 'Explain quantum computing' }
    ],
    config: {
      model: 'gpt-4o',
      temperature: 0.7,
      enableTools: true,
      enableQuantum: true
    }
  })
});
```

### **2. Test WebSocket Connection:**
```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8080?token=YOUR_JWT_TOKEN');
ws.onopen = () => console.log('Connected to streaming service');
ws.onmessage = (event) => console.log('Received:', JSON.parse(event.data));
ws.send(JSON.stringify({
  id: 'msg_1',
  type: 'chat',
  payload: {
    role: 'user',
    content: 'Hello from WebSocket!'
  }
}));
```

### **3. Test Tool Calling:**
```bash
# Test quantum analyzer tool
curl -X POST http://localhost:3001/api/tools/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "toolName": "quantum_analyzer",
    "args": {
      "problem": "Optimize quantum circuit for factorization",
      "algorithm": "shor",
      "qubits": 8
    }
  }'
```

---

## 📊 Monitoring and Analytics

### **Real-time Monitoring Access:**
1. **Navigate to monitoring dashboard** (will be added to nav)
2. **View live metrics** for system performance
3. **Check AI performance** statistics
4. **Monitor security** events and compliance
5. **Set up alerts** for critical thresholds

### **Analytics Features:**
- **Conversation analytics** with user patterns
- **Tool usage statistics** and performance
- **System resource** monitoring
- **Security event** tracking
- **Performance optimization** insights

---

## 🛠️ Development Tools

### **Browser Developer Tools:**
- **Network tab**: Monitor API calls and WebSocket connections
- **Console**: Test JavaScript functions and APIs
- **Application tab**: View localStorage, sessionStorage, and cookies
- **Sources tab**: Debug TypeScript/JavaScript code

### **API Testing:**
- **Postman**: Import API collection for testing endpoints
- **cURL**: Command-line testing of REST APIs
- **WebSocket King**: Test WebSocket connections
- **Browser DevTools**: Network monitoring and debugging

---

## 🎯 Next Steps to Get Started

### **Immediate Actions:**

1. **Start the frontend server:**
```bash
cd frontend && npm run dev
```

2. **Open browser to:**
   - `http://localhost:5173`

3. **Explore the interfaces:**
   - Click through navigation tabs
   - Try the AI conversation features
   - Test different models and configurations

4. **Check the console:**
   - Open Developer Tools (F12)
   - Monitor network requests
   - View any error messages

### **For Full Experience:**

1. **Set up backend services** (optional for basic UI testing)
2. **Configure authentication** for secure features
3. **Enable WebSocket streaming** for real-time updates
4. **Test tool calling** with quantum and MCP tools

---

## 📞 Troubleshooting

### **Common Issues:**

1. **Port conflicts:**
   - Frontend: Change port in `vite.config.ts`
   - Backend: Use different port numbers

2. **Missing dependencies:**
   - Run `npm install` in both frontend and backend directories

3. **CORS issues:**
   - Configure proxy in `vite.config.ts` for API calls

4. **WebSocket connection fails:**
   - Check JWT token validity
   - Verify WebSocket service is running

5. **AI API errors:**
   - Ensure OpenAI API key is configured
   - Check rate limits and quota

**Get started with the frontend development server to see the complete UI immediately!**