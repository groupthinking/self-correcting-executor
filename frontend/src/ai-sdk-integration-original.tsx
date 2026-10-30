import { useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { openai } from '@ai-sdk/openai';

/**
 * AI SDK 5 Beta Integration Component
 * Demonstrates the new architecture with LanguageModelV2
 */
export function AISDKIntegration() {
  const [model, setModel] = useState<string>('gpt-4o-mini');
  
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/chat',
    initialMessages: [
      {
        id: 'system',
        role: 'system',
        content: 'You are a helpful AI assistant integrated with the self-correcting executor system.',
      },
    ],
  });

  return (
    <div className="ai-sdk-integration">
      <h2>AI SDK 5 Beta Integration</h2>
      
      <div className="model-selector">
        <label htmlFor="model-select">Model:</label>
        <select 
          id="model-select"
          value={model} 
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="gpt-4o-mini">GPT-4 Turbo</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          <option value="gpt-4o">GPT-4o</option>
        </select>
      </div>

      <div className="chat-container">
        <div className="messages">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <strong>{message.role}:</strong> {message.content}
            </div>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="input-form">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask the AI assistant..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Thinking...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}

/**
 * Advanced AI SDK 5 Beta Example with Tool Calling
 * Demonstrates the new message system and agentic control
 */
export function AdvancedAISDKExample() {
  const [systemPrompt, setSystemPrompt] = useState<string>(
    'You are an advanced AI assistant with access to quantum computing tools and MCP protocols.'
  );

  const { messages, input, handleInputChange, handleSubmit, isLoading, stop } = useChat({
    api: '/api/advanced-chat',
    initialMessages: [
      {
        id: 'system',
        role: 'system',
        content: systemPrompt,
      },
    ],
    // New AI SDK 5 Beta features
    experimental_prepareStep: (step) => {
      // Custom step preparation for fine-tuned model behavior
      return {
        ...step,
        metadata: {
          timestamp: new Date().toISOString(),
          sessionId: 'self-correcting-executor-session',
        },
      };
    },
    experimental_stopWhen: (message) => {
      // Define agent termination conditions
      return message.content.includes('[TASK_COMPLETE]') || 
             message.content.includes('[ERROR_CRITICAL]');
    },
  });

  const handleSystemPromptChange = (newPrompt: string) => {
    setSystemPrompt(newPrompt);
    // Update the system message in the chat
    // This leverages the new UIMessage/ModelMessage separation
  };

  return (
    <div className="advanced-ai-sdk">
      <h2>Advanced AI SDK 5 Beta Features</h2>
      
      <div className="system-prompt-editor">
        <label htmlFor="system-prompt">System Prompt:</label>
        <textarea
          id="system-prompt"
          value={systemPrompt}
          onChange={(e) => handleSystemPromptChange(e.target.value)}
          rows={3}
          placeholder="Define the AI assistant's behavior and capabilities..."
        />
      </div>

      <div className="chat-container advanced">
        <div className="messages">
          {messages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div className="message-header">
                <strong>{message.role}</strong>
                <small>{new Date(message.createdAt || '').toLocaleTimeString()}</small>
              </div>
              <div className="message-content">{message.content}</div>
              {message.toolInvocations && (
                <div className="tool-invocations">
                  <h4>Tool Calls:</h4>
                  {message.toolInvocations.map((tool, index) => (
                    <div key={index} className="tool-call">
                      <strong>{tool.toolName}:</strong> {JSON.stringify(tool.args)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="input-form advanced">
          <input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about quantum computing, MCP protocols, or system diagnostics..."
            disabled={isLoading}
          />
          <div className="form-actions">
            <button type="submit" disabled={isLoading}>
              {isLoading ? 'Processing...' : 'Send'}
            </button>
            {isLoading && (
              <button type="button" onClick={stop} className="stop-button">
                Stop
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

// CSS styles for the components
export const aiSDKStyles = `
.ai-sdk-integration, .advanced-ai-sdk {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.model-selector, .system-prompt-editor {
  margin-bottom: 20px;
}

.model-selector label, .system-prompt-editor label {
  display: block;
  margin-bottom: 5px;
  font-weight: 600;
}

.model-selector select, .system-prompt-editor textarea {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.chat-container {
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
}

.messages {
  height: 400px;
  overflow-y: auto;
  padding: 15px;
  background: #fafafa;
}

.message {
  margin-bottom: 15px;
  padding: 10px;
  border-radius: 6px;
  background: white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.message.system {
  background: #e3f2fd;
  border-left: 4px solid #2196f3;
}

.message.user {
  background: #f3e5f5;
  border-left: 4px solid #9c27b0;
}

.message.assistant {
  background: #e8f5e8;
  border-left: 4px solid #4caf50;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.message-content {
  line-height: 1.4;
}

.tool-invocations {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid #eee;
}

.tool-call {
  background: #f5f5f5;
  padding: 5px 8px;
  border-radius: 4px;
  margin: 5px 0;
  font-family: monospace;
  font-size: 12px;
}

.input-form {
  display: flex;
  padding: 15px;
  background: white;
  border-top: 1px solid #ddd;
}

.input-form input {
  flex: 1;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin-right: 10px;
  font-size: 14px;
}

.input-form button {
  padding: 10px 20px;
  background: #2196f3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.input-form button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.form-actions {
  display: flex;
  gap: 10px;
}

.stop-button {
  background: #f44336 !important;
}

.stop-button:hover {
  background: #d32f2f !important;
}
`;