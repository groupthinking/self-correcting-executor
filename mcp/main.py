from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="UMDR MCP Server", version="1.0.0")


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "umdr-mcp-server",
        "version": "1.0.0",
    } 