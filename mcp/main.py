from fastapi import FastAPI, BackgroundTasks
from datetime import datetime
import sys
import os

# Add the project root to Python path for imports
sys.path.append('/app')

app = FastAPI(title="UMDR MCP Server", version="1.0.0")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "UMDR - Unified MCP DevContainer & Runtime",
        "version": "1.0.0", 
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "umdr-mcp-server",
        "version": "1.0.0",
    }


@app.post("/api/v1/execute")
async def execute_protocol(protocol_name: str = "default_protocol", iterations: int = 1):
    """Execute a self-correcting protocol"""
    try:
        # Import here to avoid circular imports
        from main import run_self_correcting_executor
        
        result = run_self_correcting_executor(protocol_name, iterations)
        return {
            "status": "completed",
            "protocol": protocol_name,
            "iterations": iterations,
            "final_stats": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@app.get("/api/v1/protocols")
async def list_protocols():
    """List all available protocols"""
    try:
        from protocols.loader import list_protocols
        protocols = list_protocols()
        return {
            "protocols": protocols,
            "count": len(protocols),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@app.get("/api/v1/stats")
async def get_stats(protocol_name: str = ""):
    """Get execution statistics for protocols"""
    try:
        from utils.tracker import get_protocol_stats, get_all_stats
        
        if protocol_name:
            stats = get_protocol_stats(protocol_name)
            return {
                "stats": stats,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        else:
            all_stats = get_all_stats()
            return {
                "all_stats": all_stats,
                "total_protocols": len(all_stats),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


@app.post("/api/v1/mutate")
async def force_mutation(protocol_name: str):
    """Force mutation of a specific protocol"""
    try:
        from agents.mutator import mutate_protocol
        
        result = mutate_protocol(protocol_name)
        return {
            "status": "completed",
            "protocol": protocol_name,
            "mutated": result,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        } 