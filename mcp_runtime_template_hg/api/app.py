from fastapi import FastAPI
app = FastAPI()

@app.post('/mcp/agent/run')
def run_agent(payload: dict):
    return {"status": "Agent run requested", "payload": payload}
