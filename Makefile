.PHONY: up down build logs health test clean quantum quantum-down quantum-logs quantum-test setup-dwave test-debug watch

# Default target
default: up

# Standard Development Commands
up:
	@echo "🚀 Starting development stack..."
	docker-compose up -d
	@echo "✅ Stack is running!"

down:
	@echo "🛑 Stopping development stack..."
	docker-compose down

build:
	@echo "🔨 Building containers..."
	docker-compose build

logs:
	docker-compose logs -f --tail=100

health:
	curl -f http://localhost:8080/health || echo "Health check failed"

test:
	@echo "🧪 Running standard tests..."
	docker-compose exec app python -m pytest tests/ -v

# Quantum Development Commands 
quantum:
	@echo "⚛️  Starting quantum development stack..."
	docker-compose -f docker-compose.quantum.yml up -d
	@echo "✅ Quantum stack is running!"
	@echo "📊 Frontend: http://localhost:3000"
	@echo "🔧 Backend: http://localhost:8000"
	@echo "⚛️  MCP Quantum: http://localhost:8001"
	@echo "📝 Logs: make quantum-logs"

quantum-down:
	@echo "🛑 Stopping quantum stack..."
	docker-compose -f docker-compose.quantum.yml down -v
	@echo "✅ Quantum stack stopped!"

quantum-logs:
	docker-compose -f docker-compose.quantum.yml logs -f --tail=100

quantum-test:
	@echo "⚛️  Running quantum tests..."
	docker-compose -f docker-compose.quantum.yml exec quantum-dev python -m pytest test_real_dwave_quantum.py -v
	@echo "✅ Quantum tests complete!"

quantum-build:
	@echo "⚛️  Building quantum containers..."
	docker-compose -f docker-compose.quantum.yml build

# D-Wave Setup Commands
setup-dwave:
	@echo "⚛️  Setting up D-Wave Ocean SDK..."
	@echo "📋 Please ensure you have a D-Wave Leap account"
	@echo "🔗 Visit: https://cloud.dwavesys.com/leap/"
	docker-compose -f docker-compose.quantum.yml exec quantum-dev dwave setup --auth
	@echo "✅ D-Wave setup complete!"

verify-quantum:
	@echo "⚛️  Verifying quantum connection..."
	docker-compose -f docker-compose.quantum.yml exec quantum-dev dwave ping --client qpu
	docker-compose -f docker-compose.quantum.yml exec quantum-dev dwave solvers --list

# Utility Commands
clean:
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker-compose -f docker-compose.quantum.yml down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup complete!"

clean-quantum:
	@echo "🧹 Cleaning quantum containers..."
	docker-compose -f docker-compose.quantum.yml down -v --remove-orphans
	docker rmi $(shell docker images "*quantum*" -q) 2>/dev/null || true
	@echo "✅ Quantum cleanup complete!"

test-debug:
	python test_mcp_debug_simple.py

watch:
	python guardian_linter_watchdog.py 