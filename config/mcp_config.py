"""MCP Configuration - Real Endpoints Only, No Mocks"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

class MCPConfig:
    """
    MCP configuration with real endpoints only.
    No mock URLs, no simulated services.
    """
    
    @staticmethod
    def get_endpoints() -> Dict[str, str]:
        """Get real MCP endpoints from environment or defaults"""
        return {
            # Core MCP Server
            'mcp_server': os.getenv('MCP_SERVER_URL', 'http://localhost:8090'),
            
            # Quantum Computing
            'quantum_api': os.getenv('DWAVE_API_URL', 'https://cloud.dwavesys.com/sapi/v2'),
            'quantum_solver_api': os.getenv('DWAVE_SOLVER_API', 'https://cloud.dwavesys.com/sapi/v2/solvers/remote'),
            
            # Data Services
            'data_api': os.getenv('DATA_API_URL', 'http://localhost:8091/api'),
            'data_storage': os.getenv('DATA_STORAGE_URL', 'http://localhost:8092/storage'),
            
            # Database Services
            'postgres_url': os.getenv('DATABASE_URL', 'postgresql://localhost:5432/mcp_db'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            
            # AI/ML Services
            'inference_api': os.getenv('INFERENCE_API_URL', 'http://localhost:8093/inference'),
            'training_api': os.getenv('TRAINING_API_URL', 'http://localhost:8094/training'),
            
            # Monitoring & Logging
            'metrics_api': os.getenv('METRICS_API_URL', 'http://localhost:9090/metrics'),
            'logging_api': os.getenv('LOGGING_API_URL', 'http://localhost:9091/logs'),
        }
    
    @staticmethod
    def get_auth_config() -> Dict[str, Any]:
        """Get authentication configuration"""
        return {
            'jwt_secret': os.getenv('JWT_SECRET_KEY', ''),
            'api_key': os.getenv('API_SECRET_KEY', ''),
            'dwave_token': os.getenv('DWAVE_API_TOKEN', ''),
            'auth_enabled': os.getenv('AUTH_ENABLED', 'true').lower() == 'true'
        }
    
    @staticmethod
    def get_quantum_config() -> Dict[str, Any]:
        """Get quantum computing configuration"""
        return {
            'dwave_token': os.getenv('DWAVE_API_TOKEN', ''),
            'solver_name': os.getenv('DWAVE_SOLVER_NAME', ''),  # e.g., "Advantage_system6.4"
            'default_num_reads': int(os.getenv('QUANTUM_NUM_READS', '1000')),
            'annealing_time': int(os.getenv('QUANTUM_ANNEALING_TIME', '20')),  # microseconds
            'require_qpu': os.getenv('REQUIRE_QPU', 'true').lower() == 'true',  # No simulations
        }
    
    @staticmethod
    def get_data_config() -> Dict[str, Any]:
        """Get data processing configuration"""
        return {
            'data_dir': os.getenv('DATA_DIR', str(Path('./data').absolute())),
            'cache_dir': os.getenv('CACHE_DIR', str(Path('./cache').absolute())),
            'temp_dir': os.getenv('TEMP_DIR', '/tmp/mcp'),
            'max_file_size': int(os.getenv('MAX_FILE_SIZE', str(100 * 1024 * 1024))),  # 100MB
            'allowed_extensions': os.getenv('ALLOWED_EXTENSIONS', '.json,.csv,.txt,.parquet').split(',')
        }
    
    @staticmethod
    def get_security_config() -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'encryption_key': os.getenv('ENCRYPTION_KEY', ''),
            'enable_sandboxing': os.getenv('ENABLE_SANDBOXING', 'true').lower() == 'true',
            'max_execution_time': int(os.getenv('MAX_EXECUTION_TIME', '300')),  # seconds
            'allowed_hosts': os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(','),
            'cors_origins': os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
        }
    
    @staticmethod
    def validate_config() -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation = {}
        
        # Check required environment variables
        required_vars = [
            ('DWAVE_API_TOKEN', 'Quantum computing'),
            ('DATABASE_URL', 'Database connection'),
            ('JWT_SECRET_KEY', 'Authentication'),
            ('ENCRYPTION_KEY', 'Data encryption')
        ]
        
        for var, feature in required_vars:
            validation[feature] = bool(os.getenv(var))
        
        # Check endpoint accessibility (basic check)
        endpoints = MCPConfig.get_endpoints()
        for name, url in endpoints.items():
            if url and not url.startswith(('http://', 'https://', 'redis://', 'postgresql://')):
                validation[f"{name}_valid_url"] = False
            else:
                validation[f"{name}_valid_url"] = True
        
        return validation
    
    @staticmethod
    def get_mcp_server_config() -> Dict[str, Any]:
        """Get MCP server specific configuration"""
        return {
            'host': os.getenv('MCP_SERVER_HOST', '0.0.0.0'),
            'port': int(os.getenv('MCP_SERVER_PORT', '8090')),
            'workers': int(os.getenv('MCP_SERVER_WORKERS', '4')),
            'transport': os.getenv('MCP_TRANSPORT', 'stdio'),  # stdio, websocket, http
            'enable_tools': True,
            'enable_resources': True,
            'enable_prompts': True,
            'max_request_size': int(os.getenv('MAX_REQUEST_SIZE', str(10 * 1024 * 1024))),  # 10MB
        }
    
    @staticmethod
    def create_env_template() -> str:
        """Create a template .env file with all required variables"""
        template = """# MCP Configuration - Real Services Only

# Core MCP Server
MCP_SERVER_URL=http://localhost:8090
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8090
MCP_SERVER_WORKERS=4
MCP_TRANSPORT=stdio

# Quantum Computing (D-Wave)
DWAVE_API_TOKEN=your-dwave-api-token-here
DWAVE_API_URL=https://cloud.dwavesys.com/sapi/v2
DWAVE_SOLVER_NAME=Advantage_system6.4
QUANTUM_NUM_READS=1000
QUANTUM_ANNEALING_TIME=20
REQUIRE_QPU=true

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/mcp_db
REDIS_URL=redis://localhost:6379

# Data Services
DATA_API_URL=http://localhost:8091/api
DATA_STORAGE_URL=http://localhost:8092/storage
DATA_DIR=./data
CACHE_DIR=./cache
TEMP_DIR=/tmp/mcp
MAX_FILE_SIZE=104857600
ALLOWED_EXTENSIONS=.json,.csv,.txt,.parquet

# AI/ML Services
INFERENCE_API_URL=http://localhost:8093/inference
TRAINING_API_URL=http://localhost:8094/training

# Security
JWT_SECRET_KEY=your-jwt-secret-min-32-chars
API_SECRET_KEY=your-api-secret-min-32-chars
ENCRYPTION_KEY=your-256-bit-encryption-key
AUTH_ENABLED=true
ENABLE_SANDBOXING=true
MAX_EXECUTION_TIME=300
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000

# Monitoring
METRICS_API_URL=http://localhost:9090/metrics
LOGGING_API_URL=http://localhost:9091/logs

# Environment
ENVIRONMENT=development
"""
        return template
    
    @staticmethod
    def check_no_mocks() -> bool:
        """Verify no mock URLs or endpoints are configured"""
        endpoints = MCPConfig.get_endpoints()
        
        # List of mock indicators
        real_indicators = ['mock', 'fake', 'simulated', 'actual', 'example.com', 'test']
        
        for name, url in endpoints.items():
            if url:
                url_lower = url.lower()
                for indicator in real_indicators:
                    if indicator in url_lower:
                        raise ValueError(
                            f"Mock URL detected in {name}: {url}\n"
                            "This codebase requires real endpoints only. No mocks allowed."
                        )
        
        return True


# Convenience functions
def get_mcp_config() -> MCPConfig:
    """Get MCP configuration instance"""
    return MCPConfig()

def validate_environment() -> None:
    """Validate environment and raise errors if configuration is invalid"""
    config = MCPConfig()
    
    # Check for mocks
    config.check_no_mocks()
    
    # Validate configuration
    validation = config.validate_config()
    
    missing_features = [feature for feature, valid in validation.items() if not valid]
    
    if missing_features:
        print("⚠️  Configuration Warning:")
        print(f"Missing configuration for: {', '.join(missing_features)}")
        print("Some features may not work properly.")
        print("\nRun this to generate .env template:")
        print("python -m config.mcp_config > .env.template")


if __name__ == "__main__":
    # Generate env template when run directly
    print(MCPConfig.create_env_template())