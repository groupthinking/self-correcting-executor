# Real Protocol: API Health Checker
import requests
import time

def task():
    """Check health of various API endpoints"""
    endpoints = [
        {'name': 'Local API', 'url': 'http://localhost:8080/health'},
        {'name': 'JSONactual', 'url': 'https://jsonactual.typicode.com/posts/1'},
        {'name': 'GitHub API', 'url': 'https://api.github.com/rate_limit'}
    ]
    
    results = []
    failures = 0
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(endpoint['url'], timeout=5)
            response_time = (time.time() - start_time) * 1000  # ms
            
            results.append({
                'name': endpoint['name'],
                'status': response.status_code,
                'response_time_ms': round(response_time, 2),
                'healthy': response.status_code == 200
            })
            
            if response.status_code != 200:
                failures += 1
                
        except Exception as e:
            failures += 1
            results.append({
                'name': endpoint['name'],
                'error': str(e),
                'healthy': False
            })
    
    return {
        'success': failures < len(endpoints) / 2,  # Success if less than half failed
        'action': 'api_health_check',
        'total_endpoints': len(endpoints),
        'healthy_count': len(endpoints) - failures,
        'failure_count': failures,
        'results': results
    } 