import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

def execute(protocol_context):
    """
    Checks the health of a list of API endpoints.
    """
    api_endpoints = protocol_context.get("api_endpoints")
    if not api_endpoints:
        return {"status": "error", "message": "No API endpoints provided in the protocol context."}

    results = {}
    for endpoint in api_endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            results[endpoint] = {"status": "ok", "statusCode": response.status_code}
        except Timeout:
            results[endpoint] = {"status": "error", "message": "Request timed out"}
        except ConnectionError:
            results[endpoint] = {"status": "error", "message": "Could not connect to the server"}
        except RequestException as e:
            results[endpoint] = {"status": "error", "message": str(e)}

    return {"status": "completed", "results": results}
