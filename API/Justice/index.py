from http.server import BaseHTTPRequestHandler
import json

def handler(request, response):
    response.send(json.dumps({"message": "Justice API is working!"}))
    return
