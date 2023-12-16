import requests
import json

class Chatbot:
    def __init__(self, chat_endpoint):
        self.chat_endpoint = chat_endpoint

    def run(self, query, history: list[dict] = []):
        new_message = {"user": query}
        history.append(new_message)
        
        data = {
            "approach": "ms",
            "history": history
        }
        
        headers = {'Content-type': 'application/json'}
        response = requests.post(self.chat_endpoint, data=json.dumps(data), headers=headers)
        
        response_json = json.loads(response.content)

        return response_json
