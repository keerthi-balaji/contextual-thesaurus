import requests
import json
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_word_similarity():
    test_data = {
        "sentence": "The happy dog played in the garden",
        "word": "happy",
        "top_n": 5
    }

    print("\nTesting word similarity functionality...")
    print(f"Test sentence: {test_data['sentence']}")
    print(f"Target word: {test_data['word']}\n")

    try:
        # Check model status
        status_response = requests.get('http://localhost:5000/status')
        status_data = status_response.json()
        
        print("Model Status:")
        print(f"Ready: {status_data.get('ready')}")
        print(f"Error: {status_data.get('error')}\n")

        if not status_data.get('ready'):
            print("Error: Models not ready")
            return

        # Get suggestions
        print("Requesting suggestions...")
        response = requests.post(
            'http://localhost:5000/suggest',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Response Status Code: {response.status_code}")
        
        try:
            result = response.json()
            print(f"Raw Response: {json.dumps(result, indent=2)}\n")
            
            if 'error' in result:
                print(f"Error from server: {result['error']}")
                return
                
            suggestions = result.get('suggestions', [])
            if not suggestions:
                print("No suggestions returned")
                return

            print("Suggestions:")
            print("-" * 50)
            for suggestion in suggestions:
                if isinstance(suggestion, (list, tuple)) and len(suggestion) == 3:
                    word, score, definition = suggestion
                    print(f"Word: {word}")
                    print(f"Score: {score:.4f}")
                    print(f"Definition: {definition}")
                    print("-" * 50)
                else:
                    print(f"Unexpected suggestion format: {suggestion}")
                    
        except json.JSONDecodeError as e:
            print(f"Error decoding response: {e}")
            print(f"Raw response content: {response.content}")

    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Make sure Flask is running.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logger.exception("Error during test")

if __name__ == '__main__':
    test_word_similarity()