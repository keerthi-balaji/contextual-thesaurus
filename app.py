from flask import Flask, request, jsonify, render_template
from word_similarity import ContextualThesaurus
import os
import traceback
import logging
import time  # Import time module for timeout

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

app = Flask(__name__)

# Global variables to track model status
model_ready = False
model_error = None
thesaurus = None
initialization_started = False

def initialize_models():
    global model_ready, model_error, thesaurus, initialization_started
    if initialization_started:
        return
    
    initialization_started = True
    try:
        logger.info("Loading models... This may take a few minutes on first run...")
        glove_path = os.path.join("glove.6B", "glove.6B.300d.txt")
        
        if not os.path.exists(glove_path):
            raise FileNotFoundError(f"GloVe embeddings not found at: {glove_path}")
        
        thesaurus = ContextualThesaurus(glove_path)
        model_ready = True
        logger.info("Models loaded successfully!")
    except FileNotFoundError as e:
        model_error = str(e)
        logger.error(f"File not found error: {e}")
        logger.error("Please ensure you have downloaded and extracted the GloVe embeddings.")
    except Exception as e:
        model_error = str(e)
        logger.error(f"Error loading models: {e}")
        logger.error(traceback.format_exc())

@app.before_request
def before_request():
    """Initialize models before handling any request"""
    if not model_ready and not model_error:
        initialize_models()

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/status')
def status():
    """Check model loading status"""
    return jsonify({
        'ready': model_ready,
        'error': model_error
    })

@app.route('/suggest', methods=['POST'])
def suggest():
    """Generate word suggestions based on context"""
    if not model_ready:
        return jsonify({
            'error': 'Models not ready. Please wait or refresh the page.'
        }), 503
    
    try:
        data = request.json
        sentence = data.get('sentence', '').strip()
        word = data.get('word', '').strip()
        top_n = int(data.get('top_n', 5))

        logger.debug(f"Processing request - sentence: {sentence}, word: {word}, top_n: {top_n}")

        if not sentence or not word:
            return jsonify({
                'error': 'Please provide both sentence and word.'
            }), 400

        start_time = time.time()
        timeout = 30  # 30 seconds timeout
        
        suggestions = thesaurus.get_contextual_suggestions(sentence, word, top_n)
        
        if time.time() - start_time > timeout:
            return jsonify({
                'error': 'Request timed out. Please try again.'
            }), 504

        # Convert NumPy float32 to Python float
        formatted_suggestions = [
            (word, float(score), definition) 
            for word, score, definition in suggestions
        ]

        return jsonify({'suggestions': formatted_suggestions})

    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

# Add debug route for better visibility
@app.route('/debug', methods=['GET'])
def debug_info():
    """Get debug information about the current state"""
    return jsonify({
        'model_ready': model_ready,
        'model_error': model_error,
        'thesaurus_initialized': thesaurus is not None,
        'embeddings_loaded': thesaurus.embeddings is not None if thesaurus else False,
        'bert_loaded': hasattr(thesaurus, 'model') if thesaurus else False,
        'test_word_in_vocab': 'happy' in thesaurus.embeddings if thesaurus and thesaurus.embeddings else False,
        'embeddings_size': len(thesaurus.embeddings) if thesaurus and thesaurus.embeddings else 0,
        'sample_words': list(thesaurus.embeddings.keys())[:5] if thesaurus and thesaurus.embeddings else []
    })

if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    app.run(debug=True)