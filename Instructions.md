# Contextual Thesaurus - Technical Documentation

## Prerequisites

- Python 3.7+
- 8GB RAM minimum
- Windows 10/11
- Visual Studio Code (recommended)

## Installation

1. Create and activate virtual environment:
```batch
python -m venv venv
.\venv\Scripts\activate
```

2. Install required packages:
```batch
pip install -r requirements.txt
```

3. Download GloVe embeddings:
   - Create directory: `glove.6B`
   - Download from: https://nlp.stanford.edu/data/glove.6B.zip
   - Extract `glove.6B.300d.txt` to `glove.6B` directory

## Project Structure
```
contextual-thesaurus/
│
├── static/
│   └── css/
│       └── style.css
├── templates/
│   └── index.html
├── app.py
├── word_similarity.py
└── requirements.txt
```

## API Endpoints

### GET /
- Web interface homepage
- Renders: `templates/index.html`

### GET /status
- Check model loading status
- Returns: `{"ready": boolean, "error": string|null}`

### POST /suggest
- Get word suggestions
- Body: 
  ```json
  {
    "sentence": "string",
    "word": "string",
    "top_n": number
  }
  ```
- Returns: Array of `[word, score, definition]` tuples

### GET /debug
- Get system status
- Returns: Detailed model state information

## Technical Details

### Models
- BERT: bert-base-uncased (~440MB)
- GloVe: 300d vectors (400K words)
- Cache location: `%USERPROFILE%\.cache\huggingface\transformers`

### Scoring System
- BERT contextual score: 70%
- GloVe similarity score: 30%
- Normalization: Cosine similarity

### Performance
- Request timeout: 30 seconds
- Max candidates: 100 words
- Definition cache: LRU with 1000 entries
- Memory usage: ~4GB during operation

## Development Guide

### Local Development
1. Start Flask server:
```batch
python app.py
```

2. Access application:
```
http://localhost:5000
```

### Debugging
1. Check logs:
   - Model loading: Debug level
   - API requests: Info level
   - Errors: Error level

2. Common Issues:
   - Memory errors: Check RAM usage
   - Timeout errors: Adjust timeout in app.py
   - API errors: Check network connectivity

### Testing
1. Run test script:
```batch
python test_similarity.py
```
