# contextual-thesaurus

A Python tool for finding semantically similar words using GloVe word embeddings.

## Setup

1. Download GloVe vectors from Stanford NLP's website:
   - Visit: https://nlp.stanford.edu/projects/glove/
   - Download the pre-trained word vectors (e.g., glove.6B.zip)

2. Install requirements:
```bash
pip install numpy
```

3. Extract the GloVe vectors file (e.g., glove.6B.100d.txt) to your project directory

## Usage

```python
from word_similarity import load_glove_embeddings, get_similar_words

# Load the embeddings (only needs to be done once)
embeddings = load_glove_embeddings("glove.6B.100d.txt")

# Find similar words
similar_words = get_similar_words("happy", embeddings, top_n=5)

# Print results
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
```

## Notes

- The first time loading the embeddings may take a few minutes
- Make sure you have enough RAM to load the entire embeddings file
- Vectors are normalized during loading for efficient similarity calculations