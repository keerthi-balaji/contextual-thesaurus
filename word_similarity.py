import os
import numpy as np
from typing import Dict, List, Tuple
import logging
import tqdm  # Add progress bar for loading embeddings

def load_glove_embeddings(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load GloVe word vectors from file and normalize them.
    
    Args:
        filepath: Path to GloVe vectors file
        
    Returns:
        Dictionary mapping words to their normalized vectors
    """
    embeddings_dict = {}
    try:
        # Count total lines first for progress bar
        num_lines = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f, total=num_lines, desc="Loading vectors"):
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                # Normalize the vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    embeddings_dict[word] = vector / norm
    except FileNotFoundError:
        logging.error(f"GloVe file not found at: {filepath}")
        print(f"\nPlease ensure the file exists at: {filepath}")
        print("You can download GloVe embeddings from: https://nlp.stanford.edu/data/glove.6B.zip")
    except Exception as e:
        logging.error(f"Error loading GloVe vectors: {str(e)}")
    
    return embeddings_dict

def get_similar_words(word: str, embeddings_dict: Dict[str, np.ndarray], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Find the top N most similar words to the input word using cosine similarity.
    
    Args:
        word: Input word to find similarities for
        embeddings_dict: Dictionary of word embeddings
        top_n: Number of similar words to return
        
    Returns:
        List of tuples containing (word, similarity_score)
    """
    if word not in embeddings_dict:
        return []
    
    word_vector = embeddings_dict[word]
    similarities = []
    
    for other_word, other_vector in embeddings_dict.items():
        if other_word != word:
            similarity = np.dot(word_vector, other_vector)
            similarities.append((other_word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    import argparse
    
    # Get default path using the correct folder structure
    default_path = os.path.join(
        os.path.dirname(__file__), 
        "glove.6B",
        "glove.6B.300d.txt"
    )
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Find semantically similar words using GloVe embeddings')
    parser.add_argument('--glove-path', type=str, 
                       default=default_path,
                       help='Path to GloVe vectors file')
    parser.add_argument('--word', type=str, required=True,
                       help='Word to find similarities for')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of similar words to return')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.glove_path):
        print(f"Error: File not found at {args.glove_path}")
        print("\nMake sure you have:")
        print("1. Downloaded GloVe vectors from https://nlp.stanford.edu/data/glove.6B.zip")
        print("2. Extracted them to the glove.6B directory")
        print("3. Using the correct dimension size (50d, 100d, 200d, or 300d)")
        exit(1)
    
    # Load embeddings
    print(f"Loading GloVe embeddings from {args.glove_path}...")
    embeddings = load_glove_embeddings(args.glove_path)
    print(f"Loaded {len(embeddings)} word vectors")
    
    # Get similar words
    similar_words = get_similar_words(args.word.lower(), embeddings, args.top_n)
    
    if not similar_words:
        print(f"\nWord '{args.word}' not found in vocabulary")
        print("Try another word or check spelling")
    else:
        print(f"\nWords most similar to '{args.word}':")
        print("-" * 40)
        print(f"{'Word':<20} Similarity")
        print("-" * 40)
        for word, score in similar_words:
            print(f"{word:<20} {score:.4f}")