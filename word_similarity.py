import os
import numpy as np
from typing import Dict, List, Tuple
import logging
import torch
from transformers import BertTokenizer, BertForMaskedLM
import nltk
import requests
from nltk.tokenize import word_tokenize
from tqdm import tqdm  # Add this import
from functools import lru_cache
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

class ContextualThesaurus:
    def __init__(self, glove_path: str):
        """Initialize with GloVe embeddings and BERT model."""
        logger.info("Initializing ContextualThesaurus...")
        self.embeddings = self.load_glove_embeddings(glove_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(self.device)
        self.model.eval()
        
        # Setup requests session with retries
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Initialize definition cache
        self._definition_cache = {}

        logger.info("Initialization complete")

    def load_glove_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load GloVe word vectors from file and normalize them."""
        logger.info(f"Loading GloVe embeddings from {filepath}")
        embeddings_dict = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # First count lines for progress bar
                total_lines = sum(1 for _ in open(filepath, encoding='utf-8'))
                f.seek(0)  # Reset file pointer
                
                for line in tqdm(f, total=total_lines, desc="Loading vectors"):
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        embeddings_dict[word] = vector / norm
            logger.info(f"Loaded {len(embeddings_dict)} word vectors")
            return embeddings_dict
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

    def get_contextual_suggestions(self, sentence: str, word: str, top_n: int = 5) -> List[Tuple[str, float, str]]:
        """Get context-aware word suggestions with definitions."""
        logger.debug(f"Getting suggestions for word '{word}' in sentence: {sentence}")
        
        if word not in self.embeddings:
            logger.warning(f"Word '{word}' not found in vocabulary")
            return []

        # Get similar words using GloVe first
        word_vector = self.embeddings[word]
        initial_candidates = []
        
        # First pass: Get top 100 words by GloVe similarity only
        for other_word, other_vector in self.embeddings.items():
            if other_word != word:
                try:
                    glove_score = np.dot(word_vector, other_vector)
                    initial_candidates.append((other_word, glove_score))
                except Exception as e:
                    logger.error(f"Error in GloVe scoring for {other_word}: {str(e)}")
                    continue
    
        # Sort and take top 100 by GloVe score
        initial_candidates.sort(key=lambda x: x[1], reverse=True)
        initial_candidates = initial_candidates[:100]  # Only process top 100 candidates
    
        # Second pass: Get BERT scores for top GloVe candidates
        scored_words = []
        for other_word, glove_score in tqdm(initial_candidates, desc="Processing candidates"):
            try:
                bert_score = self.get_bert_score(sentence, word, other_word)
                combined_score = 0.3 * glove_score + 0.7 * bert_score
                scored_words.append((other_word, combined_score))
            except Exception as e:
                logger.error(f"Error processing word {other_word}: {str(e)}")
                continue

        # Sort by combined score
        scored_words.sort(key=lambda x: x[1], reverse=True)
        top_words = scored_words[:top_n * 2]

        # Get definitions for top candidates
        similarities = []
        for word, score in top_words:
            try:
                definition = self.get_word_definition(word)
                if definition != "Definition not found":
                    similarities.append((word, score, definition))
                    if len(similarities) >= top_n:
                        break
            except Exception as e:
                logger.error(f"Error getting definition for {word}: {str(e)}")
                continue

        logger.debug(f"Found {len(similarities)} suggestions with definitions")
        return similarities[:top_n]

    def get_bert_score(self, sentence: str, word: str, candidate: str) -> float:
        """Calculate BERT score for candidate word in sentence context."""
        try:
            # Replace the target word with MASK token
            words = word_tokenize(sentence)
            word_idx = words.index(word)
            words[word_idx] = self.tokenizer.mask_token
            masked_sentence = ' '.join(words)

            # Encode the sentence
            inputs = self.tokenizer(masked_sentence, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            # Get probability for candidate word
            masked_idx = inputs['input_ids'][0].tolist().index(self.tokenizer.mask_token_id)
            candidate_id = self.tokenizer.convert_tokens_to_ids(candidate)
            probability = torch.softmax(predictions[0, masked_idx], dim=0)[candidate_id].item()
            
            return probability
        except Exception as e:
            logger.error(f"Error in BERT scoring: {str(e)}")
            return 0.0

    @lru_cache(maxsize=1000)
    def get_word_definition(self, word: str) -> str:
        """Get word definition from Free Dictionary API with caching and retries."""
        # Check memory cache first
        if word in self._definition_cache:
            return self._definition_cache[word]

        try:
            response = self.session.get(
                f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}",
                timeout=5  # 5 second timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                definition = data[0]['meanings'][0]['definitions'][0]['definition']
                # Cache the result
                self._definition_cache[word] = definition
                return definition
            elif response.status_code == 429:  # Rate limit
                logger.warning("Rate limited by dictionary API, waiting...")
                time.sleep(2)  # Wait 2 seconds before retry
                return self.get_word_definition(word)  # Retry once
            else:
                return "Definition not found"
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout getting definition for {word}")
            return "Definition not found"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error getting definition for {word}: {str(e)}")
            return "Definition not found"
        except Exception as e:
            logger.error(f"Error getting definition for {word}: {str(e)}")
            return "Definition not found"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Context-aware word replacement suggestions')
    parser.add_argument('--glove-path', type=str, 
                       default=os.path.join(os.path.dirname(__file__), "glove.6B", "glove.6B.300d.txt"),
                       help='Path to GloVe vectors file')
    parser.add_argument('--sentence', type=str, required=True,
                       help='Input sentence')
    parser.add_argument('--word', type=str, required=True,
                       help='Word to replace')
    parser.add_argument('--top-n', type=int, default=5,
                       help='Number of suggestions to return')
    
    args = parser.parse_args()
    
    thesaurus = ContextualThesaurus(args.glove_path)
    suggestions = thesaurus.get_contextual_suggestions(args.sentence, args.word, args.top_n)
    
    if not suggestions:
        print(f"\nNo suggestions found for '{args.word}'")
    else:
        print(f"\nSuggestions for '{args.word}' in context:")
        print("-" * 80)
        print(f"{'Word':<20} {'Score':<10} {'Definition'}")
        print("-" * 80)
        for word, score, definition in suggestions:
            print(f"{word:<20} {score:.4f}    {definition[:60]}...")