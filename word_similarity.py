import os
import numpy as np
from typing import Dict, List, Tuple
import logging
import tqdm
import torch
from transformers import BertTokenizer, BertForMaskedLM
import nltk
import requests
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt')

class ContextualThesaurus:
    def __init__(self, glove_path: str):
        """Initialize with GloVe embeddings and BERT model."""
        self.embeddings = self.load_glove_embeddings(glove_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(self.device)
        self.model.eval()

    def load_glove_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
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

    def get_word_definition(self, word: str) -> str:
        """Get word definition from Free Dictionary API."""
        try:
            response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
            if response.status_code == 200:
                data = response.json()
                return data[0]['meanings'][0]['definitions'][0]['definition']
            return "Definition not found"
        except:
            return "Definition not found"

    def get_bert_score(self, sentence: str, word: str, candidate: str) -> float:
        """Calculate BERT score for candidate word in sentence context."""
        # Replace the target word with MASK token
        words = word_tokenize(sentence)
        try:
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
        except:
            return 0.0

    def get_contextual_suggestions(self, sentence: str, word: str, top_n: int = 5) -> List[Tuple[str, float, str]]:
        """Get context-aware word suggestions with definitions."""
        if word not in self.embeddings:
            return []

        # Get similar words using GloVe
        word_vector = self.embeddings[word]
        similarities = []
        
        for other_word, other_vector in self.embeddings.items():
            if other_word != word:
                glove_score = np.dot(word_vector, other_vector)
                bert_score = self.get_bert_score(sentence, word, other_word)
                # Combine GloVe and BERT scores
                combined_score = 0.3 * glove_score + 0.7 * bert_score
                definition = self.get_word_definition(other_word)
                similarities.append((other_word, combined_score, definition))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

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