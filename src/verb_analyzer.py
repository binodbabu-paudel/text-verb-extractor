"""
Verb Analyzer Module
Extracts and analyzes verbs from text
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from collections import Counter
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VerbAnalyzer:
    """Analyze verbs in text"""
    
    def __init__(self):
        """Initialize VerbAnalyzer and download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
            nltk.data.find('taggers/universal_tagset')
        except LookupError:
            logger.info("Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('universal_tagset')
    
    def extract_verbs(self, text: str) -> Dict[str, Any]:
        """
        Extract verbs from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing verb analysis results
        """
        try:
            # Tokenize text
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # POS tagging
            tagged_words = pos_tag(words, tagset='universal')
            
            # Extract verbs
            verbs = [word.lower() for word, tag in tagged_words if tag.startswith('V')]
            verb_counts = Counter(verbs)
            
            # Analyze sentences
            sentence_analysis = []
            for i, sentence in enumerate(sentences, 1):
                sent_words = word_tokenize(sentence)
                sent_tagged = pos_tag(sent_words, tagset='universal')
                sent_verbs = [word.lower() for word, tag in sent_tagged if tag.startswith('V')]
                
                sentence_analysis.append({
                    "sentence_number": i,
                    "sentence": sentence,
                    "verbs": sent_verbs,
                    "verb_count": len(sent_verbs)
                })
            
            # Calculate statistics
            unique_verbs = list(verb_counts.keys())
            total_verb_instances = sum(verb_counts.values())
            
            return {
                "text": text,
                "sentences": sentences,
                "total_words": len(words),
                "total_sentences": len(sentences),
                "verbs": unique_verbs,
                "verb_counts": dict(verb_counts),
                "total_unique_verbs": len(unique_verbs),
                "total_verb_instances": total_verb_instances,
                "sentence_analysis": sentence_analysis,
                "sentences_with_verbs": sum(1 for s in sentence_analysis if s["verb_count"] > 0),
                "sentences_without_verbs": sum(1 for s in sentence_analysis if s["verb_count"] == 0),
                "most_common_verbs": verb_counts.most_common(10)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing verbs: {e}")
            raise
    
    def get_verb_categories(self, verbs: List[str]) -> Dict[str, List[str]]:
        """
        Categorize verbs based on common endings/patterns
        
        Args:
            verbs: List of verbs
            
        Returns:
            Dictionary of verb categories
        """
        categories = {
            "action_verbs": [],
            "linking_verbs": [],
            "helping_verbs": [],
            "irregular_verbs": [],
            "regular_verbs": []
        }
        
        # Common linking verbs
        linking_verbs = {'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 
                        'seem', 'appear', 'become', 'grow', 'turn', 'look', 
                        'feel', 'smell', 'sound', 'taste'}
        
        # Common helping verbs
        helping_verbs = {'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                        'would', 'shall', 'should', 'may', 'might', 'must', 
                        'can', 'could'}
        
        for verb in verbs:
            verb_lower = verb.lower()
            
            if verb_lower in linking_verbs:
                categories["linking_verbs"].append(verb)
            elif verb_lower in helping_verbs:
                categories["helping_verbs"].append(verb)
            elif verb.endswith('ing'):
                categories["action_verbs"].append(verb)
            elif verb.endswith('ed') and len(verb) > 2:
                categories["regular_verbs"].append(verb)
            else:
                categories["irregular_verbs"].append(verb)
        
        return categories


if __name__ == "__main__":
    # Example usage
    analyzer = VerbAnalyzer()
    sample_text = "The cat is sleeping on the mat. It dreams about chasing mice."
    results = analyzer.extract_verbs(sample_text)
    
    print(f"Found {results['total_unique_verbs']} unique verbs")
    print(f"Verbs: {results['verbs']}")
