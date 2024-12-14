import re
import spacy
from typing import List

class TextPreprocessor:
    def __init__(self, language='en'):
        """
        Initialize text preprocessor with spaCy model
        
        Args:
            language (str): Language model to use
        """
        try:
            self.nlp = spacy.load(f"{language}_core_web_sm")
        except Exception as e:
            print(f"SpaCy Model Load Error: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """
        Comprehensive text cleaning
        
        Args:
            text (str): Input text
        
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spaCy
        
        Args:
            text (str): Input text
        
        Returns:
            List[str]: List of tokens
        """
        doc = self.nlp(text)
        return [token.text for token in doc]

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens
        
        Args:
            tokens (List[str]): Input tokens
        
        Returns:
            List[str]: Filtered tokens
        """
        doc = self.nlp(' '.join(tokens))
        return [token.text for token in doc if not token.is_stop]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens
        
        Args:
            tokens (List[str]): Input tokens
        
        Returns:
            List[str]: Lemmatized tokens
        """
        doc = self.nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]

    def preprocess_pipeline(self, text: str) -> List[str]:
        """
        Complete text preprocessing pipeline
        
        Args:
            text (str): Input text
        
        Returns:
            List[str]: Preprocessed tokens
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        filtered_tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        
        return lemmatized_tokens