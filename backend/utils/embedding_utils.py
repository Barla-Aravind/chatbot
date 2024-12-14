import cohere
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        """
        Initialize Cohere embedding client
        """
        try:
            self.co = cohere.Client(os.getenv("CO_API_KEY"))
        except Exception as e:
            print(f"Cohere Initialization Error: {e}")
            raise

    def generate_embeddings(self, texts):
        """
        Generate embeddings for given texts
        
        Args:
            texts (List[str]): List of text documents
        
        Returns:
            numpy.ndarray: Embedding vectors
        """
        try:
            # Generate embeddings using Cohere
            embeddings = self.co.embed(
                texts=texts, 
                model='embed-english-v2.0',
                input_type='search_document'
            )
            
            return np.array(embeddings.embeddings)
        
        except Exception as e:
            print(f"Embedding Generation Error: {e}")
            raise

    def similarity_search(self, query, embedded_docs, top_k=3):
        """
        Perform semantic similarity search
        
        Args:
            query (str): Search query
            embedded_docs (List[np.ndarray]): List of document embeddings
            top_k (int): Number of top results to return
        
        Returns:
            List[int]: Indices of most similar documents
        """
        try:
            # Generate query embedding
            query_embedding = self.co.embed(
                texts=[query], 
                model='embed-english-v2.0',
                input_type='search_query'
            ).embeddings[0]
            
            # Compute cosine similarities
            similarities = np.dot(embedded_docs, query_embedding)
            
            # Return top-k indices
            return similarities.argsort()[-top_k:][::-1]
        
        except Exception as e:
            print(f"Similarity Search Error: {e}")
            raise