import pinecone
import os
import numpy as np
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# Import other utility classes
from .embedding_utils import EmbeddingGenerator
from .text_preprocessing import TextPreprocessor
from .dimension_reducer import DimensionReducer

# Load environment variables
load_dotenv()

class PineconeVectorStore:
    def __init__(self, 
                 index_name: Optional[str] = None, 
                 dimension: int = 768, 
                 metric: str = 'cosine'):
        """
        Initialize Pinecone Vector Store
        
        Args:
            index_name (str, optional): Name of Pinecone index
            dimension (int): Embedding dimension
            metric (str): Distance metric for vector similarity
        """
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            
            # Use default index name if not provided
            self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "pdf-qa-index")
            
            # Create or get index
            self._create_index_if_not_exists(dimension, metric)
            
            # Initialize index
            self.index = pinecone.Index(self.index_name)
            
            # Initialize utility classes
            self.embedding_generator = EmbeddingGenerator()
            self.text_preprocessor = TextPreprocessor()
            self.dimension_reducer = DimensionReducer()
        
        except Exception as e:
            print(f"Pinecone Initialization Error: {e}")
            raise

    def _create_index_if_not_exists(self, dimension: int, metric: str):
        """
        Create Pinecone index if it doesn't exist
        
        Args:
            dimension (int): Vector dimension
            metric (str): Distance metric
        """
        if self.index_name not in pinecone.list_indexes():
            print(f"Creating Pinecone index: {self.index_name}")
            pinecone.create_index(
                name=self.index_name, 
                dimension=dimension, 
                metric=metric
            )
            print(f"Index {self.index_name} created successfully")

    def upsert_documents(self, 
                         documents: List[str], 
                         reduce_dimensions: bool = False, 
                         target_dimension: int = 128):
        """
        Upsert document embeddings into Pinecone
        
        Args:
            documents (List[str]): List of documents to embed
            reduce_dimensions (bool): Whether to reduce embedding dimensions
            target_dimension (int): Target dimension for reduction
        
        Returns:
            int: Number of vectors upserted
        """
        try:
            # Preprocess documents
            preprocessed_docs = [
                ' '.join(self.text_preprocessor.preprocess_pipeline(doc)) 
                for doc in documents
            ]
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(preprocessed_docs)
            
            # Optional dimension reduction
            if reduce_dimensions:
                embeddings = self.dimension_reducer.reduce_dimensions(
                    embeddings, 
                    target_dimensions=target_dimension
                )
            
            # Prepare vectors for Pinecone
            vectors = [
                (str(idx), embedding.tolist()) 
                for idx, embedding in enumerate(embeddings)
            ]
            
            # Upsert vectors
            upsert_response = self.index.upsert(vectors)
            
            print(f"Upserted {len(vectors)} vectors to index {self.index_name}")
            return len(vectors)
        
        except Exception as e:
            print(f"Vector Upsert Error: {e}")
            raise

    def query_index(self, 
                    query: str, 
                    top_k: int = 5, 
                    include_metadata: bool = False) -> List[Tuple]:
        """
        Semantic search query on Pinecone index
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            include_metadata (bool): Whether to include metadata in results
        
        Returns:
            List[Tuple]: Top matching vector results
        """
        try:
            # Preprocess query
            preprocessed_query = ' '.join(
                self.text_preprocessor.preprocess_pipeline(query)
            )
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embeddings([preprocessed_query])[0]
            
            # Query Pinecone index
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=include_metadata
            )
            
            return query_response.matches
        
        except Exception as e:
            print(f"Pinecone Query Error: {e}")
            raise

    def delete_vectors(self, vector_ids: List[str]):
        """
        Delete specific vectors from the index
        
        Args:
            vector_ids (List[str]): List of vector IDs to delete
        """
        try:
            self.index.delete(ids=vector_ids)
            print(f"Deleted {len(vector_ids)} vectors from {self.index_name}")
        
        except Exception as e:
            print(f"Vector Deletion Error: {e}")
            raise

    def get_index_stats(self):
        """
        Retrieve Pinecone index statistics
        
        Returns:
            dict: Index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0)
            }
        
        except Exception as e:
            print(f"Index Stats Retrieval Error: {e}")
            raise

    @classmethod
    def list_available_indexes(cls):
        """
        List all available Pinecone indexes
        
        Returns:
            List[str]: Available index names
        """
        try:
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            return pinecone.list_indexes()
        
        except Exception as e:
            print(f"Index Listing Error: {e}")
            return []

# Example Usage Demonstration
def demonstrate_vector_store():
    """
    Demonstrate PineconeVectorStore functionality
    """
    try:
        # Initialize Vector Store
        vector_store = PineconeVectorStore()
        
        # Sample documents
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Neural networks are computational models inspired by biological neural networks",
            "Deep learning has revolutionized image and speech recognition"
        ]
        
        # Upsert documents
        vector_store.upsert_documents(documents)
        
        # Get index stats
        stats = vector_store.get_index_stats()
        print("Index Statistics:", stats)
        
        # Query index
        query = "What is artificial intelligence?"
        results = vector_store.query_index(query)
        
        print("\nQuery Results:")
        for match in results:
            print(f"ID: {match.id}, Score: {match.score}")
    
    except Exception as e:
        print(f"Demonstration Error: {e}")

# Run demonstration if script is executed directly
if __name__ == "__main__":
    demonstrate_vector_store()