import pinecone
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_pinecone():
    """
    Initialize Pinecone with API credentials
    """
    try:
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
    except Exception as e:
        print(f"Pinecone Initialization Error: {e}")
        raise

def create_or_get_index(index_name=None, dimension=768, metric='cosine'):
    """
    Create a new Pinecone index or return existing index
    
    Args:
        index_name (str): Name of the index
        dimension (int): Vector dimension
        metric (str): Distance metric
    
    Returns:
        pinecone.Index: Configured Pinecone index
    """
    # Use default index name if not provided
    index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "pdf-qa-index")
    
    try:
        # Initialize Pinecone
        initialize_pinecone()
        
        # Check if index exists
        if index_name not in pinecone.list_indexes():
            print(f"Creating new Pinecone index: {index_name}")
            pinecone.create_index(
                name=index_name, 
                dimension=dimension, 
                metric=metric
            )
        
        # Return the index
        return pinecone.Index(index_name)
    
    except Exception as e:
        print(f"Error creating/accessing Pinecone index: {e}")
        raise

def list_pinecone_indexes():
    """
    List all available Pinecone indexes
    
    Returns:
        List[str]: List of index names
    """
    try:
        initialize_pinecone()
        return pinecone.list_indexes()
    except Exception as e:
        print(f"Error listing Pinecone indexes: {e}")
        return []