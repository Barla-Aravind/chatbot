ifrom config.pinecone import diagnose_pinecone_connection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("Pinecone Connectivity Diagnostic")
    print("--------------------------------")
    
    result = diagnose_pinecone_connection()
    
    if result:
        print("\n✅ Pinecone connection successful!")
    else:
        print("\n❌ Pinecone connection failed. Check your configuration.")

if __name__ == "__main__":
    main()
