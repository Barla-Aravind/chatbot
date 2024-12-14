import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from utils.pdf_utils import PDFProcessor
from utils.embedding_utils import EmbeddingGenerator
from utils.pinecone_utils import PineconeVectorStore
from config.pinecone import initialize_pinecone

# Initialize FastAPI app
app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store current PDF context
current_pdf_processor = None
current_vector_store = None

class QuestionRequest(BaseModel):
    question: str

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global current_pdf_processor, current_vector_store
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process PDF
        pdf_processor = PDFProcessor()
        text = pdf_processor.extract_text_from_pdf(temp_path)
        cleaned_text = pdf_processor.clean_text(text)
        
        # Split text into chunks
        text_chunks = pdf_processor.split_text_into_chunks(cleaned_text)
        
        # Initialize Pinecone vector store
        vector_store = PineconeVectorStore()
        
        # Upsert embeddings
        vector_store.upsert_embeddings(text_chunks)
        
        # Store for future reference
        current_pdf_processor = pdf_processor
        current_vector_store = vector_store
        
        # Remove temporary file
        os.remove(temp_path)
        
        return {
            "status": "success", 
            "message": f"PDF processed. {len(text_chunks)} chunks created."
        }
    
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error processing PDF: {str(e)}"
        }

@app.post("/ask-question/")
async def ask_question(request: QuestionRequest):
    global current_vector_store
    
    if not current_vector_store:
        return {
            "status": "error", 
            "message": "No PDF uploaded. Please upload a PDF first."
        }
    
    try:
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        
        # Perform similarity search
        similar_chunk_indices = embedding_generator.similarity_search(
            request.question, 
            current_vector_store.index.describe_index_stats()
        )
        
        # Retrieve similar chunks (placeholder - you might want to enhance this)
        similar_chunks = [
            f"Chunk {idx}: Relevant information" 
            for idx in similar_chunk_indices
        ]
        
        # Generate answer (simplified - consider using a more advanced LLM)
        answer = f"Based on the document, here are some relevant insights: {' '.join(similar_chunks)}"
        
        return {
            "status": "success", 
            "question": request.question,
            "answer": answer
        }
    
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error processing question: {str(e)}"
        }

@app.get("/")
async def root():
    return {
        "message": "PDF Q&A Chatbot Backend",
        "status": "running"
    }

if __name__ == "__main__":
    # Initialize Pinecone
    initialize_pinecone()
    
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)