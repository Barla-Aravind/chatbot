import PyPDF2
import os
import re

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path):
        """
        Extract text from PDF file
        
        Args:
            file_path (str): Path to PDF file
        
        Returns:
            str: Extracted text from PDF
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
        
        except Exception as e:
            print(f"PDF Text Extraction Error: {e}")
            raise

    @staticmethod
    def clean_text(text):
        """
        Clean and preprocess extracted text
        
        Args:
            text (str): Raw extracted text
        
        Returns:
            str: Cleaned and processed text
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)
        
        return text.strip()

    @staticmethod
    def split_text_into_chunks(text, chunk_size=500, overlap=50):
        """
        Split text into manageable chunks
        
        Args:
            text (str): Input text
            chunk_size (int): Size of each text chunk
            overlap (int): Overlap between chunks
        
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks