import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def main():
    st.title("📄 PDF Q&A Chatbot")
    
    # Sidebar for PDF Upload
    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file", 
        type=['pdf']
    )
    
    # Main chat interface
    st.header("Chat with your PDF")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # PDF Upload Process
    if uploaded_file is not None:
        try:
            # Upload PDF to backend
            files = {'file': uploaded_file}
            upload_response = requests.post(
                f"{BACKEND_URL}/upload-pdf/", 
                files=files
            )
            
            if upload_response.status_code == 200:
                st.sidebar.success("PDF Uploaded Successfully!")
            else:
                st.sidebar.error("PDF Upload Failed")
        
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"Connection Error: {e}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        
        # Add to message history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })
        
        # Send question to backend
        try:
            response = requests.post(
                f"{BACKEND_URL}/ask-question/", 
                json={"question": prompt}
            )
            
            if response.status_code == 200:
                answer = response.json().get('answer', 'No answer found.')
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                
                # Add to message history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer
                })
            
            else:
                st.error("Failed to get response from backend")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")

if __name__ == "__main__":
    main()