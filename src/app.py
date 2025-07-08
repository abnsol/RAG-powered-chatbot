import gradio as gr
import os
import sys

# Import the RAG pipeline function from your rag_pipeline.py
# Make sure your current working directory allows this import,
# or adjust sys.path if running from a different location.
try:
    # Add the src directory to the Python path if running app.py directly from project root
    # This helps in importing rag_pipeline from src/
    if os.getcwd().endswith('intelligent-complaint-answering-chatbot'):
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    elif os.getcwd().endswith('src'):
        sys.path.insert(0, os.getcwd()) # If running from src/ itself

    from rag_pipeline import answer_complaint_question
    print("Successfully imported answer_complaint_question from rag_pipeline.py")
except ImportError as e:
    print(f"Error importing rag_pipeline: {e}")
    print("Please ensure rag_pipeline.py is in src/ and your Python path is correctly set.")
    print("If running from project root, ensure 'sys.path.insert(0, os.path.join(os.getcwd(), 'src'))' is active.")
    sys.exit(1)

# --- Gradio Interface Logic ---

def chat_interface(question):
    """
    Function to be called by Gradio, wrapping the RAG pipeline.
    Formats the output for display in the Gradio interface.
    """
    if not question.strip():
        return "Please enter a question.", "" # Return empty sources

    # Call your RAG pipeline function
    answer, retrieved_docs = answer_complaint_question(question)

    # Format the retrieved sources for display
    sources_text = "### Retrieved Sources:\n"
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs):
            sources_text += f"**Source {i+1}**\n"
            sources_text += f"- **Product**: {doc.metadata.get('product', 'N/A')}\n"
            sources_text += f"- **Issue**: {doc.metadata.get('issue', 'N/A')}\n"
            sources_text += f"- **Complaint ID**: {doc.metadata.get('complaint_id', 'N/A')}\n"
            sources_text += f"- **Content Snippet**: {doc.page_content[:500]}...\n\n" # Show first 500 chars
    else:
        sources_text += "No specific sources were retrieved for this query based on the context."

    # Combine answer and sources for display
    full_response = f"### CrediTrust Analyst Assistant:\n{answer}\n\n{sources_text}"
    
    return full_response

# --- Gradio Interface Definition ---

# Define the input and output components
input_text = gr.Textbox(lines=2, label="Ask a question about customer complaints:")
output_text = gr.Markdown(label="Answer & Sources")

# Create the Gradio Interface
iface = gr.Interface(
    fn=chat_interface,
    inputs=input_text,
    outputs=output_text,
    title="CrediTrust Intelligent Complaint Analysis Chatbot",
    description="Ask questions about customer complaints across various financial products. The chatbot will retrieve relevant complaint narratives and generate an insightful answer.",
    theme="soft", # A clean theme
    allow_flagging="never", # Disable flagging for internal tool
    # Add a clear button
    examples=[
        "What are the most common issues with credit card billing?",
        "Tell me about fraud complaints related to money transfers.",
        "Why are customers unhappy with personal loans?",
        "What kind of problems do people report about savings accounts?",
        "Are there any complaints regarding BNPL services?"
    ]
)

# Launch the Gradio app
if __name__ == "__main__":
    print("Launching Gradio interface...")
    print("Please ensure your RAG pipeline (rag_pipeline.py) is correctly configured and its dependencies are met.")
    print("The LLM and vector store will be loaded when the Gradio app starts.")
    iface.launch(share=True) # share=True generates a public link for easy sharing/testing