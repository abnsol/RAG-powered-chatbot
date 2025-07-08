import gradio as gr
import os
import sys

# Import the RAG pipeline function from your rag_pipeline.py
try:
    if os.getcwd().endswith('intelligent-complaint-answering-chatbot'):
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    elif os.getcwd().endswith('src'):
        sys.path.insert(0, os.getcwd())

    # Import the streaming version of the function
    from rag_pipeline import answer_complaint_question_stream
    print("Successfully imported answer_complaint_question_stream from rag_pipeline.py")
except ImportError as e:
    print(f"Error importing rag_pipeline: {e}")
    print("Please ensure rag_pipeline.py is in src/ and your Python path is correctly set.")
    sys.exit(1)

# --- Gradio Interface Logic ---

def chat_interface(question):
    """
    Function to be called by Gradio, wrapping the RAG pipeline for streaming.
    Yields partial answers and then appends sources.
    """
    if not question.strip():
        yield "Please enter a question.", "" # Return empty sources
        return

    full_answer = ""
    retrieved_sources_for_display = []

    # Call the streaming RAG pipeline function
    for partial_answer, sources in answer_complaint_question_stream(question):
        full_answer = partial_answer # Keep updating with the latest partial answer
        if sources: # Sources will only be present in the first yield
            retrieved_sources_for_display = sources
        
        # Format the output for Gradio Markdown
        current_output = f"### CrediTrust Analyst Assistant:\n{full_answer}"
        # Only append sources at the very end when all tokens are generated,
        # or yield them once at the beginning if you prefer.
        # For a clean streaming effect, we'll append sources after the loop.
        yield current_output # Yield the current partial answer

    # After the loop, append the sources to the final answer
    sources_text = "\n\n### Retrieved Sources:\n"
    if retrieved_sources_for_display:
        for i, doc in enumerate(retrieved_sources_for_display):
            sources_text += f"**Source {i+1}**\n"
            sources_text += f"- **Product**: {doc.metadata.get('product', 'N/A')}\n"
            sources_text += f"- **Issue**: {doc.metadata.get('issue', 'N/A')}\n"
            sources_text += f"- **Complaint ID**: {doc.metadata.get('complaint_id', 'N/A')}\n"
            sources_text += f"- **Content Snippet**: {doc.page_content[:500]}...\n\n"
    else:
        sources_text += "No specific sources were retrieved for this query based on the context."

    # Yield the final answer with sources
    yield f"### CrediTrust Analyst Assistant:\n{full_answer}{sources_text}"


# --- Gradio Interface Definition ---
# (No changes needed here, as the `fn` now handles streaming)
input_text = gr.Textbox(lines=2, label="Ask a question about customer complaints:")
output_text = gr.Markdown(label="Answer & Sources")

iface = gr.Interface(
    fn=chat_interface,
    inputs=input_text,
    outputs=output_text,
    title="CrediTrust Intelligent Complaint Analysis Chatbot",
    description="Ask questions about customer complaints across various financial products. The chatbot will retrieve relevant complaint narratives and generate an insightful answer.",
    theme="soft",
    allow_flagging="never",
    examples=[
        "What are the most common issues with credit card billing?",
        "Tell me about fraud complaints related to money transfers.",
        "Why are customers unhappy with personal loans?",
        "What kind of problems do people report about savings accounts?",
        "Are there any complaints regarding BNPL services?"
    ]
)

if __name__ == "__main__":
    print("Launching Gradio interface...")
    print("Please ensure your RAG pipeline (rag_pipeline.py) is correctly configured and its dependencies are met.")
    print("The LLM and vector store will be loaded when the Gradio app starts.")
    iface.launch(share=True)