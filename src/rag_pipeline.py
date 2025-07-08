import pandas as pd
import os
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # For LLM
import torch # For checking GPU availability and managing tensors

# --- Configuration
vector_store_dir = '../vector_store'

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# LLM choice: google/gemma-2b-it is a good instruction-tuned small model
# Consider 'mistralai/Mistral-7B-Instruct-v0.2' if you have more VRAM/RAM
llm_model_name = "google/gemma-2b-it" 

# --- 1. Initialize Embedding Model (Retriever Component) ---
print(f"--- Initializing Embedding Model: {embedding_model_name} ---")
try:
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"Embedding model '{embedding_model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Ensure 'sentence-transformers' and 'langchain-community' are installed.")
    sys.exit(1)

# --- 2. Load Persisted Vector Store (Retriever Component) ---
print(f"\n--- Loading Vector Store from: {vector_store_dir} ---")
try:
    # Ensure the directory exists before attempting to load
    if not os.path.exists(vector_store_dir):
        print(f"Error: Vector store directory not found at {vector_store_dir}.")
        print("Please ensure Task 2 was completed successfully and the path is correct.")
        sys.exit(1)

    vectorstore = Chroma(persist_directory=vector_store_dir, embedding_function=embedding_model)
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error loading vector store: {e}")
    print("Ensure 'chromadb' is installed and the vector store is correctly persisted.")
    sys.exit(1)

# --- 3. Initialize Language Model (Generator Component) ---
print(f"\n--- Initializing Language Model: {llm_model_name} ---")
llm_pipeline = None
try:
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map="auto", # Automatically maps model to available devices (GPU/CPU)
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Use float16 on GPU for efficiency
    )

    # Create a text generation pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500, 
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1 
    )
    print(f"LLM '{llm_model_name}' pipeline loaded successfully.")

except Exception as e:
    print(f"Error loading LLM '{llm_model_name}': {e}")
    print("Ensure 'transformers' and 'accelerate' are installed. Check your RAM/VRAM availability.")
    print("If using Colab, try a GPU runtime. If local, ensure sufficient resources.")
    sys.exit(1)

# --- 4. Define the RAG Core Function ---
def answer_complaint_question(query: str, k: int = 5):
    """
    Performs RAG to answer a question about customer complaints.

    Args:
        query (str): The user's question.
        k (int): The number of top relevant chunks to retrieve.

    Returns:
        tuple: A tuple containing:
            - str: The generated answer from the LLM.
            - list: A list of the retrieved LangChain Document objects.
    """
    if not llm_pipeline:
        return "Error: Language Model not loaded.", []

    print(f"\nUser Query: '{query}'")

    # --- Retriever Step ---
    print(f"Retrieving top {k} relevant chunks...")
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    if not retrieved_docs:
        print("No relevant documents found.")
        return "I could not find any relevant information to answer your question.", []

    # --- Context Construction ---
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"Retrieved {len(retrieved_docs)} chunks. Context length: {len(context)} characters.")

    # --- Prompt Engineering ---
    # This template guides the LLM on how to use the context.
    prompt_template = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer concisely and accurately. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {query}
Answer:"""

    # --- Generator Step ---
    print("Generating answer with LLM...")
    try:
        # For instruction-tuned models like Gemma-2b-it, it's common to wrap the prompt
        # in specific chat formats. The pipeline handles this if you provide a list of dicts.
        messages = [
            {"role": "user", "content": prompt_template}
        ]
        
        # The generate method of the pipeline expects a list of strings or dicts
        # For Gemma, it's often better to pass the formatted prompt directly if not using chat template
        # Or, if using the chat template, ensure the pipeline is configured for it.
        # For simplicity, let's use the direct prompt string for now.
        
        # The pipeline returns a list of dictionaries, each containing 'generated_text'
        response = llm_pipeline(prompt_template, num_return_sequences=1, return_full_text=False)
        
        # Extract the generated text. The exact key might vary slightly depending on the model/pipeline.
        generated_answer = response[0]['generated_text'].strip()

        print("Answer generated successfully.")
        return generated_answer, retrieved_docs

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return "An error occurred while generating the answer.", retrieved_docs

# --- Example Usage (for testing the script directly) ---
if __name__ == "__main__":
    print("\n--- Testing RAG Pipeline ---")
    # Example questions for testing
    test_questions = [
        "What are the common complaints about credit card billing?",
        "Why are people unhappy with personal loans?",
        "Are there any fraud complaints related to money transfers?",
        "What issues are customers facing with savings accounts?",
        "Tell me about complaints regarding BNPL (Buy Now, Pay Later)." # Remember BNPL limitation
    ]

    for q in test_questions:
        answer, sources = answer_complaint_question(q, k=5)
        print(f"\n--- Answer ---")
        print(answer)
        print(f"\n--- Sources Used ({len(sources)} retrieved) ---")
        for i, doc in enumerate(sources):
            print(f"Source {i+1} (Product: {doc.metadata.get('product', 'N/A')}, Issue: {doc.metadata.get('issue', 'N/A')}):")
            print(f"  Content: {doc.page_content[:300]}...") # Show first 300 chars
            print("-" * 10)
        print("=" * 50)