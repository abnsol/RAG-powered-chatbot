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

# ... (previous imports and initializations like embedding_model, vectorstore, tokenizer, model) ...

# --- 3. Initialize Language Model (Generator Component) ---
print(f"\n--- Initializing Language Model: {llm_model_name} ---")
llm_model = None # Renamed from llm_pipeline to avoid confusion with the pipeline object below
llm_tokenizer = None
try:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    # Ensure pad_token is set for generation, especially if batching or padding is involved
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llm_model.resize_token_embeddings(len(llm_tokenizer))

    print(f"LLM '{llm_model_name}' loaded successfully for streaming.")

except Exception as e:
    print(f"Error loading LLM '{llm_model_name}': {e}")
    print("Ensure 'transformers' and 'accelerate' are installed. Check your RAM/VRAM availability.")
    print("If using Colab, try a GPU runtime. If local, ensure sufficient resources.")
    sys.exit(1)

# --- 4. Define the RAG Core Function (Modified for Streaming) ---
def answer_complaint_question_stream(query: str, k: int = 5):
    """
    Performs RAG to answer a question about customer complaints, yielding tokens.

    Args:
        query (str): The user's question.
        k (int): The number of top relevant chunks to retrieve.

    Yields:
        tuple: (str, list) where str is a partial answer, and list is the retrieved docs.
               The retrieved docs are yielded only once with the first token.
    """
    if llm_model is None or llm_tokenizer is None:
        yield "Error: Language Model not loaded.", []
        return

    print(f"\nUser Query: '{query}'")

    # --- Retriever Step ---
    print(f"Retrieving top {k} relevant chunks...")
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    if not retrieved_docs:
        print("No relevant documents found.")
        yield "I could not find any relevant information to answer your question.", []
        return

    # --- Context Construction ---
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"Retrieved {len(retrieved_docs)} chunks. Context length: {len(context)} characters.")

    # --- Prompt Engineering ---
    prompt_template = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer concisely and accurately. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {query}
Answer:"""

    # --- Generator Step (Streaming) ---
    print("Generating answer with LLM (streaming)...")
    
    # Encode the prompt
    input_ids = llm_tokenizer(prompt_template, return_tensors="pt").input_ids.to(llm_model.device)

    # Initialize the streamer
    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Run generation in a separate thread to allow streaming
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        # Ensure pad_token_id is set if tokenizer has one, or model.config.eos_token_id
        pad_token_id=llm_tokenizer.pad_token_id if llm_tokenizer.pad_token_id is not None else llm_tokenizer.eos_token_id
    )
    
    thread = threading.Thread(target=llm_model.generate, kwargs=generate_kwargs)
    thread.start()

    generated_text = ""
    # Yield the retrieved docs once with the first token
    first_token_yielded = False
    for new_token in streamer:
        generated_text += new_token
        if not first_token_yielded:
            yield generated_text, retrieved_docs # Yield first token with sources
            first_token_yielded = True
        else:
            yield generated_text, [] # Yield subsequent tokens without sources (sources are static)

    print("Answer generation complete.")

# --- Example Usage (for testing the script directly - adjusted for streaming) ---
if __name__ == "__main__":
    print("\n--- Testing RAG Pipeline (Streaming) ---")
    test_questions = [
        "What are the common complaints about credit card billing?",
        "Why are people unhappy with personal loans?",
    ]

    for q in test_questions:
        print(f"\nUser Query: '{q}'")
        full_answer_text = ""
        retrieved_sources_for_display = [] # To capture sources from the first yield

        for partial_answer, sources in answer_complaint_question_stream(q, k=5):
            full_answer_text = partial_answer # Keep updating with the latest partial answer
            if sources: # Sources will only be present in the first yield
                retrieved_sources_for_display = sources
            sys.stdout.write(partial_answer + '\r') # Print partial answer, overwrite previous line
            sys.stdout.flush()
        
        print("\n" + full_answer_text) # Print final answer on a new line
        print(f"\n--- Sources Used ({len(retrieved_sources_for_display)} retrieved) ---")
        if retrieved_sources_for_display:
            for i, doc in enumerate(retrieved_sources_for_display):
                print(f"Source {i+1} (Product: {doc.metadata.get('product', 'N/A')}, Issue: {doc.metadata.get('issue', 'N/A')}):")
                print(f"  Content: {doc.page_content[:300]}...")
                print("-" * 10)
        else:
            print("No sources were retrieved or displayed for this query.")
        print("=" * 50)