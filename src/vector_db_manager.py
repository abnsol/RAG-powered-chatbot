import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
import sys 
from sentence_transformers import SentenceTransformer # For direct model loading if needed

# --- Configuration ---
cleaned_data_path = 'data/processed/filtered_complaints.csv'
vector_store_dir = 'vector_store'

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1000  # characters
chunk_overlap = 200 # characters

# --- 1. Load the cleaned and filtered dataset ---
print("--- Loading Cleaned Data ---")
try:
    df_cleaned = pd.read_csv(cleaned_data_path)
    print(f"Loaded df_cleaned. Shape: {df_cleaned.shape}")
    print(df_cleaned.head())
except FileNotFoundError:
    print(f"Error: File not found at {cleaned_data_path}. Please ensure the file exists in the specified path.")
    sys.exit(1) 

df_cleaned['Consumer complaint narrative'] = df_cleaned['Consumer complaint narrative'].astype(str).fillna('')

# --- 2. Implement Text Chunking Strategy ---
print("\n--- Implementing Text Chunking ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    add_start_index=True
)

documents_with_metadata = []
for index, row in df_cleaned.iterrows():
    narrative = row['Consumer complaint narrative']
    if narrative:
        documents_with_metadata.append({
            'page_content': narrative,
            'metadata': {
                'complaint_id': row['Complaint ID'],
                'product': row['Product'],
                'issue': row['Issue'],
                'sub_product': row['Sub-product'], 
                'company': row['Company'] 
            }
        })

all_chunks = []
for doc in documents_with_metadata:
    chunks_for_doc = text_splitter.create_documents([doc['page_content']], metadatas=[doc['metadata']])
    all_chunks.extend(chunks_for_doc)

print(f"Total number of documents processed: {len(documents_with_metadata)}")
print(f"Total number of chunks created: {len(all_chunks)}")

if all_chunks:
    print(f"\nExample chunk content (first chunk):")
    print(f"'{all_chunks[0].page_content[:400]}...'")
    print(f"Example chunk metadata (first chunk): {all_chunks[0].metadata}")
else:
    print("No chunks were created. Check your data and chunking logic.")
    sys.exit(1) 

# --- 3. Choose an Embedding Model and Embed ---
print(f"\n--- Initializing Embedding Model: {embedding_model_name} ---")
embedding_model = None 

try:
    # Try loading directly via SentenceTransformer first, which can be more robust for downloads
    # and then pass to HuggingFaceEmbeddings.
    # This also helps confirm the model can be downloaded.
    print(f"Attempting to download/load model directly using SentenceTransformer...")
    _ = SentenceTransformer(embedding_model_name) 
    print("Model loaded successfully by SentenceTransformer (and cached).")

    # Now initialize HuggingFaceEmbeddings using the locally cached model
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"HuggingFaceEmbeddings initialized with '{embedding_model_name}'.")

    # Test the embedding model
    sample_text = "This is a test sentence to generate an embedding."
    sample_embedding = embedding_model.embed_query(sample_text)
    print(f"Sample embedding generated. Dimension: {len(sample_embedding)}")

except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure 'sentence-transformers' and 'langchain-community' are installed.")
    print("If error persists, check your network connection or Hugging Face token configuration (if any).")
    sys.exit(1) # Exit if embedding model fails to load

# --- 4. Building and Persisting Vector Store (using ChromaDB) ---
print(f"\n--- Building and Persisting Vector Store (using ChromaDB) ---")

# Create directory if it doesn't exist
os.makedirs(vector_store_dir, exist_ok=True)

try:
    print("Creating ChromaDB vector store. This might take a while...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=vector_store_dir
    )
    vectorstore.persist() # Explicitly persist
    print(f"Vector store created and persisted to: {vector_store_dir}")

    # --- 5. Testing Vector Store Loading and Retrieval ---
    print("\n--- Testing Vector Store Loading and Retrieval ---")
    loaded_vectorstore = Chroma(persist_directory=vector_store_dir, embedding_function=embedding_model)
    print("Vector store loaded successfully.")

    test_query = "unauthorized transactions on my credit card"
    results = loaded_vectorstore.similarity_search(test_query, k=3) # Retrieve top 3 results

    print(f"\nTop 3 results for query: '{test_query}'")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 20)

except Exception as e:
    print(f"Error building or persisting vector store: {e}")
    print("Please ensure 'chromadb' is installed and you have enough RAM/disk space.")
    sys.exit(1)

print("\nTask 2 completed successfully!")