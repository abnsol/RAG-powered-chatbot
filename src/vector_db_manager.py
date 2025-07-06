import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# --- 1. Load the cleaned and filtered dataset ---
# Define the path to your filtered_complaints.csv
# Remember to replace 'Your_Project_Folder_Name' with your actual project folder name in Drive
cleaned_data_path = './data/processed/filtered_complaints.csv'

# Load the DataFrame
try:
    df_cleaned = pd.read_csv(cleaned_data_path)
    print(f"Loaded df_cleaned. Shape: {df_cleaned.shape}")
    print(df_cleaned.head())
except FileNotFoundError:
    print(f"Error: File not found at {cleaned_data_path}. Please check your path and ensure the file exists.")

# Ensure 'Consumer complaint narrative' is string type and handle any lingering NaNs
df_cleaned['Consumer complaint narrative'] = df_cleaned['Consumer complaint narrative'].astype(str).fillna('')

# --- 2. Implement Text Chunking Strategy ---

# as seen in the EDA if most complaints are between 50-300 words, a chunk_size of 1000 characters
# (roughly 200-250 words) with overlap might be a good starting point.

chunk_size = 1000  # characters
chunk_overlap = 200 # characters (typically 10-20% of chunk_size)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len, 
    add_start_index=True 
)

# Prepare documents for chunking with initial metadata
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
            }
        })

# Chunk the documents while preserving metadata
# text_splitter.create_documents directly accepts a list of texts and an optional list of metadatas
# iterate through our prepared documents to split them individually
all_chunks = []
for doc in documents_with_metadata:
    # create_documents returns a list of Document objects
    # Each Document object will have 'page_content' and 'metadata' attributes
    chunks_for_doc = text_splitter.create_documents([doc['page_content']], metadatas=[doc['metadata']])
    all_chunks.extend(chunks_for_doc)

print(f"\n--- Chunking Results ---")
print(f"Total number of documents processed: {len(documents_with_metadata)}")
print(f"Total number of chunks created: {len(all_chunks)}")

if all_chunks:
    print(f"\nExample chunk content (first chunk):")
    print(f"'{all_chunks[0].page_content[:400]}...'") 
    print(f"Example chunk metadata (first chunk): {all_chunks[0].metadata}")
else:
    print("No chunks were created. Check your data and chunking logic.")


## EMBEDDING MODEL SELECTION AND INITIALIZATION

from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer 

# --- 3. Choose an Embedding Model ---
# The project specifies 'sentence-transformers/all-MiniLM-L6-v2' as a good starting point.
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize the embedding model
print(f"\n--- Initializing Embedding Model: {embedding_model_name} ---")
try:
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
    print(f"Embedding model '{embedding_model_name}' loaded successfully.")
    
    # Test the embedding model with a sample text
    sample_text = "This is a test sentence to generate an embedding."
    sample_embedding = embedding_model.embed_query(sample_text)
    print(f"Sample embedding generated. Dimension: {len(sample_embedding)}")
    print(f"Sample embedding (first 10 values): {sample_embedding[:10]}")

except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Please ensure you have 'sentence-transformers' library installed.")