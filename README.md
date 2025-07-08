# Intelligent Complaint Analysis for Financial Services

## Project Overview

This project focuses on developing an internal AI tool for CrediTrust Financial, a fast-growing digital finance company in East Africa. The core of the solution is a Retrieval-Augmented Generation (RAG) powered chatbot designed to transform thousands of raw, unstructured customer complaints into actionable insights. By enabling internal stakeholders to ask plain-English questions and receive synthesized, evidence-backed answers, the tool aims to shift the company from reactive problem-solving to proactive issue identification based on real-time customer feedback.

## Business Context & Problem

CrediTrust Financial serves over 500,000 users across three countries with offerings including Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers. They receive thousands of customer complaints monthly through various channels.

**The Problem**: Product Managers like Asha (BNPL team) currently spend excessive hours manually sifting through complaints, making it difficult to identify emerging issues quickly.

### Key Performance Indicators (KPIs) for Success

- Decrease time for Product Managers to identify major complaint trends from days to minutes.
- Empower non-technical teams (Support, Compliance) to get answers without needing a data analyst.
- Shift from reactive to proactive problem-solving based on real-time customer feedback.

## Motivation

CrediTrust's internal teams face significant bottlenecks:

- Customer Support is overwhelmed by complaint volume.
- Product Managers struggle to identify frequent/critical issues across products.
- Compliance & Risk teams are reactive to violations/fraud signals.
- Executives lack visibility into emerging pain points due to scattered complaint narratives.

As a Data & AI Engineer, the goal is to build a RAG agent that:

- Allows internal users to ask plain-English questions (e.g., "Why are people unhappy with BNPL?")
- Uses semantic search (via a vector database) to retrieve relevant complaint narratives.
- Feeds retrieved narratives into an LLM to generate concise, insightful answers.
- Supports multi-product querying, enabling cross-product issue comparison.

## Data

This project utilizes complaint data from the Consumer Financial Protection Bureau (CFPB), a real-world dataset containing customer complaints across various financial products.

### Key Fields in each record:

- Consumer complaint narrative (core input for embedding and retrieval)
- Product
- Issue
- Sub-product
- Company
- Submission date and other metadata

## Project Structure

```
intelligent-complaint-answering-chatbot/
├── .git/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── processed/
│   │   └── filtered_complaints.csv
│   └── raw/
│       └── complaint.csv
├── config/
│   └── settings.yaml
├── models/
│   └── sentence-transformers_all-MiniLM-L6-v2/
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   └── 02_rag_prototyping.ipynb
├── reports/
│   ├── interim_report.md
│   └── final_report.md
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── vector_db_manager.py
│   ├── rag_pipeline.py
│   ├── app.py
│   └── utils/
│       ├── __init__.py
│       └── text_cleaners.py
├── tests/
│   ├── test_data_processor.py
│   ├── test_vector_db_manager.py
│   └── test_rag_pipeline.py
├── vector_store/
│   └── chroma.sqlite3
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Git
- Git Large File Storage (Git LFS)

### Clone the Repository

```bash
git clone https://github.com/abnsol/RAG-powered-chatbot.git
cd RAG-powered-chatbot
```

### Install Git LFS

If you don't have Git LFS installed, [follow the instructions here](https://git-lfs.github.com/). Then run:

```bash
git lfs install
git lfs track "vector_store/*.sqlite3"
git add .gitattributes
git commit -m "Configure Git LFS tracking for vector_store/*.sqlite3"
```

### Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas langchain langchain-community sentence-transformers chromadb gdown
```

### Data Acquisition (`complaint.csv`)

- **Local PC**: Place `complaint.csv` into `data/raw/`
- **Google Colab**: Use `gdown` in `src/vector_db_manager.py` to download `filtered_complaints.csv`. Replace `'YOUR_FILTERED_COMPLAINTS_CSV_FILE_ID_HERE'` with your actual Google Drive file ID.

---

## Progress & Deliverables

### Task 1: Exploratory Data Analysis and Data Preprocessing

- **Initial Dataset**: 9,609,797 rows, 18 columns.
- **Key Findings**:
  - 6.6M missing narratives (~69%) — removed.
  - >90% missing in `Tags` and `Consumer disputed?`.
  - 2.1M duplicate complaint narratives.
  - No direct “BNPL” product — handled in filtering logic.

- **Filtering**:
  - Targeted five products: Credit Card, Personal Loan, BNPL, Savings Account, Money Transfer.
  - Resulted in 1,045,146 rows → cleaned to 480,580 rows with added `narrative_length`.

- **Text Cleaning**:
  - Lowercased text.
  - Example: `a xxxx xxxx card was opened under my name by a fraudster...`

- **Distribution Analysis**:
  - Visualized complaints per product.
  - Word count histogram for chunking strategy.

- **Deliverables**:
  - Code: `src/data_processor.py`
  - Cleaned data: `data/processed/filtered_complaints.csv`

---

### Task 2: Text Chunking, Embedding, and Vector Store Indexing

- **Chunking**:
  - Used `langchain.text_splitter.RecursiveCharacterTextSplitter`
  - `chunk_size = 1000`, `chunk_overlap = 200`
  - Output: 888,532 chunks

- **Metadata**:
  - complaint_id, product, issue, sub_product, company

- **Embedding Model**:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - Output: 384-dim embeddings

- **Indexing**:
  - Used ChromaDB for vector store
  - Ongoing embedding/indexing process

- **Deliverables**:
  - Script: `src/vector_db_manager.py`
  - Saved vector store in `vector_store/`
  - Documentation in final report

---

## How to Run

### Local PC

1. Ensure required files in:
   - `data/raw/complaint.csv`
   - `data/processed/filtered_complaints.csv`

2. Activate virtual environment:

```bash
source .venv/bin/activate
```

3. Run:

```bash
python3 src/vector_db_manager.py
```

> ⚠️ This process may take hours depending on system specs.

---

### Google Colab

1. Create a new Colab notebook.
2. Install dependencies:

```python
!pip install pandas langchain langchain-community sentence-transformers chromadb gdown
```

3. Copy and paste `src/vector_db_manager.py` (Colab version).
4. Update:
   - `'YOUR_FILTERED_COMPLAINTS_CSV_FILE_ID_HERE'` with real file ID.
   - `Your_Project_Folder_Name` with actual project name.
5. Set GPU runtime (Runtime > Change runtime type > GPU).
6. Run the code cell.

---

## Challenges Encountered & Solutions

### Large File Size (GitHub Limit)

- **Issue**: `chroma.sqlite3` > 100MB
- **Solution**: Used Git LFS + `git lfs migrate` to convert history

### Git History Divergence

- **Issue**: `fatal: refusing to merge unrelated histories`
- **Solution**:

```bash
git pull origin main --allow-unrelated-histories
git push
```

---

## Future Work

- **Task 3**: Build RAG logic — retriever, generator, prompt design, evaluation.
- **Task 4**: Build interactive interface — Streamlit/Gradio chatbot with sources and streaming.

