# counterarg.

# Counter Research Generation from arXiv Data

## Overview

This project implements a pipeline to generate counterarguments to a given research input (such as a paper abstract or paragraph) using a large corpus of arXiv abstracts. The pipeline retrieves relevant documents from a vector embedding database, identifies those that refute the claims in the input text, scrapes the corresponding full papers from arxiv.org, and then uses a Large Language Model (LLM) to construct a factually grounded counterargument. If the system cannot find strong enough evidence to support a counterargument, it will not fabricate one.

## Features

1. **Data Retrieval from Kaggle**:  
   Downloads and processes a dataset of approximately 1.7 million arXiv abstracts available on Kaggle.

2. **Vector Embedding Database**:  
   Uses OpenAI embeddings (`text-embedding-ada-002`) to vectorize and store each abstract. These embeddings are then indexed using FAISS for efficient similarity search.

3. **Document Retrieval**:  
   Given an input text (a research paper, abstract, or a short paragraph), the input is split into manageable chunks. For each chunk, the system:
   - Retrieves a set of top-K similar abstracts from the vector database.
   - Uses stance detection (via an LLM prompt) to identify which of these abstracts refute the claims made in the input chunk.

4. **PDF Scraping**:  
   For each refuting abstract, the pipeline constructs the corresponding PDF URL (`https://arxiv.org/pdf/[id].pdf`), attempts to download the PDF, and extracts text from it.

5. **Counterargument Generation**:  
   Provides the LLM with the text of all refuting documents (excerpts from the downloaded PDFs). The LLM then attempts to produce a factual counterargument strictly based on the evidence provided. If no sufficient evidence is found, it responds with “No valid counterargument found.”

## Architecture

**Pipeline Steps:**

1. **Dataset Download & Loading**:
   - Uses Kaggle CLI to download and unzip the `arxiv-metadata-oai-snapshot.json`.
   - Loads the dataset into a Pandas DataFrame, retaining essential fields (`id`, `authors`, `title`, `categories`, `abstract`).

2. **Preprocessing & Embedding**:
   - Each record is concatenated into a textual snippet: `Title + Authors + Categories + Abstract`.
   - Uses OpenAI’s Embedding API to convert this text into a numerical embedding vector.
   - Stores embeddings and associated metadata (ID, title, authors, categories, abstract).

3. **Vector Database (FAISS)**:
   - Creates a FAISS index for similarity search. For demonstration, an `IndexFlatIP` (inner product index) is used.
   - Maps embeddings to their corresponding metadata.
   
4. **Query Processing**:
   - The input research text is chunked into segments, ensuring manageable token sizes.
   - For each chunk:
     - The chunk is embedded using the same embedding model.
     - A similarity search is performed on the FAISS index to find top-K similar abstracts.

5. **Stance Detection**:
   - For each retrieved candidate abstract, an LLM (GPT-4) is prompted to determine stance: "refutes", "supports", or "neutral".
   - Only those abstracts marked as "refutes" are kept.

6. **Full Text Retrieval**:
   - For each refuting abstract, the corresponding PDF is downloaded from `arxiv.org`.
   - The PDF is parsed using PyPDF2 to extract text.

7. **Counterargument Generation**:
   - All retrieved refuting texts are combined and passed to an LLM prompt.
   - The LLM must find factual evidence from these sources to formulate a counterargument.
   - If no evidence is found, no counterargument is generated.

## Requirements

- Python 3.7+
- Dependencies:
  - `pandas`
  - `requests`
  - `PyPDF2`
  - `faiss-cpu` (FAISS for vector similarity search)
  - `openai` (for embeddings and LLM calls)
  - `tqdm` (for progress bars)
  - `BeautifulSoup4` (if needed for web scraping expansions)
- `kaggle` CLI configured and `kaggle.json` API key placed in `~/.kaggle`.
- OpenAI API Key for embeddings and LLM usage.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install pandas requests PyPDF2 faiss-cpu openai tqdm beautifulsoup4
