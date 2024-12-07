# counterarg.

# Counter Research Generation from arXiv Data

## Table of Contents
1. [Project Overview](#project-overview)
2. [Core Concept](#core-concept)
3. [System Architecture](#system-architecture)
4. [Detailed Steps](#detailed-steps)
5. [Requirements & Setup](#requirements--setup)
6. [Usage](#usage)
7. [Methodological Considerations](#methodological-considerations)
8. [Performance & Scalability Considerations](#performance--scalability-considerations)
9. [Limitations & Future Work](#limitations--future-work)
10. [License & Compliance](#license--compliance)

---

## Project Overview

This repository provides a pipeline that, given a piece of input research content—ranging from a single paragraph to an entire paper abstract—attempts to generate a well-evidenced counterargument. It leverages a large corpus of arXiv abstracts to find countering perspectives and uses a Large Language Model (LLM) to synthesize a counter-research narrative supported by actual scholarly sources.

In short, if you supply a claim such as "Quantum entanglement has no significant impact on classical computation," the system will:

1. Retrieve thematically relevant documents from a massive corpus of arXiv abstracts.
2. Identify which of these documents might refute the claim.
3. Download the full PDF texts of those refuting documents from arxiv.org.
4. Provide the LLM with this evidence so it can construct a factual, evidence-based counterargument.
5. If no strong contradictory evidence is found, it will refrain from fabricating a counterargument.

---

## Core Concept

**Goal:** To produce counter-research content that challenges a given claim or viewpoint using actual scholarly papers as evidence.

**Key Idea:**  
1. **Relevance:** Identify relevant literature using vector embeddings and similarity search.
2. **Refutation Detection:** Determine which documents truly present a contradicting stance.
3. **Grounded Generation:** Equip the LLM with authentic source material to ensure that the counterargument is not a hallucination but an informed, evidence-based standpoint.
4. **Honest Failure:** If no sufficient evidence is found, the system admits it instead of making something up.

---

## System Architecture

1. **Data Source (arXiv Metadata):**  
   - A dataset of ~1.7 million arXiv abstracts (as provided by the Cornell University dataset on Kaggle).
   - Each record includes an `id`, `authors`, `title`, `categories`, and `abstract`.

2. **Embeddings & Vector Store (FAISS):**  
   - Documents are embedded into high-dimensional vectors using OpenAI’s `text-embedding-ada-002`.
   - A FAISS index efficiently retrieves the most semantically similar documents to a given query or input chunk.

3. **Stance Detection with LLM:**  
   - For each retrieved abstract, the system queries the LLM to determine if it "refutes," "supports," or is "neutral" toward the input claim.
   - Only refuting abstracts are selected for deeper analysis.

4. **Full Text Retrieval (PDF Download):**  
   - Using the `id` from the metadata, the system downloads the full PDF from `arxiv.org/pdf/<id>.pdf`.
   - Extracts textual content from the PDF for richer evidence.

5. **Counterargument Generation (LLM):**  
   - The LLM is given the input claim and a compilation of texts from refuting PDFs.
   - It attempts to synthesize these sources into a coherent counterargument.
   - If insufficient evidence is present, it outputs "No valid counterargument found."

---

## Detailed Steps

**Step 1: Data Acquisition**
- Use Kaggle CLI or API to download `arxiv-metadata-oai-snapshot.json`.
- Load this large JSON-lines file into a Pandas DataFrame.

**Step 2: Preprocessing**
- Extract essential columns: `id`, `authors`, `title`, `categories`, and `abstract`.
- Optionally, filter the dataset by categories or random sampling for demonstration (the full dataset is large and may require substantial compute resources).

**Step 3: Embeddings & Indexing**
- Convert each record into a descriptive text block: `Title, Authors, Categories, Abstract`.
- Use OpenAI’s embedding model to produce a vector embedding for each text block.
- Store these embeddings in memory or on disk and index them with FAISS for similarity queries.

**Step 4: Query Processing & Chunking**
- Input text is split into manageable chunks (for instance, a 2,000-character limit) to handle very long inputs.
- Each chunk is embedded and used as a query against the FAISS index to retrieve top-K similar abstracts.

**Step 5: Stance Detection**
- For each candidate abstract:
  - Prompt the LLM (e.g., GPT-4) with the input chunk and the candidate abstract.
  - Ask it to classify the stance: refutes, supports, or neutral.
  - Collect only those abstracts classified as "refutes" for the next step.

**Step 6: Evidence Gathering**
- For each refuting abstract:
  - Construct the PDF URL: `https://arxiv.org/pdf/<id>.pdf`.
  - Download the PDF and extract text using PyPDF2.
  - Accumulate these texts as source material for the final generation step.

**Step 7: Counterargument Generation**
- Combine all refuting PDF texts into a prompt.
- Instruct the LLM to produce a careful, accurate counterargument using only evidence from these sources.
- If no evidence is found or it is insufficient, instruct the model to state "No valid counterargument found."

---

## Requirements & Setup

**Software:**
- Python 3.7+
- Packages:
  - `pandas` for data handling
  - `requests` for HTTP requests (downloading PDFs)
  - `PyPDF2` for PDF text extraction
  - `faiss-cpu` for vector indexing and similarity search
  - `openai` for LLM and embedding API calls
  - `tqdm` for progress indication
  - `beautifulsoup4` (optional, if HTML parsing is needed)
  
**External Services:**
- **Kaggle:** Obtain an API token (kaggle.json) and place it in `~/.kaggle`.
- **OpenAI API Key:** Set `openai.api_key` or use an environment variable `OPENAI_API_KEY`.

**Installation:**
```bash
pip install pandas requests PyPDF2 faiss-cpu openai tqdm beautifulsoup4
