import os
import requests
import pandas as pd
import json
import numpy as np
import re
import time
import tempfile
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from urllib.request import urlopen
from bs4 import BeautifulSoup
import subprocess

# Note: This code is a reference implementation and may require adaptation.
# It uses OpenAI embeddings as an example. You may need appropriate API keys and installations.
# Some steps are complex and may require a considerable amount of memory and compute resources.
# Also, ensure that kaggle, faiss, PyPDF2, BeautifulSoup, requests, and openai are installed,
# and that you have your OpenAI API key set up if you use OpenAI embeddings.

# -----------------------
# 0. Setup environment and download dataset from Kaggle
# -----------------------
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser('~/.kaggle')

# This assumes you have your kaggle.json API key already placed in ~/.kaggle/kaggle.json
# and appropriate permissions. If running in a notebook, you can run:
# !kaggle datasets download -d Cornell-University/arxiv --unzip
# We'll just do it in code:
try:
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "Cornell-University/arxiv",
         "--unzip"], check=True)
except:
    # If running in an environment where kaggle CLI is unavailable or already downloaded
    pass

data_path = 'arxiv-metadata-oai-snapshot.json'
data = pd.read_json(data_path, lines=True)

# The data columns: id, submitter, authors, title, comments, journal-ref, doi, report-no,
# categories, license, abstract, update_date, authors_parsed
# We will keep: id, authors, title, categories, abstract

data = data[['id', 'authors', 'title', 'categories', 'abstract']]

# For demonstration, let's filter to a manageable subset (for a real production environment,
# you would build a scalable solution, possibly chunk indexing and external vector DB)
# For example, filter to a few categories:
# data = data[data['categories'].str.contains('cs.CL', na=False)] # as an example, or skip filtering entirely
# We'll just use a smaller subset for demonstration:
# data = data.head(10000) # NOTE: This line is optional and only for demonstration. Remove for full dataset.

# -----------------------
# 1. Create a vector embedding database
# -----------------------
# We'll use OpenAI embeddings for demonstration. You need to set your openai api key.
import openai

openai.api_key = "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"  # Replace with your key or load from env variable.

# We'll create embeddings for each abstract. This might be very large for 1.7M abstracts.
# In practice, you would need batch processing, caching, and possibly a vector store like FAISS or Chroma.
# Here we show a conceptual approach with FAISS.


import faiss


def get_embedding(text: str) -> np.ndarray:
    # Using OpenAI's text-embedding-ada-002
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'], dtype=np.float32)


# Compute embeddings for each abstract
# This is computationally expensive for large datasets. Consider streaming or subset.
# Here we do a small sample for demonstration. Remove head() in production.
sampled_data = data  # or data.head(10000)

embeddings = []
ids = []
for idx, row in tqdm(sampled_data.iterrows(), total=sampled_data.shape[0]):
    text = f"Title: {row['title']}\nAuthors: {row['authors']}\nCategories: {row['categories']}\nAbstract: {row['abstract']}"
    emb = get_embedding(text)
    embeddings.append(emb)
    ids.append(row['id'])

embeddings = np.vstack(embeddings)

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatIP(
    dimension))  # Using Inner Product. You can use L2 if desired.
index.add_with_ids(embeddings, np.array(range(len(ids))))

# We'll keep a map from intID to the actual record
id_to_metadata = {}
for i, (ind, row) in enumerate(sampled_data.iterrows()):
    id_to_metadata[i] = {
        'id': row['id'],
        'title': row['title'],
        'authors': row['authors'],
        'categories': row['categories'],
        'abstract': row['abstract']
    }


# -----------------------
# 2. Given input (research paper/abstract/paragraph), chunk and retrieve relevant abstracts
# -----------------------
def chunk_text(text, max_chunk_size=2000):
    # Simple chunking by characters. You could improve by splitting by sentences or paragraphs.
    chunks = []
    current = []
    current_len = 0
    for paragraph in text.split('\n'):
        if current_len + len(paragraph) > max_chunk_size:
            chunks.append('\n'.join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph)
    if current:
        chunks.append('\n'.join(current))
    return chunks


def retrieve_similar_docs(chunk, top_k=10):
    # Retrieve top_k similar docs for a given chunk
    emb = get_embedding(chunk)
    # Search in FAISS
    distances, indices = index.search(np.array([emb]), top_k)
    results = []
    for dist, idx_ in zip(distances[0], indices[0]):
        if idx_ == -1:
            continue
        md = id_to_metadata[idx_]
        results.append((md, dist))
    return results


# -----------------------
# 3. Find abstracts that are opposite in sentiment/similarity
# -----------------------
# We'll interpret "opposite" as those that have low semantic similarity to the input chunk
# or that somehow refute the content. This is a tricky step - "refuting" is domain-dependent.
# One naive approach:
#   - Retrieve a set of similar documents first (those with high similarity).
#   - Among them, we try to determine their stance by doing sentiment or stance classification.
#   - If a doc "refutes" or contradicts input chunk, it should show opposite standpoint.
#
# For demonstration, we will use a second embedding call:
#   We'll embed both input and candidate abstract, and consider if
#   the candidate is "opposite" if it has low similarity or if a stance detection LLM prompt says it refutes.
#
# More robust stance detection could be done by a specialized stance detection model.
#
# Here, we try a rough approach:
#   1) We have top_k similar docs (those are thematically related).
#   2) We'll use an LLM prompt to see if doc is refuting the input chunk.

def stance_detection(input_chunk, candidate_abstract):
    # We'll ask the LLM if candidate_abstract refutes or supports input_chunk.
    # This is a simplistic approach and requires an LLM call. It may be costly.
    prompt = f"""Determine if the following abstract refutes the claims or findings described in the following input text.

Input:
{input_chunk}

Candidate Abstract:
{candidate_abstract}

Your answer should be one of: "refutes", "supports", or "neutral" based solely on the information given.
If uncertain, answer "neutral".
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response['choices'][0]['message']['content'].strip().lower()
    if "refutes" in answer:
        return "refutes"
    elif "supports" in answer:
        return "supports"
    else:
        return "neutral"


# -----------------------
# 4. For each chunk, retrieve some relevant docs, then find the ones that refute the input chunk
# -----------------------
def find_refuting_docs_for_chunk(chunk, top_k=10):
    candidates = retrieve_similar_docs(chunk, top_k=top_k)
    refuting_docs = []
    for (doc, dist) in candidates:
        candidate_abstract = f"Title: {doc['title']}\nAuthors: {doc['authors']}\nCategories: {doc['categories']}\nAbstract:\n{doc['abstract']}"
        stance = stance_detection(chunk, candidate_abstract)
        if stance == "refutes":
            refuting_docs.append(doc)
    return refuting_docs


# -----------------------
# 5. Web scrape the actual arXiv papers (PDFs) for the refuting abstracts
# -----------------------
def download_arxiv_pdf(arxiv_id):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        temp_path = os.path.join(tempfile.gettempdir(), f"{arxiv_id}.pdf")
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        return temp_path
    except:
        return None


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def gather_refuting_papers(docs):
    full_texts = []
    for doc in docs:
        arxiv_id = doc['id']
        pdf_path = download_arxiv_pdf(arxiv_id)
        if pdf_path is not None:
            pdf_text = extract_text_from_pdf(pdf_path)
            full_texts.append((arxiv_id, pdf_text))
    return full_texts


# -----------------------
# 6. Using all these papers as a prompt, have a large language model generate a counterargument chunk
#    only if it has sufficient evidence.
# -----------------------
def generate_counterargument(input_chunk, refuting_papers_texts):
    # If no refuting texts, then we cannot generate a counterargument
    if len(refuting_papers_texts) == 0:
        return None

    # We'll prompt the LLM to produce a factual counterargument based only on these refuting texts.
    # We instruct it: If insufficient evidence, do not generate.
    sources_combined = ""
    for pid, ptext in refuting_papers_texts:
        # We might truncate to avoid exceeding context length
        # Let's just take first 2000 characters from each doc.
        truncated_text = ptext[:2000]
        sources_combined += f"--- BEGIN SOURCE {pid} ---\n{truncated_text}\n--- END SOURCE {pid} ---\n"

    prompt = f"""You are a meticulous researcher. You have the following input chunk of a research claim:

{input_chunk}

You have also gathered a set of research papers that provide potential counterarguments or contradictory evidence:

{sources_combined}

Your task: Construct a careful, factual counterargument to the input chunk based only on the evidence found within these sources. 
If you cannot find sufficient, verifiable evidence from these sources to counter the input chunk, respond with:
'No valid counterargument found.'

Accuracy is the most important. Do not fabricate claims. Only use information that is present in these sources. 
If unsure or no contradictory evidence is found, do not create a counterargument.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response['choices'][0]['message']['content'].strip()
    if "No valid counterargument found." in answer:
        return None
    return answer


# -----------------------
# 7. Putting it all together: Given an input text, process it through the pipeline
# -----------------------
def generate_counterarguments_for_text(input_text, top_k=10):
    chunks = chunk_text(input_text)
    all_counterarguments = []
    for chunk in chunks:
        refuting_docs = find_refuting_docs_for_chunk(chunk, top_k=top_k)
        refuting_papers = gather_refuting_papers(refuting_docs)
        counterargument = generate_counterargument(chunk, refuting_papers)
        if counterargument is not None:
            all_counterarguments.append(counterargument)
    return all_counterarguments


# -----------------------
# Example usage:
# -----------------------
input_text = """This research paper claims that quantum entanglement does not influence the speed of classical computation in any meaningful way."""
counterarguments = generate_counterarguments_for_text(input_text, top_k=5)

for c in counterarguments:
    print("Counterargument:\n", c)
