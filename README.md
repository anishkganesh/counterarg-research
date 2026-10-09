# Counterarg

Research prototypes for finding potentially opposing evidence in arXiv papers and generating a counterargument to a supplied statement.

## Overview

This repository contains two distinct approaches: a notebook that combines local semantic embeddings with sentiment filtering, and a Python script that uses OpenAI embeddings, stance classification, PDF retrieval, and generation. It is an experimental research workflow, not a hosted application or a validated fact-checking system.

## Features

- Read the arXiv metadata dataset distributed through Kaggle.
- Build an in-memory FAISS index for semantic retrieval.
- Retrieve candidate papers for a query or statement.
- Filter notebook results using a sentiment model.
- In the script, classify candidates as supporting, refuting, or neutral, download selected PDFs, and generate a response from extracted text.

## Architecture

| Implementation | Retrieval and filtering | Output |
| --- | --- | --- |
| `counter_research.ipynb` | SentenceTransformer `all-mpnet-base-v2`, FAISS, CardiffNLP sentiment pipeline | Exploratory paper retrieval and opposite-sentiment filtering |
| `counterarg.py` | OpenAI `text-embedding-ada-002`, FAISS inner-product index, GPT-4 stance prompts | A generated counterargument or `None` when evidence is unavailable |

The script embeds metadata records, retrieves candidates, checks their stance, downloads selected papers, extracts text with PyPDF2, and sends source excerpts to a language model. Only a limited excerpt from each retrieved source is used for generation.

Different sentiment does not establish logical contradiction. Likewise, a model-generated stance label or counterargument does not prove a claim false.

## Tech stack

Python, Jupyter, pandas, NumPy, FAISS, Sentence Transformers, Hugging Face Transformers, Kaggle, OpenAI, Requests, Beautiful Soup, PyPDF2, scikit-learn, and tqdm.

## Project structure

- `counter_research.ipynb` — local-model retrieval experiment.
- `counterarg.py` — API-assisted retrieval and counterargument pipeline.

No dependency manifest, web server, persisted FAISS index, or automated test suite is included.

## Run locally

Clone the repository and create a Python virtual environment. Activate it using the command appropriate to your shell.

```bash
git clone https://github.com/anishkganesh/counterarg-research.git
cd counterarg-research
python -m venv .venv
pip install jupyter pandas numpy faiss-cpu sentence-transformers transformers torch kaggle
jupyter notebook counter_research.ipynb
```

For the standalone script, install its additional dependencies:

```bash
pip install requests PyPDF2 scikit-learn tqdm beautifulsoup4 'openai<1'
```

The script uses the legacy pre-1.0 OpenAI Python interface. Installing a current SDK without adapting the code is not compatible with these calls. Model availability and provider access must be checked in your own account.

## Configuration and data

Configure Kaggle credentials locally and obtain the `Cornell-University/arxiv` dataset. The workflow expects `arxiv-metadata-oai-snapshot.json` in the working location used by the code.

`counterarg.py` contains an inline OpenAI key assignment. For a local run, replace that assignment with your own securely managed credential; exporting an environment variable alone does not override it. Do not reuse or redistribute a committed credential.

The script currently uses the full loaded metadata dataset, not the commented-out small sample. Index construction can consume substantial memory, time, and paid embedding requests. Review dataset size and the module-level execution before running it.

## Usage

The notebook is intended to be run cell by cell. Inspect intermediate retrieval and sentiment results rather than treating the final result as established evidence.

The script includes an example invocation at module level:

```bash
python counterarg.py
```

Running or importing the module can initialize the dataset/index and execute that example. Change the local example statement only after reviewing the workflow and its costs. The counterargument function can return `None` if suitable papers or a valid response are not found.

## Validation

No benchmark, accuracy evaluation, or automated tests are provided. Check retrieved paper identifiers, inspect source PDFs, assess whether evidence actually contradicts the claim, and compare generated statements with the cited excerpts. This documentation review did not download the dataset or execute paid model calls.

## Deployment

There is no deployment configuration or application server. Use this repository as a local research experiment.

## Limitations

- Metadata retrieval, sentiment, stance classification, and generation each introduce uncertainty.
- PDF extraction and network downloads can fail.
- The script index is rebuilt in memory and is not saved for reuse.
- Inner-product retrieval and limited text excerpts are not a complete evidence-verification method.

## Attribution and license

The dataset and models are supplied by their respective publishers. No standalone project license file is included; this README does not grant a new license.
