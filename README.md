
# chat-pdf

A small tool to index and chat with PDF documents locally. It provides simple scripts to create a searchable index from PDF(s) and to run retrieval/chat interactions against that index.

## What the project does

- Indexes PDF files into a local retrieval index (embeddings + metadata).
- Exposes a simple retrieval/chat workflow to ask questions against indexed PDFs.
- Includes convenience scripts and a Docker Compose setup for reproducible execution.

Key files:
- [indexing.py](indexing.py) — build the index from PDF(s)
- [retreival.py](retreival.py) — perform queries / interactive retrieval
- [requirements.txt](requirements.txt) — Python dependencies
- [docker-compose.yml](docker-compose.yml) — optional containerized run
- [Manuscript_lirias.pdf](Manuscript_lirias.pdf) — example PDF included
- [tempCodeRunnerFile.py](tempCodeRunnerFile.py) — scratch / helpers

## Why this is useful

- Quickly turn PDFs into searchable knowledge for ad-hoc Q&A.
- Lightweight and easy to extend for custom PDF preprocessing or custom models.
- Works locally or inside Docker for reproducible environments.

## Getting started

Prerequisites
- Python 3.8+
- pip
- (Optional) Docker & docker-compose

Clone / open this project, then:

1. Create and activate a virtual environment
```sh
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies
```sh
pip install -r requirements.txt
```

3. Configure environment variables
- Copy and edit [.env](.env) as needed.

4. Index your PDF(s)
```sh
python indexing.py
# or, if you want to index a specific file:
python indexing.py --pdf Manuscript_lirias.pdf
```

5. Query the index / chat
```sh
python retreival.py
# or use a single query mode if the script supports it:
python retreival.py --query "Summarize the methods section"
```

Docker (optional)
```sh
docker-compose up --build
```

Note: The exact CLI flags and behaviors are implemented in [indexing.py](indexing.py) and [retreival.py](retreival.py). Open those files to adapt the workflow to your environment.

## Usage examples

Basic interactive flow:
1. Run the indexer: `python indexing.py`
2. Start the retriever: `python retreival.py`
3. Ask natural-language questions about the PDF content.

Programmatic example (pseudo-usage):
```py
# see [retreival.py](retreival.py) for actual API
from retreival import query_index
answers = query_index("What are the main findings?")
print(answers)
```

## Where to get help

- Open an issue in this repository (Issues).
- Inspect the scripts: [indexing.py](indexing.py) and [retreival.py](retreival.py) for inline docs and usage.
- For environment/config questions, check [.env](.env).

## Files of interest

- [indexing.py](indexing.py)
- [retreival.py](retreival.py)
- [requirements.txt](requirements.txt)
- [docker-compose.yml](docker-compose.yml)
- [Manuscript_lirias.pdf](Manuscript_lirias.pdf)

---
