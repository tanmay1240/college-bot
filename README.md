# college-bot

A minimal pipeline to chunk college PDFs, generate embeddings, and load them into Weaviate for semantic search.

## Prerequisites
- Docker and Docker Compose
- Python 3.10+

## Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start Weaviate locally:
   ```bash
   cd weaviate-docker-compose && docker compose up -d && cd -
   ```

## Data
- PDFs: `data/raw_pdfs/` (already populated)
- Pre-chunked dataset (no vectors): `data/raw_pdfs/college_dataset.json`

If you need to rebuild the dataset from PDFs, adapt `scripts/make_dataset.py` to your paths or create a new chunking script. The provided JSON is ready to embed.

## Generate embeddings
Create vectors for each text chunk using a local sentence-transformers model:
```bash
python scripts/generate_embeddings.py
```
Outputs: `data/embeddings/college_dataset_with_embeddings.json`

## Upload to Weaviate
Import the embedded dataset into Weaviate with explicit vectors:
```bash
python scripts/upload_to_weaviate.py
```
Class: `CollegeDoc` with properties `text`, `source`, `page`, `chunk_id` and vectorizer `none`.

## Notes
- OPENAI keys are not required since vectors are generated locally. The uploader header is kept for compatibility but unused.
- Adjust batch sizes in scripts if needed.