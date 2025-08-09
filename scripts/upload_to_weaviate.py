import os
import json
from pathlib import Path

import weaviate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

WORKSPACE_ROOT = Path("/workspace")
EMBEDDED_DATASET_JSON = WORKSPACE_ROOT / "data/embeddings/college_dataset_with_embeddings.json"

# Connect to Weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={
        # Only necessary if your Weaviate instance expects this (e.g., when using text2vec-openai).
        # Kept for compatibility; not used when passing explicit vectors.
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
    },
)

# Create class if it doesn't exist
class_name = "CollegeDoc"
schema_class = {
    "class": class_name,
    "description": "Chunks of college documents",
    "vectorizer": "none",
    "properties": [
        {"name": "text", "dataType": ["text"]},
        {"name": "source", "dataType": ["text"]},
        {"name": "page", "dataType": ["int"]},
        {"name": "chunk_id", "dataType": ["text"]},
    ],
}

if not client.schema.contains({"classes": [{"class": class_name}]}):
    client.schema.create_class(schema_class)

# Load from JSON
if not EMBEDDED_DATASET_JSON.exists():
    raise FileNotFoundError(
        f"Embedded dataset not found at {EMBEDDED_DATASET_JSON}. Run scripts/generate_embeddings.py first."
    )

with EMBEDDED_DATASET_JSON.open("r", encoding="utf-8") as f:
    records = json.load(f)

# Upload to Weaviate in batches
client.batch.configure(batch_size=100, dynamic=True)

num_success = 0
num_skipped = 0

with client.batch as batch:
    for record in records:
        vector = record.get("embedding")
        if not vector:
            num_skipped += 1
            continue

        data_object = {
            "text": record.get("text", ""),
            "source": record.get("source", ""),
            "page": record.get("page"),
            "chunk_id": record.get("chunk_id"),
        }

        batch.add_data_object(
            data_object=data_object,
            class_name=class_name,
            vector=vector,
        )
        num_success += 1

print(f"✅ Upload complete. Inserted: {num_success}, skipped (no vector): {num_skipped}")
