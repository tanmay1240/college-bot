import weaviate
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Connect to Weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

# Create class if it doesn't exist
class_name = "CollegeDoc"
if not client.schema.contains({"classes": [{"class": class_name}]}):
    client.schema.create_class({
        "class": class_name,
        "description": "Chunks of college documents",
        "vectorizer": "none",
        "properties": [
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]}
        ]
    })

# Load from JSON
with open("college_dataset.json", "r", encoding="utf-8") as f:
    records = json.load(f)

# Upload to Weaviate
for record in records:
    client.data_object.create(
        data_object={
            "text": record["text"],
            "source": record["source"]
        },
        class_name=class_name,
        vector=record["embedding"]
    )

print("✅ Upload to Weaviate complete.")
