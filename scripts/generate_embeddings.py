import os
import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
WORKSPACE_ROOT = Path("/workspace")
INPUT_DATASET_JSON = WORKSPACE_ROOT / "data/raw_pdfs/college_dataset.json"
OUTPUT_DIR = WORKSPACE_ROOT / "data/embeddings"
OUTPUT_DATASET_JSON = OUTPUT_DIR / "college_dataset_with_embeddings.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64
NORMALIZE_EMBEDDINGS = True


def load_dataset(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found at {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def generate_embeddings_for_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    texts: List[str] = [record.get("text", "") for record in records]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
        convert_to_numpy=True,
    )

    for record, vector in zip(records, embeddings):
        record["embedding"] = vector.tolist()

    return records


def main() -> None:
    print(f"Loading dataset from: {INPUT_DATASET_JSON}")
    records = load_dataset(INPUT_DATASET_JSON)

    print(f"Generating embeddings with model: {MODEL_NAME}")
    records_with_embeddings = generate_embeddings_for_records(records)

    print(f"Saving dataset with embeddings to: {OUTPUT_DATASET_JSON}")
    save_dataset(records_with_embeddings, OUTPUT_DATASET_JSON)
    print("✅ Embeddings generated and dataset saved.")


if __name__ == "__main__":
    main()
