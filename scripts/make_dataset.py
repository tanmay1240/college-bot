import fitz  # PyMuPDF
import csv
from tqdm import tqdm

# === CONFIGURATION ===
pdf_files = [
    r"D:\college_chatbot\data\raw_pdfs\1752853618SCAN_20250710_170026887.pdf",
    r"D:\college_chatbot\data\raw_pdfs\1752853692SCAN_20250708_114455539.pdf",
    r"D:\college_chatbot\data\raw_pdfs\1753465340TIME TABLE.pdf",
    r"D:\college_chatbot\data\raw_pdfs\1753465462300625-academic-calendar-2025-26.pdf",
    r"D:\college_chatbot\data\raw_pdfs\CSE Syllabus 18-07-2025.pdf"
]

output_file = r"D:\college_chatbot\data\raw_pdfs\college_dataset.csv"
chunk_size = 300  # number of words per chunk

# === HELPER FUNCTIONS ===

def extract_text_chunks_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:
            chunks.extend(
                split_text_into_chunks(text, chunk_size, page_num + 1)
            )
    return chunks

def split_text_into_chunks(text, size, page_number):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), size):
        chunk_text = " ".join(words[i:i + size])
        chunks.append((chunk_text, page_number))
        
    return chunks

# === MAIN SCRIPT ===

def build_dataset():
    chunk_id = 1

    with open(output_file, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["chunk_id", "text_chunk", "source", "page"])

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            source_name = pdf_path.split("\\")[-1]
            chunks = extract_text_chunks_from_pdf(pdf_path)

            for chunk_text, page in chunks:
                writer.writerow([f"{chunk_id:04d}", chunk_text, source_name, page])
                chunk_id += 1

    print(f"\n✅ Dataset saved to: {output_file}")

# === RUN ===
if __name__ == "__main__":
    build_dataset()
