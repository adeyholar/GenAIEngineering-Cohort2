from fastapi import FastAPI, UploadFile, File, Form
import os
import pandas as pd
from PyPDF2 import PdfReader
import re
import glob
import uvicorn
from sentence_transformers import SentenceTransformer
import faiss

app = FastAPI()
faiss_index = None
model = None # Initialize model globally to load once

# Define paths for better management
PDF_DIR = 'pdfs'
CSV_DIR = 'csv_files'

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# Corrected: Ensure the directory for the specific PDF exists, or generally the PDF_DIR
# The commented line was problematic because os.path.dirname('/Users//BeingMrsNunu/Documents/GitHub/GenAIEngineering-Cohort2/Week1/Day_2/pdfs/IndianBudget2025.pdf')
# would try to create a very specific path. Better to create the base PDF_DIR and CSV_DIR.
# os.makedirs(os.path.dirname('/Users//BeingMrsNunu/Documents/GitHub/GenAIEngineering-Cohort2/Week1/Day_2/pdfs/IndianBudget2025.pdf'), exist_ok=True)


def chunk_pdf_to_dataframe(pdf_path, num_chunks=5):
    reader = PdfReader(pdf_path)
    data = []
    filename = os.path.basename(pdf_path)
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text.strip():
            continue
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        # Ensure actual_num_chunks is not zero if text length is very small
        actual_num_chunks = min(num_chunks, max(1, len(text) // 100)) # Simple heuristic for min 1 chunk if text is small

        if actual_num_chunks <= 1:
            data.append({'filename': filename, 'page_number': page_num, 'chunk_number': 1, 'chunk': text})
        else:
            chunk_size = len(text) // actual_num_chunks
            for i in range(actual_num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(text))
                if i == actual_num_chunks - 1:
                    end = len(text)
                chunk = text[start:end]
                data.append({'filename': filename, 'page_number': page_num, 'chunk_number': i + 1, 'chunk': chunk})
    df = pd.DataFrame(data)
    print(f"DataFrame for {filename}:")
    print(df.head()) # Print head for brevity
    
    # Ensure the directory for CSV files exists before saving
    os.makedirs(CSV_DIR, exist_ok=True)
    df.to_csv(os.path.join(CSV_DIR, filename.replace('.pdf', '.csv')), index=False)

@app.post("/chunk_pdf")
async def chunk_pdf_endpoint(pdf_file_path: str = Form(None), num_chunks: int = Form(5), file: UploadFile = File(None)):
    """
    Endpoint to chunk a PDF.
    Can either receive a file directly via upload or a path to an existing file on the server.
    """
    if file:
        # Save the uploaded file temporarily
        file_location = os.path.join(PDF_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        pdf_to_process = file_location
    elif pdf_file_path:
        pdf_to_process = pdf_file_path
        # Basic check if the file exists
        if not os.path.exists(pdf_to_process):
            return {"status": "error", "message": f"File not found at {pdf_to_process}"}
    else:
        return {"status": "error", "message": "No PDF file or path provided."}

    chunk_pdf_to_dataframe(pdf_to_process, num_chunks=num_chunks)

    global faiss_index
    faiss_index = build_faiss_index()
    return {"status": "success", "message": f"PDF chunked successfully", "file_path": pdf_to_process}


def embed_text_chunks(chunks, embedding_model_name="all-MiniLM-L6-v2"):
    global model
    if model is None:
        model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

# Function to build a FAISS index
def build_faiss_index():
    files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not files:
        print(f"No CSV files found in {CSV_DIR}. Cannot build FAISS index.")
        return None

    print(f"Found CSV files: {files}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"Total chunks loaded: {len(df)}")
    
    if df.empty or 'chunk' not in df.columns:
        print("DataFrame is empty or 'chunk' column is missing. Cannot build FAISS index.")
        return None

    # Filter out empty or NaN chunks before embedding
    df_cleaned = df.dropna(subset=['chunk'])
    df_cleaned = df_cleaned[df_cleaned['chunk'].astype(str).str.strip() != '']
    
    if df_cleaned.empty:
        print("No valid chunks found after cleaning. Cannot build FAISS index.")
        return None

    embedding_model_name = "all-MiniLM-L6-v2"
    embeddings = embed_text_chunks(list(df_cleaned['chunk']), embedding_model_name)
    print('Embeddings completed')
    
    if embeddings.shape[0] == 0:
        print("No embeddings generated. Cannot build FAISS index.")
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print('Files indexed')
    return index


@app.post("/search_chunks")
def search_chunks(search_string: str):
    global faiss_index
    if faiss_index is None:
        return {"status": "error", "message": "FAISS index not built. Please chunk a PDF first."}

    query_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([search_string], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k=5)
    
    files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not files:
        return {"status": "error", "message": f"No CSV files found in {CSV_DIR} to search against."}

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    
    # Ensure indices are within the DataFrame's bounds
    valid_indices = [idx for idx in indices[0] if idx < len(df)]
    if not valid_indices:
        return {"status": "success", "message": "No matching chunks found.", "results": []}

    matches_df = df.iloc[valid_indices]
    return {"status": "success", "message": "Search complete", "results": matches_df.to_dict(orient='records')}


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy"}

def main():
    global faiss_index
    print("Attempting to build FAISS index on startup...")
    faiss_index = build_faiss_index()
    if faiss_index:
        print("FAISS index successfully built on startup.")
    else:
        print("FAISS index could not be built on startup. Please ensure CSV files exist or chunk a PDF.")

if __name__ == "__main__":
    print('Starting services')
    main()
    uvicorn.run(app, host="localhost", port=9322) # Ensure this port matches Streamlit