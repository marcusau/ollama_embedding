import ollama
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle
import json

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text content and rich metadata from PDF file using PyMuPDF
    Returns list of dictionaries containing text and metadata
    """
    pages_data = []
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        # Get document-level metadata
        doc_metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'subject': doc.metadata.get('subject', ''),
            'keywords': doc.metadata.get('keywords', ''),
            'creator': doc.metadata.get('creator', ''),
            'producer': doc.metadata.get('producer', ''),
            'creation_date': doc.metadata.get('creationDate', ''),
            'modification_date': doc.metadata.get('modDate', ''),
            'total_pages': len(doc),
            'pdf_version': doc.version,
            'format': doc.format,
            'is_encrypted': doc.is_encrypted,
            'file_size': os.path.getsize(pdf_path)
        }
        
        # Iterate through pages
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page-specific metadata
            page_metadata = {
                'page_number': page_num + 1,
                'source': pdf_path,
                'width': page.rect.width,
                'height': page.rect.height,
                'rotation': page.rotation,
                'doc_metadata': doc_metadata,
                'images_count': len(page.get_images()),
                'links_count': len(page.get_links()),
                'has_annots': bool(page.annots),
                'fonts': list(page.get_fonts()),
            }
            
            # Get text with blocks information
            blocks = page.get_text("blocks")
            text_blocks = []
            
            for block in blocks:
                block_info = {
                    'text': block[4],
                    'block_position': {
                        'x0': block[0],
                        'y0': block[1],
                        'x1': block[2],
                        'y1': block[3]
                    }
                }
                text_blocks.append(block_info)
            
            # Store text with all metadata
            pages_data.append({
                'text': page.get_text(),
                'text_blocks': text_blocks,
                'metadata': page_metadata
            })
            
        doc.close()
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
        
    return pages_data

def chunk_text(pages_data: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict]:
    """
    Split text into smaller chunks using LangChain's RecursiveCharacterTextSplitter
    Preserves metadata for each chunk
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_with_metadata = []
    
    for page_data in pages_data:
        chunks = text_splitter.split_text(page_data['text'])
        
        # Preserve metadata for each chunk
        for chunk in chunks:
            chunks_with_metadata.append({
                'text': chunk,
                'metadata': page_data['metadata']
            })
    
    return chunks_with_metadata

def get_embeddings(chunks_with_metadata: List[Dict], model: str = "nomic-embed-text") -> List[Dict]:
    """
    Generate embeddings for text chunks using Ollama
    """
    if not model.endswith("-embed") and not model.startswith("nomic"):
        print("Warning: Using a non-embedding model. This may result in suboptimal embeddings.")
        print("Recommended: Use 'nomic-embed-text' for better embedding quality and performance.")
    
    embeddings_with_metadata = []
    
    for chunk_data in chunks_with_metadata:
        try:
            # Get embedding for chunk
            response = ollama.embeddings(model=model, prompt=chunk_data['text'])
            embedding = np.array(response['embedding'])
            
            # Store embedding with its text and metadata
            embeddings_with_metadata.append({
                'embedding': embedding,
                'text': chunk_data['text'],
                'metadata': chunk_data['metadata']
            })
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            continue
    
    return embeddings_with_metadata

def process_pdf_to_embeddings(
    pdf_path: str, 
    output_dir: str = "embeddings", 
    model: str = "nomic-embed-text",
    chunk_size: int = 1000,
    chunk_overlap: int = 100
) -> Dict:
    """
    Process a PDF file and store embeddings in FAISS index
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract text from PDF with metadata
    print(f"Extracting text from {pdf_path}")
    pages_data = extract_text_from_pdf(pdf_path)
    
    # Chunk the text using LangChain
    print("Chunking text...")
    chunks_with_metadata = chunk_text(pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings_with_metadata = get_embeddings(chunks_with_metadata, model)
    
    # Prepare data for FAISS
    embeddings_matrix = np.array([item['embedding'] for item in embeddings_with_metadata])
    dimension = embeddings_matrix.shape[1]
    
    # Create and populate FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix.astype('float32'))
    
    # Save FAISS index
    index_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_index.faiss")
    faiss.write_index(index, index_path)
    
    # Save metadata and texts separately
    metadata = [{
        'text': item['text'],
        'metadata': item['metadata']
    } for item in embeddings_with_metadata]
    
    metadata_path = os.path.join(output_dir, f"{os.path.basename(pdf_path)}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    print(f"Index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return {
        'index_path': index_path,
        'metadata_path': metadata_path,
        'dimension': dimension,
        'num_vectors': len(embeddings_with_metadata)
    }

def search_similar_chunks(query: str, index_path: str, metadata_path: str, model: str = "nomic-embed-text", k: int = 5):
    """
    Search for similar chunks using FAISS
    """
    # Load the index
    index = faiss.read_index(index_path)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get embedding for query
    response = ollama.embeddings(model=model, prompt=query)
    query_embedding = np.array([response['embedding']]).astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, k)
    
    # Return results with metadata
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        results.append({
            'text': metadata[idx]['text'],
            'metadata': metadata[idx]['metadata'],
            'distance': float(distance)
        })
    
    return results

def ask_question(question: str, context: str, model: str = "llama2:3b") -> str:
    """
    Ask a question using the LLM model
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = ollama.chat(model=model, messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])
    return response['message']['content']

if __name__ == "__main__":
    # Example usage
    pdf_path = "/Users/marcusau/Desktop/llm_project/marker_test3/2024082701155_c.pdf"
    
    # Process PDF and create FAISS index
    result = process_pdf_to_embeddings(
        pdf_path, 
        model="nomic-embed-text",
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # Example search
    query = "What is the main topic?"
    similar_chunks = search_similar_chunks(
        query,
        result['index_path'],
        result['metadata_path'],
        k=3
    )
    
    # Use the most relevant chunk for Q&A
    context = similar_chunks[0]['text']
    answer = ask_question(query, context, model="llama2:3b")
    
    print("Answer:", answer)
    print("\nSource:", similar_chunks[0]['metadata'])
