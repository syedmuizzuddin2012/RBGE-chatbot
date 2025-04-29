import os
from pdfminer.high_level import extract_text
import numpy as np
import faiss
import openai

print("All libraries loaded successfully!")

# Initialize the OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Paths
species_name = "Dicksonia antarctica"
species_directory = r"C:\Users\asus\RBGE Chatbot\library\Dicksonia antarctica"
rbge_root = r"C:\Users\asus\RBGE Chatbot"  # Main RBGE Chatbot directory
embeddings_dir = os.path.join(rbge_root, "embeddings")  # Store all embeddings in a single folder


def load_pdfs_from_directory(species_directory):
    """Load all PDFs from a directory and extract their text using PDFMiner."""
    pdf_texts = {}
    for filename in os.listdir(species_directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(species_directory, filename)
            pdf_texts[filename] = extract_text(filepath)  # Extract text with PDFMiner
    return pdf_texts

def chunk_text(text, chunk_size=300, overlap=2):
    """Chunk text into pieces of approximately chunk_size characters."""
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind(" ", 0, chunk_size)
        if split_at == -1:  # No space found, force split
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:].strip()  # Strip leading spaces for next chunk
    chunks.append(text)
    return chunks

def generate_embeddings(chunks):
    """Generate embeddings for each text chunk."""
    embeddings = []
    for chunk in chunks:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[chunk]  # Ensure input is a list
        )
        embedding = response.data[0].embedding
        embeddings.append(embedding)
    return np.array(embeddings)  # Convert list of embeddings to numpy array

def store_embeddings_and_chunks(species_name, chunks, embeddings, filenames, species_directory, embeddings_dir):
    """Store embeddings in a single embeddings folder, and text chunks in the species folder."""

    # Ensure the species directory exists (for text chunks)
    os.makedirs(species_directory, exist_ok=True)

    # Save chunks with filenames for the specific species
    chunks_filepath = os.path.join(species_directory, f"text_chunks_{species_name}.txt")
    with open(chunks_filepath, 'w') as f:
        for chunk, filename in zip(chunks, filenames):
            f.write(f"Source: {filename}\n{chunk}\n----\n")

    # Ensure the embeddings folder exists
    os.makedirs(embeddings_dir, exist_ok=True)

    # Save embeddings directly in RBGE Chatbot/embeddings/
    embeddings_filepath = os.path.join(embeddings_dir, f"faiss_{species_name}.index")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, embeddings_filepath)

    print(f"Chunks saved in {species_directory}")
    print(f"Embeddings saved in {embeddings_filepath}")

def process_allpdf_species(species_name, species_directory, embeddings_dir):
    """Process PDFs for a single species."""
    print(f"Processing species: {species_name}...")

    # Load PDFs and extract text
    pdf_texts = load_pdfs_from_directory(species_directory)

    # Process each PDF's text
    all_chunks = []
    all_filenames = []
    for filename, text in pdf_texts.items():
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_filenames.extend([filename] * len(chunks))

    # Generate embeddings
    if all_chunks:
        embeddings = generate_embeddings(all_chunks)
        store_embeddings_and_chunks(species_name, all_chunks, embeddings, all_filenames, species_directory, embeddings_dir)
    else:
        print(f"No text found in {species_name}, skipping embedding storage.")

# Run the processing function
process_allpdf_species(species_name, species_directory, embeddings_dir)

