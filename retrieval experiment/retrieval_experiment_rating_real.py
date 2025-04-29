import os
import numpy as np
import faiss
import pandas as pd
import openai
import pickle
from itertools import product
from pdfminer.high_level import extract_text

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

print("All libraries loaded successfully!")

# Function to extract text from PDFs
def load_pdfs_from_directory(directory):
    pdf_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            pdf_texts[filename] = extract_text(filepath)
    return pdf_texts

# Function to read questions from corresponding TXT files
def load_questions_from_directory(directory):
    questions = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                questions[filename.replace(".txt", ".pdf")] = [line.strip() for line in f.readlines() if line.strip()]
    return questions

# Function to chunk text with overlap
def chunk_text(text, chunk_size, overlap_percentage):
    overlap = int(chunk_size * (overlap_percentage / 100))
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks

# Function to generate embeddings in batches
def generate_embeddings_in_batches(chunks, model, batch_size=5):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        response = openai.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    return embeddings

# Function to save/load embeddings and chunks in separate folders
def save_embeddings_and_chunks(library_dir, chunk_size, overlap, embedding_model, base_embedding_dir):
    # Create a unique folder for each setting
    embedding_save_dir = os.path.join(base_embedding_dir, f"chunk_{chunk_size}_overlap_{overlap}")
    os.makedirs(embedding_save_dir, exist_ok=True)  # Ensure directory exists

    pdf_texts = load_pdfs_from_directory(library_dir)
    embeddings = {}
    chunks = {}

    for filename, text in pdf_texts.items():
        emb_file = os.path.join(embedding_save_dir, f"{filename}_embeddings.pkl")
        chunk_file = os.path.join(embedding_save_dir, f"{filename}_chunks.pkl")

        if os.path.exists(emb_file) and os.path.exists(chunk_file):
            with open(emb_file, 'rb') as f:
                embeddings[filename] = pickle.load(f)
            with open(chunk_file, 'rb') as f:
                chunks[filename] = pickle.load(f)
        else:
            new_chunks = chunk_text(text, chunk_size, overlap)
            chunk_embeddings = generate_embeddings_in_batches(new_chunks, embedding_model)
            embeddings[filename] = chunk_embeddings
            chunks[filename] = new_chunks

            with open(emb_file, 'wb') as f:
                pickle.dump(chunk_embeddings, f)
            with open(chunk_file, 'wb') as f:
                pickle.dump(new_chunks, f)

    return embeddings, chunks

# Function to build FAISS index
def build_faiss_index(embeddings):
    all_embeddings = []
    for embedding_list in embeddings.values():
        all_embeddings.extend(embedding_list)
    
    embeddings_array = np.array(all_embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_array)
    
    return faiss_index

# Function to process queries
def search_index(query, faiss_index, embeddings, chunks, k, embedding_model):
    query_embedding = generate_embeddings_in_batches([query], embedding_model)[0]
    query_embedding = np.array([query_embedding]).astype('float32')
    
    D, I = faiss_index.search(query_embedding, k)
    relevant_chunks = []
    
    for i in I[0]:
        for filename, chunk_embeddings in embeddings.items():
            if i < len(chunk_embeddings):
                chunk_text = chunks[filename][i]
                relevant_chunks.append((filename, i, chunk_text))
    return relevant_chunks

# Function to evaluate retrieved chunks
def evaluate_retrieved_chunks(query, retrieved_chunk_text):
    prompt = f"""
    You are evaluating the relevance, completeness, and accuracy of the retrieved text in relation to the user's query.

    ---
    User Query:
    {query}

    Retrieved Chunk:
    {retrieved_chunk_text}

    ---
    Provide scores on a scale from 1.0 to 10.0 (allow decimals) in this exact format:
    Relevance: X.XX  
    Completeness: X.XX  
    Accuracy: X.XX  

    No extra text before or after this format.
    """

    response = openai.chat.completions.create(
        model="gpt-4o",  # Use GPT-4o for better performance
        messages=[{"role": "system", "content": "You are an evaluator of text retrieval quality."},
                  {"role": "user", "content": prompt}]
    ).choices[0].message.content.strip()

    # Handle cases where the response is "N/A" or can't be evaluated
    if "N/A" in response or not response:
        return 0.0, 0.0, 0.0  # Assign 0 if the response is N/A or empty

    # If the evaluation is valid, parse the scores
    try:
        # Extract the scores for relevance, completeness, and accuracy
        relevance, completeness, accuracy = map(float, [score.split(": ")[1] for score in response.split("\n")])
    except ValueError:
        # In case there's any issue with parsing (e.g., malformed response), assign 0
        relevance, completeness, accuracy = 0.0, 0.0, 0.0

    return relevance, completeness, accuracy


# Function to save results to CSV
def save_results_to_csv(results, output_file='query_results.csv'):
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

# Main execution
if __name__ == "__main__":
    library_dir = r"C:\\Users\\asus\\Desktop\\Edinburgh\\Year 4\\Semester 8\\Dissertation\\chatbot\\experiment\\lib for testing"
    questions_dir = r"C:\\Users\\asus\\Desktop\\Edinburgh\\Year 4\\Semester 8\\Dissertation\\chatbot\\experiment\\questions"
    base_embedding_dir = r"C:\\Users\\asus\\Desktop\\Edinburgh\\Year 4\\Semester 8\\Dissertation\\chatbot\\experiment\\embeddings"
    embedding_model = "text-embedding-3-small"

    chunk_sizes = [200, 350, 500]  # Small, average, and large chunk sizes
    overlaps = [0, 25, 50]  # Small, average, and large overlaps
    k_values = [1, 3, 5]  # Small, average, and large kNN values

    document_questions = load_questions_from_directory(questions_dir)
    results = []

    # Get available PDFs in the library
    available_docs = set(os.listdir(library_dir))

    for chunk_size, overlap in product(chunk_sizes, overlaps):
        print(f"\n🔹 Processing Chunk Size: {chunk_size}, Overlap: {overlap}")

        embeddings, chunks = save_embeddings_and_chunks(
            library_dir, chunk_size, overlap, embedding_model, base_embedding_dir
        )

        # Loop through each document and its associated questions
        for document, questions in document_questions.items():
            if document not in available_docs:  # Skip if the document is not in the library
                print(f"⚠️ Skipping questions for {document} (not in library)")
                continue

            print(f"\n📄 Processing questions for {document}")

            # Only use embeddings and chunks for the current document
            doc_embeddings = {document: embeddings[document]}
            doc_chunks = {document: chunks[document]}

            # Build FAISS index for the current document
            faiss_index = build_faiss_index(doc_embeddings)

            # Loop through all queries and k-values
            for query, k in product(questions, k_values):
                print(f"🔍 Querying: '{query}' with kNN={k}")

                relevant_chunks = search_index(query, faiss_index, doc_embeddings, doc_chunks, k, embedding_model)

                for filename, chunk_index, retrieved_chunk_text in relevant_chunks:
                    # Evaluate retrieved chunk using the modified evaluate function
                    relevance, completeness, accuracy = evaluate_retrieved_chunks(query, retrieved_chunk_text)

                    # Append results with individual scores
                    results.append({
                        "Document": document,
                        "Chunk Size": chunk_size,
                        "Overlap": overlap,
                        "kNN": k,
                        "Query": query,
                        "Document Filename": filename,
                        "Chunk Index": chunk_index,
                        "Chunk Text": retrieved_chunk_text,
                        "Relevance": relevance,
                        "Completeness": completeness,
                        "Accuracy": accuracy
                    })

    # Save results to CSV
    save_results_to_csv(results)
    print(f"✅ Results saved in: {os.path.abspath('query_results.csv')}")

