import os
import numpy as np
import faiss
import openai
from flask import Flask, render_template, request, jsonify

# Initialize Flask app with template folder
# Change directory*
app = Flask(__name__, template_folder=r'C:\Users\asus\RBGE Chatbot\templates')

# OpenAI API key setup from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Function to load character-specific prompts from .txt files
def load_character_prompts(folder_path):
    prompts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            character = os.path.splitext(filename)[0]
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                prompts[character] = f.read().strip()
    return prompts

# Load character prompts from the 'prompts' folder
# Change directory*
prompt_folder_path = r'C:\Users\asus\RBGE Chatbot\prompts'
character_prompts = load_character_prompts(prompt_folder_path)

# Initialize dictionaries to store FAISS indices and text chunks
indices = {}
text_chunks = {}

# Directory paths
# Change directory*
species_folder_path = r'C:\Users\asus\RBGE Chatbot\library\Dicksonia antarctica'
embeddings_folder_path = r'C:\Users\asus\RBGE Chatbot\embeddings'

try:
    # Loop through document numbers 1 to 8
    for i in range(1, 9):
        # Load FAISS index from embeddings folder
        faiss_index_path = os.path.join(embeddings_folder_path, f"faiss_{i}.pdf.index")
        print(f"Trying to load FAISS index from: {faiss_index_path}")
        if os.path.exists(faiss_index_path):
            indices[i] = faiss.read_index(faiss_index_path)
            print(f"Loaded FAISS index for document {i}")
        else:
            print(f"FAISS index file not found: {faiss_index_path}")
        
        # Load text chunks from species folder
        chunks_file_path = os.path.join(species_folder_path, f"text_chunks_{i}.pdf.txt")
        print(f"Trying to load text chunks from: {chunks_file_path}")
        if os.path.exists(chunks_file_path):
            with open(chunks_file_path, 'r', encoding='utf-8') as f:
                chunks = f.read().split("\n----\n")
                text_chunks[i] = [chunk for chunk in chunks if chunk.strip()]
            print(f"Loaded text chunks for document {i} (total chunks: {len(text_chunks[i])})")
        else:
            print(f"Text chunks file not found: {chunks_file_path}")
        
        print(f"Successfully loaded data for document {i}\n")
except Exception as e:
    print(f"Error loading data: {e}")

# Route to render the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle chat API requests
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("Received data:", data)

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        user_query = data.get('message', '').strip()
        selected_species = data.get('species', '').strip()
        history = data.get('history', [])
        character = data.get('character', 'Sonny')

        if not user_query:
            return jsonify({"error": "Empty user query"}), 400
        if not selected_species:
            return jsonify({"error": "Species not selected"}), 400

        print(f"user_query: {user_query}")
        print(f"selected_species: {selected_species}")
        print(f"character: {character}")

        # For now, only support "Dicksonia antarctica"
        if selected_species != "Dicksonia antarctica":
            return jsonify({"error": f"Species {selected_species} not supported."}), 400

        # Embed the user's query
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[user_query]
        )
        query_embedding = np.array(response.data[0].embedding).reshape(1, -1)

        # Collect results from all documents
        all_results = []
        for doc_num in indices:
            index = indices[doc_num]
            chunks = text_chunks.get(doc_num, [])
            k = 1  # Top-k result per doc
            distances, indices_result = index.search(query_embedding, k)
            print(f"Document {doc_num}: distances {distances}, indices {indices_result}")
            if len(indices_result[0]) == 0:
                continue
            for idx in indices_result[0]:
                if idx < len(chunks):
                    distance = distances[0][0]
                    chunk = chunks[idx]
                    source_line = chunk.split('\n')[0]
                    source = source_line.replace('Source: ', '')
                    all_results.append((distance, chunk, source))

        if not all_results:
            return jsonify({"error": "No relevant text found"}), 404

        # Sort and get top 3 chunks
        all_results.sort(key=lambda x: x[0])
        top_results = all_results[:3]
        relevant_texts = [res[1] for res in top_results]
        sources = [res[2] for res in top_results]

        context = "\n".join(relevant_texts)
        source_info = "\n".join([f"Source: {source}" for source in sources])

        # Load system prompt based on character
        system_prompt = character_prompts.get(character, character_prompts["Sonny"])

        # Prepare GPT messages
        messages = history + [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context: {context}\n{source_info}"},
            {"role": "user", "content": user_query},
        ]
        print("Messages sent to GPT:", messages)

        # Call GPT-4o
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        gpt_response = response.choices[0].message.content

        return jsonify({"response": gpt_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
