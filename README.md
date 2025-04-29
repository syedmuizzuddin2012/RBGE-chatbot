# 🌿 Project Sprout - RBGE Chatbot

This is a chatbot developed for the Royal Botanic Garden Edinburgh (RBGE) as part of an undergraduate dissertation project. It uses vector-based retrieval and prompt engineering to simulate plant personalities and assist with visitor questions.

## 📁 Project Structure

- app.py: Main application logic for running the chatbot.
- embeddings.py: Code for generating and storing text embeddings.
- prompts: Contains prompts for all characters: Sonny, Chill and Dixie.
- templates: HTML templates for the user interface.
- embeddings : Folder storing vector database files.
- library : Source documents used for knowledge retrieval and its text chunks


## 🚀 Getting Started

### 1. Install dependencies

### bash
pip install -r requirements.txt

### 2. Generate embeddings

python embeddings.py

### 3. Start the chatbot

python app.py

