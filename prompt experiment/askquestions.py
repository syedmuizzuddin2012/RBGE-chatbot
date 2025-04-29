import os
import csv
import requests

# Paths to your folders
prompts_folder = r'C:\Users\asus\Desktop\Edinburgh\Year 4\Semester 8\Dissertation\chatbot\experiment\prompt experiment\Sonny' #change according to which character
questions_folder = r'C:\Users\asus\Desktop\Edinburgh\Year 4\Semester 8\Dissertation\chatbot\experiment\prompt experiment'

# Path to your output CSV
output_csv = 'chatbot_evaluations.csv'

# Function to load prompt styles
def load_prompts(prompt_folder):
    prompts = {}
    for filename in os.listdir(prompt_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(prompt_folder, filename), 'r', encoding='utf-8') as f:
                prompts[filename] = f.read().strip()
    return prompts

# Function to load questions
def load_questions(question_folder):
    questions = []
    for filename in os.listdir(question_folder):
        if filename.endswith('.txt'):
            with open(os.path.join(question_folder, filename), 'r', encoding='utf-8') as f:
                questions.extend(f.read().splitlines())  # Assuming each line is a question
    return questions

# Function to send request to Flask API and get the response
def get_response_from_flask(prompt, question):
    url = 'http://localhost:5000/api/chat'  # URL of your Flask API
    data = {
        'message': question,
        'species': 'Dicksonia antarctica',  # assuming you always use this species
        'character': prompt  # the prompt style (Sonny, etc.)
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"Error for prompt: {prompt}, question: {question}")
        return None

# Function to save results to CSV
def save_to_csv(results):
    header = ['prompt_style', 'question', 'response']
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # if file is empty, write header
            writer.writerow(header)
        for row in results:
            writer.writerow(row)

# Main function to run the evaluation
def run_evaluation():
    # Load prompts and questions
    prompts = load_prompts(prompts_folder)
    questions = load_questions(questions_folder)

    results = []
    
    for prompt_name, prompt in prompts.items():
        for question in questions:
            print(f"Evaluating question: '{question}' with prompt: '{prompt_name}'")
            response = get_response_from_flask(prompt, question)
            if response:
                results.append([prompt_name, question, response])
    
    # Save results to CSV
    save_to_csv(results)
    print(f"Evaluation completed. Results saved to {output_csv}")

# Run the evaluation
if __name__ == '__main__':
    run_evaluation()
