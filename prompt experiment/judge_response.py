import os
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import json
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Score schema
class ChatbotEvaluation(BaseModel):
    completeness_accuracy: int = Field(..., ge=1, le=10)
    relevance_clarity: int = Field(..., ge=1, le=10)
    emotional_expression: int = Field(..., ge=1, le=10)
    engagement: int = Field(..., ge=1, le=10)

# Evaluation function
def evaluate_output(question: str, response: str) -> ChatbotEvaluation:
    system_prompt = (
        "You are an evaluator assessing the effectiveness of chatbot responses. "
        "Score each response on the following four dimensions from 1 to 10:\n\n"
        "1. Completeness & Accuracy\n"
        "2. Relevance & Clarity\n"
        "3. Emotional Expression\n"
        "4. Engagement\n\n"
        "Respond ONLY with valid JSON in the following format:\n"
        "{"
        "\"completeness_accuracy\": 0,"
        "\"relevance_clarity\": 0,"
        "\"emotional_expression\": 0,"
        "\"engagement\": 0"
        "}"
    )

    user_message = (
        f"User Question:\n{question}\n\n"
        f"Chatbot Response:\n{response}\n\n"
        "Please score this response based on the criteria."
    )

    # Use gpt-4o-mini
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        max_tokens=150,
        temperature=0.2
    )

    try:
        raw_json = completion.choices[0].message.content.strip()
        return ChatbotEvaluation.parse_raw(raw_json)
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"❌ JSON Parsing Error: {e}")
        return None

# File processor
def evaluate_file(input_path: str, output_path: str):
    if input_path.endswith(".xlsx"):
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    if not {"question", "response"}.issubset(df.columns):
        raise ValueError("Input file must contain 'question' and 'response' columns.")

    scores = []

    for i, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        response = str(row.get("response", "")).strip()

        if not question or not response:
            print(f"❌ Skipping row {i+1}: missing question or response.")
            scores.append({
                "completeness_accuracy": None,
                "relevance_clarity": None,
                "emotional_expression": None,
                "engagement": None
            })
            continue

        print(f"✅ Evaluating row {i+1}...")

        try:
            result = evaluate_output(question, response)
            if result:
                scores.append(result.dict())
            else:
                scores.append({
                    "completeness_accuracy": None,
                    "relevance_clarity": None,
                    "emotional_expression": None,
                    "engagement": None
                })
        except Exception as e:
            print(f"❌ Error on row {i+1}: {e}")
            scores.append({
                "completeness_accuracy": None,
                "relevance_clarity": None,
                "emotional_expression": None,
                "engagement": None
            })

        time.sleep(1)

    score_df = pd.DataFrame(scores)
    full_df = pd.concat([df, score_df], axis=1)

    if output_path.endswith(".xlsx"):
        full_df.to_excel(output_path, index=False)
    else:
        full_df.to_csv(output_path, index=False)

    print(f"\n✅ Finished! Results saved to {output_path}")

# Run
input_file = "chatbot_evaluations_Dixie.csv"
output_file = "chatbot_evaluations_Dixie_scored.csv"
evaluate_file(input_file, output_file)
