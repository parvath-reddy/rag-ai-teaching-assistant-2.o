import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
from groq import Groq
from config import api_key

client = Groq(api_key=api_key)

def create_embedding(text_list):
    """
    Generate embeddings using Ollama local API.
    """
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def inference_groq_stream(prompt):
    """
    Generate streamed response from Groq API.
    Prints content as it arrives.
    """
    print("Thinking...\n")
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # enable streaming
    )

    final_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content if chunk.choices[0].delta and chunk.choices[0].delta.content else ""
        print(content, end="", flush=True)
        final_response += content
    print("\n")  # add newline after completion
    return final_response


# Load precomputed embeddings
df = joblib.load('embeddings.joblib')

# Get user query
incoming_query = input("Ask a Question: ")

# Generate embedding for the query
question_embedding = create_embedding([incoming_query])[0]

# Find similarities with precomputed embeddings
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# Get top results
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]

# Build RAG-style prompt
prompt = f'''
I am teaching Python fundamentals in my data science course. 
Here are video subtitle chunks (title, number, start, end, text):
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------
"{incoming_query}"
Answer naturally: explain where this topic is taught, mention video number and timestamps,
and guide the user to go to that particular video. 
If the question is unrelated to the course, politely say you can only answer course-related questions.
'''

# Save prompt to file (optional)
with open("prompt.txt", "w") as f:
    f.write(prompt)

# Get streamed response from Groq API
response = inference_groq_stream(prompt)

# Save final answer
with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)
