import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

# Page config
st.set_page_config(page_title="Video Q&A", page_icon="🎥", layout="wide")

# Title
st.title("🎥 Video Content Q&A System")
st.write("Ask questions about your video course content!")

@st.cache_data
def load_data():
    """Load the embeddings data"""
    try:
        df = joblib.load('embeddings.joblib')
        st.success(f"✅ Loaded {len(df)} video chunks")
        return df
    except:
        st.error("❌ embeddings.joblib not found. Run preprocess_json.py first!")
        return None

def create_embedding(text):
    """Get embedding for query"""
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": [text]
        })
        return r.json()["embeddings"][0]
    except:
        st.error("❌ Ollama not running. Start with: ollama serve")
        return None

def get_response(prompt):
    """Get AI response"""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        return r.json()["response"]
    except:
        st.error("❌ Error getting response from Ollama")
        return None

# Load data
df = load_data()

if df is not None:
    # Show stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Chunks", len(df))
    with col2:
        st.metric("🎬 Videos", df['title'].nunique())
    
    st.markdown("---")
    
    # Query input
    query = st.text_area("💬 Ask your question:", 
                        placeholder="e.g., How do I create variables in Python?",
                        height=100)
    
    if st.button("🔍 Get Answer", type="primary") and query:
        with st.spinner("Searching..."):
            # Get query embedding
            query_embedding = create_embedding(query)
            
            if query_embedding:
                # Find similar content
                embeddings_matrix = np.vstack(df['embedding'].values)
                similarities = cosine_similarity(embeddings_matrix, [query_embedding]).flatten()
                
                # Get top 5 results
                top_indices = similarities.argsort()[::-1][:5]
                results = df.iloc[top_indices]
                
                # Create prompt
                context = results[["title", "number", "start", "end", "text"]].to_json(orient="records")
                
                prompt = f'''I am teaching python fundamentals in my data science course. Here are video subtitle chunks:

{context}
---------------------------------
"{query}"
User asked this question related to the video chunks. Answer in a human way where and how much content is taught in which video (with video number and timestamp) and guide the user to go to that particular video. If unrelated question, tell them you can only answer course-related questions.'''

                # Get AI response
                with st.spinner("Generating answer..."):
                    response = get_response(prompt)
                
                if response:
                    # Show response
                    st.markdown("## 🤖 Answer")
                    st.write(response)
                    
                    # Show source chunks
                    st.markdown("## 📋 Related Content")
                    for idx, row in results.iterrows():
                        with st.expander(f"📹 {row['title']} - Video {row['number']} ({similarities[idx]:.3f} similarity)"):
                            st.write(f"**Text:** {row['text']}")
                            st.write(f"**Time:** {int(row['start']//60)}:{int(row['start']%60):02d} - {int(row['end']//60)}:{int(row['end']%60):02d}")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### 📝 How to use:
    1. Make sure Ollama is running: `ollama serve`
    2. Make sure you have the models: `ollama pull bge-m3` and `ollama pull llama3.2`
    3. Ask any question about your course content
    4. Get answers with exact video timestamps!
    """)

else:
    st.markdown("""
    ### 🚀 Setup Instructions:
    1. Run `python preprocess_json.py` to generate embeddings
    2. Start Ollama: `ollama serve`  
    3. Install models: `ollama pull bge-m3` and `ollama pull llama3.2`
    4. Refresh this page
    """)