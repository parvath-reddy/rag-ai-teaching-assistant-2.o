import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
from config import api_key

# Page configuration
st.set_page_config(
    page_title="RAG Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with greens and blues
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a1419 0%, #0d1f2d 50%, #0a1520 100%);
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .stChatMessage {
        background: transparent !important;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    div[data-testid="stChatMessageContent"] {
        background: rgba(10, 25, 35, 0.9) !important;
        border: 1px solid rgba(34, 139, 230, 0.3) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(21, 94, 117, 0.15);
    }
    
    [data-testid="stChatMessageContent"][data-testid*="user"] {
        background: linear-gradient(135deg, rgba(5, 78, 82, 0.25), rgba(4, 60, 78, 0.25)) !important;
        border: 1px solid rgba(34, 139, 230, 0.4) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #054e52 0%, #043c4e 50%, #032c3a 100%);
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.4s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(21, 94, 117, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover:before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #043c4e 0%, #032c3a 50%, #021e28 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(34, 139, 230, 0.5);
    }
    
    .video-reference {
        background: linear-gradient(135deg, rgba(10, 25, 35, 0.95), rgba(15, 30, 40, 0.95));
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid;
        border-image: linear-gradient(to bottom, #228be6, #1c7ed6, #1864ab) 1;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .video-reference:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(34, 139, 230, 0.3);
    }
    
    .timestamp-badge {
        background: linear-gradient(135deg, #0a1e28, #0f2835);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem 0.25rem 0.25rem 0;
        font-size: 0.85rem;
        color: #74c0fc;
        border: 1px solid rgba(34, 139, 230, 0.4);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(10, 25, 35, 0.9), rgba(15, 30, 40, 0.9));
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(34, 139, 230, 0.3);
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        border-color: rgba(34, 139, 230, 0.6);
        box-shadow: 0 8px 30px rgba(34, 139, 230, 0.4);
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #228be6 0%, #1c7ed6 50%, #1864ab 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(34, 139, 230, 0.5);
    }
    
    .stat-label {
        color: #74c0fc;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .welcome-section {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(10, 25, 35, 0.7), rgba(15, 30, 40, 0.7));
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 2px solid rgba(34, 139, 230, 0.3);
        box-shadow: 0 8px 32px rgba(21, 94, 117, 0.2);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .welcome-section:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(34, 139, 230, 0.15) 0%, transparent 70%);
        animation: pulse 8s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #228be6 0%, #1c7ed6 50%, #1864ab 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 0 40px rgba(34, 139, 230, 0.5);
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        color: #74c0fc;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }
    
    .sample-section-title {
        color: #a5d8ff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        background: linear-gradient(135deg, #228be6, #1c7ed6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stChatInput textarea {
        background: rgba(10, 25, 35, 0.9) !important;
        border: 2px solid rgba(34, 139, 230, 0.4) !important;
        color: #a5d8ff !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        backdrop-filter: blur(10px);
    }
    
    .stChatInput textarea:focus {
        border-color: #228be6 !important;
        box-shadow: 0 0 20px rgba(34, 139, 230, 0.5) !important;
    }
    
    h1, h2, h3 {
        color: #a5d8ff !important;
        font-weight: 700 !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(10, 20, 25, 0.95), rgba(10, 25, 35, 0.95));
        backdrop-filter: blur(10px);
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(34, 139, 230, 0.2), rgba(28, 126, 214, 0.2));
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        border: 1px solid rgba(34, 139, 230, 0.4);
        color: #74c0fc;
        font-size: 0.85rem;
        transition: all 0.3s ease;
    }
    
    .feature-badge:hover {
        background: linear-gradient(135deg, rgba(34, 139, 230, 0.3), rgba(28, 126, 214, 0.3));
        transform: scale(1.05);
    }
    
    .similarity-bar {
        height: 6px;
        background: linear-gradient(90deg, #228be6, #1c7ed6, #1864ab);
        border-radius: 3px;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(34, 139, 230, 0.5);
    }
    
    .progress-text {
        color: #4dabf7;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1419, #0d1f2d) !important;
    }
    
    .streaming-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #228be6;
        border-radius: 50%;
        margin-left: 5px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

# Functions
@st.cache_resource
def load_embeddings():
    """Load precomputed embeddings"""
    try:
        df = joblib.load('embeddings.joblib')
        return df, True
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, False

def create_embedding(text_list):
    """Generate embeddings using Ollama's BGE-M3 model"""
    try:
        # Use Ollama API for embeddings
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.info("Make sure Ollama is running with: ollama pull bge-m3")
        return None

def get_relevant_chunks(query, df, top_k=5):
    """Retrieve relevant chunks using semantic search"""
    query_embedding = create_embedding([query])
    if query_embedding is None:
        return None, None
    
    similarities = cosine_similarity(
        np.vstack(df['embedding']), 
        [query_embedding[0]]
    ).flatten()
    
    top_indices = similarities.argsort()[::-1][:top_k]
    top_similarities = similarities[top_indices]
    
    return df.loc[top_indices], top_similarities

def stream_response(prompt):
    """Stream response from Groq API"""
    try:
        client = Groq(api_key=api_key)
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error generating response: {e}"

def format_time_ago(timestamp):
    """Format timestamp to human readable"""
    now = datetime.now()
    diff = now - timestamp
    
    if diff.seconds < 60:
        return "Just now"
    elif diff.seconds < 3600:
        return f"{diff.seconds // 60} minutes ago"
    elif diff.seconds < 86400:
        return f"{diff.seconds // 3600} hours ago"
    else:
        return f"{diff.days} days ago"

# Sidebar
with st.sidebar:
    st.markdown("### Control Panel")
    
    # Stats Section
    st.markdown("---")
    st.markdown("### Live Statistics")
    
    if st.session_state.df is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(st.session_state.df)}</div>
                <div class="stat-label">Chunks</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            unique_videos = st.session_state.df['title'].nunique() if 'title' in st.session_state.df.columns else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{unique_videos}</div>
                <div class="stat-label">Videos</div>
            </div>
            """, unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(st.session_state.messages) // 2}</div>
                <div class="stat-label">Exchanges</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{st.session_state.total_queries}</div>
                <div class="stat-label">Queries</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Advanced Settings
    st.markdown("### Advanced Settings")
    
    top_k = st.slider("Context Chunks", 3, 10, 5, 
                     help="Number of relevant chunks to retrieve")
    
    temperature = st.slider("Creativity", 0.0, 1.0, 0.7, 0.1,
                           help="Response creativity (higher = more creative)")
    
    show_similarity = st.checkbox("Show Similarity Scores", value=False,
                                 help="Display relevance scores for retrieved chunks")
    
    show_metadata = st.checkbox("Show Metadata", value=False,
                               help="Display additional chunk information")
    
    st.markdown("---")
    
    # Session Info
    st.markdown("### Session Info")
    session_duration = datetime.now() - st.session_state.session_start
    st.markdown(f"""
    <div style='color: #74c0fc; font-size: 0.9rem;'>
        <strong>Started:</strong> {format_time_ago(st.session_state.session_start)}<br>
        <strong>Duration:</strong> {session_duration.seconds // 60} minutes
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Actions
    st.markdown("### Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
    
    st.markdown("---")
    
    # About
    st.markdown("### Features")
    st.markdown("""
    <div class='feature-badge'>Semantic Search</div>
    <div class='feature-badge'>Real-time Streaming</div>
    <div class='feature-badge'>Precise References</div>
    <div class='feature-badge'>Context-Aware</div>
    <div class='feature-badge'>Analytics</div>
    """, unsafe_allow_html=True)

# Load embeddings on startup
if not st.session_state.embeddings_loaded:
    with st.spinner("Loading embeddings..."):
        df, success = load_embeddings()
        if success:
            st.session_state.df = df
            st.session_state.embeddings_loaded = True
            st.success("System ready!")
            time.sleep(1)
        else:
            st.error("Failed to load embeddings. Please ensure embeddings.joblib exists.")
            st.stop()

# Welcome section (only shown when no messages)
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="welcome-section">
        <div class="welcome-title">RAG Teaching Assistant</div>
        <div class="welcome-subtitle">
            AI-Powered Course Navigation System with Semantic Search & Real-time Streaming
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="sample-section-title">Quick Start - Try These Questions</p>', unsafe_allow_html=True)
    
    sample_questions = [
        "How do I use loops in Python?",
        "Explain list comprehensions with examples",
        "What are function arguments and parameters?",
        "How to handle exceptions in Python?",
        "Explain the difference between lists and tuples",
        "What are lambda functions and when to use them?"
    ]
    
    cols = st.columns(2)
    for idx, question in enumerate(sample_questions):
        with cols[idx % 2]:
            if st.button(question, key=f"sample_{idx}", use_container_width=True):
                st.session_state.temp_query = question
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        # Show video references for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            if "chunks" in metadata and len(metadata["chunks"]) > 0:
                st.markdown("**Video References:**")
                for idx, chunk in enumerate(metadata["chunks"][:3]):
                    title = chunk.get('title', 'Unknown')
                    start = chunk.get('start', '00:00')
                    end = chunk.get('end', '00:00')
                    number = chunk.get('number', 'N/A')
                    
                    # Build the reference card HTML
                    reference_html = f"""
                    <div class="video-reference">
                        <strong>Video {number}:</strong> {title}<br>
                        <span class="timestamp-badge">{start} - {end}</span>
                    """
                    
                    # Add similarity score if enabled
                    if show_similarity and 'similarity' in chunk:
                        sim_score = chunk['similarity'] * 100
                        reference_html += f"""
                        <div class="similarity-bar" style="width: {sim_score}%"></div>
                        <div class="progress-text">Relevance: {sim_score:.1f}%</div>
                        """
                    
                    # Add metadata if enabled
                    if show_metadata and 'text' in chunk:
                        preview = chunk['text'][:100] + "..."
                        reference_html += f"""<div style="color: #4dabf7; font-size: 0.85rem; margin-top: 0.5rem; font-style: italic;">"{preview}"</div>"""
                    
                    reference_html += """</div>"""
                    
                    st.markdown(reference_html, unsafe_allow_html=True)

# Chat input
if 'temp_query' in st.session_state:
    query = st.session_state.temp_query
    del st.session_state.temp_query
else:
    query = st.chat_input("Ask about any topic from the course...")

# Process query
if query:
    st.session_state.total_queries += 1
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(query)
    
    # Get relevant chunks
    with st.spinner("Searching through course content..."):
        relevant_chunks, similarities = get_relevant_chunks(query, st.session_state.df, top_k=top_k)
    
    if relevant_chunks is not None:
        # Build prompt
        chunks_json = relevant_chunks[["title", "number", "start", "end", "text"]].to_json(orient="records")
        
        prompt = f"""You are an AI teaching assistant for a Python programming course. 

Here are the most relevant video transcript chunks:
{chunks_json}

Student Question: "{query}"

Instructions:
1. Answer the question naturally and clearly with proper formatting
2. Reference specific video numbers and timestamps where this topic is covered
3. Guide the student to watch the relevant sections
4. Use examples when appropriate
5. If the question is unrelated to the course, politely explain you can only help with course-related questions
6. Be concise but informative and engaging

Answer:"""
        
        # Stream response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Streaming with indicator
            for chunk_text in stream_response(prompt):
                full_response += chunk_text
                response_placeholder.markdown(full_response + "<span class='streaming-indicator'></span>", unsafe_allow_html=True)
            
            # Final response without indicator
            response_placeholder.markdown(full_response)
            
            # Show video references
            st.markdown("**Video References:**")
            chunks_with_similarity = relevant_chunks.head(3).copy()
            for idx, (_, chunk) in enumerate(chunks_with_similarity.iterrows()):
                # Build the reference card HTML
                reference_html = f"""
                <div class="video-reference">
                    <strong>Video {chunk.get('number', 'N/A')}:</strong> {chunk.get('title', 'Unknown')}<br>
                    <span class="timestamp-badge">{chunk.get('start', '00:00')} - {chunk.get('end', '00:00')}</span>
                """
                
                # Add similarity score if enabled
                if show_similarity and similarities is not None:
                    sim_score = similarities[idx] * 100
                    reference_html += f"""
                    <div class="similarity-bar" style="width: {sim_score}%"></div>
                    <div class="progress-text">Relevance: {sim_score:.1f}%</div>
                    """
                
                # Add metadata if enabled
                if show_metadata:
                    preview = chunk.get('text', '')[:100] + "..."
                    reference_html += f"""<div style="color: #4dabf7; font-size: 0.85rem; margin-top: 0.5rem; font-style: italic;">"{preview}"</div>"""
                
                reference_html += """</div>"""
                
                st.markdown(reference_html, unsafe_allow_html=True)
        
        # Save to session state with similarity scores
        chunks_dict = relevant_chunks.head(3).to_dict('records')
        if similarities is not None:
            for i, chunk in enumerate(chunks_dict):
                chunk['similarity'] = float(similarities[i])
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "metadata": {
                "chunks": chunks_dict,
                "timestamp": datetime.now().isoformat(),
                "query": query
            }
        })
        
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #4dabf7; font-size: 0.9rem; padding: 1rem;'>
    <strong>Powered by</strong> RAG Architecture • Groq API • BGE-M3 Embeddings • Streamlit<br>
    <span style='font-size: 0.8rem; color: #74c0fc;'>Built with care for Enhanced Learning Experience</span>
</div>
""", unsafe_allow_html=True)