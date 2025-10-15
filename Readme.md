# RAG Teaching Assistant

## https://rag-ai-teaching-assistant-2o.streamlit.app/
An advanced AI-powered teaching assistant that uses Retrieval-Augmented Generation (RAG) to provide intelligent responses based on course video transcripts.

## Features

- **Semantic Search**: Uses BGE-M3 embeddings for accurate context retrieval
- **Real-time Streaming**: Groq API integration for fast, streaming responses
- **Enhanced Chunking**: Optimized chunk sizes for better context understanding
- **Video References**: Direct timestamps linking to source material
- **Interactive UI**: Modern, responsive interface with dark theme
- **Analytics Dashboard**: Live statistics and session tracking
- **Customizable Settings**: Adjustable context window and creativity parameters

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.3 70B)
- **Embeddings**: BGE-M3 via Ollama
- **Vector Search**: Scikit-learn Cosine Similarity
- **Data Processing**: Pandas, NumPy

## Architecture
User Query → Embedding Generation → Semantic Search →
Context Retrieval → LLM Processing → Streaming Response

## Setup Instructions

### Prerequisites

- Python 3.8+
- Ollama (for embeddings)
- Groq API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-teaching-assistant.git
cd rag-teaching-assistant
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Install Ollama and pull BGE-M3 model
```bash
# Install Ollama from https://ollama.ai
ollama pull bge-m3
```

4. Configure API key
```bash
cp config_template.py config.py
# Edit config.py and add your Groq API key
```

5. Run the application
```bash
streamlit run app.py
```

## Configuration

### API Keys

Create a `config.py` file:
```python
api_key = "your_groq_api_key_here"
```

### Streamlit Cloud Deployment

Add secrets in Streamlit Cloud dashboard:
```toml
[secrets]
GROQ_API_KEY = "your_groq_api_key_here"
```

Update `app.py` to read from secrets:
```python
from streamlit import secrets
api_key = secrets["GROQ_API_KEY"]
```

## Usage

1. Launch the application
2. Select number of context chunks (3-10)
3. Adjust creativity/temperature (0.0-1.0)
4. Ask questions about the course material
5. View responses with video references and timestamps

## Project Structure

- `app.py` - Main Streamlit application
- `config.py` - API configuration (not tracked in git)
- `embeddings.joblib` - Pre-computed embeddings
- `requirements.txt` - Python dependencies

## Improvements Over v1.0

- Replaced local Ollama LLM with Groq API for 10x faster inference
- Increased chunk size from 500 to 1000 tokens for better context
- Added real-time streaming for improved UX
- Implemented session analytics and statistics
- Enhanced UI with modern dark theme
- Added similarity score visualization
- Improved error handling and user feedback

## Performance Metrics

- Response Time: ~2-3 seconds (vs 15-20s with local LLM)
- Chunk Retrieval: <1 second
- Streaming: Real-time token generation
- Context Window: Up to 10 chunks (10,000 tokens)

## Future Enhancements

- Multi-modal support (images, diagrams)
- Conversation memory
- Export chat history
- Multi-language support
- Advanced analytics dashboard
# rag-ai-teaching-assistant-2.o
