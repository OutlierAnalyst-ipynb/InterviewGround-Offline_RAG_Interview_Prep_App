# InterviewGround — Offline RAG Interview Prep App

An interview preparation tool built on Retrieval-Augmented Generation (RAG). Your resume, notes, or job description are embedded into a local vector store, and a self-hosted LLM retrieves relevant context to generate interview questions, answers, and feedback.

## Hosted App
This app is hosted and run by me — no installation, setup, or local LLM required to use it.

## Tech Stack
- **Streamlit** — UI
- **ChromaDB** — local vector store for RAG
- **Ollama** — local LLM inference (runs on the host server)
- **Docker** — containerized deployment
- **Python 3.13**

## Project Structure
```
InterviewGround/
├── InterviewGround.py     # Streamlit app
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build
└── README.md
```

## Deployment (host-side only)
```bash
# Build the image
docker build -t interviewground .

# Run the container (connects to Ollama running on the host machine)
docker run -p 8501:8501 interviewground
```
Requires Ollama running on the host, with the model pulled (e.g. `ollama pull llama3`). App served at `http://localhost:8501`.

## Environment Variables
| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | URL of the host's Ollama server |
