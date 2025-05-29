FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting Financial Assistant Multi-Agent System..."\n\
\n\
# Start all agents in background\n\
echo "📡 Starting API Agent (8000)..."\n\
uvicorn agents.api_agent.main:app --host 0.0.0.0 --port 8000 &\n\
\n\
echo "🕷️ Starting Scraping Agent (8001)..."\n\
uvicorn agents.scraping_agent.main:app --host 0.0.0.0 --port 8001 &\n\
\n\
echo "🔍 Starting Retriever Agent (8002)..."\n\
uvicorn agents.retriever_agent.main:app --host 0.0.0.0 --port 8002 &\n\
\n\
echo "📊 Starting Analysis Agent (8003)..."\n\
uvicorn agents.analysis_agent.main:app --host 0.0.0.0 --port 8003 &\n\
\n\
echo "🤖 Starting Language Agent (8004)..."\n\
uvicorn agents.language_agent.main:app --host 0.0.0.0 --port 8004 &\n\
\n\
echo "🧠 Starting Orchestrator (8005)..."\n\
uvicorn agents.orchestrator.main:app --host 0.0.0.0 --port 8005 &\n\
\n\
echo "🎙️ Starting Voice Agent (8006)..."\n\
uvicorn agents.voice_agent.main:app --host 0.0.0.0 --port 8006 &\n\
\n\
# Wait for services to start\n\
echo "⏳ Waiting for services to initialize..."\n\
sleep 15\n\
\n\
# Start Streamlit frontend\n\
echo "🎭 Starting Streamlit Frontend..."\n\
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/start.sh

# Make startup script executable
RUN chmod +x /app/start.sh

# Expose all ports
EXPOSE 8000 8001 8002 8003 8004 8005 8006 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the application
CMD ["/app/start.sh"]