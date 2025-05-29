# 🏦 Multi-Agent Financial Assistant

An AI-powered financial analysis system with 7 specialized agents providing real-time portfolio insights and market intelligence.

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit     │
│   Frontend      │
│    (8501)       │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Orchestrator  │
│     (8005)      │
└─────┬───┬───┬───┘
      │   │   │
   ┌──▼─┐ │ ┌─▼──┐
   │8000│ │ │8006│
   │API │ │ │Voice│
   └────┘ │ └────┘
        ┌─▼─┐ ┌───┐ ┌───┐
        │8001││8002││8003│
        │Scrp││Retr││Anly│
        └───┘└───┘└─┬─┘
                   │
               ┌───▼───┐
               │ 8004  │
               │Language│
               └───────┘
```

## 🤖 Agents Overview

| Agent | Port | Purpose |
|-------|------|---------|
| **🎭 Streamlit** | 8501 | Web interface and user interaction |
| **🧠 Orchestrator** | 8005 | Query routing and response coordination |
| **📡 API Agent** | 8000 | External API integrations (market data, earnings) |
| **🕷️ Scraping Agent** | 8001 | Web scraping for news and market sentiment |
| **🔍 Retriever Agent** | 8002 | Information retrieval and document search |
| **📊 Analysis Agent** | 8003 | Portfolio risk analysis and calculations |
| **🤖 Language Agent** | 8004 | AI responses using Groq/Gemini LLMs |
| **🎙️ Voice Agent** | 8006 | Text-to-speech optimization |

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API keys**
   ```bash
   set GROQ_API_KEY=your_groq_key_here
   set GEMINI_API_KEY=your_gemini_key_here
   ```

3. **Start all agents**
   ```bash
   start_agents.bat
   ```

4. **Launch web interface**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🔄 Agent Workflow

### Query Flow Example: "What's our NVDA exposure?"

1. **Streamlit** receives user query
2. **Orchestrator** routes to appropriate agents
3. **API Agent** fetches current NVDA price data
4. **Analysis Agent** calculates portfolio exposure
5. **Language Agent** generates AI response using Groq
6. **Orchestrator** combines results
7. **Streamlit** displays response to user

## 🤖 AI Integration

- **Primary LLM**: Groq (Llama 3.1 70B) - Fast responses
- **Backup LLM**: Google Gemini - Reliability
- **Fallbacks**: Context-aware responses when APIs fail

## 📁 Project Structure

```
finance-assistant/
├── streamlit_app.py
├── start_agents.bat
├── agents/
│   ├── orchestrator/main.py     # (8005)
│   ├── api_agent/main.py        # (8000)
│   ├── scraping_agent/main.py   # (8001)
│   ├── retriever_agent/main.py  # (8002)
│   ├── analysis_agent/main.py   # (8003)
│   ├── language_agent/main.py   # (8004)
│   └── voice_agent/main.py      # (8006)
├── data_loaders.py
└── requirements.txt
```

## 🔧 Configuration

### Required Environment Variables
```bash
GROQ_API_KEY=gsk_your_key_here      # Primary LLM
GEMINI_API_KEY=AIza_your_key_here   # Backup LLM
```

### Service Health Check
```bash
curl http://localhost:8005/health   # Orchestrator
curl http://localhost:8004/debug    # Language Agent status
```

## 💬 Usage Examples

- "What's our risk exposure in Asia tech stocks?"
- "How is NVDA performing today?"
- "Any earnings surprises this week?"
- "What's the market sentiment for semiconductors?"

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect at share.streamlit.io
3. Add API keys in secrets
4. Deploy

### Local Development
```bash
# Check all services
check_services.bat

# Manual agent restart
uvicorn agents.language_agent.main:app --port 8004
```