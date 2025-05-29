@echo off
set PYTHONPATH=.
start cmd /k "uvicorn agents.api_agent.main:app --port 8000"
start cmd /k "uvicorn agents.scraping_agent.main:app --port 8001"
start cmd /k "uvicorn agents.retriever_agent.main:app --port 8002"
start cmd /k "uvicorn agents.analysis_agent.main:app --port 8003"
start cmd /k "uvicorn agents.language_agent.main:app --port 8004"
start cmd /k "uvicorn agents.orchestrator.main:app --port 8005"
start cmd /k "uvicorn agents.voice_agent.main:app --port 8006"
