# Customer Service AI Agent

Production-grade customer service chat agent built with:

- LangChain (tool-mediated reasoning)
- LangGraph (durable orchestration)
- OpenAI API
- FastAPI
- Retrieval-Augmented Generation (RAG)

## Setup

1. Create virtual environment:
   python3.11 -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -e .

3. Configure environment:
   cp .env.example .env
   # Fill in your API keys

## Project Structure

- app/        Core application code
- data/       Documents and embeddings
- scripts/    One-off utilities
- tests/      Automated tests

## Status

Phase 1 complete: Data & RAG Foundations.
