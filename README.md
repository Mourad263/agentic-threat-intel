# Agentic Threat Intel

An AI-powered cybersecurity threat-intelligence assistant that generates structured SOC-style reports using local RAG, live web search, LangGraph orchestration, and a local Ollama LLM.

The system takes a cybersecurity question and returns a structured report with:

- Overview
- Attack Explanation
- Recent Examples
- IOCs
- Detection
- Mitigation
- Limitations

## Features

- FastAPI backend
- LangGraph workflow orchestration
- Local Ollama model for report generation
- Tavily live web search for recent threat intelligence
- Local RAG retrieval from cybersecurity documents
- Topic-aware output for:
  - Ransomware
  - DDoS
  - Brute force
  - Trojan
- Topic-aware validator to prevent wrong-topic contamination
- Fast-path quality gate to skip unnecessary critic/reviser steps
- Simple web UI served from FastAPI
- Optional execution trace for debugging/demo mode

## Project Workflow

1. The user submits a cybersecurity question.
2. The planner detects the topic.
3. The router decides whether to use local RAG, web search, or both.
4. The retriever searches local cybersecurity documents.
5. Tavily retrieves recent web intelligence.
6. The writer generates a structured SOC-style report.
7. The fast-path gate checks whether the draft is strong enough.
8. If strong, critic/reviser are skipped.
9. The validator cleans and finalizes the output.
10. FastAPI returns the final report to the UI.

## Tech Stack

- Python
- FastAPI
- LangGraph
- LangChain
- ChromaDB
- Ollama
- Tavily API
- HTML/CSS/JavaScript frontend

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/agentic-threat-intel.git
cd agentic-threat-intel