# 🔍 LangSmith Observability Practice

<p align="center">
  <b>Exploring LLM Observability, Tracing & Monitoring with LangSmith</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LangSmith-Observability-blueviolet?style=for-the-badge">
  <img src="https://img.shields.io/badge/LangChain-Framework-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/LangGraph-Workflows-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python">
</p>

---

## 🧠 About

This repository contains my hands-on practice with **LangSmith for LLM observability and monitoring**.

The main goal is to understand what happens inside AI applications by tracing **LLM calls, tool executions, agent workflows, RAG pipelines, and LangGraph nodes**.

Instead of looking only at the final response, LangSmith helps inspect the complete execution flow and identify performance issues, errors, and unexpected behavior.

---

## 🚀 What I Practiced

- 🔎 **LLM Tracing** — Inspect model inputs and outputs
- 📊 **Observability & Monitoring** — Monitor AI application execution
- 🤖 **AI Agents** — Trace agent reasoning and tool calls
- 🔧 **Tool Calling** — Observe tools such as web search and weather APIs
- 🧩 **LangGraph** — Monitor graph nodes and workflow execution
- 📚 **RAG Pipelines** — Trace retrieval and generation steps
- 🐞 **Debugging** — Identify errors and unexpected workflow behavior
- 📈 **Run Analysis** — Inspect latency, outputs, and execution details

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| 🐍 Python | Programming Language |
| 🦜 LangChain | LLM Application Framework |
| 🔗 LangGraph | Stateful AI Workflows |
| 🔍 LangSmith | Observability & Tracing |
| 🤖 Groq | LLM Provider |
| 📚 FAISS | pgvector | Vector Search |
| 🌐 Tavily | Web Search |
| ☁️ Open-Meteo | Weather Data |

---

## 📂 Project Focus

The repository contains different experiments and practice implementations covering:

```text
LangSmith
   │
   ├── LLM Tracing
   ├── AI Agents
   │      ├── Tool Calling
   │      ├── Web Search
   │      └── Weather Tool
   │
   ├── RAG
   │      ├── Document Loading
   │      ├── Chunking
   │      ├── Embeddings
   │      └── Retrieval
   │
   └── LangGraph
          ├── Nodes
          ├── State
          ├── Reducers
          └── Workflows