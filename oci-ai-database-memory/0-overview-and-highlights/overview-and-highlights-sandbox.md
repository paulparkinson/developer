# Overview and Highlights

Without engineered context, AI systems can suffer from **context poisoning** (repeatedly referencing a hallucination) or **context distraction** (irrelevant data confusing the model's logic). Proper engineering transforms a reactive chatbot into a proactive, reliable collaborator that understands project-wide dependencies.

**Memory context engineering** is the structured discipline of designing systems that dynamically manage what an AI model "sees" and "remembers" to ensure reliable, personalized, and efficient performance. It represents a shift from **prompt engineering** (crafting a single message) to **system design** (managing the entire lifecycle of data entering and exiting the context window).

Modern context engineering often categorizes AI memory into layers inspired by human cognitive architecture:

- **Short-Term Memory** (Session Memory): Maintains the current conversation thread. It acts as a "scratchpad" for the AI to track its reasoning and intermediate steps during a single task.
- **Long-Term Memory**: Persists across different sessions. This stores durable facts (e.g., "the user prefers Python") and episodic events (e.g., "the last deployment failed due to a specific flag").
- **Semantic vs. Episodic**:
    - **Semantic Memory**: Durable, general facts and procedures.
    - **Episodic Memory**: Recalling specific past interactions or events to maintain continuity.

--------

### What You'll Build

You'll build **Proteus**, SeerGroup Solutions' AI-powered IT support agent. Proteus uses a complete memory system with seven distinct memory types — each serving a specific cognitive function:

| Memory Type | Purpose | Storage |
| --- | --- | --- |
| **Conversational** | Chat history per support ticket thread | SQL Table |
| **Knowledge Base** | Searchable documents, runbooks, & KB articles | Vector-Enabled SQL Table |
| **Workflow** | Learned resolution patterns from past tickets | Vector-Enabled SQL Table |
| **Toolbox** | Available tool definitions for agent use | Vector-Enabled SQL Table |
| **Entity** | People, places, and systems extracted from context | Vector-Enabled SQL Table |
| **Summary** | Compressed context for long conversations | Vector-Enabled SQL Table |
| **Tool Log** | Offloaded tool outputs for lean context windows | SQL Table |

--------

### Workshop Flow

| Activity | Focus | What You'll Do |
| --- | --- | --- |
| **1. Vector Search Foundations** | Semantic similarity search | Store and query SeerGroup KB articles using embeddings and HNSW indexes |
| **2. Memory Architecture Design** | Design decisions and storage setup | Create all tables and vector stores for seven memory types |
| **3. MemoryManager** | Core abstraction | Build the `MemoryManager` class with unified read/write for all memory types |
| **4. Context Engineering & Web Search** | Context window management | Implement token tracking, auto-summarization, just-in-time retrieval, and Tavily integration |
| **5. Agent Execution** | End-to-end agent loop | Run Proteus through multi-turn IT support scenarios |

### Key Concepts

- **Memory Engineering**: Design patterns for structuring and persisting agent memory across multiple specialized stores
- **Context Engineering**: Techniques for optimizing what enters the LLM context window — and when
- **Context Window Management**: Token usage monitoring with threshold-based auto-summarization
- **Just-in-Time Retrieval**: Compact summary pointers with on-demand expansion when details are needed
- **Entity Extraction**: LLM-powered recognition of servers, services, people, and teams from conversation text
- **Log Compaction**: Marking old messages as summarized (not deleting them) to preserve audit history while keeping context lean

### Technology Stack

| Component | Role |
| --- | --- |
| **Oracle Autonomous AI Database** | Vector-enabled SQL tables with AI Vector Search |
| **LangChain OracleVS** | Vector store abstraction for embedding storage and similarity search |
| **HuggingFace Embeddings** | `sentence-transformers/paraphrase-mpnet-base-v2` for 768-dimensional vectors |
| **OpenAI API** | LLM reasoning, entity extraction, and docstring augmentation |
| **Tavily API** | Web search integration for real-time information retrieval |

### About this Workshop

In this workshop, you'll learn how to engineer memory systems that give AI agents the ability to remember, learn, and adapt across conversations. Moving beyond simple RAG, you'll implement a complete **MemoryManager** with seven distinct memory types — each serving a specific cognitive function — alongside a **context engineering layer** with token tracking and auto-summarization, and a **full agent execution loop** with IT support scenarios.

**Estimated Workshop Time:** 45 minutes

### Prerequisites

- Oracle Cloud Infrastructure (OCI) Tenancy
- Cloud Environment details provided by LiveLabs
- Visual Studio Code (VS Code) Installed
- Python 3.14+ installed
- [OpenAI API Key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
- [Tavily API Key](https://www.tavily.com/)

### Objectives

By the end of this workshop you'll have a reusable `MemoryManager` class and agent loop that demonstrates how modern AI agents maintain context, learn from interactions, and manage information across sessions.

--------

## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026
