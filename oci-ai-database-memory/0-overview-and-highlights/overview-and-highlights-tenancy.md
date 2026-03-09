# Overview and Highlights

Without engineered context, AI systems can suffer from **context poisoning** (repeatedly referencing a hallucination) or **context distraction** (irrelevant data confusing the model's logic). Proper engineering transforms a reactive chatbot into a proactive, reliable collaborator that understands project-wide dependencies.

**Memory context engineering** is the structured discipline of designing systems that dynamically manage what an AI model "sees" and "remembers" to ensure reliable, personalized, and efficient performance. It represents a shift from **prompt engineering** (crafting a single message) to **system design** (managing the entire lifecycle of data entering and exiting the context window).

Modern context engineering often categorizes AI memory into layers inspired by human cognitive architecture: 
- **Short-Term Memory** (Session Memory): Maintains the current conversation thread. It acts as a "scratchpad" for the AI to track its reasoning and intermediate steps during a single task.
- **Long-Term Memory**: Persists across different sessions. This stores durable facts (e.g., "the user prefers Python") and episodic events (e.g., "the last deployment failed due to a specific flag").
- **Semantic vs. Episodic**:
    - **Semantic Memory**: Durable, general facts and procedures.
    - **Episodic Memory**: Recalling specific past interactions or events to maintain continuity.

## What You'll Build

| Memory Type | Purpose | Storage
| --- | --- | ---
| **Conversational** | Chat history per thread | SQL Table
| **Knowledge Base** | Searchable documents & facts | Vector Store
| **Workflow** | Learned action patterns | Vector Store
| **Toolbox** | Dynamic tool definitions | Vector Store
| **Entity** | Peopl, places, systems extracted from context | Vector Store
| **Summary** | Compresse context for long conversations | Vector Store

## Key Concepts

- **Memory Engineering**: Design patterns for agent memory systems
- **Context Engineering**: Techniques for optimizing what goes into the LLM context
- **Context Window Management**: Monitor usage, auto-summarize at thresholds
- **Just-in-Time Retrieval**: Compact summaries with on-demand expansion
- **Entity Extraction**: LLM-powered entity recognition and storage

## About this Workshop

In this workshop, you'll learn how to engineer memory systems that give AI agents the ability to remember, learn, and adapt across conversations. Moving beyond simple RAG, we implement a complete **Memory Manager** with six distinct memory types—each serving a specific cognitive function.

**Estimated Workshop Time:** 45 minutes

## Pre-requisites

- Oracle Cloud Infrastructure (OCI) Tenancy
- Cloud Environment details provided by LiveLabs
- Visual Studio Code (VS Code) Installed
- Python 3.14+ installed
- [OpenAI API Key](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
- [Tavily API Key](https://www.tavily.com/)


## Objectives

By the end of this workshop you'll have a reusable MemoryLayer class and agent loop that demonstrates how modern AI agents maintain context, learn from interactions, and manage information across sessions.

## Learn More

- []()

## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026
