# Introduction

## About this Workshop

This workshop will help you build and understand comprehensive memory and context engineering systems for AI agents using Oracle AI Database, LangChain, and modern LLM technologies.

You will learn how to engineer memory systems that give AI agents the ability to remember, learn, and adapt across conversations. Moving beyond simple RAG (Retrieval-Augmented Generation), you'll implement a complete **Memory Manager** with six distinct memory types—each serving a specific cognitive function.

In this hands-on workshop, you'll work with Oracle Database Free (running locally in Docker) to build sophisticated AI agents that leverage vector embeddings, semantic search, and multiple memory systems. You'll explore how to implement conversational memory, knowledge bases, workflow patterns, semantic tool discovery, entity extraction, and context summarization.

The workshop showcases the integration between modern LLM technologies and Oracle's AI Database features, enabling you to:
- Store and query vector embeddings at scale using Oracle AI Vector Search
- Implement intelligent semantic search for documents and tools
- Build multi-layered memory systems for stateful AI interactions
- Use LLM-powered entity extraction and recognition
- Implement context window management and just-in-time retrieval
- Create dynamic tool calling systems with semantic tool discovery
- Manage long-term context and conversation summarization

Estimated Workshop Time: 120 minutes

### Objectives

* Set up Oracle AI Database Free locally using Docker
* Understand memory engineering patterns for AI agents
* Implement vector search with LangChain and Oracle AI Database
* Build a Memory Manager with six distinct memory types
* Create semantic tool discovery and dynamic tool calling systems
* Implement entity extraction using LLMs
* Apply context engineering techniques for optimal LLM performance
* Build a complete AI agent with memory, tools, and context management
 
### Prerequisites

- Python 3.10+ installed on your local machine
- Docker Desktop or Docker Engine installed and running
- Basic understanding of Python programming
- Familiarity with AI/ML concepts is helpful but not required
- HuggingFace account (free) for LLM inference API access
- Tavily API key (free tier available) for web search capabilities

### What You'll Build

| Memory Type | Purpose | Storage |
|-------------|---------|---------|
| **Conversational** | Chat history per thread | SQL Table |
| **Knowledge Base** | Searchable documents & facts | Vector Store |
| **Workflow** | Learned action patterns | Vector Store |
| **Toolbox** | Dynamic tool definitions | Vector Store |
| **Entity** | People, places, systems extracted from context | Vector Store |
| **Summary** | Compressed context for long conversations | Vector Store |

### Key Concepts Covered

- **Memory Engineering**: Design patterns for agent memory systems
- **Context Engineering**: Techniques for optimizing what goes into the LLM context
- **Context Window Management**: Monitor usage, auto-summarize at thresholds
- **Just-in-Time Retrieval**: Compact summaries with on-demand expansion
- **Dynamic Tool Calling**: Semantic tool discovery and execution
- **Entity Extraction**: LLM-powered entity recognition and storage

### Let's Get Started

You may now **proceed to the next lab.**

## Want to Learn More?

* [Oracle AI Vector Search Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/)
* [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
* [Oracle Database Free](https://www.oracle.com/database/free/)
* [HuggingFace Models](https://huggingface.co/models)
