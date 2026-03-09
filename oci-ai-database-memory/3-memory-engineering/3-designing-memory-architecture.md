# Lab 3: Designing the Memory Architecture

## Memory Types, Design Decisions, and Storage Setup

--------

### Objective

In this lab, you'll understand *why* Proteus needs six distinct memory types, learn the design principles behind each one, and create all the database tables and vector stores that will power the Memory Manager in Lab 4.

This is the **architecture** lab — we focus on design decisions before writing implementation code.

### Why Memory Engineering Matters

Without memory, an IT support agent like Proteus:
- Forgets that the user already said they're on VPN
- Can't recall that AUTH-SVC crashed last week for the same reason
- Repeats the same diagnostic steps it already tried
- Loses track of which servers, services, and people have been mentioned

With proper memory engineering, Proteus can:
- Maintain context across long troubleshooting sessions
- Reuse proven resolution patterns from past incidents
- Track infrastructure entities and their relationships
- Compress long conversations while preserving key details

#### Definitions

**Agent Memory** is the exocortex that augments an LLM — capturing, encoding, storing, linking, and retrieving information beyond the model's parametric and contextual limits. It provides the persistence and structure required for long-horizon reasoning and reliable behaviour.

**Memory Engineering** is the scaffolding and control harness that we design to move information optimally and efficiently into, through, and across all components of an AI system (databases, LLMs, applications, etc). It ensures that data is captured, transformed, organized, and retrieved in the right way at the right time — so agents can behave reliably, believably, and capably.

--------

### The Six Memory Types

Just like humans have different types of memory (short-term, long-term, procedural), AI agents benefit from specialized memory systems. Here's what we'll build for Proteus:

| Memory Type | Human Analogy | SeerGroup Use Case | Storage |
|-------------|---------------|-------------------|---------|
| **Conversational** | Short-term memory | "The user said they're on Floor 2 with a MacBook" | SQL Table |
| **Knowledge Base** | Long-term semantic memory | KB articles, runbooks, vendor docs, search results | Vector-Enabled SQL Table |
| **Workflow** | Procedural memory | "Last time AUTH-SVC crashed, we ran these 3 steps" | Vector-Enabled SQL Table |
| **Toolbox** | Skill memory | Available diagnostic tools, search APIs, scripts | Vector-Enabled SQL Table |
| **Entity** | Episodic memory | "AUTH-SVC runs on prod-cluster-3, owned by Marcus Rivera" | Vector-Enabled SQL Table |
| **Summary** | Compressed memory | "45-minute debug session condensed to 5 bullet points" | Vector-Enabled SQL Table |
| **Tool Log** | Episodic memory | "Full kubectl output stored in DB, preview in context" | SQL Table |

> **Note on Tool Log:** Tool Log is a form of episodic memory — it records *what happened* during each tool execution. Beyond keeping the context window lean, tool logs can serve as a source from which **procedural memories** (workflow patterns) and **semantic memories** (knowledge base entries) can be distilled over time.

--------

### Programmatic vs. Agent-Triggered Operations

A key design decision in memory engineering is deciding which operations run **programmatically** (always executed by the harness code) versus **agent-triggered** (the LLM chooses to invoke them during reasoning).

In Proteus's design, the harness is intentionally opinionated: memory loading and persistence are automatic, while external retrieval and context compaction are chosen by the agent.

| Operation | Programmatic | Agent-Triggered | Notes |
|-----------|:------------:|:---------------:|-------|
| `read_conversational_memory()` | ✅ | ❌ | Always loaded at turn start (unsummarized turns only) |
| `read_knowledge_base()` | ✅ | ❌ | Always loaded at turn start |
| `read_workflow()` | ✅ | ❌ | Always loaded at turn start |
| `read_entity()` | ✅ | ❌ | Always loaded at turn start |
| `read_summary_context()` | ✅ | ❌ | Always loaded at turn start (IDs + descriptions) |
| `read_toolbox()` | ✅ | ❌ | Tool schemas retrieved before model reasoning |
| `write_conversational_memory()` | ✅ | ❌ | User message (pre-loop) + assistant answer (post-loop) |
| `write_workflow()` | ✅ | ❌ | Persisted after loop when tool steps exist |
| `write_entity()` | ✅ | ❌ | Best-effort extraction around user/final assistant text |
| `write_tool_log()` | ✅ | ❌ | Full tool output offloaded to DB after every tool execution |
| Tool-call decision (`tool_choice=auto`) | ❌ | ✅ | Model decides whether to call tools |
| `search_tavily()` | ❌ | ✅ | Agent-triggered external retrieval |
| `expand_summary()` | ❌ | ✅ | Agent-triggered just-in-time summary expansion |
| `summarize_and_store()` | ❌ | ✅ | Agent-triggered context compaction primitive |
| `summarize_conversation()` | ❌ | ✅ | Agent-triggered conversation compaction for active thread |

### Why This Split Works

1. **Reliability from programmatic memory** — critical memory load/save behavior never depends on the model remembering to do it.
2. **Adaptivity from agent-triggered tools** — the model can selectively fetch, expand, or compact only when needed.
3. **Clear control boundaries** — the harness owns state integrity; the model owns strategy inside those boundaries.

## Step 1: Define Table Names

Each memory type gets its own table. SQL tables for exact-match retrieval (conversational history, tool logs); vector-enabled SQL tables for semantic search (everything else).

    ```python
    # Table names for each memory type
    CONVERSATIONAL_TABLE   = "CONVERSATIONAL_MEMORY"   # Episodic memory
    KNOWLEDGE_BASE_TABLE   = "SEMANTIC_MEMORY"          # Semantic memory
    WORKFLOW_TABLE         = "WORKFLOW_MEMORY"           # Procedural memory
    TOOLBOX_TABLE          = "TOOLBOX_MEMORY"            # Procedural memory
    ENTITY_TABLE           = "ENTITY_MEMORY"             # Semantic memory
    SUMMARY_TABLE          = "SUMMARY_MEMORY"            # Semantic memory
    TOOL_LOG_TABLE         = "TOOL_LOG"                  # Episodic memory

    ALL_TABLES = [
        CONVERSATIONAL_TABLE, KNOWLEDGE_BASE_TABLE, WORKFLOW_TABLE,
        TOOLBOX_TABLE, ENTITY_TABLE, SUMMARY_TABLE, TOOL_LOG_TABLE,
    ]

    # Drop existing tables to start fresh
    for table in ALL_TABLES:
        try:
            with vector_conn.cursor() as cur:
                cur.execute(f"DROP TABLE {table} PURGE")
        except Exception as e:
            if "ORA-00942" in str(e):
                print(f"  - {table} (not exists)")
            else:
                print(f"  ✗ {table}: {e}")

    vector_conn.commit()
    ```

    ```python
    # Model token limits (for context management in Lab 5)
    MODEL_TOKEN_LIMITS = {
        "gpt-5": 256000,
        "gpt-5-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }
    ```

## Step 2: Create the Conversational Memory Table

Unlike semantic memories backed by vector stores, conversational memory uses a traditional SQL table because we need **exact retrieval by thread ID** (not similarity search). Each support ticket gets its own `thread_id`.

The table includes a `summary_id` column — when older messages are summarized and compressed, they're marked (not deleted) with a reference to the summary that replaced them.

    ```python
    def create_conversational_history_table(conn, table_name: str = "CONVERSATIONAL_MEMORY"):
        """
        Create a table to store conversational history.

        Columns:
        - id:         Unique message identifier
        - thread_id:  Groups messages by support ticket / conversation
        - role:       'user' or 'assistant'
        - content:    The message text
        - timestamp:  When the message was stored
        - metadata:   Optional JSON metadata
        - summary_id: Links to the summary that compressed this message (NULL if unsummarized)
        """
        with conn.cursor() as cur:
            try:
                cur.execute(f"DROP TABLE {table_name}")
            except:
                pass

            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                    thread_id VARCHAR2(100) NOT NULL,
                    role VARCHAR2(50) NOT NULL,
                    content CLOB NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata CLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary_id VARCHAR2(100) DEFAULT NULL
                )
            """)

            cur.execute(f"""
                CREATE INDEX idx_{table_name.lower()}_thread_id ON {table_name}(thread_id)
            """)

            cur.execute(f"""
                CREATE INDEX idx_{table_name.lower()}_timestamp ON {table_name}(timestamp)
            """)

        conn.commit()
        print(f"✅ Table {table_name} created with indexes (thread_id, timestamp)")
        return table_name
    ```

    ```python
    CONVERSATION_HISTORY_TABLE = create_conversational_history_table(
        vector_conn, CONVERSATIONAL_TABLE
    )
    ```

## Step 3: Create the Tool Log Table

Tool call outputs during agent execution can **bloat the context window** quickly — a single web search might return thousands of tokens that are only needed once.

The `TOOL_LOG` table acts as an **experimental memory**: full tool outputs are persisted to the database and replaced in the context window with a compact one-line reference. Proteus can retrieve full outputs later if needed.

This is a form of **context offloading** — keeping the working memory lean while preserving full fidelity in durable storage.

    ```python
    def create_tool_log_table(conn, table_name: str = "TOOL_LOG"):
        """Create a table to log tool call outputs (experimental memory)."""
        with conn.cursor() as cur:
            try:
                cur.execute(f"DROP TABLE {table_name}")
            except:
                pass

            cur.execute(f"""
                CREATE TABLE {table_name} (
                    id VARCHAR2(100) DEFAULT SYS_GUID() PRIMARY KEY,
                    thread_id VARCHAR2(100) NOT NULL,
                    tool_call_id VARCHAR2(200) NOT NULL,
                    tool_name VARCHAR2(200) NOT NULL,
                    tool_args CLOB,
                    tool_output CLOB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute(
                f"CREATE INDEX idx_{table_name.lower()}_thread ON {table_name}(thread_id)"
            )
        conn.commit()
        print(f"✅ Table {table_name} created")
        return table_name


    TOOL_LOG_TABLE_NAME = create_tool_log_table(vector_conn, TOOL_LOG_TABLE)
    ```

--------

## Step 4: Create Vector-Enabled Tables for Semantic Memories

Here we create five separate OracleVS-backed vector stores — one for each semantic memory type. Each uses the same embedding model for consistency.

| Vector Store Handle | Purpose |
|---------------------|---------|
| `knowledge_base_vs` | KB articles, runbooks, vendor docs, web search results |
| `workflow_vs` | Learned resolution patterns (tool sequences that worked) |
| `toolbox_vs` | Tool definitions for semantic tool discovery |
| `entity_vs` | Extracted entities: servers, services, people, teams |
| `summary_vs` | Compressed summaries for long troubleshooting sessions |

    ```python
    knowledge_base_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=KNOWLEDGE_BASE_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )

    workflow_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=WORKFLOW_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )

    toolbox_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=TOOLBOX_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )

    entity_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=ENTITY_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )

    summary_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=SUMMARY_TABLE,
        distance_strategy=DistanceStrategy.COSINE,
    )
    ```

### Build HNSW Indexes for Each Vector Store

    ```python
    print("Creating vector indexes...")
    safe_create_index(vector_conn, knowledge_base_vs, "knowledge_base_vs_hnsw")
    safe_create_index(vector_conn, workflow_vs, "workflow_vs_hnsw")
    safe_create_index(vector_conn, toolbox_vs, "toolbox_vs_hnsw")
    safe_create_index(vector_conn, entity_vs, "entity_vs_hnsw")
    safe_create_index(vector_conn, summary_vs, "summary_vs_hnsw")
    print("✅ All indexes created!")
    ```


## Step 5: Seed the Knowledge Base with SeerGroup Data

We'll reuse the SeerGroup KB articles from Lab 2 to populate the knowledge base memory. In production, this would be a continuous ingestion pipeline from your documentation systems.

    ```python
    # Seed knowledge base memory with SeerGroup KB articles
    if "seergroup_kb_articles" in globals() and seergroup_kb_articles:
        kb_texts = [
            f"Title: {a['title']}\nContent: {a['content']}" for a in seergroup_kb_articles
        ]
        kb_meta = [
            {
                "title": a["title"],
                "category": a["category"],
                "severity": a["severity"],
                "team": a["team"],
                "source_type": "internal_kb",
            }
            for a in seergroup_kb_articles
        ]
        knowledge_base_vs.add_texts(kb_texts, kb_meta)
        print(f"✅ Seeded knowledge base memory with {len(kb_texts)} SeerGroup KB articles")
    ```

## Lab 3 Recap

You've designed and created the complete memory infrastructure for Proteus:

| What You Did | Why It Matters |
|-------------|----------------|
| Defined 6 memory types with clear purposes | Each memory serves a distinct cognitive function |
| Chose SQL vs. vector storage per type | Exact retrieval (threads) vs. semantic search (meaning) |
| Designed programmatic vs. agent-triggered split | Reliability for core operations, flexibility for strategy |
| Created conversational memory table | Thread-based chat history with summary linkage |
| Created tool log table | Context offloading for lean working memory |
| Created 5 vector-enabled tables | Semantic search across knowledge, workflows, tools, entities, summaries |
| Seeded the knowledge base | SeerGroup KB articles ready for Proteus to search |

**Key Insight**: The `summary_id` column in conversational memory enables **log compaction** — a pattern borrowed from databases where old entries are compressed but not lost. Messages are *marked* as summarized, not deleted, preserving full audit history.

**Next up**: In Lab 4, we'll implement the `MemoryManager` class that provides clean read/write interfaces for all these memory types, and build the semantic `Toolbox` for dynamic tool discovery.

## Learn More



## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026