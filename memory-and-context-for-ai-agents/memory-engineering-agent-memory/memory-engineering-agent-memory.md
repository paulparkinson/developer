# Lab 3: Memory Engineering and Agent Memory

## Introduction

In this lab, you will implement **memory engineering** patterns for AI agents with Oracle AI Database. You'll build a comprehensive **Memory Manager** class that unifies all memory operations, including the **Toolbox** for semantic tool discovery. This lab covers Part 3 of the notebook: "Memory Engineering and Agent Memory."

Memory engineering is the practice of designing, storing, and retrieving structured information that enables AI agents to maintain context, learn from interactions, and make informed decisions across sessions.

Estimated Time: 40 minutes

### Objectives

In this lab, you will:
* Understand the architecture of the Memory Manager
* Implement conversational memory using SQL tables
* Create read/write methods for vector store memories
* Implement entity extraction using LLMs
* Build workflow tracking and storage
* Test the complete memory system

### Prerequisites

* Completed Lab 2: Implement Vector Search and Memory Stores
* Vector stores and indexes created for all memory types
* Understanding of vector search concepts

## Task 1: Understanding the Memory Manager Architecture

The Memory Manager handles six types of memory:

| Memory Type | Storage | Purpose | Key Methods |
|-------------|---------|---------|-------------|
| **Conversational** | SQL Table | Chat history per thread | `read_conversational_memory()`, `write_conversational_memory()` |
| **Knowledge Base** | Vector Store | Searchable documents | `read_knowledge_base()`, `write_knowledge_base()` |
| **Workflow** | Vector Store | Action patterns | `read_workflow()`, `write_workflow()` |
| **Toolbox** | Vector Store | Tool definitions | `read_toolbox()`, `write_toolbox()` |
| **Entity** | Vector Store | Extracted entities | `read_entity()`, `write_entity()` |
| **Summary** | Vector Store | Compressed conversations | `read_summary_memory()`, `write_summary()` |

### Key Design Decisions

**Programmatic vs Agentic Operations:**

| Operation | Type | Reason |
|-----------|------|--------|
| Memory reads | Programmatic | Agent needs context to function—can't know what it doesn't know |
| Memory writes | Programmatic | Must be reliable—can't trust agent to remember to save |
| Tool calls | Agentic | Only agent knows what information is needed |

## Task 2: Create the Conversational Memory Table

First, let's create a SQL table to store conversation history.

1. **Create a file** named `create_conversation_table.py`:

    ```python
    <copy>
    import oracledb

    ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN
    )

    # Create the conversational memory table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS CONVERSATIONAL_MEMORY (
        id VARCHAR2(100) PRIMARY KEY,
        thread_id VARCHAR2(100) NOT NULL,
        role VARCHAR2(20) NOT NULL,
        content CLOB NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        summary_id VARCHAR2(100),
        INDEX idx_thread_summary (thread_id, summary_id)
    )
    """

    # Oracle doesn't support CREATE TABLE IF NOT EXISTS, so we handle exceptions
    try:
        with vector_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE CONVERSATIONAL_MEMORY (
                    id VARCHAR2(100) PRIMARY KEY,
                    thread_id VARCHAR2(100) NOT NULL,
                    role VARCHAR2(20) NOT NULL,
                    content CLOB NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary_id VARCHAR2(100)
                )
            """)
            
            # Create index
            cur.execute("""
                CREATE INDEX idx_thread_summary 
                ON CONVERSATIONAL_MEMORY (thread_id, summary_id)
            """)
            
        vector_conn.commit()
        print("✅ Created CONVERSATIONAL_MEMORY table")
    except Exception as e:
        if "ORA-00955" in str(e):  # Table already exists
            print("ℹ️ CONVERSATIONAL_MEMORY table already exists")
        else:
            print(f"⚠️ Error: {e}")

    # Verify table structure
    with vector_conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type 
            FROM user_tab_columns 
            WHERE table_name = 'CONVERSATIONAL_MEMORY'
            ORDER BY column_id
        """)
        columns = cur.fetchall()
        print("\\n📊 Table structure:")
        for col_name, data_type in columns:
            print(f"  • {col_name}: {data_type}")

    vector_conn.close()
    </copy>
    ```

2. **Run the script:**

    ```bash
    <copy>
    python create_conversation_table.py
    </copy>
    ```

## Task 3: Implement the Memory Manager Class

Now let's build the complete Memory Manager class.

1. **Create a file** named `memory_manager.py`:

    ```python
    <copy>
    import json as json_lib
    import uuid
    from datetime import datetime
    from typing import Optional, List, Dict
    from langchain.schema import Document

    class MemoryManager:
        """
        A memory manager for AI agents using Oracle AI Database.
        
        Manages 6 types of memory:
        - Conversational: Chat history per thread (SQL table)
        - Knowledge Base: Searchable documents (Vector store)
        - Workflow: Execution patterns (Vector store)
        - Toolbox: Available tools (Vector store)
        - Entity: People, places, systems (Vector store)
        - Summary: Compressed context (Vector store)
        """
        
        def __init__(self, conn, conversation_table: str, knowledge_base_vs,
                     workflow_vs, toolbox_vs, entity_vs, summary_vs):
            self.conn = conn
            self.conversation_table = conversation_table
            self.knowledge_base_vs = knowledge_base_vs
            self.workflow_vs = workflow_vs
            self.toolbox_vs = toolbox_vs
            self.entity_vs = entity_vs
            self.summary_vs = summary_vs
        
        # ==================== CONVERSATIONAL MEMORY ====================
        
        def read_conversational_memory(self, thread_id: str, limit: int = 50) -> str:
            """
            Read conversation history for a specific thread.
            Returns only messages that haven't been summarized (summary_id IS NULL).
            """
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT role, content, timestamp
                    FROM {self.conversation_table}
                    WHERE thread_id = :1 AND summary_id IS NULL
                    ORDER BY timestamp ASC
                    FETCH FIRST :2 ROWS ONLY
                """, [thread_id, limit])
                rows = cur.fetchall()
            
            if not rows:
                return "[]"
            
            messages = []
            for role, content, ts in rows:
                messages.append({
                    "role": role,
                    "content": content,
                    "timestamp": ts.isoformat() if ts else None
                })
            
            return json_lib.dumps(messages, indent=2)
        
        def write_conversational_memory(self, content: str, role: str, thread_id: str):
            """Store a conversation message."""
            msg_id = str(uuid.uuid4())
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {self.conversation_table} 
                    (id, thread_id, role, content, timestamp)
                    VALUES (:1, :2, :3, :4, CURRENT_TIMESTAMP)
                """, [msg_id, thread_id, role, content])
            self.conn.commit()
        
        def mark_as_summarized(self, thread_id: str, summary_id: str):
            """Mark messages as summarized so they're filtered out of reads."""
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {self.conversation_table}
                    SET summary_id = :1
                    WHERE thread_id = :2 AND summary_id IS NULL
                """, [summary_id, thread_id])
            self.conn.commit()
        
        # ==================== KNOWLEDGE BASE MEMORY ====================
        
        def read_knowledge_base(self, query: str, k: int = 5) -> str:
            """Retrieve relevant documents from knowledge base."""
            results = self.knowledge_base_vs.similarity_search(query, k=k)
            if not results:
                return "No relevant knowledge found."
            
            output = []
            for i, doc in enumerate(results, 1):
                output.append(f"{i}. {doc.page_content}")
                if doc.metadata:
                    output.append(f"   Source: {doc.metadata}")
            
            return "\\n".join(output)
        
        def write_knowledge_base(self, content: str, metadata: Optional[Dict] = None):
            """Store a document in the knowledge base."""
            doc = Document(page_content=content, metadata=metadata or {})
            self.knowledge_base_vs.add_documents([doc])
        
        # ==================== WORKFLOW MEMORY ====================
        
        def read_workflow(self, query: str, k: int = 3) -> str:
            """Retrieve similar execution patterns."""
            results = self.workflow_vs.similarity_search(
                query, 
                k=k,
                filter={"num_steps": {"$gt": 0}}  # Only workflows with steps
            )
            if not results:
                return "No similar workflows found."
            
            output = ["Previous similar workflows:"]
            for i, doc in enumerate(results, 1):
                output.append(f"{i}. {doc.page_content}")
            
            return "\\n".join(output)
        
        def write_workflow(self, query: str, steps: List[str], result: str):
            """Store an execution pattern."""
            workflow_text = f"Query: {query}\\nSteps: {' -> '.join(steps)}\\nResult: {result}"
            doc = Document(
                page_content=workflow_text,
                metadata={
                    "query": query,
                    "num_steps": len(steps),
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.workflow_vs.add_documents([doc])
        
        # ==================== ENTITY MEMORY ====================
        
        def read_entity(self, query: str, k: int = 5) -> str:
            """Retrieve relevant entities."""
            results = self.entity_vs.similarity_search(query, k=k)
            if not results:
                return "No relevant entities found."
            
            output = ["Relevant entities:"]
            for doc in results:
                entity_type = doc.metadata.get("type", "unknown")
                output.append(f"- [{entity_type}] {doc.page_content}")
            
            return "\\n".join(output)
        
        def write_entity(self, entity: str, entity_type: str, context: str,
                        llm_client=None, text: str = ""):
            """
            Extract and store entities using LLM.
            If llm_client is provided, uses LLM to extract entities from text.
            Otherwise, stores the provided entity directly.
            """
            if llm_client and text:
                # Use LLM to extract entities
                entities = self._extract_entities_llm(llm_client, text)
                docs = []
                for ent in entities:
                    doc = Document(
                        page_content=f"{ent['entity']}: {ent['context']}",
                        metadata={
                            "type": ent["type"],
                            "entity": ent["entity"],
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    docs.append(doc)
                if docs:
                    self.entity_vs.add_documents(docs)
            else:
                # Store provided entity
                doc = Document(
                    page_content=f"{entity}: {context}",
                    metadata={
                        "type": entity_type,
                        "entity": entity,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                self.entity_vs.add_documents([doc])
        
        def _extract_entities_llm(self, llm_client, text: str) -> List[Dict]:
            """Use LLM to extract entities from text."""
            prompt = f"""Extract entities (people, places, organizations, systems) from the following text.
            Return ONLY a JSON array with this format:
            [
                {{"entity": "name", "type": "person|place|organization|system", "context": "brief context"}}
            ]

            Text: {text}

            Return only the JSON array, no other text.
            """
            
            try:
                from openai import OpenAI
                response = llm_client.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                entities = json_lib.loads(content)
                return entities if isinstance(entities, list) else []
            except:
                return []
        
        # ==================== SUMMARY MEMORY ====================
        
        def read_summary_memory(self, summary_id: str) -> str:
            """Retrieve a specific summary by ID."""
            results = self.summary_vs.similarity_search(
                summary_id,
                k=1,
                filter={"id": summary_id}
            )
            if not results:
                return f"Summary {summary_id} not found."
            
            return results[0].page_content
        
        def read_summary_context(self) -> str:
            """
            Return compact summary references for context.
            Returns format: [Summary ID: xxx] Description
            """
            # Get all summaries (in practice, you might want to limit this)
            with self.conn.cursor() as cur:
                # Query the summary vector store's underlying table
                table_name = self.summary_vs._get_table_name()
                cur.execute(f"""
                    SELECT id, metadata
                    FROM "{table_name}"
                    ORDER BY id
                """)
                rows = cur.fetchall()
            
            if not rows:
                return ""
            
            output = ["Previous conversation summaries:"]
            for row_id, metadata_json in rows:
                metadata = json_lib.loads(metadata_json) if metadata_json else {}
                description = metadata.get("description", "No description")
                output.append(f"[Summary ID: {row_id}] {description}")
            
            return "\\n".join(output)
        
        def write_summary(self, summary_id: str, summary_text: str, 
                         description: str, thread_id: str):
            """Store a conversation summary."""
            doc = Document(
                page_content=summary_text,
                metadata={
                    "id": summary_id,
                    "description": description,
                    "thread_id": thread_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            self.summary_vs.add_documents([doc])

    print("✅ Memory Manager class defined")
    </copy>
    ```

## Task 4: Initialize and Test the Memory Manager

Let's create an instance of the Memory Manager and test its functionality.

1. **Create a file** named `test_memory_manager.py`:

    ```python
    <copy>
    import oracledb
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy
    from memory_manager import MemoryManager

    # Connection setup
    ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN
    )

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Recreate vector store objects
    knowledge_base_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="KNOWLEDGE_BASE_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    workflow_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="WORKFLOW_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    toolbox_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="TOOLBOX_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    entity_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="ENTITY_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    summary_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="SUMMARY_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    # Initialize Memory Manager
    memory_manager = MemoryManager(
        conn=vector_conn,
        conversation_table="CONVERSATIONAL_MEMORY",
        knowledge_base_vs=knowledge_base_vs,
        workflow_vs=workflow_vs,
        toolbox_vs=toolbox_vs,
        entity_vs=entity_vs,
        summary_vs=summary_vs
    )

    print("✅ Memory Manager initialized\\n")

    # ==================== TEST CONVERSATIONAL MEMORY ====================
    print("=" * 50)
    print("Testing Conversational Memory")
    print("=" * 50)

    thread_id = "test-thread-1"

    # Write some messages
    memory_manager.write_conversational_memory(
        "Hello! I need help with Oracle Database.",
        "user",
        thread_id
    )

    memory_manager.write_conversational_memory(
        "I'd be happy to help! What specific aspect of Oracle Database are you interested in?",
        "assistant",
        thread_id
    )

    memory_manager.write_conversational_memory(
        "I want to learn about vector search capabilities.",
        "user",
        thread_id
    )

    # Read back the conversation
    conversation = memory_manager.read_conversational_memory(thread_id)
    print(f"\\nConversation for thread {thread_id}:")
    print(conversation)

    # ==================== TEST KNOWLEDGE BASE ====================
    print("\\n" + "=" * 50)
    print("Testing Knowledge Base Memory")
    print("=" * 50)

    # Add some knowledge
    memory_manager.write_knowledge_base(
        "Oracle AI Vector Search enables storing and querying vector embeddings natively in the database.",
        {"source": "oracle-docs", "topic": "vector-search"}
    )

    memory_manager.write_knowledge_base(
        "Vector indexes like IVF use inverted file structures for fast approximate nearest neighbor search.",
        {"source": "documentation", "topic": "indexes"}
    )

    # Query knowledge base
    query = "How does vector search work in Oracle?"
    knowledge = memory_manager.read_knowledge_base(query, k=2)
    print(f"\\nQuery: {query}")
    print("Results:")
    print(knowledge)

    # ==================== TEST WORKFLOW MEMORY ====================
    print("\\n" + "=" * 50)
    print("Testing Workflow Memory")
    print("=" * 50)

    # Store a workflow
    memory_manager.write_workflow(
        query="Find information about vector search",
        steps=["search_web", "extract_key_points", "store_in_knowledge_base"],
        result="Successfully retrieved and stored vector search information"
    )

    # Retrieve similar workflows
    workflow_query = "lookup vector database information"
    workflows = memory_manager.read_workflow(workflow_query, k=1)
    print(f"\\nQuery: {workflow_query}")
    print("Similar workflows:")
    print(workflows)

    # ==================== TEST ENTITY MEMORY ====================
    print("\\n" + "=" * 50)
    print("Testing Entity Memory")
    print("=" * 50)

    # Store entities
    memory_manager.write_entity(
        entity="Oracle Database 23ai",
        entity_type="system",
        context="Latest version of Oracle Database with AI capabilities"
    )

    memory_manager.write_entity(
        entity="Larry Ellison",
        entity_type="person",
        context="Co-founder and CTO of Oracle Corporation"
    )

    # Query entities
    entity_query = "Oracle database systems"
    entities = memory_manager.read_entity(entity_query, k=2)
    print(f"\\nQuery: {entity_query}")
    print(entities)

    print("\\n" + "=" * 50)
    print("All Memory Manager Tests Completed!")
    print("=" * 50)

    vector_conn.close()
    </copy>
    ```

2. **Run the test:**

    ```bash
    <copy>
    python test_memory_manager.py
    </copy>
    ```

## Summary

In this lab, you successfully:
* ✅ Created a SQL table for conversational memory
* ✅ Implemented a comprehensive Memory Manager class
* ✅ Built read/write methods for all memory types
* ✅ Added entity extraction capabilities
* ✅ Tested the complete memory system

Your Memory Manager now provides a unified interface for managing all types of agent memory, making it easy to build stateful AI agents that can learn and adapt over time.

You may now **proceed to the next lab**.

## Learn More

* [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
* [Oracle Database CLOB Data Type](https://docs.oracle.com/en/database/oracle/oracle-database/23/sqlrf/Data-Types.html#GUID-8EFA29E9-E8D8-40A6-A43E-954908C954A4)
* [Memory Systems for AI Agents](https://www.anthropic.com/research/memory-systems)

## Acknowledgements

* **Author** - Paul Parkinson, Oracle Database Developer Advocate
* **Last Updated By/Date** - March 2026
