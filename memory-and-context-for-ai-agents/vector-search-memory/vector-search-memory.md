# Lab 2: Implement Vector Search and Memory Stores

## Introduction

In this lab, you will learn how to use **LangChain's Oracle Vector Store (OracleVS)** to store and search documents using semantic similarity. Vector search enables finding documents based on meaning rather than exact keyword matches.

You'll create five separate vector stores—one for each memory type that an AI agent needs: knowledge base, workflows, toolbox, entities, and summaries. Each vector store will be backed by its own Oracle table and will use embeddings for semantic search.

Estimated Time: 30 minutes

### Objectives

In this lab, you will:
* Initialize embedding models for converting text to vectors
* Create vector stores using LangChain's Oracle integration
* Build IVF (Inverted File) vector indexes for fast similarity search
* Add documents with metadata to vector stores
* Query vector stores using semantic search
* Filter results using metadata

### Prerequisites

* Completed Lab 1: Set Up Oracle AI Database Locally
* Oracle Database Free container running
* Python environment with required packages installed

## Task 1: Understanding Vector Search

Vector search (also called semantic search) works by converting text into high-dimensional vectors (embeddings) that capture meaning. Similar texts have vectors that are close together in vector space.

**Traditional keyword search:**
- Query: "AI database"
- Matches: Documents containing exactly "AI" AND "database"
- Misses: Documents with "artificial intelligence" and "data store"

**Vector/Semantic search:**
- Query: "AI database"  
- Matches: Documents about:
  - "artificial intelligence data stores"
  - "machine learning database systems"
  - "neural network storage solutions"

The key components are:

| Component | Description |
|-----------|-------------|
| **Embedding Model** | Converts text into fixed-size vectors (e.g., 768 dimensions) |
| **Vector Store** | Database that stores embeddings and supports similarity queries |
| **Distance Strategy** | Method for measuring similarity (Euclidean, Cosine, etc.) |
| **Vector Index** | Speeds up searches using approximate nearest neighbor algorithms |

## Task 2: Initialize the Embedding Model

We'll use a HuggingFace embedding model that converts text into 768-dimensional vectors.

1. **Create a new Python file** named `vector_search_demo.py`:

    ```python
    <copy>
    import oracledb
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_oracledb.vectorstores.oraclevs import create_index
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain.schema import Document

    # Connection parameters
    ORACLE_HOST = "127.0.0.1"
    ORACLE_DSN = f"{ORACLE_HOST}:1521/FREEPDB1"

    # Connect to Oracle as VECTOR user
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN,
        program="vector_search_demo"
    )

    print(f"✅ Connected as user: {vector_conn.username}")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    print("✅ Embedding model loaded")
    print(f"   Model: sentence-transformers/paraphrase-mpnet-base-v2")
    print(f"   Dimension: 768")
    </copy>
    ```

2. **Run the script:**

    ```bash
    <copy>
    python vector_search_demo.py
    </copy>
    ```

    The first time you run this, it will download the embedding model from HuggingFace (about 420MB). Subsequent runs will use the cached model.

## Task 3: Create a Simple Vector Store

Let's start with a simple example to understand how vector stores work.

1. **Add to your script:**

    ```python
    <copy>
    # Create a simple vector store
    vector_store = OracleVS(
        client=vector_conn, 
        embedding_function=embedding_model,
        table_name="VECTOR_SEARCH_DEMO",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )

    print("✅ Vector store created")
    print(f"   Table: VECTOR_SEARCH_DEMO")
    print(f"   Distance strategy: Euclidean")
    </copy>
    ```

    This creates a vector store backed by an Oracle table. The table will automatically include:
    - `id`: Unique identifier
    - `text`: The original text
    - `metadata`: JSON metadata about the document
    - `embedding`: The vector representation (VECTOR data type)

## Task 4: Add Documents to the Vector Store

Now let's add some sample documents about AI and databases.

1. **Add to your script:**

    ```python
    <copy>
    # Sample documents about AI and databases
    documents = [
        Document(
            page_content="Oracle AI Vector Search enables semantic similarity search on embeddings stored natively in Oracle Database.",
            metadata={"source": "oracle", "topic": "vector-search"}
        ),
        Document(
            page_content="LangChain provides integrations with various vector databases including Oracle, Pinecone, and Chroma.",
            metadata={"source": "langchain", "topic": "integrations"}
        ),
        Document(
            page_content="AI agents can use tool calling to retrieve relevant documents from vector stores based on semantic similarity.",
            metadata={"source": "ai-agents", "topic": "tool-calling"}
        ),
        Document(
            page_content="Vector embeddings are numerical representations of text that capture semantic meaning in high-dimensional space.",
            metadata={"source": "ml-basics", "topic": "embeddings"}
        ),
        Document(
            page_content="Oracle Database 23ai includes native support for vector data types and vector indexes for fast similarity search.",
            metadata={"source": "oracle", "topic": "database-features"}
        ),
    ]

    # Add documents to the vector store
    print("\\n📝 Adding documents to vector store...")
    ids = vector_store.add_documents(documents)
    print(f"✅ Added {len(ids)} documents")
    </copy>
    ```

2. **Run the script again:**

    ```bash
    <copy>
    python vector_search_demo.py
    </copy>
    ```

## Task 5: Create a Vector Index

For faster similarity search, we'll create an IVF (Inverted File) index. This uses approximate nearest neighbor search to speed up queries on large datasets.

1. **Add helper functions to your script:**

    ```python
    <copy>
    def cleanup_stale_vector_tables(conn):
        """Drop stale internal vector index tables (VECTOR$...) left by failed index operations."""
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM user_tables WHERE table_name LIKE 'VECTOR$%' ORDER BY table_name")
            stale_tables = cur.fetchall()
        if not stale_tables:
            return
        print(f"  🧹 Found {len(stale_tables)} internal vector table(s) — cleaning up orphans...")
        for (tbl,) in stale_tables:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'DROP TABLE "{tbl}" PURGE')
                conn.commit()
                print(f"     Dropped orphan: {tbl}")
            except Exception as e:
                err = str(e)
                if "ORA-51903" not in err:
                    print(f"     Could not drop {tbl}: {err[:80]}")

    def drop_existing_vector_indexes(conn, table_name):
        """Drop any existing vector (DOMAIN) indexes on the given table."""
        bare_table = table_name.strip('"').upper()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT index_name FROM user_indexes "
                "WHERE table_name = :1 AND index_type = 'DOMAIN'",
                [bare_table]
            )
            rows = cur.fetchall()
        for (old_idx,) in rows:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'DROP INDEX "{old_idx}" FORCE')
                conn.commit()
                print(f"  Dropped existing index: {old_idx}")
            except Exception as e:
                err = str(e)
                if "ORA-01418" not in err:
                    print(f"  Warning: could not drop index {old_idx}: {err[:80]}")

    def safe_create_index(conn, vector_store, index_name):
        """Safely create a vector index, cleaning up if needed."""
        table_name = vector_store._get_table_name()
        
        # Clean up stale tables and old indexes
        cleanup_stale_vector_tables(conn)
        drop_existing_vector_indexes(conn, table_name)
        
        try:
            create_index(
                conn,
                vector_store,
                params={
                    "idx_name": index_name,
                    "idx_type": "IVF"
                }
            )
            print(f"✅ Created index: {index_name}")
        except Exception as e:
            err_str = str(e)
            if "ORA-01408" in err_str or "already indexed" in err_str.lower():
                print(f"  ℹ️ Index {index_name} already exists or column already indexed")
            else:
                raise

    # Create the IVF index
    print("\\n🔧 Creating IVF vector index...")
    safe_create_index(vector_conn, vector_store, "demo_vs_ivf")
    </copy>
    ```

2. **Run the script:**

    ```bash
    <copy>
    python vector_search_demo.py
    </copy>
    ```

## Task 6: Query the Vector Store

Now let's search for documents using semantic similarity.

1. **Add to your script:**

    ```python
    <copy>
    # Perform similarity search
    print("\\n🔍 Searching for: 'How do I search vectors in Oracle?'")
    query = "How do I search vectors in Oracle?"
    results = vector_store.similarity_search(query, k=3)

    print(f"\\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\\n{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")

    # Search with metadata filter
    print("\\n\\n🔍 Searching with metadata filter (source='oracle'):")
    results = vector_store.similarity_search(
        query, 
        k=3,
        filter={"source": "oracle"}
    )

    print(f"\\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\\n{i}. {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    </copy>
    ```

2. **Run the script:**

    ```bash
    <copy>
    python vector_search_demo.py
    </copy>
    ```

3. **Expected output:**

    ```
    🔍 Searching for: 'How do I search vectors in Oracle?'

    Found 3 results:

    1. Oracle AI Vector Search enables semantic similarity search on embeddings stored natively in Oracle Database.
       Metadata: {'source': 'oracle', 'topic': 'vector-search'}

    2. Oracle Database 23ai includes native support for vector data types and vector indexes for fast similarity search.
       Metadata: {'source': 'oracle', 'topic': 'database-features'}

    3. Vector embeddings are numerical representations of text that capture semantic meaning in high-dimensional space.
       Metadata: {'source': 'ml-basics', 'topic': 'embeddings'}
    ```

## Task 7: Create Vector Stores for Agent Memory Types

Now let's create the five vector stores that our AI agent will use for different types of memory.

1. **Create a new file** named `create_memory_stores.py`:

    ```python
    <copy>
    import oracledb
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy

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

    # Table names for each memory type
    KNOWLEDGE_BASE_TABLE = "KNOWLEDGE_BASE_MEMORY"
    WORKFLOW_TABLE = "WORKFLOW_MEMORY"
    TOOLBOX_TABLE = "TOOLBOX_MEMORY"
    ENTITY_TABLE = "ENTITY_MEMORY"
    SUMMARY_TABLE = "SUMMARY_MEMORY"

    print("Creating vector stores for each memory type...\\n")

    # 1. Knowledge Base - stores documents and search results
    knowledge_base_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=KNOWLEDGE_BASE_TABLE,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    print(f"✅ Created {KNOWLEDGE_BASE_TABLE}")

    # 2. Workflow Memory - stores learned action patterns
    workflow_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=WORKFLOW_TABLE,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    print(f"✅ Created {WORKFLOW_TABLE}")

    # 3. Toolbox Memory - stores tool definitions for semantic discovery
    toolbox_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=TOOLBOX_TABLE,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    print(f"✅ Created {TOOLBOX_TABLE}")

    # 4. Entity Memory - stores extracted entities (people, places, systems)
    entity_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=ENTITY_TABLE,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    print(f"✅ Created {ENTITY_TABLE}")

    # 5. Summary Memory - stores compressed conversation summaries
    summary_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name=SUMMARY_TABLE,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )
    print(f"✅ Created {SUMMARY_TABLE}")

    print("\\n🎉 All memory vector stores created successfully!")
    print("\\nMemory Types:")
    print("  • Knowledge Base: Searchable documents & facts")
    print("  • Workflow: Learned action patterns")
    print("  • Toolbox: Dynamic tool definitions")
    print("  • Entity: People, places, systems")
    print("  • Summary: Compressed context")

    # Verify tables were created
    print("\\n📊 Verifying tables in database:")
    with vector_conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM user_tables 
            WHERE table_name IN (
                'KNOWLEDGE_BASE_MEMORY',
                'WORKFLOW_MEMORY',
                'TOOLBOX_MEMORY',
                'ENTITY_MEMORY',
                'SUMMARY_MEMORY'
            )
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        for (table,) in tables:
            print(f"  ✅ {table}")

    vector_conn.close()
    </copy>
    ```

2. **Run the script:**

    ```bash
    <copy>
    python create_memory_stores.py
    </copy>
    ```

## Task 8: Create Indexes for All Memory Stores

Finally, let's create IVF indexes for all memory vector stores to enable fast similarity search.

1. **Create a file** named `create_memory_indexes.py`:

    ```python
    <copy>
    import oracledb
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_oracledb.vectorstores.oraclevs import create_index
    from langchain_community.vectorstores.utils import DistanceStrategy

    # [Include the helper functions from Task 5]
    def cleanup_stale_vector_tables(conn):
        """Drop stale internal vector index tables."""
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM user_tables WHERE table_name LIKE 'VECTOR$%'")
            stale_tables = cur.fetchall()
        for (tbl,) in stale_tables:
            try:
                with conn.cursor() as cur:
                    cur.execute(f'DROP TABLE "{tbl}" PURGE')
                conn.commit()
            except: pass

    def safe_create_index(conn, vector_store, index_name):
        """Safely create a vector index."""
        try:
            create_index(
                conn,
                vector_store,
                params={
                    "idx_name": index_name,
                    "idx_type": "IVF"
                }
            )
            print(f"✅ Created index: {index_name}")
        except Exception as e:
            if "ORA-01408" in str(e) or "already indexed" in str(e).lower():
                print(f"  ℹ️ Index {index_name} already exists")
            else:
                print(f"  ⚠️ Error creating {index_name}: {e}")

    # Connection and setup
    ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN
    )

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

    # Clean up stale tables
    print("🧹 Cleaning up stale vector tables...")
    cleanup_stale_vector_tables(vector_conn)

    # Create indexes
    print("\\n🔧 Creating IVF vector indexes...\\n")
    safe_create_index(vector_conn, knowledge_base_vs, "knowledge_base_vs_ivf")
    safe_create_index(vector_conn, workflow_vs, "workflow_vs_ivf")
    safe_create_index(vector_conn, toolbox_vs, "toolbox_vs_ivf")
    safe_create_index(vector_conn, entity_vs, "entity_vs_ivf")
    safe_create_index(vector_conn, summary_vs, "summary_vs_ivf")

    print("\\n🎉 All IVF indexes created successfully!")

    # Verify indexes
    print("\\n📊 Verifying indexes in database:")
    with vector_conn.cursor() as cur:
        cur.execute("""
            SELECT index_name, table_name, index_type
            FROM user_indexes
            WHERE index_name LIKE '%_VS_IVF'
            ORDER BY index_name
        """)
        indexes = cur.fetchall()
        for idx_name, tbl_name, idx_type in indexes:
            print(f"  ✅ {idx_name} on {tbl_name} ({idx_type})")

    vector_conn.close()
    </copy>
    ```

2. **Run the script:**

    ```bash
    <copy>
    python create_memory_indexes.py
    </copy>
    ```

## Summary

In this lab, you successfully:
* ✅ Initialized embedding models for vector conversion
* ✅ Created vector stores using LangChain and Oracle integration
* ✅ Built IVF indexes for fast similarity search
* ✅ Added documents with metadata to vector stores
* ✅ Performed semantic search queries
* ✅ Created five specialized vector stores for agent memory
* ✅ Indexed all memory stores for optimal performance

You now have a complete vector search infrastructure that will serve as the foundation for your AI agent's memory system.

You may now **proceed to the next lab**.

## Learn More

* [Oracle AI Vector Search Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/)
* [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
* [Understanding Vector Embeddings](https://www.pinecone.io/learn/vector-embeddings/)
* [IVF Index Explained](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#inverted-file-ivf-indexes)

## Acknowledgements

* **Author** - Paul Parkinson, Oracle Database Developer Advocate
* **Last Updated By/Date** - March 2026
