# Lab 4: Semantic Tool Discovery and Toolbox

## Introduction

In this lab, you will implement a **Semantic Toolbox** that enables AI agents to discover and call tools dynamically based on natural language queries. Instead of hardcoding which tools an agent can use, the toolbox stores tool definitions with embeddings, allowing the agent to find relevant tools through semantic similarity search.

This is a powerful technique that combines memory engineering, context engineering, and prompt engineering to create flexible, extensible AI agents.

Estimated Time: 35 minutes

### Objectives

In this lab, you will:
* Understand semantic tool discovery concepts
* Implement LLM-powered docstring augmentation
* Generate synthetic queries for better tool retrieval
* Build a Toolbox class with tool registration
* Create and register example tools
* Test semantic tool discovery

### Prerequisites

* Completed Lab 3: Build the Memory Manager
* Understanding of function decorators in Python
* Familiarity with tool calling concepts

## Task 1: Understanding Semantic Tool Discovery

Traditional tool calling requires the agent to know about all available tools upfront. Semantic tool discovery enables:

**Traditional Approach:**
```python
tools = [search_web, calculate, send_email]  # Fixed list
agent = Agent(tools=tools)
```

**Semantic Discovery Approach:**
```python
@toolbox.register_tool()
def search_web(query: str) -> str:
    """Search the web for information."""
    ...

# Agent dynamically discovers tools based on user query
relevant_tools = toolbox.find_tools("I need to find information about AI")
# Returns: [search_web] based on semantic similarity
```

### How It Works

```
User Query → Embed Query → Vector Search → Find tools with similar docstrings → Return relevant tools
```

### The Three Engineering Disciplines

| Discipline | Technique | Purpose |
|------------|-----------|---------|
| **Memory Engineering** | Store tools as procedural memory | Tools are learned skills |
| **Memory Engineering** | Synthetic query generation | Improve discoverability |
| **Context Engineering** | Selective tool retrieval | Only relevant tools in context |
| **Prompt Engineering** | Role-based augmentation | Improve docstring quality |

## Task 2: Create the Toolbox Class

Let's build a Toolbox that can register tools, augment their docstrings, and retrieve them semantically.

1. **Create a file** named `semantic_toolbox.py`:

    ```python
    <copy>
    import inspect
    import uuid
    import json
    from typing import Callable, Optional, Dict, List
    from pydantic import BaseModel
    from langchain.schema import Document

    def get_embedding(text: str, embedding_model) -> list:
        """Get the embedding for a text using the configured embedding model."""
        return embedding_model.embed_query(text)

    class ToolMetadata(BaseModel):
        """Metadata for a registered tool."""
        name: str
        description: str
        signature: str
        parameters: dict
        return_type: str

    class Toolbox:
        """
        A toolbox for registering, storing, and retrieving tools with LLM-powered augmentation.
        
        Tools are stored with embeddings for semantic retrieval, allowing the agent to
        find relevant tools based on natural language queries.
        """
        
        def __init__(self, memory_manager, llm_client, embedding_model, 
                     model: str = "Qwen/Qwen2.5-72B-Instruct"):
            """
            Initialize the Toolbox.
            
            Args:
                memory_manager: MemoryManager instance for storing tools
                llm_client: OpenAI client for LLM augmentation
                embedding_model: Embedding model for vector conversion
                model: Model to use for augmentation
            """
            self.memory_manager = memory_manager
            self.llm_client = llm_client
            self.embedding_model = embedding_model
            self.model = model
            self._tools: Dict[str, Callable] = {}  # tool_id -> callable
            self._tools_by_name: Dict[str, Callable] = {}  # name -> callable
        
        def _call_llm(self, messages: List[Dict], temperature: float = 0.3, 
                      max_tokens: int = 500) -> str:
            """Helper to call LLM."""
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        
        def _augment_docstring(self, docstring: str) -> str:
            """
            Use LLM to improve and expand a tool's docstring.
            
            Args:
                docstring: The original docstring to augment
                
            Returns:
                An improved, more detailed docstring
            """
            if not docstring.strip():
                return "No description provided."

            prompt = f"""You are a technical writer. Improve the following function docstring to be more clear, 
comprehensive, and useful. Include:
1. A clear concise summary
2. Detailed description of what the function does
3. When to use this function
4. Any important notes or caveats

Original docstring:
{docstring}

Return ONLY the improved docstring, no other text."""

            return self._call_llm([{"role": "user", "content": prompt}])
        
        def _generate_queries(self, docstring: str, num_queries: int = 5) -> List[str]:
            """
            Generate synthetic example queries that would lead to using this tool.
            
            These queries improve retrieval by embedding both the tool description
            AND example queries.
            
            Args:
                docstring: The tool's docstring (ideally augmented)
                num_queries: Number of example queries to generate
                
            Returns:
                List of example natural language queries
            """
            prompt = f"""Based on the following tool description, generate {num_queries} diverse example queries 
that a user might ask when they need this tool. Make them natural and varied.

Tool description:
{docstring}

Return ONLY a JSON array of strings, like: ["query1", "query2", ...]"""

            response = self._call_llm([{"role": "user", "content": prompt}])
            
            try:
                queries = json.loads(response)
                return queries if isinstance(queries, list) else []
            except json.JSONDecodeError:
                return [response]
        
        def _get_tool_metadata(self, func: Callable) -> ToolMetadata:
            """
            Extract metadata from a function for storage and retrieval.
            
            Args:
                func: The function to extract metadata from
                
            Returns:
                ToolMetadata object with function details
            """
            sig = inspect.signature(func)
            
            # Extract parameter info
            parameters = {}
            for name, param in sig.parameters.items():
                param_info = {"name": name}
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = str(param.annotation)
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = str(param.default)
                parameters[name] = param_info
            
            # Extract return type
            return_type = "Any"
            if sig.return_annotation != inspect.Signature.empty:
                return_type = str(sig.return_annotation)
            
            return ToolMetadata(
                name=func.__name__,
                description=func.__doc__ or "No description",
                signature=str(sig),
                parameters=parameters,
                return_type=return_type
            )
        
        def register_tool(self, augment: bool = True, generate_queries: bool = True):
            """
            Decorator to register a tool with the toolbox.
            
            Args:
                augment: Whether to use LLM to improve the docstring
                generate_queries: Whether to generate synthetic example queries
                
            Usage:
                @toolbox.register_tool(augment=True)
                def my_tool(param: str) -> str:
                    \"\"\"Does something useful.\"\"\"
                    return result
            """
            def decorator(func: Callable) -> Callable:
                tool_id = str(uuid.uuid4())
                
                # Get metadata
                metadata = self._get_tool_metadata(func)
                
                # Optionally augment docstring
                docstring = metadata.description
                if augment and self.llm_client:
                    print(f"  🔧 Augmenting docstring for {metadata.name}...")
                    docstring = self._augment_docstring(docstring)
                
                # Optionally generate synthetic queries
                queries = []
                if generate_queries and self.llm_client:
                    print(f"  🔍 Generating queries for {metadata.name}...")
                    queries = self._generate_queries(docstring)
                
                # Build rich text for embedding
                rich_text = f"Function: {metadata.name}\\n"
                rich_text += f"Description: {docstring}\\n"
                rich_text += f"Signature: {metadata.signature}\\n"
                if queries:
                    rich_text += f"Example queries: {', '.join(queries)}"
                
                # Store in vector toolbox
                doc = Document(
                    page_content=rich_text,
                    metadata={
                        "tool_id": tool_id,
                        "name": metadata.name,
                        "signature": metadata.signature,
                        "parameters": json.dumps(metadata.parameters),
                        "return_type": metadata.return_type,
                        "queries": json.dumps(queries)
                    }
                )
                
                self.memory_manager.toolbox_vs.add_documents([doc])
                
                # Store callable reference
                self._tools[tool_id] = func
                self._tools_by_name[metadata.name] = func
                
                print(f"✅ Registered tool: {metadata.name}")
                
                return func
            
            return decorator
        
        def find_tools(self, query: str, k: int = 3) -> List[Dict]:
            """
            Find tools relevant to a query using semantic search.
            
            Args:
                query: Natural language description of what the user wants to do
                k: Number of tools to return
                
            Returns:
                List of tool metadata dictionaries
            """
            results = self.memory_manager.toolbox_vs.similarity_search(query, k=k)
            
            tools = []
            for doc in results:
                tool_info = {
                    "name": doc.metadata.get("name"),
                    "signature": doc.metadata.get("signature"),
                    "description": doc.page_content,
                    "tool_id": doc.metadata.get("tool_id")
                }
                tools.append(tool_info)
            
            return tools
        
        def execute_tool(self, tool_name: str, **kwargs):
            """
            Execute a registered tool by name.
            
            Args:
                tool_name: Name of the tool to execute
                **kwargs: Arguments to pass to the tool
                
            Returns:
                Result of the tool execution
            """
            if tool_name not in self._tools_by_name:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            func = self._tools_by_name[tool_name]
            return func(**kwargs)

    print("✅ Semantic Toolbox class defined")
    </copy>
    ```

## Task 3: Register Example Tools

Let's create some example tools and register them with our semantic toolbox.

1. **Create a file** named `register_tools.py`:

    ```python
    <copy>
    import oracledb
    import os
    from openai import OpenAI
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy
    from memory_manager import MemoryManager
    from semantic_toolbox import Toolbox

    # Setup connection and components
    ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Create vector stores
    knowledge_base_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="KNOWLEDGE_BASE_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    toolbox_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="TOOLBOX_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

    workflow_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="WORKFLOW_MEMORY",
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

    # Initialize OpenAI client for LLM
    # Note: Replace with your HuggingFace token
    HF_TOKEN = os.getenv("HF_TOKEN", "your-huggingface-token-here")
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=HF_TOKEN
    )

    # Initialize Toolbox
    toolbox = Toolbox(
        memory_manager=memory_manager,
        llm_client=client,
        embedding_model=embedding_model
    )

    print("🚀 Registering tools...\\n")

    # ==================== REGISTER TOOLS ====================

    @toolbox.register_tool(augment=True, generate_queries=True)
    def search_tavily(query: str) -> str:
        """
        Search the web using Tavily API.
        Returns search results with titles, URLs, and snippets.
        """
        # Mock implementation for demo
        return f"Mock search results for: {query}"

    @toolbox.register_tool(augment=True, generate_queries=True)
    def calculate(expression: str) -> float:
        """
        Evaluate a mathematical expression.
        Supports basic arithmetic operations.
        """
        try:
            # Safe eval for demo - use a proper math parser in production
            result = eval(expression, {"__builtins__": {}}, {})
            return float(result)
        except Exception as e:
            return f"Error: {e}"

    @toolbox.register_tool(augment=True, generate_queries=True)
    def get_current_time() -> str:
        """
        Returns the current date and time.
        """
        from datetime import datetime
        return datetime.now().isoformat()

    @toolbox.register_tool(augment=True, generate_queries=True)
    def expand_summary(summary_id: str) -> str:
        """
        Retrieve the full text of a previously stored conversation summary.
        Use this when you see a [Summary ID: xxx] reference and need the details.
        """
        return memory_manager.read_summary_memory(summary_id)

    print("\\n🎉 All tools registered successfully!")

    # Clean up
    vector_conn.close()
    </copy>
    ```

2. **Set your HuggingFace token:**

    ```bash
    <copy>
    export HF_TOKEN=your-huggingface-token-here
    </copy>
    ```

3. **Run the script:**

    ```bash
    <copy>
    python register_tools.py
    </copy>
    ```

    You'll see the toolbox augment docstrings and generate synthetic queries for each tool.

## Task 4: Test Semantic Tool Discovery

Now let's test finding tools based on natural language queries.

1. **Create a file** named `test_toolbox.py`:

    ```python
    <copy>
    import oracledb
    import os
    from openai import OpenAI
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy
    from memory_manager import MemoryManager
    from semantic_toolbox import Toolbox

    # Setup (same as register_tools.py)
    ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    toolbox_vs = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="TOOLBOX_MEMORY",
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
    )

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

    memory_manager = MemoryManager(
        conn=vector_conn,
        conversation_table="CONVERSATIONAL_MEMORY",
        knowledge_base_vs=knowledge_base_vs,
        workflow_vs=workflow_vs,
        toolbox_vs=toolbox_vs,
        entity_vs=entity_vs,
        summary_vs=summary_vs
    )

    HF_TOKEN = os.getenv("HF_TOKEN", "your-token-here")
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=HF_TOKEN
    )

    toolbox = Toolbox(
        memory_manager=memory_manager,
        llm_client=client,
        embedding_model=embedding_model
    )

    print("🔍 Testing Semantic Tool Discovery\\n")
    print("=" * 60)

    # Test queries
    test_queries = [
        "I need to find information about AI on the internet",
        "What's 42 multiplied by 17?",
        "What time is it right now?",
        "I see a summary reference and need the full details"
    ]

    for query in test_queries:
        print(f"\\nQuery: {query}")
        print("-" * 60)
        
        tools = toolbox.find_tools(query, k=2)
        
        print(f"Found {len(tools)} relevant tools:\\n")
        for i, tool in enumerate(tools, 1):
            print(f"{i}. {tool['name']}{tool['signature']}")
            print(f"   {tool['description'][:100]}...")
        
        print()

    print("=" * 60)
    print("✅ Semantic tool discovery test complete!")

    vector_conn.close()
    </copy>
    ```

2. **Run the test:**

    ```bash
    <copy>
    python test_toolbox.py
    </copy>
    ```

3. **Expected output:**

    ```
    Query: I need to find information about AI on the internet
    ------------------------------------------------------------
    Found 2 relevant tools:

    1. search_tavily(query: str) -> str
       Search the web using Tavily API. Returns search results with titles, URLs, and snippets...

    2. get_current_time() -> str
       Returns the current date and time...

    Query: What's 42 multiplied by 17?
    ------------------------------------------------------------
    Found 2 relevant tools:

    1. calculate(expression: str) -> float
       Evaluate a mathematical expression. Supports basic arithmetic operations...
    ```

## Summary

In this lab, you successfully:
* ✅ Understood semantic tool discovery concepts
* ✅ Implemented LLM-powered docstring augmentation
* ✅ Generated synthetic queries for better tool retrieval
* ✅ Built a complete Toolbox class
* ✅ Registered tools with automatic enhancement
* ✅ Tested semantic tool discovery with natural language

Your AI agent can now dynamically discover and call tools based on user intent, without hardcoded tool lists.

You may now **proceed to the next lab**.

## Learn More

* [Function Calling in LLMs](https://platform.openai.com/docs/guides/function-calling)
* [Semantic Search Explained](https://www.pinecone.io/learn/semantic-search/)
* [Python Decorators](https://realpython.com/primer-on-python-decorators/)

## Acknowledgements

* **Author** - Paul Parkinson, Oracle Database Developer Advocate
* **Last Updated By/Date** - March 2026
