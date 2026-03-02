# Lab 5: Web Access with Tavily

## Introduction

In this lab, you will implement **web access** for your AI agent using Tavily, an AI-optimized search API designed for LLM applications. You'll create an agentic tool that enables the agent to search the web and automatically store results in the knowledge base for future retrieval.

This lab demonstrates the **search-and-store pattern**, where external information discovered by the agent is persisted to memory, creating a learning agent that builds its knowledge base over time. This lab covers Part 5 of the notebook: "Web Access with Tavily."

Estimated Time: 25 minutes

### Objectives

In this lab, you will:
* Set up the Tavily API client
* Understand the search-and-store pattern
* Implement the search_tavily tool with augmentation
* Register the tool in the semantic toolbox
* Test web search and knowledge base persistence
* Demonstrate multi-turn memory with web results

### Prerequisites

* Completed Lab 3: Memory Engineering and Agent Memory
* Completed Lab 4: Context Engineering Techniques  
* Tavily API key (free tier available at [app.tavily.com](https://app.tavily.com/))
* Understanding of the Toolbox class from Lab 3

## Task 1: Understanding the Search-and-Store Pattern

Traditional web search tools return results temporarily. The **search-and-store pattern** persists search results to the agent's long-term memory.

**Traditional Approach:**
```python
results = search_web("AI news")
# Results are used once, then lost
```

**Search-and-Store Approach:**
```python
@toolbox.register_tool(augment=True)
def search_tavily(query: str):
    results = tavily_client.search(query)
    # Store each result in knowledge base
    for result in results:
        memory_manager.write_knowledge_base(result, metadata)
    return results
# Future queries can retrieve this without searching again
```

### How It Works

```
User query → Agent calls search_tavily() → Tavily API returns results
                                              ↓
                         Results stored in knowledge_base_vs with metadata
                                              ↓
                         Future conversations can access these results
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Learning Agent** | Information discovered once becomes permanent knowledge |
| **Cost Reduction** | Avoid redundant API calls for similar queries |
| **Context Enrichment** | Future conversations have richer context from past searches |
| **Audit Trail** | All search results are timestamped and traceable |

## Task 2: Set Up Tavily API

Tavily is an AI-optimized search API designed specifically for LLM applications. It returns concise, relevant results perfect for agent context windows.

1. **Get your Tavily API key:**
   - Sign up at [app.tavily.com](https://app.tavily.com/)
   - Copy your API key from the dashboard
   - Free tier includes 1,000 searches/month

2. **Install the Tavily client** (if not already installed):

    ```bash
    <copy>
    pip install tavily-python
    </copy>
    ```

3. **Set up your API key securely:**

    ```python
    <copy>
    import os
    import getpass

    # Set Tavily API key
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key: ")
    </copy>
    ```

## Task 3: Implement the search_tavily Tool

Now let's create the `search_tavily` function that searches the web and stores results in the knowledge base.

1. **Import the Tavily client:**

    ```python
    <copy>
    from tavily import TavilyClient
    from datetime import datetime

    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    </copy>
    ```

2. **Create the search_tavily tool with augmentation:**

    ```python
    <copy>
    @toolbox.register_tool(augment=True)
    def search_tavily(query: str, max_results: int = 5):
        """
        Use this function to search the web and store the results in the knowledge base.
        Call this when you need current information not available in your context.
        """
        # Call Tavily API
        response = tavily_client.search(query=query, max_results=max_results)
        results = response.get("results", [])

        # Write each result to the knowledge base
        for result in results:
            # Create the text content to embed
            text = f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}\nURL: {result.get('url', '')}"
            
            # Create metadata
            metadata = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0),
                "source_type": "tavily_search",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
            # Write to knowledge base (persistent memory)
            memory_manager.write_knowledge_base(text, metadata)

        return results
    </copy>
    ```

3. **Understanding the code:**

    | Component | Purpose |
    |-----------|---------|
    | `@toolbox.register_tool(augment=True)` | Registers tool with LLM-enhanced docstring |
    | `tavily_client.search()` | Calls Tavily API for web results |
    | Text extraction | Combines title + content + URL for embedding |
    | Metadata | Stores source info, timestamp, and search query |
    | `write_knowledge_base()` | Persists to Oracle AI Vector Store |

    The `augment=True` parameter tells the toolbox to:
    - Use an LLM to improve the docstring
    - Generate synthetic example queries
    - Store everything with enhanced embeddings

## Task 4: Test Semantic Tool Retrieval

Before using the tool in the agent, verify it can be found via semantic search.

1. **Search for the tool using natural language:**

    ```python
    <copy>
    import pprint

    # Find tools relevant to web searching
    retrieved_tools = memory_manager.read_toolbox("Search the internet", k=1)
    pprint.pprint(retrieved_tools)
    </copy>
    ```

2. **Expected output:**

    You should see the `search_tavily` tool returned with its function schema, showing:
    - Tool name: `search_tavily`
    - Augmented description
    - Parameters: `query` (string), `max_results` (integer)
    
    This confirms the tool is semantically discoverable by the agent.

## Task 5: Test Web Search and Knowledge Base Persistence

Test the search-and-store pattern with a real query.

1. **Call search_tavily directly:**

    ```python
    <copy>
    # Search for current information
    results = search_tavily("Latest Oracle Database AI features 2024", max_results=3)

    print(f"\\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\\n{i}. {result['title']}")
        print(f"   {result['url']}")
        print(f"   {result['content'][:150]}...")
    </copy>
    ```

2. **Verify knowledge base storage:**

    ```python
    <copy>
    # Query the knowledge base to see if results were stored
    kb_results = memory_manager.read_knowledge_base("Oracle Database AI features", k=3)
    print(kb_results)
    </copy>
    ```

    You should see the search results formatted as knowledge base memory, proving they're now part of the agent's permanent knowledge.

3. **Expected behavior:**

    ```
    ✅ Web search returns current results from Tavily
    ✅ Each result is embedded and stored in Oracle AI Vector Store
    ✅ Future queries can retrieve this information without new API calls
    ✅ Metadata includes source URL, timestamp, and original query
    ```

## Task 6: Demonstrate Multi-Turn Memory

Show how the agent uses stored web results in subsequent conversations.

1. **Create mock additional tools** (for the agent's toolkit):

    ```python
    <copy>
    @toolbox.register_tool()
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Use for calculations."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    print("✅ Additional tools registered")
    </copy>
    ```

2. **Test multi-turn conversation** (if agent loop is implemented):

    ```python
    <copy>
    # First turn - agent searches the web
    response1 = call_agent(
        "What are the latest AI features in Oracle Database?",
        thread_id="tavily_test"
    )

    print("\\n" + "="*50)
    print("FIRST RESPONSE:")
    print(response1)
    </copy>
    ```

3. **Follow-up query using stored results:**

    ```python
    <copy>
    # Second turn - agent uses stored knowledge base
    response2 = call_agent(
        "Can you tell me more about the first feature you mentioned?",
        thread_id="tavily_test"
    )

    print("\\n" + "="*50)
    print("FOLLOW-UP RESPONSE:")
    print(response2)
    </copy>
    ```

4. **What happens:**

    | Turn | Agent Behavior |
    |------|----------------|
    | **First** | No knowledge base results → Calls `search_tavily()` → Stores results |
    | **Second** | Finds stored results in knowledge base → Answers without new search |

    The agent **learns** from its web searches. Information discovered in one conversation is available in all future conversations.

## Task 7: Understanding the Memory Flow

The complete flow demonstrates how Tavily integrates with the agent's memory system:

```
User: "What are AI features in Oracle DB?"
       ↓
Agent: Checks knowledge base → No results found
       ↓
Agent: Calls search_tavily("Oracle Database AI features")
       ↓
Tavily: Returns 5 web results
       ↓
search_tavily: Writes each result to knowledge_base_vs with metadata
       ↓
Agent: Uses results to answer
       ↓
       [Time passes]
       ↓
User: "Tell me more about those features"
       ↓
Agent: Checks knowledge base → Finds stored Tavily results!
       ↓
Agent: Answers using stored knowledge (no new API call)
```

### Key Concepts

| Concept | Implementation |
|---------|----------------|
| **Agentic Tool** | LLM decides when to call `search_tavily()` |
| **Programmatic Storage** | Every search result is automatically saved |
| **Memory Persistence** | Oracle AI Vector Store keeps results permanently |
| **Learning Agent** | Agent builds knowledge base over time |
| **Cost Optimization** | Avoids redundant API calls for similar queries |

## Summary

In this lab, you implemented web access for your AI agent using Tavily. You learned:

✅ **Search-and-store pattern** - Persist external information to memory  
✅ **Tavily integration** - AI-optimized search for LLM applications  
✅ **Knowledge base enrichment** - Automatic storage of search results  
✅ **Multi-turn memory** - Reuse information across conversations  
✅ **Agentic behavior** - LLM decides when to search the web

The agent now has two critical capabilities:
1. **Accessing current information** via Tavily search
2. **Learning and remembering** by storing results in Oracle AI Database

This completes Part 5 of the notebook. In Lab 6, you'll integrate all components into the complete agent execution loop.

You may now **proceed to the next lab**.

## Learn More

* [Tavily Documentation](https://docs.tavily.com/)
* [Oracle AI Vector Search](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/)
* [LangChain OracleVS](https://python.langchain.com/docs/integrations/vectorstores/oracle)

## Acknowledgements

* **Author** - Oracle AI Development Team
* **Last Updated By/Date** - March 2026
