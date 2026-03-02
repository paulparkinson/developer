# Lab 6: Build the Complete AI Agent

## Introduction

In this final lab, you will integrate all the components you've built—Memory Manager, Semantic Toolbox, Context Engineering—into a complete, production-ready AI agent. This agent will have memory, can call tools dynamically, and manages its context window automatically.

You'll see how all the memory engineering, context engineering, and prompt engineering techniques come together to create a sophisticated AI system.

Estimated Time: 40 minutes

### Objectives

In this lab, you will:
* Integrate all memory systems into an agent loop
* Implement automatic context loading and management
* Build tool calling with semantic discovery
* Create multi-turn conversation handling
* Test the complete agent with real queries
* Demonstrate memory persistence across sessions

### Prerequisites

* Completed all previous labs (1-5)
* All memory systems initialized and tested
* Tools registered in the semantic toolbox
* Understanding of the complete architecture

## Task 1: Understanding the Agent Architecture

Your agent combines multiple systems:

```
┌─────────────────────────────────────────────────────────────┐
│                        AI AGENT                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌───────────────┐   ┌──────────────┐ │
│  │   Memory     │───▶│  LLM Engine   │◀──│   Toolbox    │ │
│  │   Manager    │    │               │   │   (Tools)    │ │
│  └──────────────┘    └───────────────┘   └──────────────┘ │
│        │                    │                    │         │
│        ▼                    ▼                    ▼         │
│  ┌──────────────────────────────────────────────────────┐ │
│  │          Oracle AI Database (Vector + SQL)           │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Agent Loop Flow:**

1. **Load Context (Programmatic)**
   - Read conversational memory
   - Read knowledge base
   - Read workflows
   - Read entities
   - Read summary references

2. **Send to LLM**
   - System prompt + context + user query
   - LLM decides: answer or call tools

3. **Execute Tools (Agentic)**
   - LLM requests tools by name
   - Agent discovers tools semantically
   - Execute and return results

4. **Save Results (Programmatic)**
   - Store conversation
   - Save workflows
   - Extract and store entities
   - Manage context window

## Task 2: Implement the Agent Loop

Let's build the complete agent with all integrations.

1. **Create a file** named `ai_agent.py`:

    ```python
    <copy>
    import json
    import os
    import oracledb
    from openai import OpenAI
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy
    from memory_manager import MemoryManager
    from semantic_toolbox import Toolbox
    from context_utils import calculate_context_usage
    from summarization import offload_to_summary

    # ==================== SETUP ====================

    ORACLE_DSN = "127.0.0.1:1521/FREEPDB1"
    vector_conn = oracledb.connect(
        user="VECTOR",
        password="VectorPwd_2025",
        dsn=ORACLE_DSN
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Initialize vector stores
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

    # Initialize LLM client
    HF_TOKEN = os.getenv("HF_TOKEN", "your-token-here")
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

    # ==================== REGISTER TOOLS ====================

    # Register tools if not already registered
    tools_registered = False

    try:
        # Check if tools exist
        test_tools = toolbox.find_tools("test", k=1)
        if len(test_tools) == 0:
            # Register tools
            @toolbox.register_tool(augment=False, generate_queries=False)
            def search_tavily(query: str) -> str:
                """Search the web using Tavily API. Returns search results."""
                # Mock implementation
                result = f"Search results for '{query}':\\n"
                result += "1. Oracle AI Vector Search - Semantic search in Oracle Database\\n"
                result += "2. Building AI Agents - Best practices and patterns\\n"
                result += "3. Memory Systems for AI - How agents remember and learn"
                return result

            @toolbox.register_tool(augment=False, generate_queries=False)
            def expand_summary(summary_id: str) -> str:
                """Retrieve full text of a conversation summary by ID."""
                return memory_manager.read_summary_memory(summary_id)

            @toolbox.register_tool(augment=False, generate_queries=False)
            def calculate(expression: str) -> float:
                """Evaluate a mathematical expression."""
                try:
                    result = eval(expression, {"__builtins__": {}}, {})
                    return float(result)
                except Exception as e:
                    return f"Error: {e}"

            tools_registered = True
    except:
        pass

    # ==================== AGENT IMPLEMENTATION ====================

    def call_agent(query: str, thread_id: str = "default",
                   model: str = "Qwen/Qwen2.5-72B-Instruct",
                   max_iterations: int = 5):
        """
        Main agent loop with memory, tools, and context management.
        
        Args:
            query: User's question or request
            thread_id: Conversation thread identifier
            model: LLM model to use
            max_iterations: Maximum tool calling iterations
            
        Returns:
            Final answer from the agent
        """
        
        print(f"\\n{'='*70}")
        print(f"🤖 AI AGENT")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print(f"Thread: {thread_id}")
        print(f"{'='*70}\\n")
        
        # 1. LOAD CONTEXT (Programmatic - always executed)
        print("1️⃣ Loading memory context...")
        
        conversation = memory_manager.read_conversational_memory(thread_id)
        knowledge = memory_manager.read_knowledge_base(query, k=3)
        workflows = memory_manager.read_workflow(query, k=2)
        entities = memory_manager.read_entity(query, k=3)
        summaries = memory_manager.read_summary_context()
        
        # Discover relevant tools
        relevant_tools = toolbox.find_tools(query, k=3)
        
        # Build context
        context = f"""
CONVERSATIONAL MEMORY (Thread {thread_id}):
{conversation}

KNOWLEDGE BASE:
{knowledge}

PREVIOUS WORKFLOWS:
{workflows}

RELEVANT ENTITIES:
{entities}

SUMMARY MEMORY:
{summaries}

AVAILABLE TOOLS:
"""
        for tool in relevant_tools:
            context += f"\\n- {tool['name']}{tool['signature']}: {tool['description'][:100]}"
        
        # Check context usage
        usage = calculate_context_usage(context, model)
        print(f"   Context size: {usage['tokens']:,} tokens ({usage['percent']}%)")
        
        # Auto-compact if needed
        if usage['percent'] > 80:
            print(f"   ⚠️ Context usage high - compacting...")
            context, summary_ids = offload_to_summary(context, memory_manager, client)
            usage = calculate_context_usage(context, model)
            print(f"   ✅ Compacted to {usage['tokens']:,} tokens ({usage['percent']}%)")
        
        print(f"   ✅ Context loaded\\n")
        
        # 2. BUILD SYSTEM PROMPT
        system_prompt = """You are an advanced AI assistant with access to memory systems and tools.

You have access to:
- Conversational memory: Previous messages in this thread
- Knowledge base: Relevant documents and facts
- Workflows: Patterns of previous successful actions
- Entities: Known people, places, and systems
- Summary memory: Compressed context from earlier in the conversation

Available tools are listed in your context. Use them when needed by calling functions.

When you see [Summary ID: xxx], you can call expand_summary(xxx) to get the full text.

Provide helpful, accurate responses based on your memory and available tools."""

        # 3. STORE USER MESSAGE
        memory_manager.write_conversational_memory(query, "user", thread_id)
        
        # 4. AGENT LOOP
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"CONTEXT:\\n{context}"},
            {"role": "user", "content": query}
        ]
        
        steps = []
        
        for iteration in range(max_iterations):
            print(f"2️⃣ Calling LLM (iteration {iteration + 1})...")
            
            # Prepare tools for function calling
            tools_schema = []
            for tool in relevant_tools:
                tools_schema.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"][:100],
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Query parameter"}
                            }
                        }
                    }
                })
            
            # Call LLM
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                temperature=0.7,
                max_tokens=1000
            )
            
            msg = response.choices[0].message
            messages.append({"role": "assistant", "content": msg.content, 
                           "tool_calls": msg.tool_calls})
            
            # Check if done
            if not msg.tool_calls:
                final_answer = msg.content or ""
                print(f"   ✅ Got final answer\\n")
                break
            
            # 5. EXECUTE TOOLS (Agentic - LLM decided)
            print(f"3️⃣ Executing tools...")
            
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                
                print(f"   🛠️  {tool_name}({tool_args})")
                
                try:
                    result = toolbox.execute_tool(tool_name, **tool_args)
                    steps.append(f"{tool_name}({tool_args}) → success")
                    
                    # Store search results in knowledge base
                    if tool_name == "search_tavily":
                        memory_manager.write_knowledge_base(
                            result,
                            {"source": "web-search", "query": tool_args.get("query", "")}
                        )
                except Exception as e:
                    result = f"Error: {e}"
                    steps.append(f"{tool_name}({tool_args}) → failed")
                
                print(f"   → {str(result)[:100]}...")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})
            
            print()
        else:
            final_answer = "Maximum iterations reached."
        
        # 6. SAVE RESULTS (Programmatic - always executed)
        print(f"4️⃣ Saving to memory...")
        
        # Save conversation
        memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)
        
        # Save workflow if tools were used
        if steps:
            memory_manager.write_workflow(query, steps, final_answer)
        
        # Extract and save entities
        try:
            memory_manager.write_entity("", "", "", llm_client=client, text=final_answer)
        except:
            pass
        
        print(f"   ✅ Memory updated\\n")
        
        # 7. RETURN ANSWER
        print(f"{'='*70}")
        print(f"💬 ANSWER:")
        print(f"{'='*70}")
        print(final_answer)
        print(f"{'='*70}\\n")
        
        return final_answer

    print("\\n✅ AI Agent ready!")
    print("\\nUsage: call_agent('your question here', thread_id='your-thread')")
    </copy>
    ```

## Task 3: Test the Complete Agent

Now let's test the agent with various scenarios.

1. **Create a file** named `test_agent.py`:

    ```python
    <copy>
    from ai_agent import call_agent, memory_manager

    print("\\n" + "="*70)
    print("🧪 TESTING COMPLETE AI AGENT")
    print("="*70 + "\\n")

    # ==================== TEST 1: Simple Query ====================
    print("\\nTEST 1: Simple Query (no tools needed)")
    print("-" * 70)

    call_agent(
        "What is Oracle AI Database?",
        thread_id="test-1"
    )

    # ==================== TEST 2: Tool Calling ====================
    print("\\n\\nTEST 2: Tool Calling (web search)")
    print("-" * 70)

    call_agent(
        "Search for information about building AI agents with Oracle Database",
        thread_id="test-2"
    )

    # ==================== TEST 3: Memory Persistence ====================
    print("\\n\\nTEST 3: Memory Persistence (follow-up question)")
    print("-" * 70)

    call_agent(
        "Based on what you just found, tell me more about the most relevant result",
        thread_id="test-2"  # Same thread
    )

    # ==================== TEST 4: Calculation ====================
    print("\\n\\nTEST 4: Mathematical Calculation")
    print("-" * 70)

    call_agent(
        "What is 42 multiplied by 137?",
        thread_id="test-3"
    )

    # ==================== TEST 5: Multi-turn Conversation ====================
    print("\\n\\nTEST 5: Multi-turn Conversation with Context")
    print("-" * 70)

    thread_id = "test-5"

    call_agent("My name is Alice and I work at Oracle", thread_id=thread_id)
    call_agent("What company do I work for?", thread_id=thread_id)
    call_agent("And what's my name?", thread_id=thread_id)

    # ==================== VERIFY MEMORY ====================
    print("\\n\\n" + "="*70)
    print("🔍 VERIFYING MEMORY PERSISTENCE")
    print("="*70 + "\\n")

    # Check conversational memory
    print("Conversational Memory (Thread test-5):")
    print("-" * 70)
    conversation = memory_manager.read_conversational_memory("test-5")
    print(conversation)

    # Check knowledge base
    print("\\n\\nKnowledge Base:")
    print("-" * 70)
    knowledge = memory_manager.read_knowledge_base("AI agents", k=3)
    print(knowledge)

    # Check workflows
    print("\\n\\nWorkflows:")
    print("-" * 70)
    workflows = memory_manager.read_workflow("search information", k=2)
    print(workflows)

    # Check entities
    print("\\n\\nEntities:")
    print("-" * 70)
    entities = memory_manager.read_entity("Alice Oracle", k=2)
    print(entities)

    print("\\n\\n" + "="*70)
    print("✅ ALL TESTS COMPLETED!")
    print("="*70)
    </copy>
    ```

2. **Set your HuggingFace token:**

    ```bash
    <copy>
    export HF_TOKEN=your-huggingface-token-here
    </copy>
    ```

3. **Run the tests:**

    ```bash
    <copy>
    python test_agent.py
    </copy>
    ```

## Task 4: Understanding Agent Behavior

Let's analyze what happened in the tests:

**Test 1 - Simple Query:**
- Agent loaded context (empty on first run of that thread)
- LLM generated answer without tool calling
- Answer stored in conversational memory

**Test 2 - Tool Calling:**
- Agent discovered `search_tavily` tool semantically
- LLM decided to call the tool
- Results stored in knowledge base
- Workflow pattern saved

**Test 3 - Memory Persistence:**
- Agent loaded previous conversation from test-2
- Agent retrieved knowledge from previous search
- Answered using stored context

**Test 4 - Calculation:**
- Agent discovered `calculate` tool
- Executed calculation
- Saved workflow pattern

**Test 5 - Multi-turn:**
- Agent maintained context across turns
- Extracted entities ("Alice", "Oracle")
- Answered follow-up questions using memory

## Task 5: Explore Advanced Features

Try these additional scenarios:

1. **Test Context Summarization:**

    ```python
    <copy>
    from summarization import summarize_conversation

    # Create a long conversation
    thread = "long-conversation"
    for i in range(10):
        call_agent(f"Tell me fact number {i} about databases", thread_id=thread)

    # Summarize it
    result = summarize_conversation(thread, memory_manager, client)
    print(f"Summary created: {result['id']}")

    # Query using summary reference
    call_agent("What did we discuss earlier?", thread_id=thread)
    </copy>
    ```

2. **Test Entity Extraction:**

    ```python
    <copy>
    # Mention various entities
    call_agent(
        "I'm working on a project with Bob at Microsoft using Azure and Oracle Cloud",
        thread_id="entities-test"
    )

    # Query entities
    entities = memory_manager.read_entity("Microsoft Azure Oracle", k=5)
    print("Extracted entities:")
    print(entities)
    </copy>
    ```

## Summary

Congratulations! In this lab, you successfully:
* ✅ Integrated all memory systems into a unified agent
* ✅ Implemented automatic context loading and management
* ✅ Built tool calling with semantic discovery
* ✅ Created multi-turn conversation handling
* ✅ Demonstrated memory persistence across sessions
* ✅ Built a production-ready AI agent

### What You've Built

You now have a complete AI agent with:

| Capability | Implementation |
|------------|----------------|
| **Memory** | 6 types (conversational, knowledge, workflow, toolbox, entity, summary) |
| **Context** | Automatic loading, monitoring, and compaction |
| **Tools** | Semantic discovery and dynamic execution |
| **Learning** | Workflow patterns and entity extraction |
| **Efficiency** | JIT retrieval and context summarization |

### Key Takeaways

1. **Memory Engineering** - Design memory systems that serve specific cognitive functions
2. **Context Engineering** - Optimize what goes into the LLM's context window
3. **Programmatic vs Agentic** - Balance automatic operations with LLM decision-making
4. **Vector Search** - Enable semantic similarity across all memory types
5. **Oracle AI Database** - Unified platform for SQL + vector + AI operations

## Next Steps

To extend this agent:
- Add more specialized tools
- Implement hierarchical summarization
- Add user authentication and multi-user support
- Deploy as a web service or chatbot
- Integrate with enterprise systems

## Learn More

* [Oracle AI Vector Search Documentation](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/)
* [LangChain Documentation](https://python.langchain.com/)
* [Building Production AI Systems](https://www.oreilly.com/library/view/building-llm-powered/9781098150952/)
* [Memory Systems for AI](https://www.anthropic.com/research/memory-systems)

## Acknowledgements

* **Author** - Paul Parkinson, Oracle Database Developer Advocate
* **Last Updated By/Date** - March 2026

---

**🎉 Congratulations on completing the workshop!**

You've built a sophisticated AI agent with enterprise-grade memory and context management using Oracle AI Database. This foundation can be extended to build production applications that learn, adapt, and scale.
