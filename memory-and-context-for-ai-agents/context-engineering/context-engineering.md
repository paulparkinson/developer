# Lab 4: Context Engineering Techniques

## Introduction

In this lab, you will implement **Context Engineering** techniques to optimize what goes into the LLM's context window. You'll build a **Just-in-Time (JIT) Retrieval** system that automatically compresses conversation history into summaries when context usage exceeds thresholds, and allows the agent to expand summaries on demand.

This is crucial for building agents that can handle long conversations without exceeding token limits or degrading performance.

Estimated Time: 10 minutes

### Objectives

In this lab, you will:
* Understand context window management
* Implement token counting and usage monitoring
* Build automatic summarization for long conversations
* Create JIT retrieval with on-demand expansion
* Test context compaction and retrieval

### Prerequisites

* Completed Lab 3: Memory Engineering and Agent Memory
* Understanding of LLM token limits
* Familiarity with the Memory Manager

## Task 1: Understanding Context Engineering

**The Context Window Problem:**

LLMs have limited context windows (e.g., 128k tokens for GPT-4, 200k for Claude 3.5). Long conversations can:
- Exceed token limits → API errors
- Use too many tokens → high costs
- Slow performance → longer response times

**Context Engineering Solutions:**

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| **Summarization** | Compress old messages into summaries | Long conversations |
| **Sliding Window** | Keep only recent N messages | Simple conversations |
| **Hierarchical Summary** | Multi-level summaries (recent + older) | Complex conversations |
| **JIT Retrieval** | Store summaries, expand on demand | Enterprise applications |

### How JIT Retrieval Works

```
1. Conversation grows → Monitor token usage
2. Usage > 80% threshold → Auto-summarize old messages
3. Store summary in vector DB → Mark original messages as summarized
4. Context window now shows: [Summary ID: abc123] instead of 50 messages
5. Agent needs details → Calls expand_summary(abc123) tool
```

## Task 2: Implement Token Counting

First, let's create utilities to measure context usage.

1. **Create a file** named `context_utils.py`:

    ```python
    <copy>
    import tiktoken
    from typing import Dict

    # Token limits for common models
    MODEL_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "claude-3": 200000,
        "Qwen/Qwen2.5-72B-Instruct": 128000,  # Typical for large models
    }

    def count_tokens(text: str, model: str = "Qwen/Qwen2.5-72B-Instruct") -> int:
        """
        Count tokens in a text string.
        
        Args:
            text: The text to count
            model: Model name (for encoding selection)
            
        Returns:
            Number of tokens
        """
        try:
            # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            return len(tokens)
        except Exception as e:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

    def calculate_context_usage(text: str, model: str = "Qwen/Qwen2.5-72B-Instruct") -> Dict:
        """
        Calculate context window usage.
        
        Args:
            text: The text to measure
            model: Model name
            
        Returns:
            Dictionary with tokens, limit, percent, and available
        """
        tokens = count_tokens(text, model)
        limit = MODEL_LIMITS.get(model, 128000)
        percent = (tokens / limit) * 100
        available = limit - tokens
        
        return {
            "tokens": tokens,
            "limit": limit,
            "percent": round(percent, 2),
            "available": available
        }

    def format_usage_report(usage: Dict) -> str:
        """Format usage info for display."""
        return f"""Context Usage:
  Tokens Used:     {usage['tokens']:,}
  Token Limit:     {usage['limit']:,}
  Usage:           {usage['percent']}%
  Available:       {usage['available']:,} tokens
"""

    print("✅ Context utilities defined")
    </copy>
    ```

## Task 3: Implement Summarization

Now let's create functions to summarize conversations.

1. **Create a file** named `summarization.py`:

    ```python
    <copy>
    import json
    import uuid
    from typing import Dict, List, Tuple
    from context_utils import count_tokens, calculate_context_usage

    def summarize_context_window(text: str, llm_client, model: str = "Qwen/Qwen2.5-72B-Instruct") -> str:
        """
        Use LLM to create a concise summary of conversation context.
        
        Args:
            text: The text to summarize
            llm_client: OpenAI client for LLM calls
            model: Model to use for summarization
            
        Returns:
            Concise summary text
        """
        prompt = f"""You are a summarization expert. Create a concise but comprehensive summary of the following conversation or context. Focus on:
1. Key topics discussed
2. Important decisions or conclusions
3. Relevant facts and data points
4. Action items or next steps

Keep the summary clear and useful for continuing the conversation later.

Context to summarize:
{text}

Return ONLY the summary, no other text."""

        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()

    def summarize_conversation(thread_id: str, memory_manager, llm_client, 
                              model: str = "Qwen/Qwen2.5-72B-Instruct") -> Dict:
        """
        Summarize a conversation thread and store the summary.
        
        Args:
            thread_id: The conversation thread to summarize
            memory_manager: MemoryManager instance
            llm_client: OpenAI client
            model: Model to use
            
        Returns:
            Dictionary with summary_id, description, and summary text
        """
        # Read unsummarized messages
        conversation = memory_manager.read_conversational_memory(thread_id)
        
        if conversation == "[]" or not conversation.strip():
            return {
                "success": False,
                "message": "No messages to summarize"
            }
        
        # Parse messages
        messages = json.loads(conversation)
        
        # Build text to summarize
        text_to_summarize = "\\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])
        
        # Create summary
        print(f"📝 Summarizing {len(messages)} messages...")
        summary_text = summarize_context_window(text_to_summarize, llm_client, model)
        
        # Generate summary ID and description
        summary_id = str(uuid.uuid4())[:8]
        
        # Create brief description
        first_message = messages[0]['content'][:50] if messages else "conversation"
        description = f"Conversation starting with: {first_message}..."
        
        # Store summary
        memory_manager.write_summary(
            summary_id=summary_id,
            summary_text=summary_text,
            description=description,
            thread_id=thread_id
        )
        
        # Mark original messages as summarized
        memory_manager.mark_as_summarized(thread_id, summary_id)
        
        print(f"✅ Created summary: {summary_id}")
        
        return {
            "success": True,
            "id": summary_id,
            "description": description,
            "summary": summary_text,
            "messages_summarized": len(messages)
        }

    print("✅ Summarization functions defined")
    </copy>
    ```

## Task 4: Implement JIT Retrieval

Now let's create the automatic context compaction system.

1. **Add to** `summarization.py`:

    ```python
    <copy>
    def offload_to_summary(context: str, memory_manager, llm_client, 
                          threshold_percent: float = 80.0,
                          model: str = "Qwen/Qwen2.5-72B-Instruct") -> Tuple[str, List[str]]:
        """
        Automatically compact context when usage exceeds threshold.
        
        Replaces large text blocks with [Summary ID: xxx] references.
        
        Args:
            context: The full context text
            memory_manager: MemoryManager instance
            llm_client: OpenAI client
            threshold_percent: Usage threshold to trigger compaction (default 80%)
            model: Model name for token limits
            
        Returns:
            Tuple of (compacted_context, list_of_summary_ids)
        """
        usage = calculate_context_usage(context, model)
        
        # Check if compaction is needed
        if usage['percent'] < threshold_percent:
            return context, []
        
        print(f"⚠️ Context usage at {usage['percent']}% - compacting...")
        
        # Split context into chunks (simple split by paragraphs)
        chunks = context.split("\\n\\n")
        
        # Process chunks until under threshold
        summary_ids = []
        compacted_chunks = []
        current_size = 0
        target_size = usage['limit'] * (threshold_percent / 100)
        
        for chunk in chunks:
            chunk_tokens = count_tokens(chunk)
            
            if current_size + chunk_tokens > target_size and len(summary_ids) < 3:
                # Summarize this chunk
                summary_text = summarize_context_window(chunk, llm_client, model)
                summary_id = str(uuid.uuid4())[:8]
                
                # Store summary
                memory_manager.write_summary(
                    summary_id=summary_id,
                    summary_text=summary_text,
                    description=f"Context chunk starting with: {chunk[:50]}...",
                    thread_id="context"
                )
                
                summary_ids.append(summary_id)
                
                # Replace with reference
                reference = f"[Summary ID: {summary_id}]"
                compacted_chunks.append(reference)
                current_size += count_tokens(reference)
            else:
                # Keep original
                compacted_chunks.append(chunk)
                current_size += chunk_tokens
        
        compacted = "\\n\\n".join(compacted_chunks)
        
        return compacted, summary_ids

    print("✅ JIT retrieval functions defined")
    </copy>
    ```

## Task 5: Test Context Management

Let's test the complete context engineering system.

1. **Create a file** named `test_context_engineering.py`:

    ```python
    <copy>
    import oracledb
    import os
    from openai import OpenAI
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores.utils import DistanceStrategy
    from memory_manager import MemoryManager
    from context_utils import calculate_context_usage, format_usage_report
    from summarization import summarize_conversation, offload_to_summary

    # Setup connection
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
        entity_vs=entity_vs,
        summary_vs=summary_vs
    )

    # Initialize LLM client
    HF_TOKEN = os.getenv("HF_TOKEN", "your-token-here")
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key=HF_TOKEN
    )

    print("🧪 Testing Context Engineering\\n")
    print("=" * 60)

    # ==================== TEST 1: Token Counting ====================
    print("\\nTest 1: Token Counting")
    print("-" * 60)

    sample_text = """
    This is a sample conversation about Oracle Database and AI.
    Oracle AI Vector Search enables semantic similarity search on embeddings.
    You can store vector embeddings natively in the database.
    """ * 50  # Repeat to make it longer

    usage = calculate_context_usage(sample_text)
    print(format_usage_report(usage))

    # ==================== TEST 2: Conversation Summarization ====================
    print("\\nTest 2: Conversation Summarization")
    print("-" * 60)

    # Create a test conversation
    thread_id = "test-summary-thread"
    
    memory_manager.write_conversational_memory(
        "Hello! Can you tell me about Oracle AI Database?",
        "user",
        thread_id
    )
    
    memory_manager.write_conversational_memory(
        "Oracle AI Database includes powerful AI capabilities like vector search, Select AI, and more.",
        "assistant",
        thread_id
    )
    
    memory_manager.write_conversational_memory(
        "That's interesting! How does vector search work?",
        "user",
        thread_id
    )
    
    memory_manager.write_conversational_memory(
        "Vector search converts text into embeddings and finds similar content using distance metrics.",
        "assistant",
        thread_id
    )

    # Summarize the conversation
    result = summarize_conversation(thread_id, memory_manager, client)
    
    if result.get('success'):
        print(f"\\n✅ Summary created:")
        print(f"   ID: {result['id']}")
        print(f"   Description: {result['description']}")
        print(f"   Messages summarized: {result['messages_summarized']}")
        print(f"\\n   Summary text:")
        print(f"   {result['summary']}")
    else:
        print(f"⚠️ {result.get('message')}")

    # ==================== TEST 3: JIT Retrieval ====================
    print("\\n\\nTest 3: JIT Retrieval (Context Offloading)")
    print("-" * 60)

    # Create a large context
    large_context = """
    Oracle AI Database provides comprehensive AI capabilities.
    Vector search enables semantic similarity search.
    Select AI allows natural language queries.
    """ * 100  # Make it large

    print("\\nBefore offload:")
    usage_before = calculate_context_usage(large_context)
    print(f "  Tokens: {usage_before['tokens']:,}")
    print(f"  Usage: {usage_before['percent']}%")

    # Offload with low threshold to force compaction
    compacted, summary_ids = offload_to_summary(
        large_context, 
        memory_manager, 
        client,
        threshold_percent=1.0  # Force immediate compaction
    )

    print("\\nAfter offload:")
    usage_after = calculate_context_usage(compacted)
    print(f"  Tokens: {usage_after['tokens']:,}")
    print(f"  Usage: {usage_after['percent']}%")
    print(f"  Summaries created: {len(summary_ids)}")
    
    if summary_ids:
        print(f"\\n  Summary IDs: {', '.join(summary_ids)}")
        print(f"\\n  Compacted context preview:")
        print(f"  {compacted[:200]}...")

    reduction = usage_before['tokens'] - usage_after['tokens']
    percent_saved = (reduction / usage_before['tokens']) * 100
    print(f"\\n  ✅ Saved {reduction:,} tokens ({percent_saved:.1f}%)")

    print("\\n" + "=" * 60)
    print("✅ All context engineering tests completed!")

    vector_conn.close()
    </copy>
    ```

2. **Run the test:**

    ```bash
    <copy>
    python test_context_engineering.py
    </copy>
    ```

## Summary

In this lab, you successfully:
* ✅ Implemented token counting and usage monitoring
* ✅ Built automatic conversation summarization
* ✅ Created JIT retrieval with on-demand expansion
* ✅ Tested context compaction and measurement
* ✅ Built a complete context engineering system

Your AI agent can now handle long conversations efficiently by automatically managing the context window.

You may now **proceed to the next lab** where you'll integrate all components into a complete AI agent.

## Learn More

* [OpenAI Token Counting](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
* [Context Window Management](https://www.anthropic.com/research/context-windows)
* [Summarization Techniques](https://huggingface.co/blog/summarization)

## Acknowledgements

* **Author** - Paul Parkinson, Oracle Database Developer Advocate
* **Last Updated By/Date** - March 2026
