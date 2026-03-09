# Lab 5: Context Engineering & Web Integration

## Context Window Management, Summarization, and Tavily Search

--------

### Objective

While memory engineering focuses on *what to store and retrieve*, context engineering focuses on *how to manage what's in the context window right now*. In this lab, you'll build the techniques that keep Proteus's context lean and effective, plus integrate web search so Proteus can find information beyond the internal knowledge base.

> **Context engineering** refers to the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference, including all the other information that may land there outside of the prompts.

--------

### What This Lab Covers

| Step | Function | Purpose |
|------|----------|---------|
| **1. Calculate Usage** | `calculate_context_usage()` | Monitor what % of the context window is used |
| **2. Summarize** | `summarise_context_window()` | Compress long content into summaries using LLM |
| **3. Compact** | `summarize_conversation()` / `summarize_and_store()` | Agent-triggered compaction when context gets long |
| **4. Just-in-Time Retrieval** | `expand_summary()` tool | Let Proteus expand summaries on demand |
| **5. Web Search** | `search_tavily()` tool | External retrieval with automatic knowledge base persistence |

### The Context Management Flow

```
Context built → Check usage % → Proteus may compact (summarize) → Store summary with ID
                                                               ↓
Proteus sees: [Summary ID: abc123] Brief description ← Proteus calls expand_summary("abc123") if needed
```

This approach keeps the context lean while giving Proteus access to full details when required.

--------

### Just-in-Time (JIT) Retrieval

**Just-In-Time retrieval** is the process of fetching only the information needed at the exact moment the agent requires it, based on the current task or reasoning step. Instead of loading everything upfront, the system dynamically retrieves the minimal, most relevant data on demand.

In the context of agent memory, JIT is a retrieval-control strategy where memory access is triggered by the agent's current goal. Rather than preloading large histories or the full knowledge base, the system dynamically filters, ranks, and injects only the information that materially influences the next token. This reduces context saturation, improves attention allocation, and increases reasoning fidelity.

For Proteus, this means:
- Summary pointers (ID + description) are always loaded — cheap, a few tokens each
- Full summary content is only retrieved when Proteus decides it's relevant to the current ticket
- This avoids wasting thousands of context tokens on summaries of unrelated troubleshooting sessions

--------

## Step 1: Context Window Usage Calculator

This simple utility estimates how much of the context window is being used. Proteus can check this to decide whether compaction is needed.

    ```python
    def calculate_context_usage(context: str, model: str = "gpt-4o") -> dict:
        """Calculate context window usage as percentage."""
        estimated_tokens = len(context) // 4  # ~4 chars per token
        max_tokens = MODEL_TOKEN_LIMITS.get(model, 128000)
        percentage = (estimated_tokens / max_tokens) * 100
        return {
            "tokens": estimated_tokens,
            "max": max_tokens,
            "percent": round(percentage, 1),
        }
    ```

## Step 2: Context Summarizer

When the context window grows large — after several tool calls, long conversations, or large search results — we can compress it into a summary. The full content is stored in Summary Memory, and the context window gets a compact pointer.

    ```python
    import uuid


    def summarise_context_window(
        content: str, memory_manager, llm_client, model: str = "gpt-4o"
    ) -> dict:
        """Summarise context window using LLM and store in summary memory."""
        summary_prompt = f"""
    You are compressing an AI IT support agent's context window for later retrieval.
    The content may include conversation memory, KB articles, entities, workflows, and prior summaries.

    Produce a compact summary that preserves:
    - user's reported issue and constraints
    - key diagnostic findings already established
    - important entities (server names, service names, team names, ticket IDs)
    - unresolved questions and next actions

    Output 4-7 short bullet points.
    Be faithful to the source, and do not add new facts.

    Context window content:
    {content[:3000]}
    """.strip()

        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_completion_tokens=220,
        )
        summary = response.choices[0].message.content

        desc_response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Write a short label (max 12 words) for this summary:\n{summary}",
                }
            ],
            max_completion_tokens=40,
        )
        description = desc_response.choices[0].message.content.strip()

        summary_id = str(uuid.uuid4())[:8]
        memory_manager.write_summary(summary_id, content, summary, description)

        return {"id": summary_id, "description": description, "summary": summary}
    ```

## Step 3: Context Offloader

This utility checks whether the context exceeds a threshold and, if so, automatically summarizes and replaces the content with a compact reference.

    ```python
    def offload_to_summary(
        context: str,
        memory_manager,
        llm_client,
        threshold_percent: float = 80.0,
    ) -> tuple:
        """If context exceeds threshold, summarise and return compacted version."""
        usage = calculate_context_usage(context)

        if usage["percent"] < threshold_percent:
            return context, []  # No offload needed

        result = summarise_context_window(context, memory_manager, llm_client)

        compact = f"[Summary ID: {result['id']}] {result['description']}"
        return compact, [result]
    ```

## Step 4: Register Summary Tools for the Agent

These are **agent-triggered** tools — Proteus decides when to call them based on the current context. We register them with `augment=True` for better semantic retrieval.

### Design Decision: Mark Instead of Delete

When conversation history grows large, we need to reduce context. We chose to **mark messages as summarized** rather than delete them:

| Approach | Pros | Cons |
|----------|------|------|
| **Delete summarized messages** | Simple, immediate space savings | Permanent data loss, can't audit or recover |
| **Mark as summarized (our choice)** | Preserves history, reversible, auditable | Slightly more complex queries |

Memory should be *compressed* or *forgotten*, not *erased*. The original messages remain for auditing, debugging, or reprocessing.

### The Compaction Flow

    ```
    Ticket thread has 50 messages → Context too large → summarize_conversation(thread_id)
                                                            ↓
                                1. Read unsummarized messages
                                2. LLM summarizes them
                                3. Store summary with unique ID
                                4. UPDATE messages SET summary_id = 'abc123'
                                                            ↓
                                Next read: Only new messages appear + Summary ID reference
    ```

    ```python
    @toolbox.register_tool(augment=True)
    def expand_summary(summary_id: str) -> str:
        """Expand a summary reference to full content, including the original conversation
        messages that were compacted into it. Use when you need more details from a
        [Summary ID: xxx] reference during troubleshooting."""
        summary_text = memory_manager.read_summary_memory(summary_id)

        original_msgs = memory_manager.get_messages_by_summary_id(summary_id)
        if original_msgs:
            lines = [f"[{m['role']}] {m['content']}" for m in original_msgs]
            return (
                f"Summary:\n{summary_text}\n\n"
                f"Original messages ({len(original_msgs)}):\n" + "\n".join(lines)
            )
        return summary_text


    @toolbox.register_tool(augment=True)
    def summarize_and_store(text: str) -> str:
        """Summarize a long text block and store it. Returns [Summary ID: ...] for later expansion."""
        result = summarise_context_window(text, memory_manager, client)
        return f"Stored as [Summary ID: {result['id']}] {result['description']}"


    @toolbox.register_tool(augment=True)
    def summarize_conversation(thread_id: str) -> str:
        """
        Summarize unsummarized conversation turns for a ticket thread and mark those
        turns with a summary_id. Use this when conversation memory becomes long and
        you need context compaction during a troubleshooting session.
        """
        unsummarized = memory_manager.get_unsummarized_messages(thread_id, limit=200)
        if not unsummarized:
            return "No unsummarized conversation turns found."

        full_text = "\n".join([f"[{m['role']}] {m['content']}" for m in unsummarized])
        result = summarise_context_window(full_text, memory_manager, client)

        message_ids = [m["id"] for m in unsummarized]
        memory_manager.mark_as_summarized(thread_id, result["id"], message_ids=message_ids)

        return f"Conversation summarized as [Summary ID: {result['id']}] {result['description']}"
    ```

## Step 5: Web Search with Tavily

Proteus needs to search external sources when the internal knowledge base doesn't have the answer — for example, looking up a new Kubernetes CVE, a vendor advisory, or an unfamiliar error message.

We use [Tavily](https://tavily.com/), an AI-optimized search API designed for LLM applications.

### The Search-and-Store Pattern

When Proteus calls `search_tavily()`, it doesn't just return results — it **persists them to the knowledge base**:

    ```
    Proteus calls search_tavily("CrowdStrike Falcon sensor error 0x80070005")
        ↓
    Tavily API returns results
        ↓
    Each result is written to knowledge_base_vs with metadata (title, URL, timestamp)
        ↓
    Future tickets can retrieve this information without searching again
    ```

This pattern means Proteus **learns** from its searches. Information discovered once becomes part of the agent's long-term memory.

    ```python
    set_env_securely("TAVILY_API_KEY", "Tavily API Key: ")
    ```

    ```python
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


    @toolbox.register_tool(augment=True)
    def search_tavily(query: str, max_results: int = 5):
        """
        Search the web for information not available in the internal knowledge base.
        Results are automatically stored in the knowledge base for future reference.
        Use for vendor advisories, CVEs, error messages, and external documentation.
        """
        response = tavily_client.search(query=query, max_results=max_results)
        results = response.get("results", [])

        for result in results:
            text = (
                f"Title: {result.get('title', '')}\n"
                f"Content: {result.get('content', '')}\n"
                f"URL: {result.get('url', '')}"
            )

            metadata = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "score": result.get("score", 0),
                "source_type": "tavily_search",
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }

            memory_manager.write_knowledge_base(text, metadata)

        return results
    ```

### Verify Tool Retrieval

Let's confirm that Proteus can find the search tool when needed:

    ```python
    import pprint

    retrieved_tools = memory_manager.read_toolbox("Search the internet for vendor documentation")
    pprint.pprint(retrieved_tools)
    ```

## Lab 5 Recap

| What You Built | Why It Matters |
|---------------|----------------|
| `calculate_context_usage()` | Monitor context consumption and trigger compaction proactively |
| `summarise_context_window()` | LLM-powered compression that preserves key diagnostic details |
| `offload_to_summary()` | Automatic threshold-based context offloading |
| `expand_summary()` tool | JIT retrieval — Proteus expands only the summaries it needs |
| `summarize_conversation()` tool | Log compaction for long troubleshooting threads |
| `search_tavily()` tool | External search with automatic knowledge base persistence |

**Key Insight**: The search-and-store pattern means Proteus builds institutional knowledge over time. The first time a SeerGroup employee asks about a specific error, Proteus searches externally. The second time, Proteus finds the answer in its own knowledge base — no external call needed, faster and cheaper.

**Next up**: In Lab 6, we'll wire everything together into the `call_agent` harness and run Proteus through real IT support scenarios.

## Learn More



## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026