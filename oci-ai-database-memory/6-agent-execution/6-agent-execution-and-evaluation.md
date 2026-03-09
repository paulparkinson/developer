# Lab 6: Agent Execution & Evaluation

## Running Proteus Through IT Support Scenarios

--------

## Objective

This is where everything comes together. You'll build the **turn-level agent harness** that integrates all memory types, context engineering, and tool calling — then run Proteus through realistic IT support scenarios and compare the engineered approach against a naive baseline.

--------

## Step 1: The Agent System Prompt

The system prompt tells Proteus how to use its memory systems and tools. Notice it establishes a **priority order** for memory types — this is critical for reliable behavior.

    ```python
    import json as json_lib
    from openai import OpenAI

    client = OpenAI()

    # Persistent context-window tracker — survives across call_agent() invocations
    context_size_history = []  # list of (run_label, iteration, estimated_tokens)


    AGENT_SYSTEM_PROMPT = """
    # System Instructions
    You are Proteus, SeerGroup Solutions' AI IT Support Agent. You have access to memory systems and
    diagnostic tools to help resolve internal support tickets.

    IMPORTANT: The user's input contains CONTEXT retrieved from multiple memory systems.
    Each memory section has a Purpose and When-to-use guide — follow them.

    ## Memory Priority Order
    1. **Conversation Memory** — check what the user already reported and what you already suggested.
    2. **Knowledge Base Memory** — cite facts from stored KB articles, runbooks, and incident reports
    before searching externally.
    3. **Entity Memory** — resolve named references ("that server", "the team we discussed") from here.
    4. **Workflow Memory** — reuse proven resolution sequences for similar past tickets.
    5. **Summary Memory** — expand a summary ID only when you need specific details from an older session.

    ## Tool Output Handling
    Tool call outputs are logged to a Tool Log table and replaced with compact references in context.
    The preview in each [Tool Log ...] reference contains enough to reason about the result.
    If you need the full output, it can be retrieved from the database — but prefer working with
    the preview and the knowledge base (where search results are also stored).

    ## Context Management
    If conversation memory is getting long or repetitive, call summarize_conversation(thread_id)
    to compact it. Use summarization tools at your discretion when they improve context quality.

    When answering:
    1. FIRST, use the context provided in the input
    2. Expand summary IDs just-in-time when needed
    3. Use external search tools only if memory context is insufficient
    4. Keep responses evidence-based and aligned with SeerGroup's internal documentation
    5. Always mention the relevant team to escalate to when appropriate
    """
    ```

## Step 2: Tool Execution and OpenAI Chat Wrapper

    ```python
    def execute_tool(tool_name: str, tool_args: dict) -> str:
        """Execute a tool by looking it up in the toolbox."""
        if tool_name not in toolbox._tools_by_name:
            return f"Error: Tool '{tool_name}' not found"
        return str(toolbox._tools_by_name[tool_name](**tool_args) or "Done")


    def call_openai_chat(messages: list, tools: list = None, model: str = "gpt-4o"):
        """Call OpenAI Chat Completions API with tools."""
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)
    ```

## Step 3: The Turn-Level Agent Harness

This is the core of Proteus. Each call to `call_agent()` represents one **agent run** (one user turn handled). Within a run, the **tool-call loop** repeats: model reasoning → optional tool calls → harness executes tools → model observes results → repeat until a final answer.

### The Flow

    ```
    1. BUILD CONTEXT (programmatic)
    ├── Read conversational memory (unsummarized turns only)
    ├── Read knowledge base (relevant KB articles)
    ├── Read workflow memory (past resolution patterns)
    ├── Read entity memory (servers, services, people)
    └── Read summary context (available summary IDs + descriptions)

    2. GET TOOLS (programmatic)
    └── Retrieve semantically relevant tools from toolbox

    3. STORE USER MESSAGE (programmatic)
    └── Persist the user message + best-effort entity extraction

    4. WITHIN-RUN TOOL-CALL LOOP (up to max_iterations, within time budget)
    ├── Call LLM with context + tool schemas
    ├── If tool calls → execute tools and append results
    ├── If tools changed memory (search/compaction) → rebuild context
    └── If no tool calls → finalize answer

    5. GUARDED STOP
    └── If budget hit → force a final best-effort answer (no tools)

    6. SAVE RESULTS (programmatic)
    ├── Write workflow (if tools were used)
    ├── Best-effort entity extraction on final answer
    └── Store assistant response in conversational memory
    ```

    ```python
    import time


    def call_agent(
        query: str,
        thread_id: str = "1",
        max_iterations: int = 10,
        max_execution_time_s: float = 60.0,
    ) -> str:
        """Turn-level agent harness: build context, run tool-call loop, persist results.

        Appends (run_label, iteration, tokens) to the global context_size_history list
        so context growth can be visualised across multiple runs.
        """
        thread_id = str(thread_id)
        steps = []
        run_label = f"Run {len(set(r for r, _, _ in context_size_history)) + 1}"

        start_time = time.time()
        timed_out = False

        # ── 1. Build context from memory ──
        print("\n" + "=" * 50)
        print("🧠 BUILDING CONTEXT...")

        def build_context() -> str:
            """Rebuild the full context from the current memory state."""
            ctx = f"# Support Ticket\n{query}\n\n"
            ctx += memory_manager.read_conversational_memory(thread_id) + "\n\n"
            ctx += memory_manager.read_knowledge_base(query) + "\n\n"
            ctx += memory_manager.read_workflow(query) + "\n\n"
            ctx += memory_manager.read_entity(query) + "\n\n"
            ctx += memory_manager.read_summary_context(query) + "\n\n"
            return ctx

        context = build_context()

        print("==== CONTEXT WINDOW ====\n")
        print(context)

        # ── 2. Check context usage ──
        usage = calculate_context_usage(context)
        print(f"📊 Context: {usage['percent']}% ({usage['tokens']}/{usage['max']} tokens)")
        if usage["percent"] > 80:
            print(
                "⚠️ Context >80% — Proteus may call summarize_conversation(thread_id) for compaction."
            )

        # ── 3. Get tools ──
        dynamic_tools = memory_manager.read_toolbox(query, k=5)

        # Ensure summary tools are always available for discretionary compaction/JIT
        summary_tool_candidates = memory_manager.read_toolbox(
            "summarize conversation compact context expand summary memory", k=5
        )
        must_have = {"expand_summary", "summarize_conversation", "summarize_and_store"}
        existing = {t.get("function", {}).get("name") for t in dynamic_tools}

        for tool in summary_tool_candidates:
            name = tool.get("function", {}).get("name")
            if name in must_have and name not in existing:
                dynamic_tools.append(tool)
                existing.add(name)

        print(f"🔧 Tools: {[t['function']['name'] for t in dynamic_tools]}")

        # ── 4. Store user message & extract entities ──
        memory_manager.write_conversational_memory(query, "user", thread_id)
        try:
            memory_manager.write_entity("", "", "", llm_client=client, text=query)
        except:
            pass

        # ── 5. Within-run tool-call loop ──
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        final_answer = ""

        tool_schema_tokens = len(json_lib.dumps(dynamic_tools)) // 4 if dynamic_tools else 0

        print("\n🤖 TOOL-CALL LOOP")
        for iteration in range(max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Track context window size
            total_chars = sum(len(m.get("content", "") or "") for m in messages)
            est_tokens = (total_chars // 4) + tool_schema_tokens
            context_size_history.append((run_label, iteration + 1, est_tokens))

            if max_execution_time_s is not None:
                elapsed = time.time() - start_time
                if elapsed > max_execution_time_s:
                    timed_out = True
                    print(
                        f"\n⏱️ Time limit reached ({elapsed:.1f}s > {max_execution_time_s:.1f}s). Finalizing..."
                    )
                    break

            response = call_openai_chat(messages, tools=dynamic_tools)
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments or "{}"
                    try:
                        tool_args = json_lib.loads(raw_args)
                    except Exception as e:
                        result = f"Error: invalid JSON arguments for {tool_name}: {e}"
                        print(f"🛠️ {tool_name}(<invalid args>)")
                        steps.append(f"{tool_name}(<invalid args>) → failed")
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": result}
                        )
                        continue

                    if not isinstance(tool_args, dict):
                        result = f"Error: arguments for {tool_name} must be a JSON object."
                        print(f"🛠️ {tool_name}(<non-object args>)")
                        steps.append(f"{tool_name}(<non-object args>) → failed")
                        messages.append(
                            {"role": "tool", "tool_call_id": tc.id, "content": result}
                        )
                        continue

                    # Ensure conversation compaction targets the active ticket thread
                    if tool_name == "summarize_conversation":
                        tool_args["thread_id"] = thread_id

                    args_display = {
                        k: (v[:50] + "..." if isinstance(v, str) and len(v) > 50 else v)
                        for k, v in tool_args.items()
                    }
                    print(f"🛠️ {tool_name}({args_display})")

                    if max_execution_time_s is not None:
                        elapsed = time.time() - start_time
                        if elapsed > max_execution_time_s:
                            timed_out = True
                            result = f"Error: time limit reached before executing {tool_name}."
                            steps.append(f"{tool_name}({args_display}) → timed out")
                            messages.append(
                                {"role": "tool", "tool_call_id": tc.id, "content": result}
                            )
                            break

                    try:
                        result = execute_tool(tool_name, tool_args)
                        steps.append(f"{tool_name}({args_display}) → success")
                    except Exception as e:
                        result = f"Error: {e}"
                        steps.append(f"{tool_name}({args_display}) → failed")

                    print(f"   → {result[:200]}...")

                    # Offload tool output to TOOL_LOG table
                    compact_result = memory_manager.write_tool_log(
                        thread_id, tc.id, tool_name, raw_args, str(result)
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": compact_result}
                    )

                    # If tools changed memory state, refresh context
                    if tool_name in {
                        "search_tavily",
                        "summarize_conversation",
                        "summarize_and_store",
                    }:
                        context = build_context()
                        if len(messages) >= 2 and messages[1].get("role") == "user":
                            messages[1]["content"] = context
                        usage = calculate_context_usage(context)
                        print(
                            f"   Refreshed context: {usage['percent']}% "
                            f"({usage['tokens']}/{usage['max']} tokens)"
                        )

                if timed_out:
                    break
            else:
                final_answer = msg.content or ""
                print(f"\n✅ DONE ({len(steps)} tool calls)")
                break

        # ── Guarded stop ──
        if not final_answer:
            reason = "time limit" if timed_out else "iteration limit"
            print(f"\n⚠️ Stopped due to {reason}. Generating best-effort final answer...")
            try:
                final_messages = messages + [
                    {
                        "role": "user",
                        "content": "Finalize your answer using the context and tool outputs so far. Do not call tools.",
                    }
                ]
                final_resp = call_openai_chat(final_messages, tools=None)
                final_answer = final_resp.choices[0].message.content or ""
            except Exception as e:
                final_answer = f"Error: unable to finalize answer: {e}"

        # ── 6. Save results ──
        if steps:
            memory_manager.write_workflow(query, steps, final_answer)
        try:
            memory_manager.write_entity("", "", "", llm_client=client, text=final_answer)
        except:
            pass
        memory_manager.write_conversational_memory(final_answer, "assistant", thread_id)

        print("\n" + "=" * 50 + f"\n💬 ANSWER:\n{final_answer}\n" + "=" * 50)
        return final_answer
    ```

## Step 4: Run Proteus Through IT Support Scenarios

### Scenario 1: Simple Ticket — "I can't log in"

    ```python
    call_agent(
        "Hi, I can't log in to any of our internal apps this morning. "
        "Jira, Confluence, everything is giving me an error.",
        thread_id="TICKET-2025-001",
    )
```

### Scenario 2: Follow-up on the Same Ticket

Watch how Proteus uses conversational memory from the previous turn:

    ```python
    call_agent(
        "I tried restarting my browser and clearing cookies like you suggested, "
        "but it's still not working. My colleague on Floor 2 is having the same issue.",
        thread_id="TICKET-2025-001",
    )
```

### Scenario 3: Infrastructure Issue Requiring Web Search

    ```python
    call_agent(
        "Our deployment pipeline is failing with an error about a Kubernetes "
        "admission webhook timeout. This started after the cluster upgrade last night.",
        thread_id="TICKET-2025-002",
    )
```

### Scenario 4: Cross-Referencing Past Incidents

Proteus should recognize the AUTH-SVC pattern from its knowledge base:

    ```python
    call_agent(
        "AUTH-SVC is showing CrashLoopBackOff again. Did this happen before? "
        "What did we do last time?",
        thread_id="TICKET-2025-003",
    )
    ```

### Visualize Context Window Growth

    ```python
    import matplotlib.pyplot as plt

    if context_size_history:
        tokens = [t for _, _, t in context_size_history]

        plt.figure(figsize=(8, 3))
        plt.plot(range(1, len(tokens) + 1), tokens, marker="o")
        plt.xlabel("Global Iteration (across all runs)")
        plt.ylabel("Estimated Tokens")
        plt.title("Context Window Size Over Agent Iterations")
        plt.tight_layout()
        plt.show()
    else:
        print("No iterations recorded — run call_agent() first.")
    ```
## Key Takeaways

### Agent Architecture Concepts

In OpenAI-style framing:
- An **agent run** (one user turn handled) is what `call_agent(...)` executes
- Within a run, the **tool-call loop** repeats: model reasoning → optional tool calls → harness executes tools → model observes results → repeat until a final answer

An **agent harness** is the runtime scaffolding around that loop. In this workshop, it is a **memory-based agent harness** where:
- Context is assembled from multiple memory types each run
- Tools are discovered and executed within the run
- Outputs are written back into memory for future runs
- Summaries compact context while preserving continuity

### The Practical Takeaway

Strong agents are not just model prompts. They are **run + harness systems**, and memory engineering is the control layer that makes them reliable, stateful, and scalable. The key discipline is deciding:
- What should be stored, retrieved, summarized, and refreshed
- How to keep context windows relevant, not just large
- How to treat memory as an evolving system that improves agent reliability over time

--------

## Workshop Complete 🎉

You've built a complete memory-powered AI agent from the ground up:

| Lab | What You Built |
|----------|---------------|
| **1** | Set up your Jupyter environment in VS Code, connected to Oracle Database, and created a VECTOR user |
| **2** | Vector search foundations with OracleVS and SeerGroup KB data |
| **3** | Memory architecture — 6 types, SQL + vector stores, design principles |
| **4** | MemoryManager class with unified read/write for all memory types |
| **5** | Context engineering — usage tracking, summarization, JIT retrieval, web search |
| **6** | Agent harness and IT support scenarios |

Proteus now demonstrates how modern AI agents maintain context, learn from interactions, and manage information across sessions — all backed by Oracle AI Database 26ai as the converged storage layer for relational, vector, and semantic data.

## Learn More



## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026