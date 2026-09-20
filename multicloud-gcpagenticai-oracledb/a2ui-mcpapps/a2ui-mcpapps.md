# Develop A2UI and MCPApps

## Introduction

Build a governed review experience for Oracle inventory recommendations. The same Oracle transaction is presented through A2UI to Gemini Enterprise and through an MCP App dashboard to compatible hosts. The model can request a read-only recommendation; only an explicit user action can approve or reject it.

### Objectives

- Understand the relationship between A2A, MCP, A2UI, and MCP Apps.
- Render an Oracle recommendation with an allowlisted A2UI catalog.
- Register a `ui://` MCP App resource.
- Enforce actor-bound, short-lived approval at the service and database layers.

### Prerequisites

- Completed Lab 5.
- The agent service and Oracle inventory recommendation view available.
- Node.js 20+ for the MCP App sample.
- Gemini Enterprise preview access for A2UI, or an MCP Apps-compatible host.

## Task 1: Trace the contracts

```text
Host -> A2A or MCP -> agent service -> Oracle MCP Toolkit -> Oracle Database
                         |                 |
                         +-- A2UI/MCP App  +-- governed recommendation and audit
```

Use A2A for agent-to-agent messages, MCP for tool discovery and invocation, A2UI for portable declarative UI updates, and MCP Apps for a sandboxed `ui://` resource.

## Task 2: Render a read-only recommendation

1. Start the agent service and request an inventory recommendation for `SKU-500`.
2. Confirm the response contains product, source warehouse, destination warehouse, quantity, and recommendation ID returned by Oracle.
3. Emit only the supported A2UI messages: `createSurface`, `updateComponents`, and `updateDataModel`.
4. Use the fixed catalog and component allowlist from the source project. Build DOM text with `textContent`; never inject model-generated HTML or scripts.

The browser must render the recommendation but must not accept client-selected source, target, product, or quantity values.

## Task 3: Run the MCP App

From the source `a2ui_mcpapps_mcptoolkit/mcp-app` project:

```bash
npm install
npm run build
npm run dev
```

Register the server resource as a `ui://` MCP App with the host. Keep the model-visible tool read-only. The app-only tools for approve and reject must require an explicit user interaction.

## Task 4: Test approval safeguards

1. Request a recommendation and record its ID.
2. Approve it in the UI without editing product, source, target, or quantity.
3. Confirm the service binds approval to the authenticated actor and a short-lived, single-use nonce.
4. Attempt to reuse the nonce, change the quantity, or approve as another actor. Each attempt must fail.
5. Reject a recommendation and confirm no write tool is called.

Oracle performs the final row locks, current-stock revalidation, audit insert, and reservation in one stored procedure transaction.

## Task 5: Test failure and security cases

- Send malformed A2UI component data; the allowlist must reject it.
- Remove the actor identity; approval must fail.
- Expire the approval token; approval must fail.
- Attempt a model-visible write; no write-capable model tool should exist.
- Inspect logs for bearer tokens, passwords, or wallet paths; none may appear.

## Conclusion

A2UI and MCP Apps make the workflow usable without moving authority into the model or browser. Continue to Lab 7 for the lakehouse data path that supplies broader analytical context.

## Acknowledgements

- A2UI v0.9.1 specification
- MCP Apps extension documentation
- Oracle MCP Toolkit integration plan and security notes
