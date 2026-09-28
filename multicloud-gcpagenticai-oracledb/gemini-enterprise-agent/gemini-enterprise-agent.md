# Setup and Use Oracle AI Database Agent for Gemini Enterprise Apps

## Introduction

Gemini Enterprise can discover Oracle-backed agents through A2A agent cards. In this lab you will deploy or identify the public HTTPS host for the workshop agents, import the Oracle graph, spatial, Select AI, and inventory-action cards, and test each agent with a self-contained prompt.

### Objectives

- Understand the agent-card and runtime endpoints.
- Import Oracle AI Database agents into Gemini Enterprise.
- Test graph, spatial, Select AI, and inventory-action flows.
- Recognize the ADC fallback behavior of the action agent.

### Prerequisites

- Completed Labs 1 and 2.
- A deployed HTTPS agent host with the four agent-card endpoints.
- Gemini Enterprise access with permission to add custom agents.
- Oracle AI Database credentials and permission to configure the managed agent and sample data.

## Task 1: Prepare the Oracle AI Database agent

Clone the sample repository and enter its directory:

```bash
git clone https://github.com/paulparkinson/oracle-ai-database-gcp-gemini.git
cd oracle-ai-database-gcp-gemini
```

1. Review the [SQL assets README](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini/blob/main/sql/README.md) and select the setup scripts for your database and identity configuration. The manager-schema and regional Deep Data Security configurations are mutually exclusive for the same users.
2. Connect with SQLcl as the documented database administrator or application schema and execute the required setup scripts from the [`sql/` directory](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini/tree/main/sql) in the order described by the README.
3. After installing the Oracle AI Database Agent scripts, run `@sql/verify_oracle_ai_database_agent.sql` as the application schema. Confirm the `ORACLE_AI_DATABASE_AGENT` team and its expected tools are listed.

## Task 2: Identify the agent endpoints

Replace `YOUR_PUBLIC_AGENT_HOST` with the HTTPS host supplied for the workshop:

```text
https://YOUR_PUBLIC_AGENT_HOST/agent-card-graph.json
https://YOUR_PUBLIC_AGENT_HOST/agent-card-spatial.json
https://YOUR_PUBLIC_AGENT_HOST/agent-card-select-ai.json
https://YOUR_PUBLIC_AGENT_HOST/agent-card-action.json
```

Open each URL in a browser or use:

```bash
curl -fsS https://YOUR_PUBLIC_AGENT_HOST/agent-card-graph.json | jq .
```

Confirm that every card advertises an HTTPS URL and a distinct agent name.

## Task 3: Import the agents

1. In Gemini Enterprise, open **Agents**.
2. Select **Add agent** and choose the custom A2A import flow.
3. Import each card URL from Task 1.
4. Leave authentication empty only when the workshop endpoint is intentionally public. Private database access is covered in Lab 5.
5. Open each imported agent in a new chat so the prompt is routed to the intended agent.

## Task 4: Test database-backed analysis

Select the graph agent and submit:

```text
Use the Oracle Database property graph to show supply-chain dependencies for SKU-500 and render the graph as an image.
```

Expected result: a graph image and a text summary mentioning `SKU-500`.

Select the spatial agent and submit:

```text
Show a map for SKU-500 and highlight warehouse hotspots plus the best relief route.
```

Expected result: a hotspot map and a summary of the pressure and relief locations.

Select the Select AI agent and submit:

```text
Using only the Oracle inventory risk demo tables, list the top products at risk of stockouts next quarter, including stockout probability, projected revenue impact, and primary region.
```

Use complete questions instead of conversational follow-ups. The current Select AI path is most reliable when each request contains its own product and business context.

## Task 5: Test the action recommendation

Select the inventory-action agent and submit:

```text
What inventory action should we take for SKU-500 given the current supply risk? Gather graph, spatial, and external evidence first, then recommend the safest next move and say whether approval is required.
```

Expected result: a recommendation with source, destination, quantity, and an explicit approval requirement. The current VM may report deterministic fallback when its ADC token is stale; this is expected and does not mean the database path failed.

## Task 6: Troubleshoot and clean up

- A 404 usually means the card URL or runtime path is wrong.
- A completed response with fallback metadata indicates the action agent could not refresh VM ADC.
- A Select AI response should identify the database profile or database-backed source.
- Re-import an agent card after a deployment change; cards currently report version `0.0.1`.

## Conclusion

Gemini Enterprise can coordinate several Oracle AI Database agents, but each agent should have a narrow contract and a separately testable data source. Continue to Lab 5 to protect private A2A traffic and preserve the user identity at the database boundary.

## Acknowledgements

- Gemini Enterprise agent registration documentation
- Oracle AI Database Gemini Enterprise setup runbook
- [Oracle AI Database Agent for Gemini Enterprise demo](https://www.youtube.com/watch?v=lU8UAwmBMeQ)
