# Develop A2UI and MCPApps (charts, spatial, graph, ...)

## Introduction

Build two complementary lanes in the inventory application. MCP Apps provide
interactive, primarily read-only exploration for spatial hotspots and
property-graph dependencies. A2UI provides the agent-driven decision lane for
inventory recommendations, transfer review, and—once the governed write path
is enabled—explicit approval and execution. These are separate host paths over
the same Oracle-backed domain; do not force one UI protocol to do both jobs.

The graph example below shows another useful result shape: a property-graph traversal for `SKU-500`, with its supplier-to-warehouse path, a weather alert, and database-derived metrics. Use it as a visual reference when extending the lab's agent experience beyond text and recommendation cards.

![Oracle property graph result for SKU-500, showing six nodes, five relationship edges, and a traversal summary.](images/oracle-graph-dependency-view.svg)

*Example graph view: the Oracle property graph is the source of the traversal; A2UI or an MCP App controls how a compatible host presents it.*

### Objectives

- Understand the relationship between A2A, MCP, A2UI, and MCP Apps.
- Render an Oracle recommendation with an allowlisted A2UI catalog.
- Extend the existing Oracle Supply-Chain MCP connector with separate `ui://` resources for spatial and graph exploration.
- Keep the transfer MCP App disabled; transfer review belongs to the A2UI lane.
- Understand the implementation boundary between the current draft flow and a
  real actor-bound, short-lived approval/write path.

### Prerequisites

- Completed Lab 5.
- The agent service and Oracle inventory recommendation view available.
- Node.js 20+ for the MCP App sample.
- Gemini Enterprise preview access for A2UI, or an MCP Apps-compatible host.

## Task 1: Trace the contracts

```text
Gemini Enterprise -- A2A --> inventory-action coordinator -- Oracle/A2A/MCP --> Oracle AI Database
    ^                                      |
    +------------ A2UI review ------------+

MCP-compatible host -- MCP --> graph/spatial tool -- ui:// resource --> sandboxed MCP App
```

Keep the responsibilities distinct:

- **A2A** carries tasks and messages between Gemini Enterprise and a remote agent. In the supplied implementation, an A2A adapter calls the shared service and returns the result to Gemini Enterprise.
- **MCP** discovers and invokes tools or reads resources. The Oracle Database MCP Java Toolkit exposes named database operations; it is not the A2A transport or the UI.
- **A2UI** sends a declarative component tree and data model for the host to validate and render with its own approved native components. It is not generated HTML or JavaScript.
- **MCP Apps** associate an MCP tool with a developer-built `ui://` resource. A compatible host loads that UI in a sandbox and mediates its tool calls through a host bridge.

The supplied code-deep-dive demonstrates separate host adapters over the same governed backend. Gemini Enterprise receives A2UI v0.8 DataParts over A2A v0.3; the MCP Apps host loads developer-built `ui://` resources. These are distinct host paths: negotiate the version, component catalog, and MCP Apps bridge with each host instead of assuming the payloads are interchangeable.

## Task 2: Render graph, chart, and spatial results

1. Open the spatial MCP App and select `SKU-500`. Verify warehouse hotspots and the schematic source/destination connection from the managed Oracle agent. This line is not road routing or an approved transfer.
2. Open the graph MCP App and traverse the supplier → plant → port → warehouse path for `SKU-500`. Verify that the graph data comes from the governed graph/relational service.
3. Ask the A2A inventory-action agent: `Recommend an inventory action for SKU-500. Gather graph, spatial, and external evidence first, then render the transfer review as A2UI.`
4. Verify that the agent returns the proposed source, destination, quantity, policy result, and A2UI review controls.
5. For A2UI, map structured agent results into the host's advertised catalog and supported component types. Use host-rendered native controls; do not put arbitrary markup, scripts, or model-generated component definitions into the surface.
6. Keep visualization data access behind bounded server contracts. An MCP App is presentation code and must not choose a different product, route, quantity, or database operation than the service returned.

The visualization is a presentation of Oracle results, not an authority boundary. The browser or MCP App must not choose a different product, source, target, or transfer quantity than the governed service returned.

### What the local toolkit dashboard should show

The full-stack toolkit includes a runnable local dashboard that makes the
surface split visible before a Gemini Enterprise registration. Start it from
the toolkit repository:

```bash
cd "$HOME/oracle-ai-database-fullstack-toolkit"
mvn test
mvn -pl runtime -am spring-boot:run
```

Open `http://localhost:8080`. Confirm these entries:

1. `inventory-spatial-mcpapp` has MCP and MCP APP enabled.
2. `inventory-graph-mcpapp` has MCP and MCP APP enabled.
3. `inventory-transfer-a2ui` has A2A and A2UI enabled, while MCP and MCP APP
   are disabled.
4. `approve-inventory-transfer`, `reserve-inventory-transfer-id`, and
   `count-inventory-transfers` are visible as MCP definitions imported from the
   toolkit catalog. Expose the write tool only through authenticated approval.

![Toolkit dashboard showing the governed approval MCP definition and its PL/SQL contract.](images/toolkit-dashboard-approve-mcp.png)

![Toolkit dashboard showing the spatial MCP App projection.](images/toolkit-spatial-mcpapp.png)

![Toolkit dashboard showing the graph MCP App projection.](images/toolkit-graph-mcpapp.png)

![Toolkit dashboard showing the A2UI transfer projection with MCP App disabled.](images/toolkit-transfer-a2ui-output.png)

The dashboard emits descriptors and example A2UI messages. It does not replace
the production MCP server, MCP Apps-compatible host, Gemini Enterprise A2A
registration, or database approval procedure.

## Task 3: Run the Oracle Supply-Chain MCP App from the correct repository

The maintained implementation is in
`oracle-ai-database-gcp-gemini/mcp-app`. In read-only mode it registers
`list-inventory-items` and `show-inventory-spatial-hotspots` on
the same **Oracle Supply Chain MCP App** connector. It is not a second
connector.

```bash
cd "$HOME/src/github.com/paulparkinson/oracle-ai-database-gcp-gemini/mcp-app"
npm ci --ignore-scripts
npm run build
```

For the GCP deployment, from the repository root run:

```bash
./deploy/gcp/deploy-oracle-agent-and-mcp-app.sh
```

Click **Reload custom actions** on the existing **Oracle Supply Chain MCP App**
connector, enable the catalog and spatial actions, and start a new conversation.
The Toolkit transfer-dashboard action is no longer offered in read-only mode.
The spatial path is deliberately:

```text
MCP tool(sku)
  -> Java gateway /api/inventory/spatial-hotspots
  -> private Oracle A2A relay
  -> managed Oracle AI Database Agent
  -> validated GeoJSON
  -> MapLibre MCP App
```

The tool does not accept model-passed hotspot evidence and does not fall back
to the MCP Toolkit, static demo rows, or Select AI for spatial reads. Ask:

```text
Show the spatial hotspot map for SKU-500.
```

The gateway requests database rows containing PRODUCT_ID and WAREHOUSE_ID and
filters by product before constructing GeoJSON. It rejects conflicting rows,
invalid coordinates and hotspot scores outside 0–1. Source, destination and
relay roles stay distinct. Connections are schematic, not road routes.
Approval belongs to the separate A2UI/Toolkit workflow.

Ask `Use List-inventory-items to list the managed Oracle inventory catalog and its scope.`
The audited SC_PRODUCTS catalog contains SKU-500, SKU-700, SKU-900,
SKU-APAC-210 and SKU-APAC-420. Try `Show the spatial hotspot map for SKU-700.`
Do not use the Toolkit SUPPLY_PRODUCTS transfer recommendations as this catalog.

Test `Show the spatial hotspot map for SKU-501.` A NO_DATA result must say
risk is **unknown**, not safe/stable, and must not promise monitoring. An error
must not become an invented diagnosis or trigger a Toolkit fallback.
HOTSPOT_SCORE is a score, not a stockout probability.

Inspect the returned scope, per-row SKU/warehouse IDs and A2A task ID.
These establish the server's managed-agent call path; they are not a signed
database audit record. For independent SQL execution proof, correlate the task
with Oracle query/audit records rather than trusting Gemini's narration.
Run the implementation's `mcp-app/test/live-evidence.mjs` against the deployed
MCP endpoint for read-only catalog, multi-SKU and no-data contract checks.

## Task 4: Keep approval and database authority server-side

1. Request a recommendation and record its draft ID.
2. Review it in A2UI without editing product, source, destination, or quantity.
3. In the current repository state, confirm that the result remains a draft and no database write is called.
4. If you implement the write extension, bind approval to the authenticated actor and a short-lived, single-use handle bound to the exact recommendation.
5. Test replay, expiry, actor mismatch, changed route/quantity, insufficient stock, and rollback; each must fail without a partial write.

The MCP App is untrusted presentation code: its iframe must not receive database credentials, wallet files, or authority to select a transfer. For a future write-enabled A2UI flow, the service must validate the approval and Oracle must perform final row locks, current-stock revalidation, audit insert, and transfer update in one transaction. This keeps the database, not the model or UI, as the final execution authority.

## Task 5: Test failure and security cases

- Send malformed or unsupported A2UI component data; the host-side allowlist must reject it.
- Attempt an unapproved image or network origin in the MCP App; its content security policy must block it.
- Confirm the MCP App cannot access the host DOM, cookies, or local storage, and that host communication uses the MCP Apps bridge.
- Remove the actor identity; approval must fail.
- Expire the approval token; approval must fail.
- Attempt a model-visible write; no write-capable model tool should exist. The current draft implementation must not claim execution.
- Inspect logs for bearer tokens, passwords, or wallet paths; none may appear.

Keep the host's A2UI catalog allowlisted and the MCP App's content security policy narrowly scoped. Sandboxing protects the host boundary; it does not replace server authentication, input validation, database authorization, or transaction checks.

## Conclusion

A2UI and MCP Apps make the workflow usable without moving authority into the model or browser. Continue to Lab 7 to compare MCP server and application options for Oracle AI Database.

For reusable implementation guidance, see the
[`inventory-ui-architecture` agent skill](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini/tree/main/.agents/skills/inventory-ui-architecture)
and the maintained
[`INVENTORY_UI_ARCHITECTURE.md`](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini/blob/main/docs/INVENTORY_UI_ARCHITECTURE.md).
These instructions can be supplied to ChatGPT, Claude, or another coding
agent when extending the workshop application.

## Acknowledgements

*All Done! You may proceed to the next lab.*

- **Authors/Contributors** - Paul Parkinson, Architect and Dev Advocate, Oracle AI Database
- **Last Updated By/Date** - Paul Parkinson, October 2026
