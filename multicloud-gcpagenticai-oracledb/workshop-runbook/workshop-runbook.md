# Complete workshop runbook

This page is the operational index for running the workshop from an empty Google
Cloud/Oracle Database@Google Cloud environment. The individual labs explain the
concepts; this page makes the order, repositories, commands, verification
points, screenshots, and cleanup explicit.

## 0. What you will build

~~~text
Google Cloud project
├── VPC + VM (SQLcl, Java agents, optional MCP host)
└── Oracle Database@Google Cloud Autonomous Database
    ├── FINANCIAL inventory/risk/graph/spatial data
    ├── Oracle AI Database Agent and A2A endpoint
    └── governed transfer recommendation and approval procedures

Gemini Enterprise
├── Marketplace Oracle AI Database Agent
├── custom A2A graph/spatial/action agents
└── optional MCP/MCP App integration
~~~

Use three separate checkouts:

~~~bash
export WORKSHOP_REPO="$HOME/multicloud-gcpagenticai-oracledb"
export APP_REPO="$HOME/oracle-ai-database-gcp-gemini"
export TOOLKIT_REPO="$HOME/oracle-ai-database-fullstack-toolkit"
~~~

Never commit wallets, passwords, OAuth secrets, API keys, Maven caches, or
generated build output.

## 1. Prerequisites and clone

You need Google Cloud billing and Oracle Database@Google Cloud access, an OCI
tenancy linked to the Marketplace offer, Gemini Enterprise permissions, and
git, gcloud, ssh, scp, Java 17+, Maven, Python 3, SQLcl, jq, and Node.js 20+
if using an MCP Apps host.

~~~bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT

git clone https://github.com/paulparkinson/multicloud-gcpagenticai-oracledb.git "$WORKSHOP_REPO"
git clone https://github.com/paulparkinson/oracle-ai-database-gcp-gemini.git "$APP_REPO"
git clone https://github.com/paulparkinson/oracle-ai-database-fullstack-toolkit.git "$TOOLKIT_REPO"
~~~

## 2. Lab order

| Order | Lab | Outcome | Required? |
| --- | --- | --- | --- |
| 1 | gcp-started | Link Google Cloud Marketplace and OCI | Yes |
| 2 | adb-provisioning-databases | Create private Autonomous Database and wallet | Yes |
| 3 | gcp-get-started | Create VM, clone source, seed Oracle data | Yes |
| 4 | gemini-cli | Validate SQLcl MCP locally | Optional |
| 5 | gemini-enterprise-agent | Register Marketplace Oracle AI Database Agent | Yes |
| 6 | a2a-agents | Expose private Oracle A2A through Cloud Run | Yes for private A2A |
| 7 | a2ui-mcpapps | Add graph/spatial MCP Apps and transfer A2UI | Yes for UI demo |
| 8 | mcp-options | Compare SQLcl, Java Toolkit, Toolbox, and MCP Apps | Optional |
| 9 | agent-memory | Add actor-bound, expiring memory | Optional |
| 10 | deep-data-security | Apply regional row filtering | Optional |
| 11 | lakehouse | Add governed analytical context | Optional |

## 3. Provision network, database, and wallet

1. Complete Marketplace/OCI onboarding in gcp-started.
2. Create app-network with public subnet 10.1.0.0/24.
3. Create ODBG network odbg-network in us-east4.
4. Create client subnet db-subnet with 10.2.0.0/24.
5. Create the Autonomous Database with private endpoint access only.
6. Download the wallet and copy it to the VM:

~~~bash
scp -i "$SSH_KEY" Wallet_*.zip "$VM_USER@$VM_HOST:$HOME/wallet/"
ssh -i "$SSH_KEY" "$VM_USER@$VM_HOST"
unzip -o "$HOME/wallet/Wallet_*.zip" -d "$HOME/wallet"
export TNS_ADMIN="$HOME/wallet"
~~~

Record the service alias from tnsnames.ora. Never commit wallet files.

## 4. Seed the database

On the private VM, verify the wallet connection:

~~~bash
cd "$APP_REPO"
export TNS_ADMIN="$HOME/wallet"
sql ADMIN@YOUR_SERVICE_ALIAS
~~~

As ADMIN, verify or create FINANCIAL, then run:

~~~sql
@sql/admin_prepare_paulparkdb_demo.sql
~~~

Reconnect as FINANCIAL and run in order:

~~~sql
@sql/setup_supply_chain_graph_schema.sql
@sql/seed_supply_chain_graph_data.sql
@sql/setup_inventory_risk_demo_schema.sql
@sql/seed_inventory_risk_demo_data.sql
~~~

Verify:

~~~sql
select object_name, object_type, status
from user_objects
where object_name like 'SC_%' or object_name = 'SUPPLY_CHAIN_GRAPH'
order by object_type, object_name;
~~~

The expected core products are SKU-500, SKU-700, and SKU-900.

## 5. Configure Oracle AI Database Agent

1. Create or verify a narrow Select AI profile over the SC_* objects.
2. Install and verify the Oracle AI Database Agent team using the scripts in
   $APP_REPO/sql and the pinned installer described in Lab 4.
3. Enable database A2A on the database resource.
4. Register OAuth with this exact redirect URI:

~~~text
https://vertexaisearch.cloud.google.com/oauth-redirect
~~~

5. Add the Marketplace Oracle AI Database Agent in Gemini Enterprise and test:

~~~text
Which products are at risk of stockouts next quarter? Include stockout probability, projected revenue impact, and primary region.
~~~

Expect database-backed values for the seeded products. If the response is
generic, fix database/A2A authentication before continuing.

## 6. Deploy and verify custom A2A agents

On the VM:

~~~bash
cd "$APP_REPO/oracle_agent_java"
./build.sh
~~~

Configure the ignored .env with database, wallet, public host, OAuth, and model
values. Start locally first, then enable HTTPS/systemd from
$APP_REPO/deploy/gcp/.

Verify every card before importing it:

~~~bash
for path in agent-card-graph.json agent-card-spatial.json agent-card-select-ai.json agent-card-inventory-system.json agent-card-action.json; do
  curl -fsS "https://YOUR_PUBLIC_AGENT_HOST/$path" | jq .name
done
~~~

Register the inventory-system, graph, spatial, and action cards. Test one
read-only graph request and one action recommendation. The action must remain a
draft until the governed write path is explicitly enabled.

## 7. Run the full-stack toolkit

~~~bash
cd "$TOOLKIT_REPO"
mvn test
mvn -pl runtime -am spring-boot:run
~~~

Open http://localhost:8080 and verify:

- inventory-spatial-mcpapp: MCP + MCP App enabled.
- inventory-graph-mcpapp: MCP + MCP App enabled.
- inventory-transfer-a2ui: A2A + A2UI enabled; MCP App disabled.
- approve-inventory-transfer: named PL/SQL operation imported from the MCP
  catalog, not unrestricted SQL.

Inspect projections:

~~~bash
curl -s http://localhost:8080/api/tools | jq
curl -s http://localhost:8080/api/tools/inventory-spatial-mcpapp/mcp-app | jq
curl -s http://localhost:8080/api/tools/inventory-graph-mcpapp/mcp-app | jq
curl -s http://localhost:8080/api/tools/inventory-transfer-a2ui/a2ui/example | jq
curl -s http://localhost:8080/api/tools/approve-inventory-transfer/mcp | jq
~~~

The toolkit supplies descriptors and A2UI examples. A compatible MCP Apps host
and production MCP server are still required to render a live ui:// app.

![Toolkit MCP approval projection](../a2ui-mcpapps/images/toolkit-dashboard-approve-mcp.png)

![Toolkit A2UI transfer projection](../a2ui-mcpapps/images/toolkit-transfer-a2ui-output.png)

## 8. Verify both UI paths

### Explore with MCP Apps

The live Oracle Supply-Chain MCP App connector is maintained in the existing
`oracle-ai-for-sustainable-dev/a2ui_mcpapps_mcptoolkit` project. It owns the
transfer dashboard and the spatial extension; do not create a second connector.
Enable these read-only actions in the existing connector:

- `show-inventory-transfer-dashboard`
- `show-inventory-spatial-hotspots`

Ask:

```text
Show the spatial hotspot map for SKU-500.
```

The spatial tool calls the Oracle Database MCP Java Toolkit operation
`get-inventory-spatial-hotspots`, returns GeoJSON, and renders it with MapLibre
GL JS. Confirm the map shows source and destination warehouses plus the relief
route. The MCP App keeps credentials server-side and cannot invoke the
approval procedure.

### Decide with A2UI

In Gemini Enterprise ask:

~~~text
Recommend an inventory action for SKU-500. Gather graph, spatial, and external evidence first, then render the transfer review as A2UI.
~~~

Confirm the A2UI surface shows evidence, source, destination, quantity, policy,
draft ID, and approval state. The current repository workflow stops at draft;
the database write requires authenticated approval bound to the exact governed
recommendation and the named approval procedure.

## 9. Optional labs

- Gemini CLI: start SQLcl with sql -mcp, add the local MCP server, list tools,
  run a read-only query, and verify writes are not exposed.
- MCP options: inspect the Java Toolkit catalog and compare SQLcl MCP with
  Google MCP Toolbox. Keep only named, bounded tools.
- Agent memory: bind records to actor, session, authorization scope, and expiry;
  query current Oracle data again instead of trusting memory.
- Deep Data Security: run the regional setup script, test NA/APAC users, and
  verify identical prompts return only authorized rows.
- Lakehouse: load approved signals, create a governed view, join bounded recent
  data, and keep operational inventory authoritative.

Every optional lab must end with a negative test: wrong actor, expired data,
malformed input, or unauthorized region must fail without leaking data or
creating a write.

## 10. Cleanup and troubleshooting

Remove Gemini Enterprise agents/OAuth resources, disable temporary Cloud Run
relays, stop or delete the VM if it is not shared, revoke temporary database
users and credentials, and remove only workshop-owned database/bucket objects.

- Private connection fails: check TNS_ADMIN, wallet alias, VM subnet, firewall,
  and that the client is inside the VPC.
- Card returns 404: check service path/port and local card discovery before
  checking Gemini Enterprise.
- MCP App does not render: confirm the host supports MCP Apps and the MCP server
  advertises the matching ui:// resource; the toolkit dashboard alone is not an
  MCP Apps host.
- Approval is rejected: verify current recommendation state, actor, single-use
  approval, and database procedure grants.

When this runbook and a lab differ, treat checked-in source scripts and current
agent cards as authoritative, then update both documents. Never add a command
that requires a credential, wallet, or generated artifact to be committed.
