# Understand and Develop with various MCP options

## Introduction

Choose an MCP implementation based on the tools an agent needs, where those tools run, and how much control the application requires. This lab compares Oracle database-focused MCP options and shows how to expose narrow, governed business operations without confusing tool access with agent communication or user-interface rendering.

### Objectives

- Distinguish MCP tools, A2A agent communication, A2UI, and MCP Apps.
- Compare SQLcl MCP, Oracle Database MCP Java Toolkit, and Google MCP Toolbox.
- Define an allowlisted MCP toolset for Oracle AI Database.
- Select the right MCP and UI approach for a Gemini Enterprise workflow.

### Prerequisites

- Completed Labs 4 through 6.
- Oracle AI Database access and the sample data from the earlier labs.
- Java 21 or later for the Java MCP Toolkit example; use the versions specified by its repository.
- Gemini Enterprise access or another compatible MCP host for optional host testing.

## Task 1: Separate the protocols

```text
Gemini Enterprise -- A2A --> agent
agent -- MCP --> approved tools --> Oracle AI Database
agent -- A2UI --> host-rendered native controls
MCP host -- MCP Apps --> sandboxed ui:// resource
```

- **A2A** carries tasks and messages between an agent host and a remote agent. Lab 5 uses this path.
- **MCP** lets a client discover and call tools or read resources from a server.
- **A2UI** is a declarative UI contract that a compatible host validates and renders with its own components.
- **MCP Apps** pair an MCP tool with a developer-built UI resource that a compatible host loads in a sandbox. Lab 6 demonstrates this UI path.

A2A, MCP, A2UI, and MCP Apps solve different problems and can be combined. For example, Gemini Enterprise can call an agent over A2A, while that agent uses MCP to call a separate Oracle tool server.

## Task 2: Compare MCP options

| Option | Best fit | Design consideration |
| --- | --- | --- |
| Oracle SQLcl MCP | SQL and database workflows using SQLcl's MCP server | Keep database credentials and privileges scoped to the local or hosted client; do not expose unrestricted SQL to an agent by default. |
| Oracle Database MCP Java Toolkit | Java services that need named, reusable Oracle operations | Define explicit toolsets and parameterized operations; expose only the operations required by the application. |
| Google MCP Toolbox for Databases | Google ADK agents and declarative database tools | Configure data sources and tools separately, then review each configured query and its access scope. |
| MCP Apps | Interactive UI inside an MCP-compatible host | This is a UI extension, not a database connector; the MCP server still authenticates and authorizes every tool call. |

The built-in Oracle AI Database Agent and its A2A endpoint are another option when the database agent team should own the database-side reasoning. They are not interchangeable with an MCP server.

## Task 3: Design a bounded Oracle toolset

Review the [Oracle Database MCP Java Toolkit sample](https://github.com/oracle-devrel/oracle-ai-for-sustainable-dev/tree/main/a2ui_mcpapps_mcptoolkit/oracle-db-mcp-toolkit) and the [A2UI and MCP Apps walkthrough](https://paul-parkinson.medium.com/develop-a2ui-and-mcp-apps-with-oracle-ai-database-and-the-java-mcp-toolkit-running-in-google-gemini-b495abf9b949).

For the inventory workflow, use named operations such as:

- `find-stockout-transfer-recommendations` to return a bounded, read-only result set.
- `get-stockout-transfer-details` to retrieve one current recommendation.
- `approve-inventory-transfer` only behind an explicit, authenticated user action and database revalidation.

1. Inspect the sample Toolkit configuration and identify its datasource, tool definitions, parameters, and enabled toolset.
2. Confirm every query uses bind parameters and limits result size. Enable only the named tools the application needs; do not enable unrestricted query, administration, or arbitrary write tools.
3. Keep approval inputs bound to an authenticated actor and an exact recommendation. The service and Oracle AI Database must validate current state and enforce the transaction; a model-generated response or UI payload is not authorization.
4. Test tool discovery and a read-only call before enabling any write path. Verify authentication, timeouts, audit records, and failure behavior.

The sample uses the Java MCP Toolkit as a separately governed database-tool service. An agent can reuse the same narrow contracts without receiving database credentials or arbitrary SQL access.

## Task 4: Choose and test a UI path

Use Lab 6's A2UI path when the host should render validated native components from its own catalog. Use its MCP App path when a compatible host should load a developer-built `ui://` experience in a sandbox. Neither path changes which database operations the MCP server permits.

1. In Gemini Enterprise, invoke the A2A agent and verify its database-backed response.
2. If using the MCP App sample, register the MCP server with a compatible host and confirm the host discovers only the intended dashboard tool and UI resource.
3. Compare the host-rendered result with the MCP tool result. Confirm approval remains an explicit user action and that the database is the final authority for writes.

## Task 5: Record the implementation choice

For your application, record the selected MCP server, the enabled tool names, authentication method, database principal, and any UI extension. Confirm that the design:

- grants only the database privileges needed by enabled tools;
- validates and bounds every tool input and result;
- keeps credentials out of prompts, UI resources, and source control;
- applies Deep Data Security at the database boundary where required; and
- provides audit evidence for sensitive reads and writes.

## Conclusion

MCP standardizes tool access, but it does not make a server or tool safe by itself. A narrow tool contract, least-privileged database access, and database-side validation preserve clear authority boundaries across SQLcl, Java services, ADK, A2A agents, and MCP-compatible hosts.

## Acknowledgements

*All Done! You may proceed to the next lab.*

- **Authors/Contributors** - Paul Parkinson, Architect and Dev Advocate, Oracle AI Database
- **Last Updated By/Date** - Paul Parkinson, October 2026
