# Oracle AI Database + GCP agentic AI workshop

Start with the [A2UI and MCP Apps lab](a2ui-mcpapps/a2ui-mcpapps.md) for the
current inventory exploration/review split. It includes real Gemini Enterprise
screenshots, dynamic catalog/SKU prompts, provenance verification and the
Java gateway/server-side OAuth rationale.

The application lives in
[oracle-ai-database-gcp-gemini](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini),
not this documentation repository or `oracle-ai-for-sustainable-dev`.

- [Canonical managed-agent runbook](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini/blob/main/docs/MCP_APP_ORACLE_AGENT_SPATIAL.md):
  local/cloud steps, raw evidence checks, OAuth lifecycle and dated test results.
- [User-facing inventory skill](https://github.com/paulparkinson/oracle-ai-database-gcp-gemini/blob/main/.agents/skills/inventory-ui-architecture/SKILL.md):
  give this file and its references to ChatGPT/Claude; the lab includes a
  copy/paste prompt. Maintain this single canonical skill rather than a divergent
  workshop copy.

Current MCP actions are catalog and spatial reads via the managed Oracle AI
Database Agent, with no Toolkit/static fallback. These are live reads of
seeded demo data. The separate A2A/A2UI flow currently creates transfer drafts
and review controls; do not describe it as a verified committed inventory write.
