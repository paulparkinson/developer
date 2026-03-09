# Lab 2: Vector Search Foundations

## with Oracle AI Database 26ai and LangChain OracleVS

--------

### Objective

In this lab, you'll learn how to store and search documents using **semantic similarity** — finding results based on meaning rather than exact keyword matches. This is the foundation that powers every memory type we'll build in later labs.

You'll work with **SeerGroup Solutions' IT knowledge base**: internal documentation, runbooks, and incident reports that our AI support agent "Proteus" will use to resolve tickets.

#### What You'll Learn

| Step | Description |
|------|-------------|
| **1. Initialize Embeddings** | Load a HuggingFace embedding model to convert text into vectors |
| **2. Create Vector-Enabled Table** | Set up an Oracle-backed vector store with cosine distance via `OracleVS` |
| **3. Create Index** | Build an HNSW (Hierarchical Navigable Small World) index for fast similarity search |
| **4. Add Documents** | Store SeerGroup IT knowledge base articles with metadata |
| **5. Query** | Search for similar documents using natural language |
| **6. Filter Results** | Use metadata filters to narrow down search results |

#### Key Components

- **`OracleVS`**: LangChain abstraction over Oracle vector-enabled SQL tables
- **`HuggingFaceEmbeddings`**: Converts text to 768-dimensional vectors
- **`DistanceStrategy.COSINE`**: Measures vector similarity using cosine distance
- **HNSW Index**: Graph-based ANN index for fast and accurate nearest-neighbor retrieval

--------

## Step 1: Download the .ipynb and open in VS Code

The following notebook contains all the code for the remainder of the workshop. Download it and open with VS Code.

[Download the notebook](https://raw.githubusercontent.com/enschilling/workshop-dev/refs/heads/main/oci-ai-database-memory-context-engineering/files/workshop-complete.ipynb)


## Step 2: Connect to Oracle AI Database and Initialize Embeddings

Your environment has been pre-configured with Oracle AI Database 26ai running on OCI. In the first lab, you established a connection to the database and created the `VECTOR` uesr. We'll leverage connection details from the previous lab to keep building.

1. Run the following code block to create a `vector_conn` function that will be used during the workshop.

    ```python
    import oracledb
    import time
    import logging

    %store -r adb_dsn vector_user vector_password

    def connect_to_oracle(
        max_retries=3,
        retry_delay=5,
        user=vector_user,
        password=vector_password,
        dsn=adb_dsn,
        program="seergroup.proteus.workshop",
    ):
        """
        Connect to Oracle database with retry logic and helpful error messages.
        """
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Connection attempt {attempt}/{max_retries}...")
                conn = oracledb.connect(
                    user=user, password=password, dsn=dsn, program=program
                )
                print("✓ Connected successfully!")

                with conn.cursor() as cur:
                    cur.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'")
                    banner = cur.fetchone()[0]
                    print(f"\n{banner}")

                return conn

            except oracledb.OperationalError as e:
                error_msg = str(e)
                print(f"✗ Connection failed (attempt {attempt}/{max_retries})")

                if "DPY-4011" in error_msg or "Connection reset by peer" in error_msg:
                    print("  → Database may still be starting. Retrying...")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    else:
                        raise
                else:
                    raise

        raise ConnectionError("Failed to connect after all retries")

    vector_conn = connect_to_oracle()
    print("Using user:", vector_conn.username)
    ```

2. Initialize Embeddings and Create a Vector-Enabled Table

We'll use the `sentence-transformers/paraphrase-mpnet-base-v2` model to convert text into 768-dimensional vectors. OracleVS handles the table creation, embedding storage, and similarity search under the hood.

    ```python
    from langchain_oracledb.vectorstores import OracleVS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_oracledb.vectorstores.oraclevs import create_index
    from langchain_community.vectorstores.utils import DistanceStrategy

    # Suppress verbose logging from langchain_oracledb
    logging.getLogger("langchain_oracledb").setLevel(logging.CRITICAL)

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-mpnet-base-v2"
    )

    # Create the vector-enabled SQL table via OracleVS
    vector_store = OracleVS(
        client=vector_conn,
        embedding_function=embedding_model,
        table_name="VECTOR_SEARCH_DEMO",
        distance_strategy=DistanceStrategy.COSINE,
    )
    ```

3. Create an HNSW Index

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbor index. It provides fast, accurate similarity search at scale — essential when your knowledge base grows to thousands or millions of documents.

    ```python
    def safe_create_index(conn, vs, idx_name):
        """Create index, skipping if it already exists."""
        try:
            create_index(
                client=conn,
                vector_store=vs,
                params={"idx_name": idx_name, "idx_type": "HNSW"},
            )
            print(f"  ✅ Created index: {idx_name}")
        except Exception as e:
            if "ORA-00955" in str(e):
                print(f"  ⏭️ Index already exists: {idx_name} (skipped)")
            else:
                raise


    safe_create_index(vector_conn, vector_store, "oravs_hnsw")
    ```

## Step 2: Ingest SeerGroup IT Knowledge Base

In a real deployment, this data would come from Confluence, ServiceNow, or internal wikis. Here we'll seed the knowledge base with representative IT support articles that Proteus will use to resolve tickets.

    ```python
    # SeerGroup IT Knowledge Base — internal docs, runbooks, and incident reports
    seergroup_kb_articles = [
        {
            "title": "VPN Connectivity Troubleshooting Guide",
            "content": "When users report VPN connectivity issues, first verify they are running "
                    "GlobalProtect v6.2 or later. Check if the user's machine has a valid "
                    "certificate by running 'certutil -viewstore My' on Windows. Common causes "
                    "include expired certificates (renew via InternalCA portal), split-tunnel "
                    "misconfiguration (verify route table with 'route print'), and DNS resolution "
                    "failures (test with 'nslookup auth-svc.seergroup.internal'). Escalate to "
                    "Network Ops if the issue persists beyond basic troubleshooting.",
            "category": "networking",
            "severity": "medium",
            "team": "Network Ops",
        },
        {
            "title": "AUTH-SVC Service Recovery Runbook",
            "content": "AUTH-SVC is SeerGroup's central authentication service running on "
                    "prod-cluster-3 (Kubernetes). If AUTH-SVC is unresponsive: 1) Check pod "
                    "status: 'kubectl get pods -n auth -l app=auth-svc'. 2) Review logs: "
                    "'kubectl logs -n auth -l app=auth-svc --tail=200'. 3) If OOMKilled, "
                    "increase memory limit in helm values and redeploy. 4) If CrashLoopBackOff, "
                    "check for config map changes in the last 24h. 5) As last resort, rollback: "
                    "'helm rollback auth-svc -n auth'. Owner: Platform Team. SLA: P1 — 15min response.",
            "category": "infrastructure",
            "severity": "critical",
            "team": "Platform Team",
        },
        {
            "title": "New Employee Laptop Provisioning",
            "content": "Standard provisioning for new SeerGroup employees: 1) Image laptop with "
                    "SeerGroup-Win11-Enterprise or SeerGroup-macOS-14 base image. 2) Enroll in "
                    "Intune MDM. 3) Install standard software bundle: GlobalProtect VPN, "
                    "Microsoft 365, Slack, Zoom, CrowdStrike Falcon. 4) Create AD account and "
                    "add to department security group. 5) Provision Okta SSO with MFA (require "
                    "hardware key for engineering roles). 6) Send welcome email with setup guide. "
                    "Average provisioning time: 2 hours. Escalation: IT Service Desk Manager.",
            "category": "onboarding",
            "severity": "low",
            "team": "IT Service Desk",
        },
        {
            "title": "Database Connection Pool Exhaustion",
            "content": "If applications report 'connection pool exhausted' errors against Oracle "
                    "databases: 1) Check current sessions: 'SELECT COUNT(*) FROM v$session WHERE "
                    "username = <app_user>'. 2) Identify long-running queries: check v$sql for "
                    "queries exceeding 30s. 3) Check HikariCP pool settings — default max is 10, "
                    "recommended 20-30 for production services. 4) Look for connection leaks: "
                    "enable leak detection with 'leakDetectionThreshold: 60000' in application "
                    "config. 5) If immediate relief needed, kill idle sessions older than 30min. "
                    "Owner: DBA Team. Related: APP-CONFIG-2024-017.",
            "category": "database",
            "severity": "high",
            "team": "DBA Team",
        },
        {
            "title": "Slack Integration Bot Failures",
            "content": "SeerGroup's internal Slack bots (TicketBot, DeployBot, AlertBot) run on "
                    "the automation-cluster in the 'bots' namespace. Common failure modes: "
                    "1) Token expiration — Slack bot tokens expire every 12 hours; check if "
                    "token refresh cron job is running. 2) Rate limiting — Slack API limits to "
                    "1 message/second per channel; batch notifications. 3) Webhook URL changes "
                    "after Slack workspace migration — update in Vault at "
                    "'secret/bots/slack-webhooks'. Contact: Automation Team (Sarah Chen, lead).",
            "category": "integrations",
            "severity": "medium",
            "team": "Automation Team",
        },
        {
            "title": "CrowdStrike Falcon Sensor Troubleshooting",
            "content": "If CrowdStrike Falcon sensor shows as inactive or fails to check in: "
                    "1) Verify service is running: 'sc query csfalconservice' (Windows) or "
                    "'sudo systemctl status falcon-sensor' (Linux). 2) Check proxy settings — "
                    "sensor needs outbound access to ts01-b.cloudsink.net on port 443. "
                    "3) Verify Customer ID (CID) matches SeerGroup's tenant: check "
                    "'HKLM\\SYSTEM\\CrowdStrike\\{9b03c1d9-3138}\\CU'. 4) If reinstall needed, "
                    "use the latest installer from the Falcon console — never use cached "
                    "installers as they embed the old CID. Escalation: Security Ops.",
            "category": "security",
            "severity": "high",
            "team": "Security Ops",
        },
        {
            "title": "Kubernetes Pod Scheduling Failures",
            "content": "When pods are stuck in Pending state on SeerGroup clusters: 1) Check events: "
                    "'kubectl describe pod <pod> -n <ns>' — look for 'Insufficient cpu' or "
                    "'Insufficient memory'. 2) Review node capacity: 'kubectl top nodes'. "
                    "3) Check for taints blocking scheduling: 'kubectl describe node <node> | "
                    "grep Taint'. 4) Verify resource requests aren't over-provisioned — the "
                    "Platform Team recommends requests at 50% of limits for non-critical services. "
                    "5) If cluster is at capacity, request node scale-up via #platform-requests "
                    "Slack channel. prod-cluster-3 has auto-scaling enabled; staging does not.",
            "category": "infrastructure",
            "severity": "high",
            "team": "Platform Team",
        },
        {
            "title": "Email Delivery Delays and Bounces",
            "content": "SeerGroup uses Microsoft 365 for email with a custom transport rule for DLP. "
                    "If users report delivery delays: 1) Check message trace in Exchange Admin "
                    "Center (admin.microsoft.com). 2) Verify SPF/DKIM/DMARC records are current: "
                    "'nslookup -type=txt seergroup.com'. 3) Common cause: DLP policy scan adds "
                    "2-5min delay for messages with attachments over 10MB. 4) For bounces, check "
                    "if recipient domain is on the block list (maintained by Security Ops). "
                    "5) External relay issues — check Mimecast dashboard for queue depth. "
                    "Contact: Messaging Team (part of IT Service Desk).",
            "category": "email",
            "severity": "medium",
            "team": "IT Service Desk",
        },
        {
            "title": "Incident Report: AUTH-SVC Outage 2025-01-15",
            "content": "Duration: 47 minutes (08:12 - 08:59 UTC). Impact: All SSO logins failed "
                    "across SeerGroup. Root cause: A config map update pushed an invalid OIDC "
                    "issuer URL, causing AUTH-SVC pods to crash on startup (CrashLoopBackOff). "
                    "Detection: AlertBot triggered P1 alert at 08:14 via Slack #incidents. "
                    "Resolution: Platform Team rolled back the config map change, pods recovered "
                    "automatically. Prevention: Added config validation pre-hook to CI/CD pipeline. "
                    "Action items: (1) Add dry-run config validation. (2) Implement canary "
                    "deployment for auth services. (3) Review change approval process for "
                    "auth-critical configs. Owner: Marcus Rivera, Platform Team Lead.",
            "category": "incident_report",
            "severity": "critical",
            "team": "Platform Team",
        },
        {
            "title": "Printer and Peripheral Setup Guide",
            "content": "SeerGroup offices use HP LaserJet Enterprise printers managed via HP Web "
                    "Jetadmin. To add a printer: 1) Connect to 'seergroup-corp' WiFi or wired "
                    "LAN. 2) Open Settings > Printers > Add via IP. 3) Floor 1: 10.20.1.50, "
                    "Floor 2: 10.20.2.50, Floor 3: 10.20.3.50. 4) Use driver 'HP Universal "
                    "Print Driver' (pre-installed on SeerGroup images). For USB peripherals "
                    "(monitors, docks): Intune policy auto-installs drivers. If a dock isn't "
                    "recognized, run 'devmgmt.msc' and scan for hardware changes. For persistent "
                    "issues, check that USB-C port supports DisplayPort Alt Mode.",
            "category": "hardware",
            "severity": "low",
            "team": "IT Service Desk",
        },
    ]

    # Prepare texts and metadata for ingestion
    texts = []
    metadata_list = []

    for article in seergroup_kb_articles:
        text = f"Title: {article['title']}\nContent: {article['content']}"
        texts.append(text)
        metadata_list.append({
            "title": article["title"],
            "category": article["category"],
            "severity": article["severity"],
            "team": article["team"],
            "source_type": "internal_kb",
        })

    # Ingest into the vector-enabled table
    vector_store.add_texts(texts=texts, metadatas=metadata_list)
    print(f"✅ Ingested {len(texts)} SeerGroup KB articles into VECTOR_SEARCH_DEMO")
    ```

## Step 4: Querying with Natural Language

Now let's see the power of semantic search. Unlike keyword search, vector similarity finds documents based on *meaning*. A query about "login problems" will match articles about AUTH-SVC and SSO — even if those exact words don't appear in the query.

### Basic Similarity Search

    ```python
    query = "Users can't log in to any applications this morning"

    results = vector_store.similarity_search(query, k=3)

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print("Title:", doc.metadata.get("title", "N/A"))
        print("Team:", doc.metadata.get("team", "N/A"))
        print("Text:", doc.page_content[:200], "...")
        print()
    ```

> **🔍 Notice** how the search returns AUTH-SVC and authentication-related articles even though the query didn't mention "auth", "SSO", or "Kubernetes". That's semantic similarity at work.

### Search with Relevance Scores

Scores let Proteus gauge confidence. A score close to 0 means high similarity (cosine distance); a score near 1 means low relevance.

    ```python
    query = "The VPN keeps disconnecting on my laptop"

    results = vector_store.similarity_search_with_score(query, k=3)

    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Team:  {doc.metadata.get('team', 'N/A')}")
        print("------")
    ```

## Step 5: Filtered Search with Metadata

In a real IT support system, Proteus needs to narrow results by category, severity, or owning team. OracleVS supports metadata filters that combine with vector similarity.

### Filter by Category

    ```python
    query = "something is wrong with the kubernetes cluster"

    docs = vector_store.similarity_search(
        query,
        k=3,
        filter={"category": {"$eq": "infrastructure"}},
    )

    for doc in docs:
        print("Title:", doc.metadata.get("title", "N/A"))
        print("Severity:", doc.metadata.get("severity", "N/A"))
        print("Text:", doc.page_content[:150], "...")
        print("------")
    ```

### Filter by Severity (Critical Issues Only)

    ```python
    query = "What are the most important incidents I should know about?"

    docs = vector_store.similarity_search(
        query,
        k=5,
        filter={"severity": {"$eq": "critical"}},
    )

    for doc in docs:
        print("Title:", doc.metadata.get("title", "N/A"))
        print("Team:", doc.metadata.get("team", "N/A"))
        print("------")
    ```

### Filter by Team

    ```python
    query = "What does the Platform Team own?"

    docs = vector_store.similarity_search(
        query,
        k=5,
        filter={"team": {"$eq": "Platform Team"}},
    )

    for doc in docs:
        print("Title:", doc.metadata.get("title", "N/A"))
        print("Category:", doc.metadata.get("category", "N/A"))
        print("------")
    ```

## Lab 2 Recap

You've now built the search foundation that Proteus will rely on:

| What You Did | Why It Matters |
|-------------|----------------|
| Created a vector-enabled SQL table | Documents stored with embeddings for semantic retrieval |
| Built an HNSW index | Fast approximate nearest-neighbor search at scale |
| Ingested SeerGroup KB articles | Real IT support data with rich metadata |
| Queried with natural language | "Users can't log in" finds AUTH-SVC articles without keyword matching |
| Applied metadata filters | Narrow results by category, severity, or team |

**Next up**: In Lab 3, we'll design the complete memory architecture that gives Proteus six distinct types of memory — each with a specific purpose and storage strategy.

## Learn More



## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026