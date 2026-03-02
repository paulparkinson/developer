# Lab 1: Set Up Oracle AI Database Locally

## Introduction

In this lab, you will set up **Oracle Database Free** locally using Docker. Oracle Database Free is a converged database that combines relational, document, graph, and vector data in a single engine—making it ideal for AI applications that need semantic search, embeddings storage, and vector similarity queries.

This local setup gives you a fully functional Oracle database for development and testing without needing cloud infrastructure.

Estimated Time: 20 minutes

### Objectives

In this lab, you will:
* Install and configure Docker for Oracle Database
* Pull and run the Oracle Database Free container
* Configure Vector Memory for AI operations
* Create a dedicated user for vector operations
* Test the database connection

### Prerequisites

* Docker Desktop (Mac/Windows) or Docker Engine (Linux) installed
* At least 4GB of RAM available for Docker
* At least 10GB of free disk space
* Python 3.10+ with pip installed

## Task 1: Install Required Python Packages

First, let's install the necessary Python packages for working with Oracle AI Database and LangChain.

1. Open a terminal or command prompt.

2. Install the required packages:

    ```bash
    <copy>
    pip install -qU langchain-oracledb sentence-transformers langchain-openai langchain tavily-python huggingface_hub openai transformers torch
    </copy>
    ```

    This installs:
    - `langchain-oracledb`: Oracle integration for LangChain
    - `sentence-transformers`: For creating embeddings
    - `langchain`: Core LangChain library
    - `tavily-python`: Web search tool
    - `huggingface_hub`: Access to HuggingFace models
    - `openai`: For LLM API calls
    - `transformers` and `torch`: For running local models

## Task 2: Start the Oracle Database Free Container

Before running the setup code, ensure you have Docker installed and running.

1. **Verify Docker is running:**

    ```bash
    <copy>
    docker info
    </copy>
    ```

    If Docker is not running, start Docker Desktop (Mac/Windows) or Docker Engine (Linux).

2. **Pull the Oracle Free container image:**

    ```bash
    <copy>
    docker pull ghcr.io/gvenzl/oracle-free:23.26.0
    </copy>
    ```

    This downloads the Oracle Database Free 23c image which includes vector search capabilities.

3. **Run the Oracle container:**

    ```bash
    <copy>
    docker run -d \
      --name oracle-free \
      -p 1521:1521 -p 5500:5500 \
      --shm-size=1g \
      -e ORACLE_PASSWORD=OraclePwd_2025 \
      ghcr.io/gvenzl/oracle-free:23.26.0
    </copy>
    ```

    This command:
    - Creates a container named `oracle-free`
    - Exposes port 1521 (database) and 5500 (EM Express)
    - Sets SYS/SYSTEM password to `OraclePwd_2025`
    - Allocates 1GB shared memory

4. **Check container logs (optional):**

    ```bash
    <copy>
    docker logs -f oracle-free
    </copy>
    ```

    Wait for the message "DATABASE IS READY TO USE!" (this may take 2-3 minutes on first start).

    Press `Ctrl+C` to stop following the logs.

## Task 3: Automated Database Setup

Now we'll use Python to automatically configure the database for AI operations. This script handles everything: checking Docker status, waiting for the database to be ready, configuring vector memory, creating users, and testing connections.

1. **Create a new Python file** named `setup_database.py` with the following content:

    ```python
    <copy>
    import subprocess
    import time
    import sys
    import oracledb

    IMAGE = "ghcr.io/gvenzl/oracle-free:23.26.0"
    CONTAINER = "oracle-free"
    PDB = "FREEPDB1"
    PORT = 1521
    ORACLE_HOST = "127.0.0.1"
    ORACLE_DSN = f"{ORACLE_HOST}:{PORT}/{PDB}"

    def sh(*cmd, check=True, capture=True, text=True):
        return subprocess.run(cmd, check=check, capture_output=capture, text=text)

    def ensure_running(container=CONTAINER, image=IMAGE, oracle_pwd=None):
        """Ensure the Oracle Free container is running, creating it if needed."""
        # Docker reachable?
        r = sh("docker", "info", check=False)
        if r.returncode != 0:
            raise RuntimeError("Docker is not running. Please start Docker and try again.")

        # Container exists?
        r = sh("docker", "ps", "-a", "--filter", f"name=^{container}$", "--format", "{{.Names}}", check=False)
        if r.stdout.strip() == container:
            r2 = sh("docker", "inspect", "-f", "{{.State.Running}}", container, check=False)
            if r2.stdout.strip().lower() != "true":
                print(f"Starting existing container '{container}'...")
                sh("docker", "start", container)
            else:
                print(f"Container '{container}' is already running.")
            return

        # Otherwise create it
        if not oracle_pwd:
            raise ValueError("Set oracle_pwd (SYS/SYSTEM password) to create a fresh container.")
        print(f"Creating new container '{container}' from {image}...")
        r = sh(
            "docker", "run", "-d",
            "--name", container,
            "-p", f"{PORT}:1521",
            "-p", "5500:5500",
            "--shm-size=1g",
            "-e", f"ORACLE_PASSWORD={oracle_pwd}",
            image,
            check=False
        )
        if r.returncode != 0:
            err = (r.stderr or "").strip()
            if "port is already allocated" in err or "address already in use" in err:
                raise RuntimeError(
                    f"Port {PORT} is already in use. Another container may be running on that port.\\n"
                    f"Try: docker ps  (to see what is using the port)\\n"
                    f"     docker stop <name> && docker rm <name>  (to free the port)"
                )
            elif "is already in use" in err:
                raise RuntimeError(
                    f"Container name '{container}' is already in use by a stopped container.\\n"
                    f"Try: docker rm {container}  (then re-run this script)"
                )
            else:
                raise RuntimeError(f"docker run failed (exit {r.returncode}):\\n{err}")
        print(f"Container '{container}' created.")

    def wait_ready(container=CONTAINER, pdb=PDB, timeout_s=600):
        """Deterministic readiness check: instance OPEN + PDB READ WRITE."""
        print("Waiting for database to be ready (this may take a few minutes on first start)...")
        deadline = time.time() + timeout_s
        cmd = (
            "export ORACLE_SID=FREE; "
            "sqlplus -s / as sysdba <<'SQL'\\n"
            "set heading off feedback off pages 0 verify off echo off\\n"
            "select status from v$instance;\\n"
            f"select open_mode from v$pdbs where name='{pdb}';\\n"
            "exit\\n"
            "SQL"
        )
        while time.time() < deadline:
            r = sh("docker", "exec", container, "bash", "-lc", cmd, check=False)
            out = (r.stdout or "").upper()
            if "OPEN" in out and "READ WRITE" in out:
                print("✅ Database is ready!")
                return
            time.sleep(3)
        raise TimeoutError(f"DB not ready after {timeout_s}s. Try: docker logs -f {container}")

    def get_vector_memory_size(container=CONTAINER) -> int:
        """Return vector_memory_size in bytes (0 if unset)."""
        cmd = (
            "export ORACLE_SID=FREE; "
            "sqlplus -s / as sysdba <<'SQL'\\n"
            "set heading off feedback off pages 0 verify off echo off\\n"
            "select value from v$parameter where name='vector_memory_size';\\n"
            "exit\\n"
            "SQL"
        )
        r = sh("docker", "exec", container, "bash", "-lc", cmd, check=False)
        raw = (r.stdout or "").strip()
        tokens = [t for t in raw.split() if t.isdigit()]
        return int(tokens[-1]) if tokens else 0

    def configure_vector_memory(container=CONTAINER, size="1G", timeout_s=600):
        """
        Ensure Vector Pool is enabled for vector index operations by setting VECTOR_MEMORY_SIZE
        (SPFILE) and restarting the DB if needed.
        """
        current = get_vector_memory_size(container)
        if current and current > 0:
            print(f"Vector memory already configured: {current} bytes")
            return False  # no restart needed

        print(f"Configuring vector memory to {size}...")
        sql = f"""
    alter system set vector_memory_size={size} scope=spfile sid='*';
    shutdown immediate;
    startup;
    """
        sh("docker", "exec", container, "bash", "-lc", f"export ORACLE_SID=FREE; sqlplus -s / as sysdba <<'SQL'\\n{sql}\\nSQL")
        wait_ready(container=container, pdb=PDB, timeout_s=timeout_s)
        return True  # restart happened

    def create_vector_user(container=CONTAINER, vector_pwd="VectorPwd_2025"):
        """Create the VECTOR user with necessary privileges."""
        print("Creating VECTOR user...")
        sql = f"""
    ALTER SESSION SET CONTAINER = {PDB};
    BEGIN
      EXECUTE IMMEDIATE 'CREATE USER VECTOR IDENTIFIED BY "{vector_pwd}"';
    EXCEPTION
      WHEN OTHERS THEN
        IF SQLCODE != -01920 AND SQLCODE != -01921 AND SQLCODE != -01918 AND SQLCODE != -00955 THEN
          RAISE;
        END IF;
    END;
    /
    GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW TO VECTOR;
    GRANT UNLIMITED TABLESPACE TO VECTOR;
    """
        sh("docker", "exec", container, "bash", "-lc",
           f'export ORACLE_SID=FREE; echo "{sql}" | sqlplus -s / as sysdba')

    def test_connection(vector_pwd="VectorPwd_2025"):
        """Test connection as VECTOR user."""
        print("Testing connection...")
        conn = oracledb.connect(user="VECTOR", password=vector_pwd, dsn=f"127.0.0.1:{PORT}/{PDB}")
        with conn.cursor() as cur:
            cur.execute("select sys_context('userenv','con_name') from dual")
            con_name = cur.fetchone()[0]
            cur.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'")
            banner = cur.fetchone()[0]
            print(f"\\n{banner}")
        conn.close()
        return con_name

    def setup(oracle_pwd=None, vector_pwd="VectorPwd_2025", vector_memory_size="1G"):
        """Main setup function."""
        print("🚀 Starting Oracle AI Database setup...\\n")
        
        ensure_running(oracle_pwd=oracle_pwd)
        wait_ready()

        restarted = configure_vector_memory(size=vector_memory_size)
        if restarted:
            print(f"🧠 Enabled Vector Pool (VECTOR_MEMORY_SIZE={vector_memory_size}) and restarted DB")

        create_vector_user(vector_pwd=vector_pwd)
        con_name = test_connection(vector_pwd=vector_pwd)
        
        print(f"\\n✅ Ready! Connected as VECTOR to container: {con_name}")
        print(f"DSN: 127.0.0.1:{PORT}/{PDB}")
        print("\\n🎉 Setup complete! You can now proceed to the next lab.")

    if __name__ == "__main__":
        setup(oracle_pwd="OraclePwd_2025")
    </copy>
    ```

2. **Run the setup script:**

    ```bash
    <copy>
    python setup_database.py
    </copy>
    ```

    The script will:
    - ✅ Check if Docker is running
    - ✅ Check if Oracle container exists and is healthy
    - ✅ Wait for database to be ready (with progress indicator)
    - ✅ Configure vector memory (1GB) for vector operations
    - ✅ Create the VECTOR user with proper privileges
    - ✅ Test the connection

3. **Expected output:**

    ```
    🚀 Starting Oracle AI Database setup...

    Container 'oracle-free' is already running.
    Waiting for database to be ready (this may take a few minutes on first start)...
    ✅ Database is ready!
    Configuring vector memory to 1G...
    🧠 Enabled Vector Pool (VECTOR_MEMORY_SIZE=1G) and restarted DB
    Creating VECTOR user...
    Testing connection...

    Oracle Database 23ai Free Release 23.0.0.0.0 - Develop, Learn, and Run for Free

    ✅ Ready! Connected as VECTOR to container: FREEPDB1
    DSN: 127.0.0.1:1521/FREEPDB1

    🎉 Setup complete! You can now proceed to the next lab.
    ```

## Task 4: Verify the Setup

Let's verify that everything is working correctly.

1. **Create a test connection script** named `test_connection.py`:

    ```python
    <copy>
    import oracledb

    ORACLE_HOST = "127.0.0.1"
    ORACLE_DSN = f"{ORACLE_HOST}:1521/FREEPDB1"

    def test_vector_user():
        """Test connection as VECTOR user and verify capabilities."""
        conn = oracledb.connect(
            user="VECTOR",
            password="VectorPwd_2025",
            dsn=ORACLE_DSN
        )
        
        with conn.cursor() as cur:
            # Check version
            cur.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'")
            print(f"Connected to: {cur.fetchone()[0]}")
            
            # Check vector memory
            cur.execute("SELECT value FROM v$parameter WHERE name='vector_memory_size'")
            vm_size = cur.fetchone()[0]
            print(f"Vector Memory Size: {vm_size}")
            
            # Check current user
            cur.execute("SELECT user FROM dual")
            current_user = cur.fetchone()[0]
            print(f"Current User: {current_user}")
            
            # Check tablespace quota
            cur.execute("SELECT tablespace_name, max_bytes FROM user_ts_quotas")
            quotas = cur.fetchall()
            if quotas:
                for ts, quota in quotas:
                    print(f"Tablespace {ts}: {'UNLIMITED' if quota == -1 else f'{quota} bytes'}")
            else:
                print("Tablespace: UNLIMITED (no explicit quotas)")
        
        conn.close()
        print("\\n✅ All checks passed!")

    if __name__ == "__main__":
        test_vector_user()
    </copy>
    ```

2. **Run the test:**

    ```bash
    <copy>
    python test_connection.py
    </copy>
    ```

## Troubleshooting

### Docker Container Issues

If you see an error about the container name being in use:

```bash
<copy>
docker rm oracle-free
</copy>
```

Then re-run the setup script.

### Port Conflicts

If port 1521 is already in use:

1. Check what's using the port:
    ```bash
    <copy>
    docker ps
    </copy>
    ```

2. Stop the conflicting container:
    ```bash
    <copy>
    docker stop <container-name>
    </copy>
    ```

### Connection Failures

If you cannot connect to the database:

1. Check if the container is running:
    ```bash
    <copy>
    docker ps
    </copy>
    ```

2. Check container logs:
    ```bash
    <copy>
    docker logs oracle-free
    </copy>
    ```

3. Wait a few more minutes—Oracle takes time to initialize on first start.

## Summary

In this lab, you successfully:
* ✅ Installed required Python packages
* ✅ Started Oracle Database Free in Docker
* ✅ Configured vector memory for AI operations
* ✅ Created a dedicated VECTOR user
* ✅ Verified the database connection

You now have a fully functional Oracle AI Database running locally, ready for building AI agents with memory and context capabilities.

You may now **proceed to the next lab**.

## Learn More

* [Oracle Database Free](https://www.oracle.com/database/free/)
* [Oracle AI Vector Search](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/)
* [Docker Installation](https://docs.docker.com/get-docker/)
* [Python oracledb Driver](https://python-oracledb.readthedocs.io/)

## Acknowledgements

* **Author** - Paul Parkinson, Oracle Database Developer Advocate
* **Last Updated By/Date** - March 2026
