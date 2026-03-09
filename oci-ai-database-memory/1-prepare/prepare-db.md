# Lab 1: Prepare Development Environment

## Introduction

As part of the Sandbox environment, an Autonomous Database running Oracle AI Database 26ai has already been provisioned for you. You'll need Microsoft Visual Studio Code installed on your own laptop / PC to complete the workshop.

In this section you'll get your IDE prepared, connect to the Oracle Autonomous Database, then create a VECTOR user for use in the subsequent labs.

**Estimated Time:** 15 minutes

### Objectives

In this lab, you will:

1. Log into the provided OCI tenancy
2. Review and connect to the Autonomous Database that was provisioned for you
3. Create a dedicated user for vector operations

### Prerequisites

This lab assumes you have:

- Provisioned the Workshop using the LiveLabs Sandbox
- Retrieved your account credentials from the LiveLabs UI
- Installed Microsoft VS Code Community Edition

## Task 1: Log into the OCI Tenancy

... 
    DO WE HAVE: existing instructions we can pull to show how to 
    retrieve credentials in LiveLabs and log into the tenancy?
...

## Task 2: Locate and review the Oracle AI Database 26ai instance

1. Update ACL to include your IP (click the slide button) and save.

    SCREENSHOT

2. Update mTLS - not required

    SCREENSHOT

3. Click the **[Database connection]** button at the top of the screen. Locate and copy the full **Connection string** for the TNS name that corresponds with `medium`. (Paste in a note pad for future reference)

    SCREENSHOT

4. Time to set up VS Code

## Task 3: Open VS Code and Create Jupyter environment

For this lab, you will create a new `.ipynb` file and copy in the code to prep the database. in future labs, we've provided the `.ipynb` file to streamline these activities.

1. Open VS Code and close any existing tabs. 


2. Create a new Workspace: **File** -> **Open Folder**

3. Select an existing folder where you'd like to create your workspace, or create and select a new folder.

    SCREENSHOT

4. Once the folder is selected, click **[Select folder]**

5. Be sure to save all Jupyter notebooks (`.ipynb` files) into this folder for the duration of the workshop.

6. Press `Ctrl+Shift+P` and begin typing "Jupyter". It should auto-populate the option to `Create: New Jupyter Notebook`. Select this option.

7. Paste the following and press the Run arrow on the left side of the code block:

    ```
    <copy>
    pip install --upgrade pip
    </copy>
    ```

3. Press the **[+ Code]** button at the top of the notebookt add a new Code block. Paste the following and run the code block:

    ```
    <copy>
    pip install -qU oracledb langchain-oracledb langchain langchain-openai langchain-huggingface ipywidgets tavily-python sentence-transformers ipykernel jupyter numpy 
    </copy>
    ```

    >NOTE: This will perform a quiet install that takes 3-5 minutes. No output will be diplayed until it completes.

4. Test to make sure the Python modules were installed properly. Create a new code block, then paste and run the following:

    ```
    from openai import OpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    import oracledb
    ```

    >NOTE: No output is displayed unless there are errors. If you see a green check mark, all is well!


## Task 4: Connect to the database and create VECTOR user

1. Now test the connection using the OracleDB driver for Python. When you run this code block, VS Code will prompt you for three values. After entering these three values, the code block will finish executing.

    * Your autonomous database admin password
    * Your chosen VECTOR user password (recommended: MemoryContext_2026)
    * The DSN for your Autonomous DB _medium listener

    SCREENSHOT of VS Code prompt for input

    ```python
    <copy>

    import oracledb
    import os
    import getpass

    # Store username, password, and DSN
    adb_password = getpass.getpass("Database password: ")
    vector_password = getpass.getpass("Enter password for VECTOR (e.g. MemoryContext_2026): ")
    adb_dsn = os.environ.get("DB_DSN", input("Database DSN copied earlier: "))
    adb_user = "ADMIN"
    vector_user = "VECTOR"


    try:
        connection = oracledb.connect(
            user=adb_user,
            password=adb_password,
            dsn=adb_dsn
        )

        print("Connection Successful")

        with connection.cursor() as cursor:
            cursor.execute("SELECT banner FROM v$version WHERE banner LIKE 'Oracle%'")
            banner = cursor.fetchone()[0]
            print(f"\n{banner}")
            
    except oracledb.Error as error:
        print(f"Error connecting to the database: {error}")

    finally:
        if 'connection' in locals() and connection:
            connection.close()
            print("Connection Closed.")
    </copy>
    ```

    SCREENSHOT of successful output

3. Create the VECTOR user.

    ```python
    <copy>

    def setup_oracle_adb(
        create_vector_user=True,
        vector_user=vector_user,
        vector_password=vector_password 
    ) -> bool:
        """
        Connect to Oracle Autonomous Database and create VECTOR user
        
        Args:
            create_vector_user: Whether to create a VECTOR user
            vector_user: Username for the vector user
            vector_password: Password for the vector user
        
        Returns:
            bool: True if setup succeeded, False otherwise
        """
        print("=" * 60)
        print("🚀 ORACLE AUTONOMOUS DATABASE SETUP")
        print("=" * 60)
        
        # Step 1: Install/verify oracledb
        print("\n[1/3] Checking oracledb driver...")
        try:
            import oracledb
            print(f"   ✅ oracledb version {oracledb.__version__} installed")
        except ImportError:
            print("   ❌ oracledb not installed!")
            print("   💡 Install with: pip install oracledb")
            return False
        
        # Step 2: Test connection with wallet
        print("\n[2/3] Testing connection to Autonomous Database...")
        
        try:
            # Configure wallet location for thin mode
            connection = oracledb.connect(
                user=adb_user,
                password=adb_password,
                dsn=adb_dsn
            )
            
            with connection.cursor() as cursor:
                cursor.execute("SELECT banner FROM v$version WHERE ROWNUM = 1")
                version = cursor.fetchone()[0]
                print(f"   ✅ Connected successfully!")
                print(f"   📊 {version}")
            
        except oracledb.Error as e:
            error = e.args[0]
            print(f"   ❌ Connection failed: {error.message}")
            
            # Provide helpful hints based on error code
            if "ORA-01017" in str(error):
                print("   💡 Invalid username/password. Check your credentials.")
            elif "ORA-12154" in str(error):
                print("   💡 TNS could not resolve service name. Check tns_alias parameter.")
            elif "ORA-28759" in str(error):
                print("   💡 Failed to open wallet. Check wallet_password.")
            elif "ORA-28858" in str(error):
                print("   💡 Wallet file error. Re-download the wallet from OCI.")
            
            return False
        
        # Step 3: Create VECTOR user (optional)
        if create_vector_user:
            print(f"\n[3/3] Creating {vector_user} user...")
            
            try:
                with connection.cursor() as cursor:
                    # Check if user exists
                    cursor.execute(
                        "SELECT COUNT(*) FROM all_users WHERE username = :1",
                        [vector_user.upper()]
                    )
                    user_exists = cursor.fetchone()[0] > 0
                    
                    if user_exists:
                        print(f"   ✅ {vector_user} user already exists")
                    else:
                        # Create user with appropriate privileges
                        cursor.execute(f'CREATE USER {vector_user} IDENTIFIED BY "{vector_password}"')
                        cursor.execute(f"GRANT CONNECT, RESOURCE TO {vector_user}")
                        cursor.execute(f"GRANT UNLIMITED TABLESPACE TO {vector_user}")
                        cursor.execute(f"GRANT CREATE TABLE, CREATE SEQUENCE, CREATE VIEW TO {vector_user}")
                        
                        # For vector operations in ADB
                        try:
                            cursor.execute(f"GRANT EXECUTE ON DBMS_VECTOR TO {vector_user}")
                            cursor.execute(f"GRANT EXECUTE ON DBMS_VECTOR_CHAIN TO {vector_user}")
                        except oracledb.Error:
                            # These grants may not be available in all ADB versions
                            pass
                        
                        connection.commit()
                        print(f"   ✅ {vector_user} user created successfully")
                        
            except oracledb.Error as e:
                print(f"   ⚠️  Could not create user: {e.args[0].message}")
                print("   💡 You may need DBA privileges to create users.")
        else:
            print("\n[3/3] Skipping user creation (create_vector_user=False)")
        
        connection.close()
        
        # Success summary
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        print(f"""
    Connection Details:
        Service: {adb_dsn}
        Admin User: {adb_user}
    """)
        
        if create_vector_user:
            print(f"""    
    Vector User Credentials:
        User: {vector_user}
        Password: {vector_password}
    """)

        return True

    setup_oracle_adb()
    <copy>
    ```

    SCREENSHOT code output

You may now proceed to the next lab.

## Learn More



## Acknowledgements

- **Author** - Richmond Alake
- **Contributors** - Eli Schilling
- **Last Updated By/Date** - Published February, 2026