import os
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake

# -----------------------------------------------------------------------------
# Database connection validation and table loading function
# -----------------------------------------------------------------------------
def validate_and_connect_database(credentials):
    # Extract credentials
    db_user = credentials["DB_USER"]
    db_password = credentials["DB_PASSWORD"]
    db_host = credentials["DB_HOST"]
    db_port = credentials["DB_PORT"]
    db_name = credentials["DB_NAME"]
    groq_api_key = credentials["GROQ_API_KEY"]

    # Encode password for special characters
    encoded_password = db_password.replace('@', '%40')

    # Create database engine
    connection_string = f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(connection_string)

    try:
        with engine.connect() as connection:
            # Initialize LLM with explicit temperature setting
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                api_key=groq_api_key,
                temperature=0.2  # Add low temperature for more deterministic results
            )

            # Inspect database
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables_views = tables + views

            sdf_list = []
            table_info = {}
            failed_tables = []

            for table in all_tables_views:
                try:
                    # Get sample data with limit
                    query = f'SELECT * FROM "public"."{table}" LIMIT 10'
                    df = pd.read_sql_query(query, engine)
                    
                    # Create SmartDataframe with simplified config to avoid 'prompt_id' error
                    try:
                        # Try the most recent PandasAI interface
                        sdf = SmartDataframe(df, name=f"public.{table}", config={"llm": llm})
                    except TypeError:
                        # Fallback for older versions that might need different format
                        sdf = SmartDataframe(df, name=f"public.{table}", llm=llm)
                    
                    sdf_list.append(sdf)
                    
                    # Save table metadata
                    table_info[table] = {
                        "columns": list(df.columns),
                        "row_count": len(df),
                        "sample_loaded": True
                    }
                except Exception as e:
                    st.error(f"Error loading table {table}: {str(e)}")
                    table_info[table] = {
                        "sample_loaded": False,
                        "error": str(e)
                    }
                    failed_tables.append(table)

            # Create SmartDatalake with simplified config
            try:
                # Try the most recent PandasAI interface
                datalake = SmartDatalake(sdf_list, config={"llm": llm})
            except TypeError:
                # Fallback for older versions that might need different format
                datalake = SmartDatalake(sdf_list, llm=llm)
            
            return datalake, table_info, engine, failed_tables
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None, []

# -----------------------------------------------------------------------------
# Process input query function
# -----------------------------------------------------------------------------
def process_query(datalake, prompt):
    with st.spinner("Processing query..."):
        try:
            # Get response from PandasAI
            result = datalake.chat(prompt)
            
            # Display results based on type
            st.subheader("Result:")
            if isinstance(result, pd.DataFrame):
                st.dataframe(result)
            elif isinstance(result, str) and (result.endswith('.png') or result.endswith('.jpg')) and os.path.exists(result):
                st.image(result)
            elif isinstance(result, str):
                st.markdown(result)
            else:
                st.write(result)
                
            return True
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            return False

# -----------------------------------------------------------------------------
# Main View with Direct Input and Display
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    
    # Initialize session state
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    
    # Sidebar for Database Credentials
    with st.sidebar:
        st.header("🔐 Database Credentials")
        
        # Database credential inputs
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        connect_button = st.button("Connect to Database")

        # Display loaded table information
        if st.session_state.get("database_loaded", False):
            st.header("📊 Table Information")
            
            # Add expander for successfully loaded tables
            st.subheader("Accessible Tables")
            loaded_tables = [table for table, info in st.session_state.table_info.items() 
                            if info.get('sample_loaded', False)]
            if loaded_tables:
                for table in loaded_tables:
                    with st.expander(table):
                        info = st.session_state.table_info[table]
                        st.write(f"Columns: {', '.join(info['columns'])}")
                        st.write(f"Row Count: {info['row_count']}")
            else:
                st.warning("No tables could be loaded.")
            
            # Add expander for tables with access issues (if any)
            if 'failed_tables' in st.session_state and st.session_state.failed_tables:
                st.subheader("Tables with Access Issues")
                for table in st.session_state.failed_tables:
                    with st.expander(f"❌ {table}"):
                        st.error(st.session_state.table_info[table].get('error', 'Unknown error'))
            else:
                st.success("All tables loaded successfully!")

    # Main content area
    st.title("🔍 Direct Database Explorer")

    # Process database connection
    if connect_button and all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
        credentials = {
            "DB_USER": db_user,
            "DB_PASSWORD": db_password,
            "DB_HOST": db_host,
            "DB_PORT": db_port,
            "DB_NAME": db_name,
            "GROQ_API_KEY": groq_api_key
        }
        with st.spinner("Connecting to database and loading tables..."):
            datalake, table_info, engine, failed_tables = validate_and_connect_database(credentials)

        if datalake and table_info:
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.failed_tables = failed_tables
            st.session_state.database_loaded = True
            
            st.success("Database connected successfully!")

    # Direct input interface (replaces chat)
    if st.session_state.get("database_loaded", False):
        st.header("📝 Data Query")
        
        # Create an input form
        with st.form(key="query_form"):
            query_input = st.text_area("Enter your query:", 
                                      placeholder="Example: What are the top 5 products by sales?",
                                      height=100)
            submit_button = st.form_submit_button("Process Query")
        
        # Process form submission
        if submit_button and query_input:
            st.divider()
            st.subheader("📊 Your Query:")
            st.info(query_input)
            process_query(st.session_state.datalake, query_input)
    else:
        st.info("Please connect to a database first to start querying data.")

if __name__ == "__main__":
    main()
