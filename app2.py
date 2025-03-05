import os
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.responses.response_parser import ResponseParser

# -----------------------------------------------------------------------------
# Custom Response Parser untuk Streamlit
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return ""

    def format_plot(self, result):
        st.image(result["value"])
        return ""

    def format_other(self, result):
        st.write(result["value"])
        return ""

# -----------------------------------------------------------------------------
# Database connection validation and table loading function
# -----------------------------------------------------------------------------
def validate_and_connect_database(credentials):
    try:
        db_user = credentials["DB_USER"]
        db_password = credentials["DB_PASSWORD"]
        db_host = credentials["DB_HOST"]
        db_port = credentials["DB_PORT"]
        db_name = credentials["DB_NAME"]
        groq_api_key = credentials["GROQ_API_KEY"]

        encoded_password = db_password.replace('@', '%40')
        engine = create_engine(
            f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        )

        with engine.connect() as connection:
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                api_key=groq_api_key,
                temperature=0.2
            )

            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables_views = tables + views

            sdf_list = []
            table_info = {}

            for table in all_tables_views:
                try:
                    df = pd.read_sql_query(f'SELECT * FROM "public"."{table}" LIMIT 10', engine)
                    sdf = SmartDataframe(
                        df,
                        name=f"public.{table}",
                        config={
                            "llm": llm,
                            "response_parser": StreamlitResponse(context=st),
                            "enable_cache": False
                        }
                    )
                    sdf_list.append(sdf)
                    table_info[table] = {
                        "columns": list(df.columns),
                        "row_count": len(df),
                        "sample_loaded": True
                    }
                except Exception as e:
                    table_info[table] = {
                        "sample_loaded": False,
                        "error": str(e)
                    }

            datalake = SmartDatalake(
                sdf_list,
                config={
                    "llm": llm,
                    "response_parser": StreamlitResponse(context=st),
                    "enable_cache": False
                }
            )
            
            return datalake, table_info, engine

    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Direct Database Explorer", layout="wide")
    st.title("🔍 Direct Database Explorer")

    # Initialize session state
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False

    # Sidebar for Database Credentials
    with st.sidebar:
        st.header("🔐 Database Credentials")
        credentials = {
            "DB_USER": st.text_input("PostgreSQL Username"),
            "DB_PASSWORD": st.text_input("PostgreSQL Password", type="password"),
            "DB_HOST": st.text_input("PostgreSQL Host", value="localhost"),
            "DB_PORT": st.text_input("PostgreSQL Port", value="5432"),
            "DB_NAME": st.text_input("Database Name"),
            "GROQ_API_KEY": st.text_input("Groq API Key", type="password")
        }
        
        if st.button("Connect to Database"):
            with st.spinner("Connecting to database..."):
                datalake, table_info, engine = validate_and_connect_database(credentials)
                
                if datalake and table_info:
                    st.session_state.datalake = datalake
                    st.session_state.table_info = table_info
                    st.session_state.database_loaded = True
                    st.success("Database connected successfully!")

    # Main content area
    if st.session_state.get("database_loaded", False):
        st.header("📝 Direct Query Interface")
        
        # Query input
        query = st.text_area("Enter your data query:", height=100)
        
        if st.button("Execute Query"):
            st.divider()
            with st.spinner("Processing your query..."):
                try:
                    # Direct execution with StreamlitResponse integration
                    result = st.session_state.datalake.chat(query)
                    
                    # Handle text responses
                    if isinstance(result, str):
                        st.subheader("Query Result")
                        st.markdown(result)
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

        # Table information
        st.header("📊 Database Schema Info")
        for table, info in st.session_state.table_info.items():
            with st.expander(f"Table: {table}"):
                if info.get('sample_loaded'):
                    st.write(f"Columns: {', '.join(info['columns'])}")  # Perbaikan di sini
                    st.write(f"Sample rows: {info['row_count']}")
                else:
                    st.error(f"Failed to load table: {info.get('error', 'Unknown error')}")

    else:
        st.info("Please connect to a database using the sidebar credentials")

if __name__ == "__main__":
    main()
