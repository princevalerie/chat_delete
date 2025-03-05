import os
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.responses.response_parser import ResponseParser

# -----------------------------------------------------------------------------
# Custom Streamlit Response Parser
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def format_dataframe(self, result):
        """Handle dataframe output"""
        st.dataframe(result["value"])
        return ""

    def format_plot(self, result):
        """Handle visualization output"""
        st.image(result["value"])
        return ""

    def format_other(self, result):
        """Handle other text-based output"""
        st.write(result["value"])
        return ""

# -----------------------------------------------------------------------------
# Database Connection & Initialization
# -----------------------------------------------------------------------------
def initialize_database(credentials):
    try:
        # Create database engine
        encoded_password = credentials["DB_PASSWORD"].replace('@', '%40')
        engine = create_engine(
            f"postgresql://{credentials['DB_USER']}:{encoded_password}"
            f"@{credentials['DB_HOST']}:{credentials['DB_PORT']}"
            f"/{credentials['DB_NAME']}"
        )

        with engine.connect() as conn:
            # Initialize LLM
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                api_key=credentials["GROQ_API_KEY"],
                temperature=0.2
            )

            # Get database schema
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables = tables + views

            # PandasAI configuration
            pandasai_config = {
                "llm": llm,
                "response_parser": StreamlitResponse,
                "enable_cache": False,
                "verbose": True
            }

            # Load sample data
            sdf_list = []
            table_info = {}
            
            for table in all_tables:
                try:
                    df = pd.read_sql_query(f'SELECT * FROM "public"."{table}" LIMIT 10', engine)
                    sdf = SmartDataframe(
                        df,
                        name=f"public.{table}",
                        config=pandasai_config
                    )
                    sdf_list.append(sdf)
                    
                    table_info[table] = {
                        "columns": df.columns.tolist(),
                        "row_count": len(df),
                        "sample_loaded": True
                    }
                except Exception as e:
                    table_info[table] = {
                        "sample_loaded": False,
                        "error": str(e)
                    }

            # Create datalake
            datalake = SmartDatalake(sdf_list, config=pandasai_config)
            
            return datalake, table_info, engine

    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None, None, None

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart DB Explorer", layout="wide")
    st.title("🔍 Smart Database Explorer")

    # Initialize session state
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Database Configuration")
        credentials = {
            "DB_USER": st.text_input("Username"),
            "DB_PASSWORD": st.text_input("Password", type="password"),
            "DB_HOST": st.text_input("Host", value="localhost"),
            "DB_PORT": st.text_input("Port", value="5432"),
            "DB_NAME": st.text_input("Database Name"),
            "GROQ_API_KEY": st.text_input("Groq API Key", type="password")
        }
        
        if st.button("🚀 Connect Database"):
            with st.spinner("Initializing database connection..."):
                datalake, table_info, engine = initialize_database(credentials)
                
                if datalake:
                    st.session_state.datalake = datalake
                    st.session_state.table_info = table_info
                    st.session_state.db_initialized = True
                    st.success("Database connected successfully!")
                else:
                    st.error("Failed to connect to database")

    # Main Interface
    if st.session_state.get("db_initialized"):
        # Query Section
        st.header("📝 Direct Query Interface")
        query = st.text_area("Enter your query:", height=100)
        
        if st.button("🔍 Execute Query"):
            with st.spinner("Analyzing data..."):
                try:
                    result = st.session_state.datalake.chat(query)
                    if isinstance(result, str) and not result:
                        st.info("Query executed successfully")
                except Exception as e:
                    st.error(f"Query execution failed: {str(e)}")

        # Schema Information
        st.header("📊 Database Schema")
        for table, info in st.session_state.table_info.items():
            with st.expander(f"📑 {table}"):
                if info["sample_loaded"]:
                    st.markdown(f"**Columns:** `{', '.join(info['columns'])}`")
                    st.markdown(f"**Sample Rows:** {info['row_count']}")
                else:
                    st.error(f"❌ Load failed: {info.get('error', 'Unknown error')}")

    else:
        st.info("💡 Please configure database credentials in the sidebar to begin")

if __name__ == "__main__":
    main()
