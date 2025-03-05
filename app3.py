import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Optional, Union, Any
from pandasai.connectors import MySQLConnector, PostgreSQLConnector
from pandasai import SmartDataframe
from langchain_community.llms import Ollama

# Add timeout for database connections
TIMEOUT = 10  # seconds

st.set_page_config(layout="wide", page_title="Database Explorer")  # Panggilan yang benar (hanya sekali)

def fetch_tables(connector):
    """
    Fetch list of tables from the database.
    """
    try:
        if isinstance(connector, MySQLConnector):
            # For MySQL
            query = "SHOW TABLES"
            tables_df = connector.query(query)
            if tables_df.empty:
                return []
            return tables_df.iloc[:, 0].tolist()

        elif isinstance(connector, PostgreSQLConnector):
            # For PostgreSQL
            schema_query = """
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
            """
            schemas_df = connector.query(schema_query)
            schemas = schemas_df['schema_name'].tolist()
            all_tables = []

            for schema_name in schemas:
                table_query = """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = %s
                """
                tables_df = connector.query(table_query, (schema_name,))
                tables = tables_df['table_name'].tolist()
                all_tables.extend([f"{schema_name}.{table}" for table in tables])

            return all_tables

        else:
            st.error("Unsupported database connector.")
            return []

    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []


def load_table_data(connector, table_name):
    """
    Load data from a specific table.
    """
    try:
        if isinstance(connector, PostgreSQLConnector):
            # Split into schema and table for PostgreSQL
            if '.' in table_name:
                schema, table = table_name.split('.', 1)
                query = f'SELECT * FROM "{schema}"."{table}"'
            else:
                query = f'SELECT * FROM "{table_name}"'
        else:
            query = f"SELECT * FROM {table_name}"

        df = connector.query(query)
        return df  # Return directly as it's already a DataFrame

    except Exception as e:
        st.error(f"Error loading table data: {e}")
        return pd.DataFrame()


def validate_connection_params(config):
    return all(config.values())


def main():
    # Initialize session state
    if 'connector' not in st.session_state:
        st.session_state.connector = None
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None

    # Sidebar configuration
    with st.sidebar:
        st.title("Database Configuration")
        db_type = st.radio("Select Database", ["MySQL", "PostgreSQL"])
        
        # Connection parameters
        config = {}
        if db_type == "MySQL":
            config = {
                "host": st.text_input("Host", "localhost"),
                "port": st.number_input("Port", value=3306),
                "database": st.text_input("Database"),
                "user": st.text_input("Username"),  # Tetap menggunakan "user" untuk MySQL
                "password": st.text_input("Password", type="password")
            }
        else:
            config = {
                "host": st.text_input("Host", "localhost"),
                "port": st.number_input("Port", value=5432),
                "database": st.text_input("Database"),
                "username": st.text_input("Username"),  # Perbaikan: "username" untuk PostgreSQL
                "password": st.text_input("Password", type="password"),
                "table": st.text_input("Table")  # Tambahkan parameter "table"
            }

        if st.button("Connect"):
            if validate_connection_params(config):
                with st.spinner("Connecting to database..."):
                    try:
                        connector_class = MySQLConnector if db_type == "MySQL" else PostgreSQLConnector
                        st.session_state.connector = connector_class(config=config)
                        st.success("Connected successfully!")
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")
            else:
                st.error("Please fill all connection parameters")

    # Main content
    if st.session_state.connector:
        tables = fetch_tables(st.session_state.connector)
        
        if tables:
            selected_table = st.selectbox("Select Table", tables)
            
            if selected_table:
                with st.spinner("Loading table data..."):
                    df = load_table_data(st.session_state.connector, selected_table)
                    st.dataframe(df)
                
                # AI Assistant
                st.header("🤖 AI Database Assistant")
                try:
                    llm = Ollama(model="mistral")
                    df_connector = SmartDataframe(st.session_state.connector, config={"llm": llm})
                    
                    prompt = st.text_input("Ask a question about your data:")
                    if st.button("Generate"):
                        if prompt:
                            with st.spinner("Generating response..."):
                                try:
                                    response = df_connector.chat(prompt)
                                    display_response(response)
                                except Exception as e:
                                    st.error(f"Error generating response: {str(e)}")
                        else:
                            st.warning("Please enter a question")
                except Exception as e:
                    st.error(f"Error initializing AI assistant: {str(e)}")
        else:
            st.warning("No tables found in the database")


def display_response(response: Any) -> None:
    """Display different types of responses appropriately."""
    if isinstance(response, pd.DataFrame):
        st.dataframe(response)
    elif isinstance(response, plt.Figure):
        st.pyplot(response)
    elif isinstance(response, go.Figure):
        st.plotly_chart(response)
    elif isinstance(response, str):
        st.write(response)
    else:
        st.write("Result:", response)


if __name__ == "__main__":
    main()
