import os
from pathlib import Path

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.responses.response_parser import ResponseParser

# -----------------------------------------------------------------------------
# Custom Response Parser for Streamlit
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
        self.image_response = None
        self.text_response = None
        self.dataframe_response = None

    def format_dataframe(self, result):
        self.dataframe_response = result["value"]
        st.dataframe(result["value"])
        return result["value"]

    def format_plot(self, result):
        try:
            # Store image path instead of direct object to avoid serialization issues
            if isinstance(result["value"], str) and os.path.exists(result["value"]):
                self.image_response = result["value"]
                st.image(result["value"])
            else:
                self.image_response = result["value"]
                st.image(result["value"])
        except Exception as e:
            st.error(f"Error displaying plot: {e}")
        return result["value"]

    def format_other(self, result):
        self.text_response = str(result["value"])
        st.write(result["value"])
        return result["value"]

# -----------------------------------------------------------------------------
# PandasAI Response Handling Function
# -----------------------------------------------------------------------------
def handle_pandasai_response(response, response_parser):
    """
    Comprehensive handler for PandasAI responses

    Args:
        response: Raw response from PandasAI
        response_parser: Custom Streamlit response parser

    Returns:
        dict: Parsed message entry with different response types
    """
    message_entry = {"role": "assistant"}
    
    try:
        # Handle different response types
        if response is None:
            message_entry["text"] = "No response generated."
            st.write("No response generated.")
        
        # 1. Dataframe Handling
        elif isinstance(response, pd.DataFrame):
            message_entry["dataframe"] = response
            st.dataframe(response)
        
        # 2. Dataframe from Response Parser
        elif response_parser.dataframe_response is not None:
            message_entry["dataframe"] = response_parser.dataframe_response
            st.dataframe(response_parser.dataframe_response)
        
        # 3. Image/Plot Handling
        elif response_parser.image_response is not None:
            message_entry["image"] = response_parser.image_response
            # Image already displayed in format_plot
        
        # 4. Text/String Handling
        elif isinstance(response, str):
            message_entry["text"] = response
            st.markdown(response)
        elif response_parser.text_response is not None:
            message_entry["text"] = response_parser.text_response
            st.markdown(response_parser.text_response)
        
        # 5. Numeric/Scalar Value Handling
        elif isinstance(response, (int, float, bool)):
            message_entry["text"] = str(response)
            st.write(response)
        
        # 6. List Handling
        elif isinstance(response, list):
            message_entry["text"] = "\n".join(map(str, response))
            st.write(response)
        
        # 7. Dictionary Handling
        elif isinstance(response, dict):
            formatted_dict = "\n".join([f"**{k}**: {v}" for k, v in response.items()])
            message_entry["text"] = formatted_dict
            st.write(formatted_dict)
        
        # 8. Catch-all for unexpected types
        else:
            message_entry["text"] = f"Unhandled response type: {type(response)}"
            st.write(f"Unhandled response type: {type(response)}")
    
    except Exception as e:
        message_entry["text"] = f"Error processing response: {str(e)}"
        st.error(f"Error processing response: {str(e)}")
    
    return message_entry

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
                    
                    # Create SmartDataframe with LLM and ResponseParser configuration
                    response_parser = StreamlitResponse(st)
                    sdf = SmartDataframe(
                        df, 
                        name=f"public.{table}", 
                        config={
                            "llm": llm, 
                            "response_parser": response_parser,
                            "save_charts": True,  # Ensure charts are saved
                            "enforce_panel_edit": False,  # Prevent Panel editing which can cause issues
                            "open_charts": False  # Don't try to open charts automatically
                        }
                    )
                    sdf_list.append(sdf)
                    
                    # Save table metadata
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
                    failed_tables.append(table)

            # Create SmartDatalake from SmartDataframe list with explicit configs
            datalake = SmartDatalake(
                sdf_list, 
                config={
                    "llm": llm, 
                    "response_parser": StreamlitResponse(st),
                    "save_charts": True,
                    "enforce_panel_edit": False,
                    "open_charts": False
                }
            )
            
            return datalake, table_info, engine, failed_tables
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None, []

# -----------------------------------------------------------------------------
# Main View and Database Chat Logic
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    
    # Initialize session state
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processing_query' not in st.session_state:
        st.session_state.processing_query = False
    
    # Function to handle query submission
    def process_query(prompt):
        if prompt:
            # Add user message to history
            st.session_state.messages.append({
                "role": "user", 
                "text": prompt
            })
            st.session_state.processing_query = True
            # Set to rerun the app
            st.session_state.pending_query = prompt
    
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
    st.title("🔍 Smart Database Explorer")

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

    # Chat interface
    if st.session_state.get("database_loaded", False):
        st.header("💬 Database Chat")

        # Display message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], 
                             avatar="🤖" if message["role"] == "assistant" else "👤"):
                if "image" in message and message["image"] is not None:
                    try:
                        st.image(message["image"])
                    except Exception as e:
                        st.error(f"Failed to display image: {e}")
                if "text" in message and message["text"] is not None:
                    st.markdown(message["text"])
                if "dataframe" in message and message["dataframe"] is not None:
                    st.dataframe(message["dataframe"])

        # Process any pending query
        if "pending_query" in st.session_state and st.session_state.pending_query:
            prompt = st.session_state.pending_query
            st.session_state.pending_query = None  # Clear pending query
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Generating response..."):
                    try:
                        response_parser = StreamlitResponse(st)
                        response_parser.image_response = None
                        response_parser.text_response = None
                        response_parser.dataframe_response = None

                        answer = st.session_state.datalake.chat(prompt)
                        
                        message_entry = {
                            "role": "assistant",
                            "text": response_parser.text_response,
                            "image": response_parser.image_response,
                            "dataframe": response_parser.dataframe_response
                        }
                        
                        # Remove None values
                        message_entry = {k: v for k, v in message_entry.items() if v is not None}
                        
                        if not any(k in message_entry for k in ["text", "image", "dataframe"]):
                            if isinstance(answer, str):
                                message_entry["text"] = answer
                            elif isinstance(answer, pd.DataFrame):
                                message_entry["dataframe"] = answer
                            else:
                                message_entry["text"] = str(answer)
                        
                        st.session_state.messages.append(message_entry)
                        st.session_state.processing_query = False

                    except Exception as e:
                        error_msg = f"Error processing chat: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "text": error_msg
                        })
                        st.session_state.processing_query = False

        # Chat input
        if not st.session_state.processing_query:
            prompt = st.chat_input("Ask a question about your data")
            if prompt:
                # Display user message immediately
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                process_query(prompt)
                st.rerun()  # Use st.rerun() instead of experimental_rerun

if __name__ == "__main__":
    main()
