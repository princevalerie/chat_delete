import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
import joblib

def save_datalake(datalake, table_info, filename='datalake_cache.joblib'):
    """Save datalake and table info to a file"""
    try:
        joblib.dump({
            'datalake': datalake,
            'table_info': table_info
        }, filename)
        st.success(f"Database cache saved to {filename}")
    except Exception as e:
        st.error(f"Error saving datalake cache: {e}")

def load_datalake(filename='datalake_cache.joblib'):
    """Load datalake and table info from a file"""
    try:
        cached_data = joblib.load(filename)
        st.success("Database cache loaded successfully")
        return cached_data['datalake'], cached_data['table_info']
    except Exception as e:
        st.error(f"Error loading datalake cache: {e}")
        return None, None

def validate_and_connect_database(user, password, host, port, db, groq_api_key):
    """Validate database connection and initialize everything in one step"""
    try:
        # URL encode special characters in password
        encoded_password = password.replace('@', '%40')
        
        # Create database engine
        engine = create_engine(
            f"postgresql://{user}:{encoded_password}@{host}:{port}/{db}"
        )
        
        # Test connection
        with engine.connect() as connection:
            # Initialize LLM
            os.environ["GROQ_API_KEY"] = groq_api_key
            llm = ChatGroq(model_name="Llama3-8b-8192", api_key=groq_api_key)
            
            # Inspect database
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema='public')
            views = inspector.get_view_names(schema='public')
            all_tables_views = tables + views
            
            # Load tables
            sdf_list = []
            table_info = {}
            
            # Enhanced debugging and loading
            st.subheader("🔍 Table Column Information")
            for table in all_tables_views:
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)
                    
                    # Detailed column information
                    st.write(f"Table: {table}")
                    st.write("Columns:", list(df.columns))
                    st.write("Sample Data:")
                    st.dataframe(df.head())
                    
                    # Create SmartDataframe
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm}
                    sdf_list.append(sdf)
                    
                    # Store table metadata
                    table_info[table] = {
                        'columns': list(df.columns),
                        'row_count': len(df)
                    }
                    
                except Exception as e:
                    st.warning(f"Failed to load data from public.{table}: {e}")
            
            # Create SmartDatalake
            datalake = SmartDatalake(sdf_list, config={"llm": llm})
            
            return datalake, table_info, engine
    
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

def render_response(response):
    """Comprehensive response rendering with enhanced debugging"""
    try:
        # Extensive debugging output
        st.write("Debug - Response Type:", type(response))
        st.write("Debug - Response Content:", str(response)[:500])  # First 500 chars
        
        # Detailed rendering logic
        if response is None:
            st.warning("No response generated.")
            return

        if isinstance(response, pd.DataFrame):
            st.dataframe(response)
        elif isinstance(response, str):
            st.write(response)
        elif isinstance(response, (list, dict)):
            st.write(response)
        elif isinstance(response, (int, float, bool)):
            st.write(str(response))
        else:
            st.write("Unhandled response type:", str(response))
    
    except Exception as e:
        st.error(f"Error rendering response: {e}")
        st.write("Raw response:", response)

def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    st.title("🔍 Smart Database Explorer")
    
    # Sidebar for credentials
    with st.sidebar:
        st.header("🔐 Database Credentials")
        
        # Database credentials inputs
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        
        # Groq API Key
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            connect_button = st.button("Connect to Database")
        with col2:
            load_cache_button = st.button("Load Cached Database")
    
    # Connection Logic
    if connect_button and all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
        with st.spinner("Connecting to database and loading tables..."):
            # Attempt to connect and load database
            datalake, table_info, engine = validate_and_connect_database(
                db_user, db_password, db_host, db_port, db_name, groq_api_key
            )
        
        if datalake and table_info:
            # Save to cache
            save_datalake(datalake, table_info)
            
            # Store in session state
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.database_loaded = True
    
    # Load Cached Database
    if load_cache_button:
        cached_datalake, cached_table_info = load_datalake()
        if cached_datalake and cached_table_info:
            st.session_state.datalake = cached_datalake
            st.session_state.table_info = cached_table_info
            st.session_state.database_loaded = True
    
    # Chat interface
    if st.session_state.get('database_loaded', False):
        st.header("💬 Database Chat")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display table information
        st.subheader("📊 Loaded Tables")
        for table, info in st.session_state.table_info.items():
            with st.expander(table):
                st.write(f"Columns: {', '.join(info['columns'])}")
                st.write(f"Row Count: {info['row_count']}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"]=="assistant" else "👤"):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Generating response..."):
                    try:
                        # Enhanced debugging
                        st.write("Debug - Attempting to chat with datalake")
                        response = st.session_state.datalake.chat(prompt)
                        
                        # Render and display response
                        render_response(response)
                        
                        # Add to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": str(response)
                        })
                    
                    except Exception as e:
                        st.error(f"Detailed error in chat processing: {e}")
                        # Log full error traceback
                        import traceback
                        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
