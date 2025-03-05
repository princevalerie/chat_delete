import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
import joblib

def auto_validate_and_connect_database():
    """Automatically validate and connect to database using environment variables"""
    try:
        # Retrieve credentials from environment variables or Streamlit secrets
        db_user = st.secrets.get("DB_USER", os.getenv("DB_USER"))
        db_password = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))
        db_host = st.secrets.get("DB_HOST", os.getenv("DB_HOST", "localhost"))
        db_port = st.secrets.get("DB_PORT", os.getenv("DB_PORT", "5432"))
        db_name = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))
        groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        
        # Validate credentials
        if not all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
            st.error("Missing database or API credentials. Please set environment variables or Streamlit secrets.")
            return None, None, None
        
        # URL encode special characters in password
        encoded_password = db_password.replace('@', '%40')
        
        # Create database engine
        engine = create_engine(
            f"postgresql://{db_user}:{encoded_password}@{db_host}:{port}/{db_name}"
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
            
            for table in all_tables_views:
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)
                    
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
            
            # Save datalake cache
            joblib.dump({
                'datalake': datalake,
                'table_info': table_info
            }, 'datalake_cache.joblib')
            
            return datalake, table_info, engine
    
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

def render_response(response):
    try:
        if response is None:
            st.warning("No response generated.")
            return

        if isinstance(response, plt.Figure):
            st.pyplot(response)
        elif isinstance(response, go.Figure):
            st.plotly_chart(response)
        elif isinstance(response, pd.DataFrame):
            st.dataframe(response)
        elif isinstance(response, str):
            st.write(response)
        elif isinstance(response, (list, dict, int, float, bool)):
            st.write(str(response))
        else:
            st.write("Unhandled response type:", str(response))
    
    except Exception as e:
        st.error(f"Error rendering response: {e}")
        st.write("Raw response:", response)

def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    st.title("🔍 Smart Database Explorer")
    
    # Automatically load database on startup
    if 'datalake' not in st.session_state:
        with st.spinner("Connecting to database and loading tables..."):
            datalake, table_info, _ = auto_validate_and_connect_database()
        
        if datalake and table_info:
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.database_loaded = True
    
    # Display table information
    if st.session_state.get('database_loaded', False):
        st.subheader("📊 Loaded Tables")
        for table, info in st.session_state.table_info.items():
            with st.expander(table):
                st.write(f"Columns: {', '.join(info['columns'])}")
                st.write(f"Row Count: {info['row_count']}")
    
    # Chat interface
    if st.session_state.get('database_loaded', False):
        st.header("💬 Database Chat")
        
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
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
                        import traceback
                        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
