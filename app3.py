import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from urllib.parse import quote_plus
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake

def validate_and_connect_database(credentials):
    """Validate database connection and initialize everything"""
    try:
        # Extract credentials
        db_user = credentials["DB_USER"]
        db_password = credentials["DB_PASSWORD"]
        db_host = credentials["DB_HOST"]
        db_port = credentials["DB_PORT"]
        db_name = credentials["DB_NAME"]
        groq_api_key = credentials["GROQ_API_KEY"]
        
        # Properly encode credentials
        encoded_user = quote_plus(db_user)
        encoded_password = quote_plus(db_password)
        
        # Create database engine
        engine = create_engine(
            f"postgresql://{encoded_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        )
        
        # Initialize LLM
        llm = ChatGroq(model_name="Llama3-8b-8192", api_key=groq_api_key)
        
        # Inspect database
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        views = inspector.get_view_names(schema='public')
        all_tables_views = tables + views
        
        # Load tables and views
        sdf_list = []
        table_info = {}
        
        for table in all_tables_views:
            query = f'SELECT * FROM "public"."{table}"'
            try:
                df = pd.read_sql_query(query, engine)
                
                # Create SmartDataframe without passing LLM here
                sdf = SmartDataframe(df, name=f"public.{table}")
                sdf_list.append(sdf)
                
                # Store table metadata
                table_info[table] = {
                    'columns': list(df.columns),
                    'row_count': len(df)
                }
                
            except Exception as e:
                st.warning(f"Failed to load data from public.{table}: {e}")
        
        # Create SmartDatalake and pass LLM here
        datalake = SmartDatalake(sdf_list, llm=llm)
        
        return datalake, table_info, engine

    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

# Use Streamlit caching for resource management
@st.cache_resource(show_spinner=False)
def get_datalake(credentials):
    datalake, table_info, engine = validate_and_connect_database(credentials)
    return datalake, table_info

def render_response(response):
    """Function to render different response types"""
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
        else:
            st.write("Unknown response type:", str(response))
    
    except Exception as e:
        st.error(f"Error rendering response: {e}")
        st.write("Raw response:", response)

def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    st.title("🔍 Smart Database Explorer")
    
    # Database credentials sidebar
    with st.sidebar:
        st.header("🔐 Database Credentials")
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        connect_button = st.button("Connect to Database")
    
    # Database connection logic
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
            datalake, table_info = get_datalake(credentials)
        
        if datalake and table_info:
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.database_loaded = True
    
    # Chat interface
    if st.session_state.get('database_loaded', False):
        st.header("💬 Database Chat")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display loaded tables
        st.subheader("📊 Loaded Tables")
        for table, info in st.session_state.table_info.items():
            with st.expander(table):
                st.write(f"Columns: {', '.join(info['columns'])}")
                st.write(f"Row Count: {info['row_count']}")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])
        
        # User input and response handling
        if prompt := st.chat_input("Ask a question about your data"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Generating response..."):
                    try:
                        response = st.session_state.datalake.chat(prompt)
                        render_response(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": str(response)
                        })
                    
                    except Exception as e:
                        st.error(f"Error processing request: {e}")
                        import traceback
                        st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
