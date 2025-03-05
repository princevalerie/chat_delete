import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
import plotly.express as px
import plotly.graph_objs as go

def set_page_style():
    """Set custom page style"""
    st.set_page_config(
        page_title="Smart Database Explorer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f2ff;
        border-left: 4px solid #1e90ff;
    }
    .ai-message {
        background-color: #f0f0f0;
        border-left: 4px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

def validate_credentials(user, password, host, port, db):
    """Validate database connection credentials"""
    try:
        engine = create_engine(
            f"postgresql://{user}:{password.replace('@', '%40')}@{host}:{port}/{db}"
        )
        with engine.connect() as connection:
            return True
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return False

def load_database_tables(user, password, host, port, db, schema='public'):
    """Load tables from PostgreSQL database into SmartDataframes"""
    try:
        engine = create_engine(
            f"postgresql://{user}:{password.replace('@', '%40')}@{host}:{port}/{db}"
        )
        
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema=schema)
        views = inspector.get_view_names(schema=schema)
        all_tables_views = tables + views
        
        sdf_list = []
        table_info = {}
        
        for table in all_tables_views:
            query = f'SELECT * FROM "{schema}"."{table}"'
            try:
                df = pd.read_sql_query(query, engine)
                sdf = SmartDataframe(df, name=f"{schema}.{table}")
                sdf_list.append(sdf)
                
                # Store table metadata
                table_info[table] = {
                    'columns': list(df.columns),
                    'row_count': len(df)
                }
                
            except Exception as e:
                st.warning(f"Gagal memuat data dari {schema}.{table}: {e}")
        
        return sdf_list, table_info
    
    except Exception as e:
        st.error(f"Database loading error: {e}")
        return [], {}

def initialize_llm(api_key):
    """Initialize Groq LLM"""
    try:
        os.environ["GROQ_API_KEY"] = api_key
        llm = ChatGroq(model_name="Llama3-8b-8192", api_key=api_key)
        return llm
    except Exception as e:
        st.error(f"LLM initialization error: {e}")
        return None

def render_response(response):
    """Render response based on its type"""
    if isinstance(response, pd.DataFrame):
        st.dataframe(response)
    elif isinstance(response, (go.Figure, plt.Figure)):
        st.pyplot(response) if isinstance(response, plt.Figure) else st.plotly_chart(response)
    elif isinstance(response, str):
        st.markdown(response)
    else:
        st.write(response)

def main():
    set_page_style()
    st.title("🔍 Smart Database Explorer")
    
    # Sidebar for credentials
    with st.sidebar:
        st.header("🔐 Database Credentials")
        
        # Database credentials inputs
        db_user = st.text_input("PostgreSQL Username")
        db_password = st.text_input("PostgreSQL Password", type="password")
        db_host = st.text_input("PostgreSQL Host", value="localhost")
        db_port = st.text_input("PostgreSQL Port", value="5432")
        db_name = st.text_input("Database Name")
        
        # Groq API Key
        groq_api_key = st.text_input("Groq API Key", type="password")
        
        # Connect button
        connect_button = st.button("Connect to Database")
    
    # Database connection and table loading
    if connect_button:
        if validate_credentials(db_user, db_password, db_host, db_port, db_name):
            # Initialize LLM
            llm = initialize_llm(groq_api_key)
            
            if llm:
                # Load database tables
                with st.spinner("Loading tables..."):
                    sdf_list, table_info = load_database_tables(
                        db_user, db_password, db_host, db_port, db_name
                    )
                
                if sdf_list:
                    # Configure SmartDataframes with LLM
                    for sdf in sdf_list:
                        sdf.config = {"llm": llm}
                    
                    # Create SmartDatalake
                    datalake = SmartDatalake(sdf_list)
                    
                    # Store in session state
                    st.session_state.datalake = datalake
                    st.session_state.table_info = table_info
                    st.success("Database connected and tables loaded successfully!")
                    
                    # Display table information
                    st.subheader("Loaded Tables")
                    for table, info in table_info.items():
                        with st.expander(table):
                            st.write(f"Columns: {', '.join(info['columns'])}")
                            st.write(f"Row Count: {info['row_count']}")
    
    # Chat interface
    if 'datalake' in st.session_state:
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
                        render_response(response)
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": str(response)
                        })
                    
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
