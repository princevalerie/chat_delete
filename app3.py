import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake

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

def debug_table_columns(engine, table_name):
    """
    Retrieve and display table column information
    """
    try:
        # Query to get column information
        query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = '{table_name}';
        """
        columns_df = pd.read_sql_query(query, engine)
        return columns_df
    except Exception as e:
        st.error(f"Error retrieving columns for {table_name}: {e}")
        return None

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
            
            # Debugging section to display all table columns
            st.subheader("🔍 Table Column Information")
            for table in all_tables_views:
                columns_df = debug_table_columns(engine, table)
                if columns_df is not None:
                    with st.expander(f"Columns in {table}"):
                        st.dataframe(columns_df)
                
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm}  # Configure LLM for each dataframe
                    sdf_list.append(sdf)
                    
                    # Store table metadata
                    table_info[table] = {
                        'columns': list(df.columns),
                        'row_count': len(df)
                    }
                    
                except Exception as e:
                    st.warning(f"Failed to load data from public.{table}: {e}")
            
            # Create SmartDatalake with only config parameter
            datalake = SmartDatalake(sdf_list, config={"llm": llm})
            
            return datalake, table_info, engine
    
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None


def render_response(response):
    """
    Render response with comprehensive handling of different output types
    Supports multiple output types from PandasAI
    """
    try:
        # Handle various possible output types
        if response is None:
            st.warning("No response generated.")
            return

        # Check if it's a file path (for saved plots or files)
        if isinstance(response, str):
            # Check file extension
            if response.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg')):
                st.image(response)
            elif response.lower().endswith(('.csv', '.xlsx')):
                st.download_button(
                    label="Download File", 
                    data=open(response, 'rb').read(), 
                    file_name=os.path.basename(response)
                )
            else:
                st.write(response)
        
        # Pandas DataFrame
        elif isinstance(response, pd.DataFrame):
            st.dataframe(response)
        
        # Matplotlib Figure
        elif isinstance(response, plt.Figure):
            st.pyplot(response)
        
        # Plotly Figure
        elif isinstance(response, go.Figure):
            st.plotly_chart(response)
        
        # List or Dict
        elif isinstance(response, (list, dict)):
            st.write(response)
        
        # Numeric or Boolean values
        elif isinstance(response, (int, float, bool)):
            st.write(str(response))
        
        # Default fallback
        else:
            try:
                # Attempt to convert to string if possible
                st.write(str(response))
            except Exception as conversion_error:
                st.error(f"Unable to render response. Type: {type(response)}")
                st.error(f"Error details: {conversion_error}")
    
    except Exception as e:
        st.error(f"Unexpected error rendering response: {e}")
        st.error(f"Response type: {type(response)}")
        st.write("Raw response:", response)

def main():
    set_page_style()
    st.title("🔍 Smart Database Explorer")
    
    # Initialize session state variables if not exists
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    
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
        
        # Connect Button
        connect_button = st.button("Connect to Database")
    
    # Connection Logic
    if connect_button and all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
        with st.spinner("Connecting to database and loading tables..."):
            # Attempt to connect and load database
            datalake, table_info, engine = validate_and_connect_database(
                db_user, db_password, db_host, db_port, db_name, groq_api_key
            )
        
        if datalake and table_info:
            # Store in session state
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.engine = engine
            st.session_state.database_loaded = True
            
            # Display table information
            st.subheader("📊 Loaded Tables")
            for table, info in table_info.items():
                with st.expander(table):
                    st.write(f"Columns: {', '.join(info['columns'])}")
                    st.write(f"Row Count: {info['row_count']}")
    
    # Chat interface
    if st.session_state.database_loaded:
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
