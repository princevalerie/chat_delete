import os
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.responses import ResponseParser
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Custom Streamlit Response Parser
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context=None):
        if context is None:
            context = {}  # Or set a meaningful default context if required
        super().__init__(context)
        self.plot_counter = 0


    def format_dataframe(self, result):
        """Handle dataframe output"""
        try:
            logger.info(f"Displaying dataframe with shape: {result['value'].shape}")
            st.dataframe(result["value"])
            return None  # Return None instead of empty string
        except Exception as e:
            logger.error(f"Error displaying dataframe: {str(e)}")
            st.error(f"Error displaying dataframe: {str(e)}")
            return None

    def format_plot(self, result):
        """Handle visualization output"""
        try:
            logger.info(f"Displaying plot: {type(result['value'])}")
            # For matplotlib figures
            if isinstance(result["value"], plt.Figure):
                st.pyplot(result["value"])
            # For file paths (which is more common with PandasAI)
            elif isinstance(result["value"], str) and os.path.exists(result["value"]):
                self.plot_counter += 1
                plot_path = result["value"]
                logger.info(f"Plot saved at: {plot_path}")
                st.image(plot_path, caption=f"Generated Plot {self.plot_counter}")
            # For raw image data
            else:
                st.image(result["value"])
            return None  # Return None instead of empty string
        except Exception as e:
            logger.error(f"Error displaying plot: {str(e)}")
            st.error(f"Error displaying plot: {str(e)}")
            return None

    def format_other(self, result):
        """Handle other text-based output"""
        try:
            # Only display if it's not a text response
            if not isinstance(result["value"], str):
                logger.info(f"Displaying other type result: {type(result['value'])}")
                st.write(result["value"])
            return None  # Return None instead of empty string
        except Exception as e:
            logger.error(f"Error displaying other result: {str(e)}")
            st.error(f"Error displaying other result: {str(e)}")
            return None

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

            # PandasAI configuration - Updated to use instance of StreamlitResponse
            pandasai_config = {
                "llm": llm,
                "response_parser": StreamlitResponse(),  # Instantiate the class
                "enable_cache": False,
                "verbose": True,
                # Add explicit configuration for visualization
                "save_charts": True,
                "save_charts_path": "./temp_charts/"
            }

            # Create temp charts directory if it doesn't exist
            os.makedirs("./temp_charts/", exist_ok=True)

            # Load sample data
            sdf_list = []
            table_info = {}
            
            for table in all_tables:
                try:
                    df = pd.read_sql_query(f'SELECT * FROM "public"."{table}"', engine)
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
# Debug Function
# -----------------------------------------------------------------------------
def debug_result(result):
    """Helper function to debug query results"""
    with st.expander("Debug Information (Toggle to hide)"):
        st.markdown("### Debug Information")
        st.write(f"Result type: {type(result)}")
        st.write(f"Result value: {result}")
        if hasattr(result, '__dict__'):
            st.write(f"Result attributes: {result.__dict__}")

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
        st.markdown("Enter your query about the data. Include visualization instructions if you want a chart.")
        
        # Example queries
        with st.expander("Example Queries"):
            st.markdown("""
            - Show me the top 5 records from the customers table
            - Create a bar chart of monthly sales
            - What's the average value in the revenue column?
            - Show me a pie chart of product categories by sales
            - Plot the trend of transactions over time
            """)
            
        query = st.text_area("Enter your query:", height=100)
        
        col1, col2 = st.columns([1, 5])
        with col1:
            execute_button = st.button("🔍 Execute Query", use_container_width=True)
        with col2:
            show_debug = st.checkbox("Show debug information")
        
        if execute_button:
            with st.spinner("Analyzing data..."):
                try:
                    # Clear previous images (helpful for development)
                    if os.path.exists("./temp_charts/"):
                        for file in os.listdir("./temp_charts/"):
                            if file.endswith('.png') or file.endswith('.jpg'):
                                try:
                                    os.remove(os.path.join("./temp_charts/", file))
                                except:
                                    pass
                    
                    # Execute query
                    result = st.session_state.datalake.chat(query)
                    
                    # Debug information if requested
                    if show_debug:
                        debug_result(result)
                    
                    # Handle text response (only if we don't have visual output)
                    # You can comment this out if you never want text output
                    if isinstance(result, str) and not result:
                        st.info("Query executed successfully")
                        
                except Exception as e:
                    st.error(f"Query execution failed: {str(e)}")
                    logger.error(f"Query execution error: {str(e)}")

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
