import os
import pickle
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.responses import ResponseParser

# -----------------------------------------------------------------------------
# Custom Streamlit Response Parser
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context):
        super().__init__(context)
        self.plot_counter = 0
        self.context = context

    def format_dataframe(self, result):
        """Handle dataframe output"""
        try:
            self.context.dataframe(result["value"])
        except Exception as e:
            self.context.error(f"Error displaying dataframe: {str(e)}")

    def format_plot(self, result):
        """Handle visualization output"""
        try:
            if isinstance(result["value"], plt.Figure):
                self.context.pyplot(result["value"])
            elif isinstance(result["value"], str) and os.path.exists(result["value"]):
                self.plot_counter += 1
                self.context.image(result["value"], caption=f"Generated Plot {self.plot_counter}")
            else:
                self.context.image(result["value"])
        except Exception as e:
            self.context.error(f"Error displaying plot: {str(e)}")

    def format_other(self, result):
        """Handle other text-based output"""
        try:
            self.context.write(result["value"])
        except Exception as e:
            self.context.error(f"Error displaying other result: {str(e)}")

# -----------------------------------------------------------------------------
# Database Connection & Initialization dengan Caching
# -----------------------------------------------------------------------------
def load_database_cache(credentials, cache_path="db_cache.pkl"):
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    datalake, table_info, _ = initialize_database(credentials)
    if datalake:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((datalake, table_info), f)
        except Exception:
            pass
    return datalake, table_info

def initialize_database(credentials):
    try:
        encoded_password = credentials["DB_PASSWORD"].replace('@', '%40')
        engine = create_engine(
            f"postgresql://{credentials['DB_USER']}:{encoded_password}"
            f"@{credentials['DB_HOST']}:{credentials['DB_PORT']}"
            f"/{credentials['DB_NAME']}"
        )

        with engine.connect() as conn:
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
                api_key=credentials["GROQ_API_KEY"],
                temperature=0.2
            )

            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables = tables + views

            pandasai_config = {
                "llm": llm,
                "response_parser": StreamlitResponse(st),
                "enable_cache": False,
                "save_charts": True,
                "save_charts_path": "./temp_charts/"
            }

            os.makedirs("./temp_charts/", exist_ok=True)

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
                datalake, table_info = load_database_cache(credentials)
                
                if datalake:
                    st.session_state.datalake = datalake
                    st.session_state.table_info = table_info
                    st.session_state.db_initialized = True
                    st.success("Database connected successfully!")
                else:
                    st.error("Failed to connect to database")

    if st.session_state.get("db_initialized"):
        st.header("📝 Natural Language Query Interface")
        
        with st.expander("Example Queries"):
            st.markdown("""
            - Show me the top 5 records from customers
            - Create a bar chart of monthly sales
            - What's the average revenue?
            - Show me a pie chart of product categories
            """)
            
        query = st.chat_input("Ask your data question...")
        
        if query:
            with st.spinner("Analyzing..."):
                try:
                    if os.path.exists("./temp_charts/"):
                        for file in os.listdir("./temp_charts/"):
                            if file.endswith(('.png', '.jpg')):
                                os.remove(os.path.join("./temp_charts/", file))
                    
                    st.session_state.datalake.chat(query)
                    
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")

        st.header("📊 Database Schema")
        for table, info in st.session_state.table_info.items():
            with st.expander(f"📑 {table}"):
                if info["sample_loaded"]:
                    st.markdown(f"**Columns:** `{', '.join(info['columns'])}`")
                    st.markdown(f"**Rows:** {info['row_count']:,}")
                else:
                    st.error(f"❌ Load failed: {info.get('error', 'Unknown error')}")

    else:
        st.info("💡 Please configure database credentials in the sidebar")

if __name__ == "__main__":
    main()
