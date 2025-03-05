import os
import pickle
from pathlib import Path

from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

# =============================================================================
# Custom ResponseParser dan Callback untuk Streamlit
# =============================================================================
class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return
# =============================================================================
# Fungsi untuk validasi koneksi database dan memuat tabel
# =============================================================================
def validate_and_connect_database(credentials):
    """Validasi koneksi database dan inisialisasi komponen-komponen utama."""
    try:
        # Ekstrak kredensial
        db_user = credentials["DB_USER"]
        db_password = credentials["DB_PASSWORD"]
        db_host = credentials["DB_HOST"]
        db_port = credentials["DB_PORT"]
        db_name = credentials["DB_NAME"]
        groq_api_key = credentials["GROQ_API_KEY"]

        # Encode password untuk karakter spesial
        encoded_password = db_password.replace('@', '%40')

        # Buat database engine
        engine = create_engine(
            f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"
        )

        # Uji koneksi dan inisialisasi komponen
        with engine.connect() as connection:
            # Inisialisasi LLM
            llm = ChatGroq(model_name="Llama3-8b-8192", api_key=groq_api_key)

            # Inspeksi database
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema='public')
            views = inspector.get_view_names(schema='public')
            all_tables_views = tables + views

            # Load tabel dan view
            sdf_list = []
            table_info = {}

            for table in all_tables_views:
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)

                    # Buat SmartDataframe dengan konfigurasi LLM dan ResponseParser
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm, "response_parser": StreamlitResponse(st)}
                    sdf_list.append(sdf)

                    # Simpan metadata tabel
                    table_info[table] = {
                        'columns': list(df.columns),
                        'row_count': len(df)
                    }
                except Exception as e:
                    st.warning(f"Gagal memuat data dari public.{table}: {e}")

            # Buat SmartDatalake dari list SmartDataframe dengan ResponseParser
            datalake = SmartDatalake(sdf_list, config={"llm": llm, "response_parser": StreamlitResponse(st)})

            return datalake, table_info, engine

    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

# =============================================================================
# Caching dengan Pickle untuk data tabel dari database
# =============================================================================
def load_database_cache(credentials, cache_path="db_cache.pkl"):
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                datalake, table_info = pickle.load(f)
            return datalake, table_info
        except Exception as e:
            st.warning(f"Gagal memuat cache: {e}. Memuat ulang data dari database.")
    # Jika cache tidak ada atau gagal dibaca, muat dari database
    datalake, table_info, engine = validate_and_connect_database(credentials)
    if datalake is not None and table_info is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((datalake, table_info), f)
        except Exception as e:
            st.warning(f"Gagal menyimpan cache: {e}")
    return datalake, table_info

# =============================================================================
# Tampilan Utama dan Logika Chat Database
# =============================================================================
def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    st.title("🔍 Smart Database Explorer")

    # Sidebar untuk kredensial database
    with st.sidebar:
        st.header("🔐 Database Credentials")

        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")

        connect_button = st.button("Connect to Database")

    # Logika koneksi database dan caching dengan pickle
    if connect_button and all([db_user, db_password, db_host, db_port, db_name, groq_api_key]):
        credentials = {
            "DB_USER": db_user,
            "DB_PASSWORD": db_password,
            "DB_HOST": db_host,
            "DB_PORT": db_port,
            "DB_NAME": db_name,
            "GROQ_API_KEY": groq_api_key
        }
        with st.spinner("Menghubungkan ke database dan memuat tabel..."):
            datalake, table_info = load_database_cache(credentials)

        if datalake and table_info:
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.database_loaded = True

    # Tampilan antarmuka chat database
    if st.session_state.get('database_loaded', False):
        st.header("💬 Database Chat")

        # Inisialisasi history chat
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Tampilkan informasi tabel yang telah dimuat
        st.subheader("📊 Loaded Tables")
        for table, info in st.session_state.table_info.items():
            with st.expander(table):
                st.write(f"Columns: {', '.join(info['columns'])}")
                st.write(f"Row Count: {info['row_count']}")

        # Tampilkan history chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"])

        # Input chat
        if prompt := st.chat_input("Ask a question about your data"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Generate response dari datalake; output akan langsung dirender oleh StreamlitResponse
            with st.spinner("Generating response..."):
                try:
                    st.session_state.datalake.chat(prompt)
                    # Log history respons sebagai catatan (output telah dirender secara langsung)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Output telah dirender pada tampilan."
                    })
                except Exception as e:
                    st.error(f"Error dalam memproses chat: {e}")

if __name__ == "__main__":
    main()
