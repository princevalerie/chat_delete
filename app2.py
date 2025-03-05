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
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

# -----------------------------------------------------------------------------
# Custom Response Parser untuk Streamlit
# -----------------------------------------------------------------------------
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
        self.image_response = None
        self.text_response = None

    def format_dataframe(self, result):
        self.text_response = result["value"].to_string()
        st.dataframe(result["value"])
        return result["value"]

    def format_plot(self, result):
        self.image_response = result["value"]
        st.image(result["value"])
        return result["value"]

    def format_other(self, result):
        self.text_response = str(result["value"])
        st.write(result["value"])
        return result["value"]

# -----------------------------------------------------------------------------
# Fungsi validasi koneksi database dan pemuatan tabel
# -----------------------------------------------------------------------------
def validate_and_connect_database(credentials):
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

        with engine.connect() as connection:
            # Inisialisasi LLM
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

            # Inspeksi database
            inspector = inspect(engine)
            tables = inspector.get_table_names(schema="public")
            views = inspector.get_view_names(schema="public")
            all_tables_views = tables + views

            sdf_list = []
            table_info = {}

            for table in all_tables_views:
                query = f'SELECT * FROM "public"."{table}"'
                try:
                    df = pd.read_sql_query(query, engine)
                    # Buat SmartDataframe dengan konfigurasi LLM dan ResponseParser
                    response_parser = StreamlitResponse(st)
                    sdf = SmartDataframe(df, name=f"public.{table}", 
                                          config={"llm": llm, "response_parser": response_parser})
                    sdf_list.append(sdf)
                    # Simpan metadata tabel
                    table_info[table] = {
                        "columns": list(df.columns),
                        "row_count": len(df)
                    }
                except Exception as e:
                    st.warning(f"Gagal memuat data dari public.{table}: {e}")

            # Buat SmartDatalake dari list SmartDataframe
            datalake = SmartDatalake(sdf_list, config={"llm": llm, "response_parser": StreamlitResponse})
            return datalake, table_info, engine

    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None, None, None

# -----------------------------------------------------------------------------
# Caching dengan Pickle untuk data tabel dari database
# -----------------------------------------------------------------------------
def load_database_cache(credentials, cache_path="db_cache.pkl"):
    cache_file = Path(cache_path)
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                datalake, table_info = pickle.load(f)
            return datalake, table_info
        except Exception as e:
            st.warning(f"Gagal memuat cache: {e}. Memuat ulang data dari database.")
    datalake, table_info, engine = validate_and_connect_database(credentials)
    if datalake is not None and table_info is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump((datalake, table_info), f)
        except Exception as e:
            st.warning(f"Gagal menyimpan cache: {e}")
    return datalake, table_info

# -----------------------------------------------------------------------------
# Tampilan Utama dan Logika Chat Database
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    
    # Inisialisasi session state
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Layout dengan sidebar dan main content
    col1, col2 = st.columns([1, 3])

    with col1:
        # Sidebar untuk kredensial database
        st.header("🔐 Database Credentials")
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        connect_button = st.button("Connect to Database")

        # Tampilkan tabel yang dimuat
        if st.session_state.database_loaded:
            st.header("📊 Loaded Tables")
            for table, info in st.session_state.table_info.items():
                with st.expander(table):
                    st.write(f"Columns: {', '.join(info['columns'])}")
                    st.write(f"Row Count: {info['row_count']}")

    with col2:
        st.title("🔍 Smart Database Explorer")

        # Proses koneksi database
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
                st.success("Database connected successfully!")

        # Chat interface
        if st.session_state.get("database_loaded", False):
            st.header("💬 Database Chat")

            # Tampilkan riwayat pesan
            for message in st.session_state.messages:
                with st.chat_message(message["role"], 
                                     avatar="🤖" if message["role"] == "assistant" else "👤"):
                    # Tangani berbagai jenis output
                    if "image" in message:
                        st.image(message["image"])
                    if "text" in message:
                        st.markdown(message["text"])

            # Input chat
            if prompt := st.chat_input("Ask a question about your data"):
                # Tambahkan pesan pengguna
                st.session_state.messages.append({
                    "role": "user", 
                    "text": prompt
                })

                # Tampilkan pesan pengguna
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)

                # Siapkan response parser khusus
                response_parser = StreamlitResponse(st)

                # Proses chat dengan datalake
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Generating response..."):
                        try:
                            # Jalankan chat dan dapatkan respons
                            answer = st.session_state.datalake.chat(prompt)

                            # Tambahkan respons ke daftar pesan
                            message_entry = {"role": "assistant"}

                            # Tangani respons gambar
                            if response_parser.image_response is not None:
                                message_entry["image"] = response_parser.image_response
                                st.image(response_parser.image_response)

                            # Tangani respons teks
                            if response_parser.text_response is not None:
                                message_entry["text"] = response_parser.text_response
                                st.markdown(response_parser.text_response)

                            # Tambahkan ke riwayat pesan
                            st.session_state.messages.append(message_entry)

                        except Exception as e:
                            st.error(f"Error dalam memproses chat: {e}")

if __name__ == "__main__":
    main()
