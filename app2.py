import os
import pickle
from pathlib import Path

from PIL import Image
import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import PermissionError, SQLAlchemyError
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser

# [Previous code for StreamlitResponse remains the same]

def validate_and_connect_database(credentials):
    # [Previous implementation remains the same]

def main():
    st.set_page_config(page_title="Smart Database Explorer", layout="wide")
    
    # Inisialisasi session state
    if 'database_loaded' not in st.session_state:
        st.session_state.database_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar for Database Credentials
    with st.sidebar:
        st.header("🔐 Database Credentials")
        
        # Input kredensial database
        db_user = st.text_input("PostgreSQL Username", key="db_user")
        db_password = st.text_input("PostgreSQL Password", type="password", key="db_password")
        db_host = st.text_input("PostgreSQL Host", value="localhost", key="db_host")
        db_port = st.text_input("PostgreSQL Port", value="5432", key="db_port")
        db_name = st.text_input("Database Name", key="db_name")
        groq_api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
        connect_button = st.button("Connect to Database")

        # Tampilkan informasi tabel yang dimuat
        if st.session_state.get("database_loaded", False):
            st.header("📊 Table Information")
            
            # Tambahkan expander untuk tabel yang berhasil dimuat
            st.subheader("Accessible Tables")
            loaded_tables = [table for table, info in st.session_state.table_info.items() if info['sample_loaded']]
            if loaded_tables:
                for table in loaded_tables:
                    with st.expander(table):
                        info = st.session_state.table_info[table]
                        st.write(f"Columns: {', '.join(info['columns'])}")
                        st.write(f"Row Count: {info['row_count']}")
            else:
                st.warning("No tables could be loaded.")
            
            # Tambahkan expander untuk tabel dengan permission error
            st.subheader("Tables with Access Issues")
            failed_tables = [table for table, info in st.session_state.table_info.items() if not info['sample_loaded']]
            if failed_tables:
                for table in failed_tables:
                    with st.expander(f"❌ {table}"):
                        st.error(st.session_state.table_info[table]['error'])
            else:
                st.success("All tables loaded successfully!")

    # Main content area
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
            datalake, table_info, engine, permission_errors = validate_and_connect_database(credentials)

        if datalake and table_info:
            st.session_state.datalake = datalake
            st.session_state.table_info = table_info
            st.session_state.database_loaded = True
            
            # Berikan feedback tentang koneksi
            if permission_errors:
                st.warning(f"Database connected with {len(permission_errors)} table access issues.")
            else:
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
                if "dataframe" in message:
                    st.dataframe(message["dataframe"])

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
                        # Reset response parser
                        response_parser.image_response = None
                        response_parser.text_response = None
                        response_parser.dataframe_response = None

                        # Jalankan chat dan dapatkan respons
                        answer = st.session_state.datalake.chat(prompt)

                        # Tangani respons menggunakan fungsi khusus
                        message_entry = handle_pandasai_response(answer, response_parser)
                        
                        # Tambahkan ke riwayat pesan
                        st.session_state.messages.append(message_entry)

                    except Exception as e:
                        st.error(f"Error dalam memproses chat: {e}")

if __name__ == "__main__":
    main()
