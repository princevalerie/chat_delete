import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sqlalchemy import create_engine, inspect
from langchain_groq import ChatGroq
from pandasai import SmartDataframe, SmartDatalake

def validate_and_connect_database(credentials):
    """Validasi koneksi ke database dan inisialisasi komponen-komponennya."""
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
            # Inisialisasi LLM menggunakan ChatGroq
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
                    
                    # Buat SmartDataframe
                    sdf = SmartDataframe(df, name=f"public.{table}")
                    sdf.config = {"llm": llm}
                    sdf_list.append(sdf)
                    
                    # Simpan metadata tabel
                    table_info[table] = {
                        'columns': list(df.columns),
                        'row_count': len(df)
                    }
                    
                except Exception as e:
                    print(f"Warning: Gagal memuat data dari public.{table}: {e}")
            
            # Buat SmartDatalake dari list SmartDataframe
            datalake = SmartDatalake(sdf_list, config={"llm": llm})
            
            return datalake, table_info, engine
    
    except Exception as e:
        print(f"Error: Database connection error: {e}")
        return None, None, None

def main():
    print("=== Smart Database Explorer ===")
    print("Masukkan kredensial database Anda.\n")
    
    db_user = input("PostgreSQL Username: ")
    db_password = input("PostgreSQL Password: ")
    db_host = input("PostgreSQL Host [default: localhost]: ") or "localhost"
    db_port = input("PostgreSQL Port [default: 5432]: ") or "5432"
    db_name = input("Database Name: ")
    groq_api_key = input("Groq API Key: ")
    
    credentials = {
        "DB_USER": db_user,
        "DB_PASSWORD": db_password,
        "DB_HOST": db_host,
        "DB_PORT": db_port,
        "DB_NAME": db_name,
        "GROQ_API_KEY": groq_api_key
    }
    
    print("\nMenghubungkan ke database dan memuat tabel...")
    datalake, table_info, engine = validate_and_connect_database(credentials)
    
    if datalake and table_info:
        print("\nKoneksi berhasil! Berikut adalah tabel yang telah dimuat:\n")
        for table, info in table_info.items():
            print(f"Table: {table}")
            print(f"  Columns  : {', '.join(info['columns'])}")
            print(f"  Row Count: {info['row_count']}")
            print("-" * 40)
    else:
        print("Gagal terhubung ke database.")
        return
    
    print("\n=== Chat Interface ===")
    print("Ketik 'exit' untuk keluar.\n")
    
    while True:
        prompt = input("Your question about the data: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break
        
        try:
            response = datalake.chat(prompt)
            # Tampilkan response secara langsung
            print("\nResponse:")
            print(response)
            print("-" * 40)
        except Exception as e:
            print(f"Error dalam memproses chat: {e}")
    
if __name__ == "__main__":
    main()
