import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
conn = psycopg.connect(os.getenv("SUPABASE_DB_URL"))
cur = conn.cursor()
cur.execute(
    "select left(content, 180), metadata->>'source' "
    "from embeddings_agente_4_qualidade_leite limit 3"
)
print(cur.fetchall())
conn.close()
