import os
import oracledb
from dotenv import load_dotenv

load_dotenv()

conn = oracledb.connect(
    user=os.getenv("ORACLE_USER"),
    password=os.getenv("ORACLE_PWD"),
    dsn=os.getenv("ORACLE_DSN"),
)
cur = conn.cursor()
cur.execute("SELECT banner FROM v$version WHERE ROWNUM = 1")
print("Connecté !", cur.fetchone()[0])
conn.close()