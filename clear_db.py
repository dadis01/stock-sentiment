from database.db import get_connection
conn = get_connection()
cur = conn.cursor()
cur.execute("TRUNCATE TABLE sentiments, articles, prices RESTART IDENTITY CASCADE;")
conn.commit()
print("Database cleared.")
conn.close()
