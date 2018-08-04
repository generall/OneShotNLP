import sqlite3
import sys

conn = sqlite3.connect(sys.argv[2])

cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS mentions')

cur.execute("""
CREATE VIRTUAL TABLE mentions USING fts3(
    concept STRING,
    left_context STRING,
    mention TEXT,
    right_context STRING
)
""")

with open(sys.argv[1]) as fd:
    cur.execute("begin")
    for line in fd:
        concept, left_context, mention, right_context = line.strip('\n').split('\t')
        query = 'insert into mentions (concept, left_context, mention, right_context) values (?, ?, ?, ?)'
        cur.execute(query, (concept, left_context, mention, right_context))
    cur.execute("commit")

cur.execute("rollback")