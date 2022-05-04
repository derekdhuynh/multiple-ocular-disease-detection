import sqlite3
import os

con = sqlite3.connect(
    '../instance/ocularnn.sqlite',
    detect_types=sqlite3.PARSE_DECLTYPES
)
con.row_factory = sqlite3.Row
# cur = con.cursor()

#rows = con.execute('SELECT * FROM user').fetchall()

user_info = con.execute(
    "SELECT * FROM user WHERE username=?",
    ("derek",)
).fetchone()

print(tuple(user_info))
