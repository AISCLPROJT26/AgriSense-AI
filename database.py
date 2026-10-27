import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "agrisense.db")


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    connection = get_db()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            guest_name TEXT,
            activity TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_history(guest_name, activity, result):
    connection = get_db()

    try:
        connection.execute("""
            INSERT INTO history
            (guest_name, activity, result)
            VALUES (?, ?, ?)
        """, (
            guest_name,
            activity,
            result
        ))

        connection.commit()
    finally:
        connection.close()