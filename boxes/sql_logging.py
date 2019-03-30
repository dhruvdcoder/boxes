import sqlite3
from sqlite3 import Connection
from .learner import Recorder
from typing import List


def write_dict_(conn: Connection, table_name: str, d: dict):
    # Convert tensors to values we can write to SQL
    for k, v in d.items():
        if hasattr(v, 'dtype'):
            d[k] = v.item()

    c = conn.cursor()
    c.execute(f"insert into {table_name} ({','.join(d.keys())}) values ({':' +', :'.join(d.keys())})", d)
    conn.commit()
    return c.lastrowid # this is guaranteed to be the ID generated by the previous operation

def create_or_update_table_and_cols_(conn: Connection, table_name: str, cols: List[str]):
    c = conn.cursor()
    c.execute(f"create table if not exists {table_name} ({','.join(cols)})")
    for col in cols:
        try:
            c.execute(f"alter table {table_name} add column {col}")
        except:
            pass
    conn.commit()

def save_recorder_to_sql_(conn: Connection, rec: Recorder, table_name: str, training_instance_id: int):
    cols = [
        "[training_instance_id] INTEGER NOT NULL",
        "[epochs] REAL", # Note: we add this explicitly because it is actually the index label of the dataframe
    ]
    cols += [f"[{col}] REAL" for col in rec._data.columns]
    cols += ["FOREIGN KEY(training_instance_id) REFERENCES training_instance(id)"]
    create_or_update_table_and_cols_(conn, table_name, cols)

    rec.dataframe()["training_instance_id"] = training_instance_id
    rec.dataframe().to_sql(table_name, conn, if_exists="append", index_label="epochs")