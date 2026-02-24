import gzip
import pandas as pd
import os

def is_gzipped(filename):
    with open(filename, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'

# List all file paths (add all your files here)
file_paths = [
    r"C:\Users\rgolande\Downloads\part-00000-of-00500.csv.gz",
    r"C:\Users\rgolande\Downloads\part-00001-of-00500.csv.gz",
    r"C:\Users\rgolande\Downloads\task_usage_part-00002-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00003-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00004-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00005-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00006-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00007-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00008-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00009-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00010-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00011-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00012-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00013-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00014-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00015-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00016-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00017-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00018-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00019-of-00500.csv",
    r"C:\Users\rgolande\Downloads\task_usage_part-00020-of-00500.csv"
]

# According to the schema, assign column names
col_names = [
    "start_time", "end_time", "job_id", "task_index", "machine_id",
    "cpu_rate", "canonical_memory_usage", "assigned_memory_usage", "unmapped_page_cache",
    "total_page_cache", "maximum_memory_usage", "disk_io_time", "local_disk_space_usage",
    "maximum_cpu_rate", "maximum_disk_io_time", "cycles_per_instruction",
    "memory_accesses_per_instruction", "sample_portion", "aggregation_type",
    "sampled_cpu_usage"
]

frames = []
# Read all rows from each shard (no nrows limit)
for fp in file_paths:
    if os.path.exists(fp):
        if fp.endswith('.gz') or is_gzipped(fp):
            with gzip.open(fp, 'rt') as f:
                frames.append(pd.read_csv(f, names=col_names))
        else:
            frames.append(pd.read_csv(fp, names=col_names))
    else:
        print(f"File not found: {fp}")

df = pd.concat(frames, ignore_index=True)
print(f"Total raw rows loaded: {len(df)}")

# Convert start_time from microseconds to datetime
if 'start_time' in df.columns:
    df['start_time'] = pd.to_datetime(df['start_time'], unit='us')
    df.set_index('start_time', inplace=True)
    # Per-minute aggregation
    per_minute = df.resample('1T').agg({
        'cpu_rate': ['mean', 'max'],
        'canonical_memory_usage': 'mean'
    })
    print("\nPer-minute aggregated stats (first 5):")
    print(per_minute.head())
    # Save to CSV in the repo
    output_path = os.path.join(os.path.dirname(__file__), 'per_minute_agg.csv')
    per_minute.to_csv(output_path)
    print(f"\nPer-minute aggregated data saved to: {output_path}")
    print(f"\nTotal per-minute rows: {len(per_minute)}")
else:
    print("\n'start_time' column not found. Please check the schema.")

# For very large files, use pd.read_csv(..., chunksize=100000) and process in batches.
