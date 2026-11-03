"""One-off: columns of MBPP+ / HumanEval+ and their overlap with the practice set."""
import json

import pandas as pd

mp = pd.read_parquet("/opt/wsp/eval/raw/mbppplus.parquet")
hp = pd.read_parquet("/opt/wsp/eval/raw/humanevalplus.parquet")
for name, df in [("mbpp+", mp), ("humaneval+", hp)]:
    print(name, len(df), list(df.columns))
    row = df.iloc[0]
    for c in df.columns:
        print(f"   {c}: {str(row[c])[:160]!r}")

practice = [json.loads(l) for l in open("/opt/wsp/practice/practice.jsonl")]
mbpp_ids = {int(p["id"].split("/")[1]) for p in practice if p["source"] == "mbpp"}
plus_ids = set(int(x) for x in mp["task_id"])
print("mbpp+ ids:", min(plus_ids), "-", max(plus_ids), "| overlap with practice MBPP:", len(mbpp_ids & plus_ids),
      "of", len(mbpp_ids))
