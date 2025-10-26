# ===============================================================
# 📘 process_mtgan_data.py
# Tiền xử lý dữ liệu MTGAN Synthetic MIMIC-III
# ===============================================================
import pandas as pd
import numpy as np
import json, pickle
from collections import defaultdict, Counter
from pathlib import Path

INPUT_CSV = "Dipole-master/preprocessing/decoded_synthetic_mimic3.csv"
OUT_SEQ = "preprocessing/seqs.pkl"
OUT_VOCAB = "preprocessing/vocab.json"

print("📂 Loading:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)
print("Shape:", df.shape)

# Gom lượt khám cho từng bệnh nhân
df = df.sort_values(["patient_id", "admission_index"])
seqs = defaultdict(list)
for pid, g in df.groupby("patient_id"):
    visits = []
    for _, row in g.iterrows():
        codes = [c.strip() for c in str(row["codes"]).split(",") if c.strip()]
        visits.append(codes)
    seqs[int(pid)] = visits

# Tạo vocab
counter = Counter()
for visits in seqs.values():
    for codes in visits:
        counter.update(codes)
vocab = [c for c, _ in counter.most_common()]
code2idx = {c: i for i, c in enumerate(vocab)}

print(f"✅ Patients: {len(seqs)} | Vocab size: {len(vocab)}")

# Lưu dữ liệu
with open(OUT_SEQ, "wb") as f:
    pickle.dump(dict(seqs), f)
with open(OUT_VOCAB, "w") as f:
    json.dump(code2idx, f, indent=2)

print("✅ Saved:")
print(" - Sequences:", OUT_SEQ)
print(" - Vocab:", OUT_VOCAB)
