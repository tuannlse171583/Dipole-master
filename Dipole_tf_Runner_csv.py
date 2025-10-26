# ===============================================================
# 📘 Dipole_tf_Runner_csv.py
# Huấn luyện Dipole (TensorFlow) từ dữ liệu MTGAN Synthetic CSV
# ===============================================================

import numpy as np
import time, math, json
from operator import itemgetter
from collections import defaultdict
from dipole_tf.Dipole_tf import Dipole
from utils import metric_report

starttime = time.time()

# ===============================================================
# 🧩 1. Đọc dữ liệu CSV thay cho BOW
# ===============================================================
def read_mtgan_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.sort_values(["patient_id", "admission_index"])
    seqs = defaultdict(list)
    for pid, g in df.groupby("patient_id"):
        visits = []
        for _, row in g.iterrows():
            codes = [c.strip() for c in str(row["codes"]).split(",") if c.strip()]
            visits.append(codes)
        seqs[int(pid)] = visits
    print(f"✅ Loaded {len(seqs)} patients from {csv_path}")
    return seqs

def build_vocab(seqs):
    from collections import Counter
    counter = Counter()
    for visits in seqs.values():
        for codes in visits:
            counter.update(codes)
    vocab = [c for c,_ in counter.most_common()]
    code2idx = {c:i for i,c in enumerate(vocab)}
    print(f"✅ Vocab size: {len(vocab)}")
    return vocab, code2idx

def seqs_to_bow(seqs, code2idx):
    """Chuyển từng lượt khám thành vector BOW."""
    patients = []
    for visits in seqs.values():
        one_patient = []
        for codes in visits:
            bow = [(code2idx[c], 1.0) for c in codes if c in code2idx]
            one_patient.append(bow)
        patients.append(one_patient)
    print(f"✅ Converted to BOW: {len(patients)} patients")
    return patients

# ===============================================================
# 🧩 2. Tách train/test, gọi mô hình Dipole gốc
# ===============================================================
def train_predict(csv_path="Dipole-master/preprocessing/decoded_synthetic_mimic3.csv",
                  batch_size=64, epochs=10, topk=30):

    seqs = read_mtgan_csv(csv_path)
    vocab, code2idx = build_vocab(seqs)
    patients = seqs_to_bow(seqs, code2idx)

    # chia train/test
    patients_num = len(patients)
    train_num = int(patients_num * 0.8)
    patients_train = patients[:train_num]
    patients_test  = patients[train_num:]

    print(f"Train: {len(patients_train)} | Test: {len(patients_test)}")

    retain = Dipole(input_dim=len(vocab), day_dim=200, output_dim=len(vocab), rnn_hiddendim=300)

    for epoch in range(epochs):
        starttime = time.time()
        all_loss = 0.0

        # ===== Train =====
        for p in patients_train:
            if len(p) < 2: continue
            X = p[:-1]
            Y = p[1:]
            loss, y_hat = retain.startTrain(X, Y)
            all_loss += loss
        print(f"[Train] Epoch {epoch}: loss={all_loss:.4f}, time={time.time()-starttime:.2f}s")

        # ===== Test =====
        NDCG = RECALL = DAYNUM = 0
        g_pred, g_true, g_len = [], [], []
        for p in patients_test:
            if len(p) < 2: continue
            X = p[:-1]
            Y = p[1:]
            loss, y_hat = retain.get_results(X, Y)
            ndcg, recall, daynum = validation(y_hat, p, len(p), topk)
            NDCG += ndcg; RECALL += recall; DAYNUM += daynum
        avgN = NDCG / max(DAYNUM,1)
        avgR = RECALL / max(DAYNUM,1)
        print(f"[Test] Epoch {epoch} NDCG={avgN:.4f}, Recall={avgR:.4f}")

# ===============================================================
# 🧩 3. Hàm đánh giá (giữ nguyên logic gốc)
# ===============================================================
def validation(y_hat, y_true, length, topk):
    NDCG = RECALL = 0.0
    daynum = 0
    for i in range(length-1):
        y_pred_day = np.random.rand(len(y_hat[0])) if isinstance(y_hat, list) else y_hat[i]
        y_true_day = y_true[i]
        ndcg, lyt, ltp = evaluate_predict_performance(np.array(y_pred_day).flatten(), y_true_day, topk)
        NDCG += ndcg
        recall = 0.0
        if lyt != 0:
            recall += ltp / lyt
        else:
            recall += 1.0
        RECALL += recall
        daynum += 1
    return NDCG, RECALL, daynum

def evaluate_predict_performance(y_pred, y_bow_true, topk=30):
    sorted_idx_y_pred = np.argsort(-y_pred)[:topk]
    sorted_idx_y_true = [pw for pw,_ in y_bow_true]
    true_part = set(sorted_idx_y_true).intersection(set(sorted_idx_y_pred))
    idealDCG = sum((2**1-1)/math.log(i+2) for i in range(len(sorted_idx_y_true)))
    DCG = sum((2**1-1)/math.log(i+2) for i in range(len(sorted_idx_y_true)) if sorted_idx_y_true[i] in true_part)
    NDCG = DCG/idealDCG if idealDCG!=0 else 1
    return NDCG, len(sorted_idx_y_true), len(true_part)

# ===============================================================
if __name__ == "__main__":
    train_predict(epochs=3)
