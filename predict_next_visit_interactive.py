# ===============================================================
# 📘 predict_next_visit_interactive.py
# Dự đoán mã bệnh của lượt khám kế tiếp (ứng dụng demo)
# ===============================================================
import torch
import numpy as np
import json, pickle
from train_dipole import Dipole, visits_to_multihot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load model và vocab đã huấn luyện ===
CKPT_PATH = "dipole_best.pt"
SEQ_PATH = "preprocessing/seqs.pkl"
VOCAB_PATH = "preprocessing/vocab.json"

print("📦 Loading model and vocab...")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
with open(VOCAB_PATH) as f:
    code2idx = json.load(f)
idx2code = {v: k for k, v in code2idx.items()}
V = len(code2idx)

model = Dipole(V).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print("✅ Model loaded with vocab size:", V)

# === Hàm dự đoán từ mã ICD đầu vào ===
def predict_next_visit(input_codes, topk=10):
    """
    input_codes: list[str] — danh sách mã ICD của lượt khám hiện tại
    """
    # Chuyển thành tensor [1, T=1, V]
    X = visits_to_multihot([input_codes], code2idx)
    X = torch.tensor(np.stack(X)).unsqueeze(0).float().to(DEVICE)
    L = torch.tensor([1]).long().to(DEVICE)

    with torch.no_grad():
        logits = model(X, L)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    top_idx = np.argsort(-probs)[:topk]
    pred_codes = [(idx2code[i], probs[i]) for i in top_idx]
    return pred_codes

# === Vòng lặp tương tác ===
print("\n🩺 Nhập danh sách mã ICD (phân tách bằng dấu phẩy), ví dụ:")
print("4280, 4019, 25000  →  Dự đoán lượt khám kế tiếp\n")

while True:
    try:
        line = input("👉 Nhập mã bệnh (hoặc 'q' để thoát): ").strip()
        if line.lower() in ["q", "quit", "exit"]:
            break
        input_codes = [x.strip() for x in line.split(",") if x.strip()]
        if not input_codes:
            continue

        print(f"🔍 Dự đoán lượt khám kế tiếp cho: {input_codes}")
        preds = predict_next_visit(input_codes, topk=10)
        print("🧠 Top-10 mã dự đoán:")
        for c, p in preds:
            print(f"   {c:>8} : {p:.3f}")
        print("-" * 50)
    except KeyboardInterrupt:
        break
