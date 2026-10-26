"""Melatih model Regresi Logistik untuk prediksi pembelian asuransi perjalanan.
Jalankan:  python train_model.py
Output  :  model/model.joblib, model/scaler.joblib, model/metrics.json, static/img/*.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score)

NAVY, TEAL, ORANGE, GREY = "#0f2a47", "#12a594", "#f08a24", "#8a97a6"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GREY,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.titleweight": "bold", "axes.titlecolor": NAVY})

# 1. Muat data -------------------------------------------------------------
df = pd.read_csv("data/TravelInsurancePrediction.csv").drop(columns=["Unnamed: 0"])
print("Ukuran data:", df.shape, "| missing value:", int(df.isna().sum().sum()))

# 2. Pra-pemrosesan --------------------------------------------------------
# Kategorikal biner -> angka 0/1
df["Employment Type"] = (df["Employment Type"] == "Private Sector/Self Employed").astype(int)
for c in ["GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]:
    df[c] = (df[c] == "Yes").astype(int)
df = df.rename(columns={"Employment Type": "PrivateSector"})

FITUR = ["Age", "AnnualIncome", "FamilyMembers", "ChronicDiseases",
         "PrivateSector", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]
NUMERIK = ["Age", "AnnualIncome", "FamilyMembers"]
X, y = df[FITUR], df["TravelInsurance"]

# 3. Cek keseimbangan kelas ------------------------------------------------
dist = y.value_counts().sort_index()
pct_beli = dist[1] / len(y) * 100
print(f"Kelas 0: {dist[0]} | Kelas 1: {dist[1]} ({pct_beli:.1f}%) -> cukup seimbang, SMOTE tidak dipakai")

fig, ax = plt.subplots(figsize=(5, 3.6))
b = ax.bar(["Tidak Beli (0)", "Beli (1)"], dist.values, color=[GREY, TEAL], width=.55)
for r in b: ax.text(r.get_x()+r.get_width()/2, r.get_height()+15, int(r.get_height()), ha="center", color=NAVY, fontweight="bold")
ax.set_title("Distribusi Target"); ax.set_ylabel("Jumlah pelanggan")
fig.tight_layout(); fig.savefig("static/img/distribusi.png", dpi=160, transparent=False, facecolor="white"); plt.close()

# 4. Split + scaling -------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_tr[NUMERIK])
def prep(d):
    d = d.copy(); d[NUMERIK] = scaler.transform(d[NUMERIK]); return d
X_tr_s, X_te_s = prep(X_tr), prep(X_te)

# 5. Bandingkan dua konfigurasi (tanpa dan dengan class_weight) -------------
hasil = {}
for nama, cw in [("default", None), ("balanced", "balanced")]:
    m = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42).fit(X_tr_s, y_tr)
    p = m.predict(X_te_s)
    hasil[nama] = (m, recall_score(y_te, p), f1_score(y_te, p))
    print(f"{nama:9s} recall={hasil[nama][1]:.3f}  f1={hasil[nama][2]:.3f}")
pilih = "balanced" if hasil["balanced"][1] > hasil["default"][1] else "default"
model = hasil[pilih][0]
print("Model dipilih:", pilih)

# 6. Evaluasi --------------------------------------------------------------
pred = model.predict(X_te_s); proba = model.predict_proba(X_te_s)[:, 1]
cm = confusion_matrix(y_te, pred); tn, fp, fn, tp = cm.ravel()
auc = roc_auc_score(y_te, proba)
cv_auc = cross_val_score(LogisticRegression(max_iter=1000, class_weight=model.class_weight),
                         prep(X), y, cv=5, scoring="roc_auc")
print(classification_report(y_te, pred, target_names=["Tidak Beli", "Beli"]))
print("AUC:", round(auc, 3), "| CV AUC:", cv_auc.round(3), cv_auc.mean().round(3))

fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.imshow(cm, cmap="GnBu")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, v, ha="center", va="center", fontsize=18, fontweight="bold", color="white" if v > cm.max()/2 else NAVY)
ax.set_xticks([0, 1], ["Tidak Beli", "Beli"]); ax.set_yticks([0, 1], ["Tidak Beli", "Beli"])
ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Confusion Matrix")
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("static/img/confusion_matrix.png", dpi=160, facecolor="white"); plt.close()

fpr, tpr, _ = roc_curve(y_te, proba)
fig, ax = plt.subplots(figsize=(4.6, 3.8))
ax.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"AUC = {auc:.2f}")
ax.plot([0, 1], [0, 1], "--", color=GREY, label="Tebakan acak")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("Kurva ROC")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("static/img/roc.png", dpi=160, facecolor="white"); plt.close()

# Koefisien (pengaruh fitur) — fitur numerik sudah distandarisasi, jadi bisa dibandingkan
koef = pd.Series(model.coef_[0], index=FITUR).sort_values()
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.barh(koef.index, koef.values, color=[ORANGE if v < 0 else TEAL for v in koef.values])
ax.axvline(0, color=GREY, lw=1); ax.set_title("Pengaruh Fitur (Koefisien)"); ax.set_xlabel("Koefisien")
fig.tight_layout(); fig.savefig("static/img/koefisien.png", dpi=160, facecolor="white"); plt.close()

# 7. Simpan ---------------------------------------------------------------
joblib.dump(model, "model/model.joblib"); joblib.dump(scaler, "model/scaler.joblib")
metrics = {
    "n_data": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    "pct_beli": round(pct_beli, 1), "model": "LogisticRegression" + (" (class_weight=balanced)" if pilih == "balanced" else ""),
    "accuracy": round(accuracy_score(y_te, pred), 3), "precision": round(precision_score(y_te, pred), 3),
    "recall": round(recall_score(y_te, pred), 3), "f1": round(f1_score(y_te, pred), 3),
    "auc": round(auc, 3), "cv_auc": round(cv_auc.mean(), 3),
    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    "fitur": FITUR, "numerik": NUMERIK,
    "koef": {k: round(float(v), 3) for k, v in koef.items()},
}
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k not in ("koef", "fitur")}, indent=1))
