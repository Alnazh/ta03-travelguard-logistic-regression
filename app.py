"""Aplikasi web Flask: Prediksi Pembelian Asuransi Perjalanan (Regresi Logistik)."""
import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Muat model, scaler, dan metrik hasil pelatihan
model = joblib.load(os.path.join(BASE, "model", "model.joblib"))
scaler = joblib.load(os.path.join(BASE, "model", "scaler.joblib"))
with open(os.path.join(BASE, "model", "metrics.json")) as f:
    METRICS = json.load(f)

FITUR, NUMERIK = METRICS["fitur"], METRICS["numerik"]
_df = pd.read_csv(os.path.join(BASE, "data", "TravelInsurancePrediction.csv"))
RANGE = {  # rentang data latih, dipakai untuk petunjuk & peringatan input
    "Age": (int(_df.Age.min()), int(_df.Age.max())),
    "AnnualIncome": (int(_df.AnnualIncome.min()), int(_df.AnnualIncome.max())),
    "FamilyMembers": (int(_df.FamilyMembers.min()), int(_df.FamilyMembers.max())),
}
NAMA = {
    "Age": "Usia", "AnnualIncome": "Pendapatan tahunan", "FamilyMembers": "Jumlah anggota keluarga",
    "ChronicDiseases": "Penyakit kronis", "PrivateSector": "Bekerja di sektor swasta/wiraswasta",
    "GraduateOrNot": "Lulusan perguruan tinggi", "FrequentFlyer": "Sering terbang",
    "EverTravelledAbroad": "Pernah ke luar negeri",
}
DEFAULT = {"Age": 30, "AnnualIncome": 900000, "FamilyMembers": 4, "ChronicDiseases": 0,
           "PrivateSector": 1, "GraduateOrNot": 1, "FrequentFlyer": 0, "EverTravelledAbroad": 0}


def parse_form(form):
    """Ambil & validasi input. Mengembalikan (nilai, daftar_error)."""
    nilai, err = {}, []
    batas = {"Age": (17, 100), "AnnualIncome": (0, 100_000_000), "FamilyMembers": (1, 20)}
    for k in NUMERIK:
        try:
            v = float(form.get(k, ""))
        except ValueError:
            err.append(f"{NAMA[k]} harus berupa angka."); continue
        lo, hi = batas[k]
        if not lo <= v <= hi:
            err.append(f"{NAMA[k]} harus antara {lo:,} dan {hi:,}.".replace(",", "."))
        nilai[k] = v
    for k in ["ChronicDiseases", "PrivateSector", "GraduateOrNot", "FrequentFlyer", "EverTravelledAbroad"]:
        v = form.get(k)
        if v not in ("0", "1"):
            err.append(f"Pilih salah satu jawaban untuk: {NAMA[k]}.")
        else:
            nilai[k] = int(v)
    return nilai, err


def prediksi(nilai):
    x = pd.DataFrame([nilai])[FITUR]
    x[NUMERIK] = scaler.transform(x[NUMERIK])
    p = float(model.predict_proba(x)[0, 1])
    # kontribusi tiap fitur = koefisien x nilai (sudah distandarisasi)
    kontribusi = {f: float(model.coef_[0][i] * x.iloc[0][f]) for i, f in enumerate(FITUR)}
    urut = sorted(kontribusi.items(), key=lambda t: t[1])
    naik = [(NAMA[f], c) for f, c in reversed(urut) if c > 0.05][:3]
    turun = [(NAMA[f], c) for f, c in urut if c < -0.05][:3]
    return p, naik, turun


@app.route("/", methods=["GET", "POST"])
def index():
    hasil, errors, form_val = None, [], dict(DEFAULT)
    if request.method == "POST":
        nilai, errors = parse_form(request.form)
        form_val.update({k: (int(v) if float(v).is_integer() else v) for k, v in nilai.items()})
        if not errors:
            p, naik, turun = prediksi(nilai)
            luar = [NAMA[k] for k in NUMERIK if not RANGE[k][0] <= nilai[k] <= RANGE[k][1]]
            hasil = {
                "berpotensi": p >= 0.5, "persen": round(p * 100, 1),
                "label": "Berpotensi Membeli" if p >= 0.5 else "Kurang Berpotensi Membeli",
                "naik": naik, "turun": turun, "luar_rentang": luar,
            }
    return render_template("index.html", hasil=hasil, errors=errors, f=form_val,
                           m=METRICS, rng=RANGE, scroll=request.method == "POST")


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
