# 🩺 VitalsCheck — Data Scientist Handoff

> Platform edukasi dan deteksi dini Penyakit Tidak Menular (PTM) berbasis prediksi risiko kesehatan.

---

## 👥 Tim Data Scientist

| Nama |
|------|
| Henokh William Christianos Lase |
| Nicholas Howard Gouwtama |

---

## 📌 Tentang Proyek

VitalsCheck adalah platform berbasis web yang memungkinkan pengguna memasukkan **13 variabel kesehatan** untuk mendapatkan prediksi risiko terhadap **5 Penyakit Tidak Menular (PTM)** utama:

| Target Label | Penyakit |
|---|---|
| `Diabetes_binary` | Diabetes |
| `HighBP` | Hipertensi (Tekanan Darah Tinggi) |
| `HighChol` | Kolesterol Tinggi |
| `Stroke` | Stroke |
| `HeartDiseaseorAttack` | Penyakit Jantung / Serangan Jantung |

---

## 🔄 Alur Pipeline Data

```
Sumber Data (BRFSS 2015)
        ↓
  Gathering Data
        ↓
  Assessing Data
  (deteksi missing values, duplikat, outlier)
        ↓
  Cleaning Data
  (hapus duplikat + winsorizing outlier BMI/MentHlth/PhysHlth)
        ↓
  Feature Selection
  (drop 4 kolom: AnyHealthcare, NoDocbcCost, Education, Income)
        ↓
  EDA (Univariate, Multivariate, Kategorikal & Numerikal)
        ↓
  Train-Test Split (MultilabelStratifiedShuffleSplit 80:20)
        ↓
  Scaling (MinMaxScaler — fit train only, transform test)
        ↓
  SMOTE (per label, train only)
        ↓
  📦 Export → diserahkan ke AI Engineer
```

---

## 📁 Struktur Folder

```
Proyek VitalsCheck/
├── data/
│   ├── diabetes_binary_health_indicators_BRFSS2015.csv   ← data mentah (sumber)
│   │
│   ├── X_test.csv                  ← fitur test set (shared semua model)
│   ├── y_test.csv                  ← label test set (shared semua model)
│   │
│   ├── X_train_smote_Diabetes_binary.csv
│   ├── y_train_smote_Diabetes_binary.csv
│   ├── X_train_smote_HighBP.csv
│   ├── y_train_smote_HighBP.csv
│   ├── X_train_smote_HighChol.csv
│   ├── y_train_smote_HighChol.csv
│   ├── X_train_smote_Stroke.csv
│   ├── y_train_smote_Stroke.csv
│   ├── X_train_smote_HeartDiseaseorAttack.csv
│   └── y_train_smote_HeartDiseaseorAttack.csv
│
├── dashboard/
│   ├── dashboard.py                ← Streamlit dashboard (EDA & visualisasi)
│   └── data_dashboard.csv          ← data untuk dashboard
│
├── Proyek_PTM.ipynb                ← notebook utama (full pipeline)
├── requirements.txt                ← dependensi Python
├── url_dashboard.txt               ← link dashboard live
└── README.md                       ← dokumen ini
```

---

## 📊 Detail File Data untuk AI Engineer

### Fitur Input (13 Variabel)

Semua file `X_train_smote_*.csv` dan `X_test.csv` memiliki **13 kolom** berikut:

| Kolom | Tipe | Keterangan |
|---|---|---|
| `CholCheck` | Binary (0/1) | Cek kolesterol dalam 5 tahun terakhir |
| `BMI` | Numerik (scaled) | Indeks Massa Tubuh — sudah di-MinMaxScale |
| `Smoker` | Binary (0/1) | Perokok ≥100 batang seumur hidup |
| `PhysActivity` | Binary (0/1) | Aktif fisik/olahraga dalam 30 hari terakhir |
| `Fruits` | Binary (0/1) | Konsumsi buah harian |
| `Veggies` | Binary (0/1) | Konsumsi sayur harian |
| `HvyAlcoholConsump` | Binary (0/1) | Konsumsi alkohol berat |
| `GenHlth` | Numerik (scaled) | Kondisi kesehatan umum (1=baik, 5=buruk) |
| `MentHlth` | Numerik (scaled) | Hari kesehatan mental buruk (30 hari) |
| `PhysHlth` | Numerik (scaled) | Hari kesehatan fisik buruk (30 hari) |
| `DiffWalk` | Binary (0/1) | Kesulitan berjalan/naik tangga |
| `Sex` | Binary (0/1) | Jenis kelamin (0=Perempuan, 1=Laki-laki) |
| `Age` | Numerik (scaled) | Kelompok usia (1=18-24 s.d. 13=80+) |

> ⚠️ **Catatan Scaling:** Kolom `BMI`, `GenHlth`, `MentHlth`, `PhysHlth`, dan `Age` sudah di-scaling menggunakan `MinMaxScaler` yang di-fit pada data train. Nilai di test set bisa sedikit keluar dari range [0, 1] — ini **normal dan benar**.

---

### Ukuran Dataset per Label

| Label | Baris X_train (setelah SMOTE) | Baris X_test |
|---|---|---|
| `Diabetes_binary` | ~311.002 | ~45.895 |
| `HighBP` | ~200.342 | ~45.895 |
| `HighChol` | ~205.006 | ~45.895 |
| `Stroke` | ~350.704 | ~45.895 |
| `HeartDiseaseorAttack` | ~329.218 | ~45.895 |

> SMOTE diterapkan per label untuk menyeimbangkan class (50:50). Setiap label memiliki X_train dan y_train tersendiri karena jumlah sampel berbeda.

---

### Cara Penggunaan File

**Untuk training 1 model per penyakit:**
```python
import pandas as pd

# Contoh untuk model Diabetes
X_train = pd.read_csv('data/X_train_smote_Diabetes_binary.csv')
y_train = pd.read_csv('data/y_train_smote_Diabetes_binary.csv').squeeze()

# Test set shared untuk semua model
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')['Diabetes_binary']
```

**Untuk prediksi input user baru (13 variabel):**
```python
from sklearn.preprocessing import MinMaxScaler
import joblib

# Kolom yang perlu di-scaling (urutan HARUS sama)
kolom_scale = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']



# Input user → pastikan urutan kolom sama dengan X_train
user_input = pd.DataFrame([{
    'CholCheck': 1, 'BMI': 27.5, 'Smoker': 0,
    'PhysActivity': 1, 'Fruits': 1, 'Veggies': 1,
    'HvyAlcoholConsump': 0, 'GenHlth': 2, 'MentHlth': 0,
    'PhysHlth': 0, 'DiffWalk': 0, 'Sex': 1, 'Age': 7
}])
user_input[kolom_scale] = scaler.transform(user_input[kolom_scale])
```

---

## ⚙️ Setup & Instalasi

```bash
# Install semua dependensi
pip install -r requirements.txt

# Jalankan notebook (pipeline lengkap)
jupyter notebook Proyek_PTM.ipynb

# Jalankan dashboard lokal
cd dashboard
streamlit run dashboard.py
```

### Dependensi Utama

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
iterative-stratification>=0.1.8
```

---

## 🌐 Dashboard Live

Akses dashboard EDA & visualisasi tanpa instalasi:

👉 **https://projectptm.streamlit.app/**

---

