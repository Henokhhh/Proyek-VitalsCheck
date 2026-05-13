import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Konfigurasi Halaman
st.set_page_config(
    page_title="VitalsCheck Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI LOAD DATA ---
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'data_dashboard.csv')
    df = pd.read_csv(csv_path)
    return df

# --- KONSTANTA ---
kolom_penyakit = {
    'Diabetes_binary':      'Diabetes',
    'HighBP':               'Hipertensi',
    'HighChol':             'Kolesterol Tinggi',
    'Stroke':               'Stroke',
    'HeartDiseaseorAttack': 'Penyakit Jantung'
}

kolom_faktor_binary = ['CholCheck', 'Smoker', 'PhysActivity', 'Fruits',
                       'Veggies', 'HvyAlcoholConsump', 'DiffWalk', 'Sex']

kolom_faktor_numerikal = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']

label_faktor = {
    'CholCheck'         : 'Cek Kolesterol',
    'BMI'               : 'Indeks Massa Tubuh (BMI)',
    'Smoker'            : 'Perokok',
    'PhysActivity'      : 'Aktif Fisik/Olahraga',
    'Fruits'            : 'Konsumsi Buah Harian',
    'Veggies'           : 'Konsumsi Sayur Harian',
    'HvyAlcoholConsump' : 'Konsumsi Alkohol Berat',
    'GenHlth'           : 'Kondisi Kesehatan Umum',
    'MentHlth'          : 'Hari Kesehatan Mental Buruk',
    'PhysHlth'          : 'Hari Kesehatan Fisik Buruk',
    'DiffWalk'          : 'Kesulitan Berjalan',
    'Sex'               : 'Jenis Kelamin',
    'Age'               : 'Kelompok Usia'
}

usia_label = {
    1:'18-24', 2:'25-29', 3:'30-34', 4:'35-39', 5:'40-44',
    6:'45-49', 7:'50-54', 8:'55-59', 9:'60-64', 10:'65-69',
    11:'70-74', 12:'75-79', 13:'80+'
}

genhlth_label = {1:'Sangat Baik', 2:'Baik', 3:'Cukup', 4:'Buruk', 5:'Sangat Buruk'}

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("🩺 VitalsCheck")
st.sidebar.markdown("Deteksi Dini & Analisis Penyakit Tidak Menular (PTM)")
menu = st.sidebar.radio("Navigasi Menu", ["Beranda & EDA", "Simulasi Prediksi PTM"])


# ============================================================
# MENU 1: BERANDA & EDA
# ============================================================
if menu == "Beranda & EDA":
    st.title("📊 Beranda & Eksplorasi Data")
    st.markdown("""
    Dashboard ini menampilkan ringkasan data Penyakit Tidak Menular (PTM) 
    dari indikator kesehatan BRFSS, serta hubungan antar faktor risiko kesehatan.
    """)

    try:
        df = load_data()

        # ── Key Metrics ──────────────────────────────────────────
        st.subheader("Ringkasan Data")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Responden", f"{len(df):,}")
        col2.metric("Fitur Kesehatan", "13 Variabel")
        col3.metric("Target Prediksi", "5 Penyakit")

        st.markdown("---")

        # ── Distribusi 5 Penyakit ────────────────────────────────
        st.subheader("Distribusi 5 Penyakit Utama")
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        colors_disease = ['#2ECC71', '#E74C3C']

        for i, (col, nama) in enumerate(kolom_penyakit.items()):
            ax = axes[i]
            counts = df[col].value_counts().sort_index()
            bars = ax.bar(['Tidak', 'Ya'], counts.values, color=colors_disease)
            ax.set_title(nama, fontsize=12, fontweight='bold')
            ax.set_ylim(0, max(counts.values) * 1.2)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:,}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # ── Cuplikan Data ────────────────────────────────────────
        st.subheader("Cuplikan Dataset Bersih")
        st.dataframe(df.head())

        st.markdown("---")

        # ============================================================
        # EDA UNIVARIATE
        # ============================================================
        st.header("🔍 EDA Univariate")

        # -- Tab navigasi EDA --
        tab1, tab2, tab3 = st.tabs([
            "📈 Faktor Numerikal",
            "📊 Faktor Binary",
            "🎂 Distribusi Usia & Jenis Kelamin"
        ])

        # ── Tab 1: Faktor Numerikal ──────────────────────────────
        with tab1:
            st.subheader("Statistik Deskriptif Faktor Numerikal")

            # Tabel statistik
            stats_rows = []
            for col in kolom_faktor_numerikal:
                stats_rows.append({
                    'Faktor': label_faktor[col],
                    'Min':    round(df[col].min(), 1),
                    'Rata-rata': round(df[col].mean(), 2),
                    'Median': round(df[col].median(), 1),
                    'Max':    round(df[col].max(), 1),
                    'Std':    round(df[col].std(), 2),
                })
            st.dataframe(pd.DataFrame(stats_rows).set_index('Faktor'), use_container_width=True)

            st.markdown("#### Distribusi per Faktor Numerikal")
            fig, axes = plt.subplots(1, 5, figsize=(22, 4))
            palette_num = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6']
            for i, col in enumerate(kolom_faktor_numerikal):
                ax = axes[i]
                ax.hist(df[col], bins=20, color=palette_num[i], alpha=0.85, edgecolor='white')
                ax.set_title(label_faktor[col], fontsize=9, fontweight='bold')
                ax.set_xlabel(col, fontsize=8)
                ax.set_ylabel('Frekuensi', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Tab 2: Faktor Binary ─────────────────────────────────
        with tab2:
            st.subheader("Proporsi Faktor Binary (% = Ya)")

            rows = []
            for col in kolom_faktor_binary:
                pct = df[col].mean() * 100
                rows.append({'Faktor': label_faktor[col], '% Ya': round(pct, 1), '% Tidak': round(100 - pct, 1)})
            df_bin = pd.DataFrame(rows).set_index('Faktor')
            st.dataframe(df_bin, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            y_pos = range(len(kolom_faktor_binary))
            labels_bin = [label_faktor[c] for c in kolom_faktor_binary]
            pcts = [df[c].mean() * 100 for c in kolom_faktor_binary]
            bars = ax.barh(labels_bin, pcts, color='#3498DB', alpha=0.85)
            ax.set_xlabel('Persentase (% = Ya)', fontsize=10)
            ax.set_xlim(0, 110)
            for bar, pct in zip(bars, pcts):
                ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}%', va='center', fontsize=9)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_title('Proporsi Faktor Binary di Seluruh Responden', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Tab 3: Usia & Jenis Kelamin ──────────────────────────
        with tab3:
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Distribusi Kelompok Usia")
                usia_dist = df['Age'].value_counts().sort_index()
                usia_labels = [usia_label[int(k)] for k in usia_dist.index]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(usia_labels, usia_dist.values, color='#3498DB', alpha=0.85)
                ax.set_xlabel('Kelompok Usia')
                ax.set_ylabel('Jumlah Responden')
                ax.set_title('Distribusi Kelompok Usia', fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_b:
                st.subheader("Distribusi Jenis Kelamin")
                sex_counts = df['Sex'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(
                    sex_counts.values,
                    labels=['Perempuan', 'Laki-laki'],
                    colors=['#E91E8C', '#3498DB'],
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops=dict(edgecolor='white', linewidth=2)
                )
                ax.set_title('Distribusi Jenis Kelamin', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        st.markdown("---")

        # ============================================================
        # EDA MULTIVARIATE
        # ============================================================
        st.header("🔗 EDA Multivariate")

        tab_mv1, tab_mv2 = st.tabs([
            "🌡️ Heatmap Korelasi",
            "📐 Korelasi per Penyakit"
        ])

        kolom_faktor = ['CholCheck', 'BMI', 'Smoker', 'PhysActivity', 'Fruits',
                        'Veggies', 'HvyAlcoholConsump', 'GenHlth', 'MentHlth',
                        'PhysHlth', 'DiffWalk', 'Sex', 'Age']

        # ── Tab Heatmap ──────────────────────────────────────────
        with tab_mv1:
            col_h1, col_h2 = st.columns(2)

            with col_h1:
                st.subheader("Korelasi Faktor vs 5 Penyakit")
                corr_matrix = pd.DataFrame()
                for col in kolom_penyakit.keys():
                    corr_matrix[kolom_penyakit[col]] = df[kolom_faktor].corrwith(df[col])
                corr_matrix.index = [label_faktor[f] for f in kolom_faktor]

                fig, ax = plt.subplots(figsize=(8, 7))
                sns.heatmap(
                    corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    center=0, linewidths=0.5, ax=ax,
                    annot_kws={'size': 8}, cbar_kws={'shrink': 0.8}
                )
                ax.set_title('Korelasi Faktor Risiko terhadap Penyakit', fontweight='bold', pad=10)
                plt.xticks(rotation=30, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_h2:
                st.subheader("Korelasi Antar 5 Penyakit")
                corr_penyakit = df[list(kolom_penyakit.keys())].corr()
                corr_penyakit.index   = list(kolom_penyakit.values())
                corr_penyakit.columns = list(kolom_penyakit.values())

                fig, ax = plt.subplots(figsize=(6, 5))
                mask = np.triu(np.ones_like(corr_penyakit, dtype=bool))
                sns.heatmap(
                    corr_penyakit, annot=True, fmt='.3f',
                    cmap='Blues', mask=mask, linewidths=0.5,
                    ax=ax, annot_kws={'size': 9}, cbar_kws={'shrink': 0.8}
                )
                ax.set_title('Korelasi Antar Penyakit', fontweight='bold', pad=10)
                plt.xticks(rotation=30, ha='right', fontsize=8)
                plt.yticks(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # ── Tab Korelasi per Penyakit ────────────────────────────
        with tab_mv2:
            st.subheader("Faktor Risiko & Pelindung per Penyakit")
            penyakit_pilihan = st.selectbox(
                "Pilih Penyakit",
                list(kolom_penyakit.keys()),
                format_func=lambda x: kolom_penyakit[x]
            )

            corr = df[kolom_faktor].corrwith(df[penyakit_pilihan]).sort_values(ascending=False)
            labels_corr = [label_faktor[f] for f in corr.index]
            colors_corr = ['#E74C3C' if v >= 0 else '#3498DB' for v in corr.values]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(labels_corr, corr.values, color=colors_corr, alpha=0.85)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xlabel('Nilai Korelasi')
            ax.set_title(f'Korelasi Faktor terhadap {kolom_penyakit[penyakit_pilihan]}',
                         fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            for bar, val in zip(bars, corr.values):
                ax.text(val + (0.003 if val >= 0 else -0.003),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center',
                        ha='left' if val >= 0 else 'right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            col_r, col_p = st.columns(2)
            with col_r:
                st.markdown("🔴 **Faktor Risiko** (korelasi positif)")
                risiko = corr[corr >= 0.03]
                for f, v in risiko.items():
                    st.markdown(f"- **{label_faktor[f]}**: `+{v:.4f}`")
            with col_p:
                st.markdown("🔵 **Faktor Pelindung** (korelasi negatif)")
                pelindung = corr[corr < 0]
                for f, v in pelindung.items():
                    st.markdown(f"- **{label_faktor[f]}**: `{v:.4f}`")

        st.markdown("---")

        # ============================================================
        # EDA KATEGORIKAL & NUMERIKAL
        # ============================================================
        st.header("📦 EDA Kategorikal & Numerikal")

        tab_kn1, tab_kn2, tab_kn3 = st.tabs([
            "📊 Boxplot Numerikal per Penyakit",
            "📋 Proporsi Faktor Binary per Penyakit",
            "📅 % Penderita per Kelompok Usia"
        ])

        # ── Tab Boxplot ──────────────────────────────────────────
        with tab_kn1:
            penyakit_box = st.selectbox(
                "Pilih Penyakit",
                list(kolom_penyakit.keys()),
                format_func=lambda x: kolom_penyakit[x],
                key='box_select'
            )
            nama_box = kolom_penyakit[penyakit_box]

            fig, axes = plt.subplots(1, 5, figsize=(20, 5))
            for i, faktor in enumerate(kolom_faktor_numerikal):
                ax = axes[i]
                data_tidak = df[df[penyakit_box] == 0.0][faktor]
                data_ya    = df[df[penyakit_box] == 1.0][faktor]
                bp = ax.boxplot([data_tidak, data_ya], labels=['Tidak', 'Ya'],
                                patch_artist=True,
                                medianprops=dict(color='black', linewidth=2),
                                flierprops=dict(marker='o', markersize=2, alpha=0.2))
                bp['boxes'][0].set_facecolor('#3498DB')
                bp['boxes'][0].set_alpha(0.7)
                bp['boxes'][1].set_facecolor('#E74C3C')
                bp['boxes'][1].set_alpha(0.7)
                ax.set_title(label_faktor[faktor], fontsize=8, fontweight='bold')
                ax.set_ylabel(faktor, fontsize=8)
                ax.spines[['top', 'right']].set_visible(False)
            fig.suptitle(f'Distribusi Numerikal – {nama_box}', fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Tabel rata-rata
            st.markdown(f"#### Rata-rata per Kelompok – {nama_box}")
            rows_box = []
            for faktor in kolom_faktor_numerikal:
                stats = df.groupby(penyakit_box)[faktor].mean().round(2)
                tidak, ya = stats[0.0], stats[1.0]
                diff = ya - tidak
                rows_box.append({
                    'Faktor': label_faktor[faktor],
                    'Tidak Sakit': tidak,
                    'Sakit': ya,
                    'Selisih': round(diff, 2),
                    'Keterangan': '↑ lebih tinggi' if diff > 0 else '↓ lebih rendah'
                })
            st.dataframe(pd.DataFrame(rows_box).set_index('Faktor'), use_container_width=True)

        # ── Tab Proporsi Binary per Penyakit ────────────────────
        with tab_kn2:
            penyakit_bin = st.selectbox(
                "Pilih Penyakit",
                list(kolom_penyakit.keys()),
                format_func=lambda x: kolom_penyakit[x],
                key='bin_select'
            )
            nama_bin = kolom_penyakit[penyakit_bin]

            props_tidak = [df[df[penyakit_bin]==0.0][c].mean()*100 for c in kolom_faktor_binary]
            props_ya    = [df[df[penyakit_bin]==1.0][c].mean()*100 for c in kolom_faktor_binary]
            labels_bin2 = [label_faktor[c] for c in kolom_faktor_binary]

            x = np.arange(len(labels_bin2))
            w = 0.35
            fig, ax = plt.subplots(figsize=(14, 5))
            b1 = ax.bar(x - w/2, props_tidak, w, label='Tidak Sakit', color='#3498DB', alpha=0.85)
            b2 = ax.bar(x + w/2, props_ya,    w, label='Sakit',        color='#E74C3C', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_bin2, rotation=20, ha='right', fontsize=9)
            ax.set_ylabel('% Memiliki Faktor')
            ax.set_ylim(0, 115)
            ax.set_title(f'Proporsi Faktor Binary – {nama_bin}', fontweight='bold')
            ax.legend()
            ax.spines[['top', 'right']].set_visible(False)
            for bar in list(b1) + list(b2):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{bar.get_height():.1f}%', ha='center', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Tabel selisih
            rows_bin = []
            for c in kolom_faktor_binary:
                props = df.groupby(penyakit_bin)[c].mean() * 100
                diff  = props[1.0] - props[0.0]
                rows_bin.append({
                    'Faktor': label_faktor[c],
                    'Tidak Sakit (%)': round(props[0.0], 1),
                    'Sakit (%)':       round(props[1.0], 1),
                    'Selisih (%)':     round(diff, 1),
                    'Arah':            '↑' if diff > 0 else '↓'
                })
            st.dataframe(pd.DataFrame(rows_bin).set_index('Faktor'), use_container_width=True)

        # ── Tab % Sakit per Usia ──────────────────────────────────
        with tab_kn3:
            penyakit_usia = st.selectbox(
                "Pilih Penyakit",
                list(kolom_penyakit.keys()),
                format_func=lambda x: kolom_penyakit[x],
                key='usia_select'
            )
            nama_usia = kolom_penyakit[penyakit_usia]

            ct = df.groupby(['Age', penyakit_usia]).size().unstack(fill_value=0)
            ct.columns = ['Tidak', 'Ya']
            ct['Total']   = ct.sum(axis=1)
            ct['% Sakit'] = (ct['Ya'] / ct['Total'] * 100).round(1)
            ct.index = [usia_label[int(i)] for i in ct.index]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(ct.index, ct['% Sakit'], marker='o', color='#E74C3C',
                    linewidth=2.5, markersize=7)
            ax.fill_between(ct.index, ct['% Sakit'], alpha=0.15, color='#E74C3C')
            for i, (usia, row) in enumerate(ct.iterrows()):
                ax.text(i, row['% Sakit'] + 0.5, f"{row['% Sakit']:.1f}%",
                        ha='center', fontsize=8)
            ax.set_xlabel('Kelompok Usia')
            ax.set_ylabel('% Penderita')
            ax.set_title(f'% Penderita per Kelompok Usia – {nama_usia}', fontweight='bold')
            ax.spines[['top', 'right']].set_visible(False)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.dataframe(
                ct[['Tidak', 'Ya', 'Total', '% Sakit']].rename(
                    columns={'Tidak': 'Tidak Sakit', 'Ya': 'Sakit'}),
                use_container_width=True
            )

    except FileNotFoundError:
        st.error("File 'data_dashboard.csv' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan dashboard.py.")


# ============================================================
# MENU 2: SIMULASI PREDIKSI PTM
# ============================================================
elif menu == "Simulasi Prediksi PTM":
    st.title("🩺 Simulasi Prediksi 5 PTM")
    st.markdown("Masukkan indikator kesehatan Anda di bawah ini untuk melihat estimasi risiko berdasarkan data historis.")

    with st.form("form_prediksi"):
        st.subheader("Informasi Dasar & Fisik")
        col1, col2, col3 = st.columns(3)

        with col1:
            age_dict = {
                1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
                6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
                11: '70-74', 12: '75-79', 13: '80+'
            }
            age_label = st.selectbox("Kelompok Usia", list(age_dict.values()))
            age = list(age_dict.keys())[list(age_dict.values()).index(age_label)]

            sex = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"])
            sex_val = 1.0 if sex == "Laki-laki" else 0.0

            bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

        with col2:
            genhlth  = st.slider("Kondisi Kesehatan Umum (1=Sangat Baik, 5=Buruk)", 1, 5, 2)
            menthlth = st.slider("Hari Kesehatan Mental Buruk (30 hari terakhir)", 0, 30, 0)
            physhlth = st.slider("Hari Kesehatan Fisik Buruk (30 hari terakhir)", 0, 30, 0)
            diffwalk = st.radio("Kesulitan Berjalan/Naik Tangga?", ["Tidak", "Ya"])
            diffwalk_val = 1.0 if diffwalk == "Ya" else 0.0

        with col3:
            cholcheck    = st.radio("Cek Kolesterol (5 thn terakhir)?", ["Tidak", "Ya"], index=1)
            cholcheck_val = 1.0 if cholcheck == "Ya" else 0.0

            smoker     = st.radio("Perokok (>=100 batang seumur hidup)?", ["Tidak", "Ya"])
            smoker_val = 1.0 if smoker == "Ya" else 0.0

            physactivity = st.radio("Aktif Olahraga (30 hari terakhir)?", ["Tidak", "Ya"], index=1)
            physact_val  = 1.0 if physactivity == "Ya" else 0.0

            fruits     = st.radio("Konsumsi Buah Harian?", ["Tidak", "Ya"], index=1)
            fruits_val = 1.0 if fruits == "Ya" else 0.0

            veggies     = st.radio("Konsumsi Sayur Harian?", ["Tidak", "Ya"], index=1)
            veggies_val = 1.0 if veggies == "Ya" else 0.0

            alcohol     = st.radio("Konsumsi Alkohol Berat?", ["Tidak", "Ya"])
            alcohol_val = 1.0 if alcohol == "Ya" else 0.0

        submitted = st.form_submit_button("Deteksi Risiko PTM")

    if submitted:
        st.markdown("---")
        st.subheader("Hasil Analisis Prediksi VitalsCheck")

        input_data = pd.DataFrame({
            'CholCheck':        [cholcheck_val],
            'BMI':              [bmi],
            'Smoker':           [smoker_val],
            'PhysActivity':     [physact_val],
            'Fruits':           [fruits_val],
            'Veggies':          [veggies_val],
            'HvyAlcoholConsump':[alcohol_val],
            'GenHlth':          [genhlth],
            'MentHlth':         [menthlth],
            'PhysHlth':         [physhlth],
            'DiffWalk':         [diffwalk_val],
            'Sex':              [sex_val],
            'Age':              [age]
        })

        st.info("Catatan: Hasil di bawah ini merupakan SIMULASI UI. Hubungkan model ML Anda (.pkl) ke dalam kode untuk hasil nyata.")

        risk_score = (bmi/40) + (age/13) + (genhlth/5) + smoker_val + diffwalk_val
        mock_pred = {
            'Diabetes':         'Tinggi' if risk_score > 2.5 else 'Rendah',
            'Hipertensi':       'Tinggi' if (risk_score + cholcheck_val) > 3.0 else 'Rendah',
            'Kolesterol Tinggi':'Tinggi' if (age > 8 and cholcheck_val) else 'Rendah',
            'Stroke':           'Tinggi' if (risk_score > 3.5 and smoker_val) else 'Rendah',
            'Penyakit Jantung': 'Tinggi' if (risk_score > 3.0 and sex_val == 1.0) else 'Rendah'
        }

        cols = st.columns(5)
        for i, (penyakit, risiko) in enumerate(mock_pred.items()):
            with cols[i]:
                if risiko == 'Tinggi':
                    st.error(f"**{penyakit}**\n\nRisiko {risiko} ⚠️")
                else:
                    st.success(f"**{penyakit}**\n\nRisiko {risiko} ✅")

        st.caption("Peringatan: VitalsCheck adalah alat edukasi deteksi dini dan tidak menggantikan diagnosis medis dari dokter profesional.")
