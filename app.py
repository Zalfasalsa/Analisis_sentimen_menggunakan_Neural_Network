import re
import os
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Analisis Sentimen Kesehatan",
    page_icon="🏥",
    layout="wide"
)

DATA_FILE = "tweet_data_kesehatan .csv"


# ============================================================
# KAMUS NORMALISASI
# ============================================================

normalisasi_dict = {
    "yg": "yang",
    "yng": "yang",
    "dgn": "dengan",
    "dg": "dengan",
    "dr": "dari",
    "utk": "untuk",
    "untk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "krena": "karena",
    "tdk": "tidak",
    "gak": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "bgt": "banget",
    "klo": "kalau",
    "kalo": "kalau",
    "kl": "kalau",
    "aja": "saja",
    "jgn": "jangan",
    "sm": "sama",
    "udh": "sudah",
    "udah": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "belom": "belum",
    "dpt": "dapat",
    "dapet": "dapat",
    "jd": "jadi",
    "org": "orang",
    "orng": "orang",
    "bbrp": "beberapa",
    "byk": "banyak",
    "bnyk": "banyak",
    "mksh": "terima kasih",
    "makasih": "terima kasih",
    "trims": "terima kasih",
    "thx": "terima kasih",
    "pls": "tolong",
    "info": "informasi"
}


# ============================================================
# CLEANSING
# ============================================================

def cleansing_text(text):

    text = str(text).lower()

    # Hapus RT
    text = re.sub(r"\brt\b", " ", text)

    # Hapus URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Hapus mention
    text = re.sub(r"@\w+", " ", text)

    # Hapus tanda #
    # contoh #kesehatan -> kesehatan
    text = re.sub(r"#(\w+)", r"\1", text)

    # Hapus angka
    text = re.sub(r"\d+", " ", text)

    # Hapus tanda baca
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)

    # Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# NORMALISASI
# ============================================================

def normalisasi_text(text):

    text = str(text).lower().strip()

    # Mengurangi huruf yang berulang
    # contoh: baguuuusss -> baguuss
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    tokens = text.split()

    hasil = []

    for token in tokens:

        if token in normalisasi_dict:
            hasil.append(normalisasi_dict[token])

        else:
            hasil.append(token)

    return " ".join(hasil)


# ============================================================
# MEMBACA DATASET
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):

        return None

    # Dataset menggunakan separator ;
    data = pd.read_csv(
        DATA_FILE,
        sep=";"
    )

    # Cek kolom
    if "full_text" not in data.columns:

        raise ValueError(
            "Kolom 'full_text' tidak ditemukan."
        )

    if "sentimen" not in data.columns:

        raise ValueError(
            "Kolom 'sentimen' tidak ditemukan."
        )

    # Hapus data kosong
    data = data.dropna(
        subset=[
            "full_text",
            "sentimen"
        ]
    ).copy()

    # Ubah tipe data
    data["full_text"] = (
        data["full_text"]
        .astype(str)
    )

    data["sentimen"] = (
        data["sentimen"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Label yang digunakan
    valid_labels = [
        "positif",
        "negatif",
        "netral"
    ]

    data = data[
        data["sentimen"].isin(valid_labels)
    ].copy()

    # Cleansing
    data["clean_text"] = (
        data["full_text"]
        .apply(cleansing_text)
    )

    # Normalisasi
    data["normalisasi_text"] = (
        data["clean_text"]
        .apply(normalisasi_text)
    )

    # Hapus teks kosong
    data = data[
        data["normalisasi_text"]
        .str.strip() != ""
    ].copy()

    data.reset_index(
        drop=True,
        inplace=True
    )

    return data


# ============================================================
# TRAINING MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    X = data["normalisasi_text"]

    y = data["sentimen"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    vectorizer = TfidfVectorizer(

        ngram_range=(1, 2),

        min_df=1,

        sublinear_tf=True,

        # L2 NORMALIZATION
        norm="l2"
    )

    X_train_tfidf = (
        vectorizer
        .fit_transform(X_train)
    )

    X_test_tfidf = (
        vectorizer
        .transform(X_test)
    )

    # ========================================================
    # NEURAL NETWORK
    # ========================================================

    model = MLPClassifier(

        hidden_layer_sizes=(64, 32),

        activation="relu",

        solver="adam",

        alpha=0.0001,

        learning_rate_init=0.001,

        max_iter=500,

        random_state=42,

        early_stopping=True,

        validation_fraction=0.10,

        n_iter_no_change=20
    )

    # Training
    model.fit(
        X_train_tfidf,
        y_train
    )

    # Prediksi
    y_pred = model.predict(
        X_test_tfidf
    )

    # ========================================================
    # EVALUASI
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(

        y_test,

        y_pred,

        labels=[
            "negatif",
            "netral",
            "positif"
        ]
    )

    report = classification_report(

        y_test,

        y_pred,

        digits=3,

        zero_division=0
    )

    return (
        vectorizer,
        model,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        report,
        model.loss_curve_,
        len(X_train),
        len(X_test)
    )


# ============================================================
# JUDUL APLIKASI
# ============================================================

st.title(
    "🏥 Analisis Sentimen Tweet Kesehatan"
)

st.write(
    "Klasifikasi sentimen teks kesehatan menggunakan "
    "Neural Network (MLP), TF-IDF, dan Normalisasi."
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    data = load_data()

    if data is None:

        st.error(
            "Dataset tidak ditemukan."
        )

        st.info(
            "Pastikan file berikut berada "
            "dalam folder yang sama dengan app.py:"
        )

        st.code(
            "tweet_data_kesehatan .csv"
        )

        st.stop()

except Exception as e:

    st.error(
        f"Terjadi kesalahan: {e}"
    )

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner(
    "Sedang melatih model Neural Network..."
):

    (
        vectorizer,
        model,
        accuracy,
        precision,
        recall,
        f1,
        cm,
        report,
        loss_curve,
        n_train,
        n_test
    ) = train_model(data)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "⚙️ Informasi Model"
)

st.sidebar.write(
    "**Dataset:** Tweet Kesehatan"
)

st.sidebar.write(
    "**Metode:** Neural Network / MLP"
)

st.sidebar.write(
    "**Fitur:** TF-IDF"
)

st.sidebar.write(
    "**Normalisasi:** Kata + L2"
)

st.sidebar.write(
    f"**Jumlah Data:** {len(data)}"
)

st.sidebar.write(
    f"**Data Training:** {n_train}"
)

st.sidebar.write(
    f"**Data Testing:** {n_test}"
)


# ============================================================
# TAB
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(

    [
        "📊 Data",
        "🧹 Cleansing & Normalisasi",
        "🔮 Prediksi",
        "📈 Evaluasi"
    ]

)


# ============================================================
# TAB 1 - DATA
# ============================================================

with tab1:

    st.subheader(
        "Dataset Tweet Kesehatan"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Data",
        len(data)
    )

    col2.metric(
        "Negatif",
        int(
            (
                data["sentimen"]
                == "negatif"
            ).sum()
        )
    )

    col3.metric(
        "Netral",
        int(
            (
                data["sentimen"]
                == "netral"
            ).sum()
        )
    )

    col4.metric(
        "Positif",
        int(
            (
                data["sentimen"]
                == "positif"
            ).sum()
        )
    )

    st.subheader(
        "Data Dataset"
    )

    st.dataframe(
        data,
        use_container_width=True,
        height=400
    )

    st.subheader(
        "Distribusi Sentimen"
    )

    st.bar_chart(
        data["sentimen"]
        .value_counts()
    )


# ============================================================
# TAB 2 - CLEANSING & NORMALISASI
# ============================================================

with tab2:

    st.subheader(
        "Hasil Cleansing dan Normalisasi"
    )

    st.info(
        "Data diproses melalui cleansing terlebih dahulu, "
        "kemudian normalisasi kata sebelum masuk ke TF-IDF."
    )

    tampil = data[
        [
            "full_text",
            "clean_text",
            "normalisasi_text",
            "sentimen"
        ]
    ]

    st.dataframe(
        tampil,
        use_container_width=True,
        height=450
    )

    jumlah_berubah = int(
        (
            data["clean_text"]
            != data["normalisasi_text"]
        ).sum()
    )

    st.write(
        "Jumlah teks yang berubah setelah normalisasi:",
        jumlah_berubah
    )

    # Download
    csv_data = data.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(

        label="⬇️ Download Data Normalisasi",

        data=csv_data,

        file_name=(
            "tweet_data_kesehatan_normalisasi.csv"
        ),

        mime="text/csv"
    )

    st.subheader(
        "Kamus Normalisasi"
    )

    kamus = pd.DataFrame(

        list(
            normalisasi_dict.items()
        ),

        columns=[
            "Kata Tidak Baku",
            "Kata Normalisasi"
        ]
    )

    st.dataframe(
        kamus,
        use_container_width=True
    )


# ============================================================
# TAB 3 - PREDIKSI
# ============================================================

with tab3:

    st.subheader(
        "🔮 Prediksi Sentimen Teks Kesehatan"
    )

    teks = st.text_area(

        "Masukkan teks kesehatan:",

        value=(
            "Pelayanan puskesmas "
            "sangat bagus dan petugasnya ramah"
        ),

        height=120,

        placeholder=(
            "Contoh: pelayanan rumah sakit "
            "sangat mengecewakan"
        )
    )

    if st.button(
        "🔍 Prediksi Sentimen",
        type="primary"
    ):

        if not teks.strip():

            st.warning(
                "Silakan masukkan teks terlebih dahulu."
            )

        else:

            # Cleansing
            clean = cleansing_text(
                teks
            )

            # Normalisasi
            normal = normalisasi_text(
                clean
            )

            # TF-IDF
            tfidf = vectorizer.transform(
                [normal]
            )

            # Prediksi
            prediksi = model.predict(
                tfidf
            )[0]

            # Probabilitas
            probabilitas = (
                model.predict_proba(
                    tfidf
                )[0]
            )

            # --------------------------------------------
            # HASIL PENGOLAHAN
            # --------------------------------------------

            st.subheader(
                "Hasil Pengolahan Teks"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.text_area(
                    "Teks Asli",
                    teks,
                    height=100,
                    disabled=True
                )

            with col2:

                st.text_area(
                    "Cleansing",
                    clean,
                    height=100,
                    disabled=True
                )

            with col3:

                st.text_area(
                    "Normalisasi",
                    normal,
                    height=100,
                    disabled=True
                )

            # --------------------------------------------
            # HASIL SENTIMEN
            # --------------------------------------------

            st.subheader(
                "Hasil Prediksi"
            )

            if prediksi == "positif":

                st.success(
                    f"😊 Sentimen: "
                    f"**{prediksi.upper()}**"
                )

            elif prediksi == "negatif":

                st.error(
                    f"😞 Sentimen: "
                    f"**{prediksi.upper()}**"
                )

            else:

                st.info(
                    f"😐 Sentimen: "
                    f"**{prediksi.upper()}**"
                )

            # --------------------------------------------
            # PROBABILITAS
            # --------------------------------------------

            hasil_prob = pd.DataFrame({

                "Sentimen":
                    model.classes_,

                "Probabilitas":
                    probabilitas

            })

            hasil_prob[
                "Persentase"
            ] = (

                hasil_prob[
                    "Probabilitas"
                ] * 100

            ).round(2)

            st.subheader(
                "Probabilitas Prediksi"
            )

            st.dataframe(

                hasil_prob[
                    [
                        "Sentimen",
                        "Persentase"
                    ]
                ],

                use_container_width=True,

                hide_index=True
            )

            st.bar_chart(

                hasil_prob.set_index(
                    "Sentimen"
                )[
                    "Persentase"
                ]
            )

    # ========================================================
    # CONTOH PREDIKSI
    # ========================================================

    st.divider()

    st.subheader(
        "📝 Contoh Prediksi Teks Kesehatan"
    )

    contoh = [

        "Pelayanan puskesmas sangat bagus dan petugasnya ramah",

        "Informasi kesehatan dari pemerintah sangat membantu masyarakat",

        "Pelayanan rumah sakit sangat mengecewakan dan lambat"

    ]

    for i, teks_contoh in enumerate(
        contoh,
        start=1
    ):

        clean = cleansing_text(
            teks_contoh
        )

        normal = normalisasi_text(
            clean
        )

        tfidf = vectorizer.transform(
            [normal]
        )

        prediksi = model.predict(
            tfidf
        )[0]

        st.write(
            f"**Contoh {i}:** {teks_contoh}"
        )

        st.write(
            f"**Cleansing:** {clean}"
        )

        st.write(
            f"**Normalisasi:** {normal}"
        )

        st.write(
            f"**Prediksi:** "
            f"**{prediksi.upper()}**"
        )

        st.divider()


# ============================================================
# TAB 4 - EVALUASI
# ============================================================

with tab4:

    st.subheader(
        "📈 Evaluasi Model"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

    col4.metric(
        "F1-Score",
        f"{f1 * 100:.2f}%"
    )

    # Classification report
    st.subheader(
        "Classification Report"
    )

    st.code(
        report
    )

    # Confusion Matrix
    st.subheader(
        "Confusion Matrix"
    )

    cm_df = pd.DataFrame(

        cm,

        index=[
            "Negatif",
            "Netral",
            "Positif"
        ],

        columns=[
            "Negatif",
            "Netral",
            "Positif"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # Training loss
    st.subheader(
        "Training Loss"
    )

    loss_df = pd.DataFrame({

        "Loss":
            loss_curve

    })

    st.line_chart(
        loss_df
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Analisis Sentimen Tweet Kesehatan | "
    "Neural Network (MLP) + TF-IDF + Normalisasi"
)