import base64
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image, ImageOps

from flowers import FLOWERS, GROW, ORDER, THEME, illustration, photo_data_uri, theme_css
from model import count_params, grad_cam, load_checkpoint, overlay_heatmap, predict

# ─────────────────────────── Konfigurasi ───────────────────────────
APP_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = APP_DIR / "flower_cnn.pth"
SAMPLES_DIR = APP_DIR / "assets" / "samples"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Bloomora · Pengenal Bunga", page_icon="🌸", layout="wide",
                   initial_sidebar_state="collapsed")


# Warna tiap kelas diambil dari warna kelopak bunganya sendiri
PETAL = {
    "Chrysanthemum": "#D9A21B",
    "Hibiscus": "#C8305F",
    "Marigold": "#E8701A",
    "Petuniya": "#7A4CB5",
    "Rose": "#B8233A",
}
LOCAL_NAME = {
    "Chrysanthemum": "Krisan",
    "Hibiscus": "Kembang sepatu",
    "Marigold": "Gemitir",
    "Petuniya": "Petunia",
    "Rose": "Mawar",
}
INK, MUTED, LEAF, PAPER, LINE = "#1E2B24", "#66746B", "#2F5D46", "#F2F5F0", "#DCE3DB"

# Hasil dari notebook (dipakai bila checkpoint tidak membawa histori sendiri)
NB_HISTORY = {
    "train_loss": [.4674, .2608, .1832, .1762, .1167, .1221, .1167, .0920, .1124, .1168, .0760, .0869, .1037, .0800, .0842,
                   .0585, .0454, .0602, .0694, .0589, .0351, .0253, .0408, .0301, .0231, .0315, .0322, .0265, .0338, .0265],
    "val_loss": [.4560, .1938, .3539, .1633, .1173, .1955, .1114, .0814, .1103, .0658, .1535, .1312, .1006, .1066, .0597,
                 .0414, .0543, .0445, .0420, .0897, .0786, .0625, .0558, .0670, .0577, .0471, .0542, .0523, .0434, .0484],
    "train_acc": [.8276, .9112, .9357, .9408, .9653, .9663, .9622, .9684, .9582, .9602, .9776, .9704, .9673, .9724, .9755,
                  .9857, .9888, .9806, .9755, .9786, .9888, .9959, .9878, .9929, .9949, .9908, .9898, .9898, .9898, .9918],
    "val_acc": [.8429, .9429, .8810, .9429, .9619, .9286, .9524, .9762, .9619, .9714, .9429, .9524, .9714, .9667, .9857,
                .9905, .9714, .9857, .9857, .9714, .9810, .9810, .9810, .9810, .9810, .9857, .9762, .9762, .9857, .9762],
}
NB_REPORT = pd.DataFrame({
    "Kelas": ["Chrysanthemum", "Hibiscus", "Marigold", "Petuniya", "Rose"],
    "Precision": [.96, .94, 1.00, 1.00, .97],
    "Recall": [.98, 1.00, .98, 1.00, .93],
    "F1-score": [.97, .97, .99, 1.00, .95],
    "Support": [44, 29, 51, 46, 40],
})
NB_DATASET = {"Chrysanthemum": 291, "Hibiscus": 197, "Marigold": 335, "Petuniya": 309, "Rose": 268}
NB_TEST_ACC, NB_BEST_VAL = 0.9762, 0.9905

# ─────────────────────────── Gaya ───────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block');

html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, label {{
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
.stApp {{ background: {PAPER}; color: {INK}; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 4rem; }}
#MainMenu, footer, [data-testid="stAppDeployButton"] {{ visibility: hidden; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif !important; color: {INK}; font-weight: 600; letter-spacing: -0.01em; }}

.brand {{ display:flex; align-items:baseline; gap:.9rem; margin-bottom:.2rem; }}
.brand .logo {{ font-family:'Fraunces',serif; font-weight:600; font-size:2.6rem; line-height:1; color:{INK}; letter-spacing:-.02em; }}
.brand span {{ color:{MUTED}; font-size:.98rem; }}
.petals {{ display:flex; gap:6px; margin:.9rem 0 1.6rem; }}
.petals i {{ width:22px; height:10px; border-radius:10px 0 10px 0; display:block; }}

/* Tab sebagai pil lembut */
.stTabs [role="tablist"] {{ gap:.3rem; border-bottom:none !important; box-shadow:none !important; background:#E6ECE4;
  padding:.3rem; border-radius:14px; width:fit-content; max-width:100%; overflow-x:auto; margin-bottom:1rem; }}
.stTabs [role="tab"] {{ padding:.45rem 1.1rem; border-radius:10px; color:{MUTED}; }}
.stTabs [role="tab"] p {{ font-weight:500; font-size:.95rem; }}
.stTabs [role="tab"][aria-selected="true"] {{ background:white; color:{INK}; box-shadow:0 1px 2px rgba(30,43,36,.08); }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display:none !important; }}

/* Area unggah */
[data-testid="stFileUploaderDropzone"] {{ background:white; border:1.5px dashed #B9C6B8; border-radius:18px; padding:1.6rem; }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{LEAF}; }}

.stButton>button, .stDownloadButton>button {{ border-radius:999px; border:1px solid {LINE}; background:white; color:{INK}; font-weight:500; padding:.45rem 1.2rem; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{LEAF}; color:{LEAF}; }}
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible {{ outline:2px solid {LEAF}; outline-offset:2px; }}

/* Kartu hasil utama */
.verdict {{ background:white; border-radius:24px; padding:1.8rem 2rem 1.6rem; position:relative; overflow:hidden; }}
.verdict::before {{ content:""; position:absolute; right:-60px; top:-60px; width:200px; height:200px;
  background:var(--c); opacity:.12; border-radius:100px 0 100px 0; transform:rotate(12deg); }}
.verdict small {{ color:{MUTED}; font-size:.9rem; }}
.verdict .name {{ font-family:'Fraunces',serif; font-size:3.3rem; line-height:1.05; color:var(--c); margin:.3rem 0 .1rem; font-weight:600; }}
.verdict .local {{ color:{MUTED}; font-size:1.05rem; margin-bottom:1.3rem; }}
.meter {{ height:8px; background:#EEF1ED; border-radius:8px; overflow:hidden; }}
.meter b {{ display:block; height:100%; background:var(--c); border-radius:8px; animation:grow .9s cubic-bezier(.2,.8,.2,1); }}
@keyframes grow {{ from {{ width:0 }} }}
@media (prefers-reduced-motion: reduce) {{ .meter b {{ animation:none; }} }}
.meter-row {{ display:flex; justify-content:space-between; margin-top:.5rem; font-size:.92rem; color:{MUTED}; }}
.meter-row strong {{ color:{INK}; font-size:1.15rem; }}

.note {{ border-left:3px solid #D9A21B; background:#FBF6E8; padding:.8rem 1rem; border-radius:0 12px 12px 0; margin-top:1rem; font-size:.93rem; }}
.empty {{ text-align:center; color:{MUTED}; padding:3rem 1rem; }}
.empty .big {{ font-family:'Fraunces',serif; font-size:1.5rem; color:{INK}; margin-bottom:.3rem; }}

.chip {{ display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600; color:white; }}
.thumb-cap {{ font-size:.85rem; color:{MUTED}; margin-top:.3rem; }}

.stat {{ background:white; border-radius:18px; padding:1.1rem 1.3rem; }}
.stat .v {{ font-family:'Fraunces',serif; font-size:2rem; line-height:1.1; }}
.stat .l {{ color:{MUTED}; font-size:.88rem; }}

[data-testid="stImage"] img {{ border-radius:18px; }}
[data-testid="stSidebar"] {{ background:white; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Model ───────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(blob: bytes | None, path: str | None, version: str = ""):
    # `version` (ukuran + waktu ubah file) membuat cache ikut diperbarui saat flower_cnn.pth diganti
    src = io.BytesIO(blob) if blob else path
    return load_checkpoint(src, DEVICE)


def color_of(name):
    return PETAL.get(name, LEAF)


def prob_chart(probs, classes, height=230):
    order = np.argsort(probs)
    fig = go.Figure(go.Bar(
        x=probs[order] * 100, y=[classes[i] for i in order], orientation="h",
        marker=dict(color=[color_of(classes[i]) for i in order], cornerradius=6),
        text=[f"{probs[i]*100:.1f}%" for i in order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(height=height, margin=dict(l=0, r=40, t=6, b=0), bargap=.38,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans", color=INK, size=13),
                      xaxis=dict(visible=False, range=[0, 112]), yaxis=dict(showgrid=False))
    return fig


def line_chart(a, b, la, lb, ytitle, pct=False):
    ep = list(range(1, len(a) + 1))
    fig = go.Figure()
    for ys, name, col, dash in [(a, la, "#9AAE9F", "solid"), (b, lb, LEAF, "solid")]:
        fig.add_trace(go.Scatter(x=ep, y=ys, name=name, mode="lines",
                                 line=dict(color=col, width=2.4 if col == LEAF else 2, dash=dash, shape="spline")))
    fig.update_layout(height=280, margin=dict(l=0, r=10, t=30, b=0), title=dict(text=ytitle, font=dict(size=15, family="Fraunces")),
                      plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Plus Jakarta Sans", color=INK),
                      legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                      xaxis=dict(title="Epoch", showgrid=False), yaxis=dict(gridcolor="#EEF1ED", tickformat=".0%" if pct else None))
    return fig


def page_identify():

    st.markdown("## Kenali bunga dari foto")
    with st.sidebar:
        st.markdown("### Pengaturan")
        up_weights = st.file_uploader("Bobot model (.pth)", type=["pth", "pt"],
                                      help="Kosongkan untuk memakai flower_cnn.pth di folder aplikasi.")
        threshold = st.slider("Batas keyakinan", 0.30, 0.95, 0.60, 0.05,
                              help="Di bawah angka ini, hasil diberi tanda kurang yakin.")
        use_tta = st.toggle("Prediksi + cermin (TTA)", value=True,
                            help="Rata-rata prediksi gambar asli dan versi cerminnya. Sedikit lebih lambat, biasanya lebih stabil.")
        show_cam = st.toggle("Tampilkan area fokus model", value=True,
                             help="Grad-CAM: bagian gambar yang paling memengaruhi keputusan model.")
        st.caption(f"Perangkat: `{DEVICE}`")

    # Muat model
    model = classes = meta = None
    load_error = None
    try:
        if up_weights is not None:
            model, classes, meta = get_model(up_weights.getvalue(), None)
        elif DEFAULT_WEIGHTS.exists():
            st_ = DEFAULT_WEIGHTS.stat()
            model, classes, meta = get_model(None, str(DEFAULT_WEIGHTS), f"{st_.st_size}-{st_.st_mtime_ns}")
    except Exception as e:  # noqa: BLE001
        load_error = str(e)

    if "history" not in st.session_state:
        st.session_state.history = []

    if model is None:
        if load_error:
            st.error(f"Model tidak bisa dimuat: {load_error}. Pastikan file berasal dari ImprovedCNN di notebook.")
        else:
            st.markdown(f"""
<div class="verdict" style="--c:{LEAF}">
  <div class="name" style="font-size:2rem">Model belum dipasang</div>
  <div class="local">Simpan bobot dari notebook sebagai <code>flower_cnn.pth</code> di folder yang sama dengan
  <code>app.py</code>, atau unggah lewat panel Pengaturan di kiri.</div>
</div>""", unsafe_allow_html=True)
            st.code("""# Tambahkan di akhir notebook, setelah training
torch.save({
    "state_dict": best_model_wts,
    "class_names": class_names,
    "history": {"train_loss": train_losses, "val_loss": val_losses,
                "train_acc": train_accs, "val_acc": val_accs},
    "test_acc": float(test_acc),
}, "flower_cnn.pth")""", language="python")

    tab_one, tab_batch, tab_hist, tab_about = st.tabs(["Identifikasi", "Banyak foto", "Riwayat", "Tentang model"])


    def log(name, label, conf):
        st.session_state.history.insert(0, {
            "Waktu": datetime.now().strftime("%H:%M:%S"), "File": name,
            "Prediksi": label, "Keyakinan": round(float(conf) * 100, 2),
        })


    # ─────────────────────────── Tab 1: satu foto ───────────────────────────
    with tab_one:
        src = st.radio("Sumber foto", ["Unggah", "Tempel (Ctrl+V)", "Kamera", "Contoh foto"], horizontal=True,
                       label_visibility="collapsed")
        st.markdown(PHOTO_GUIDE, unsafe_allow_html=True)
        if src == "Unggah":
            file = st.file_uploader("Pilih foto bunga", type=["jpg", "jpeg", "png", "webp"], key="single")
        elif src == "Kamera":
            file = st.camera_input("Ambil foto bunga")
        elif src == "Contoh foto":
            st.markdown('<div class="paste-hint">Foto berikut diambil dari data uji, yaitu foto yang tidak digunakan '
                        'saat melatih model. Pilih salah satu untuk mencobanya.</div>', unsafe_allow_html=True)
            samples = sorted(SAMPLES_DIR.glob("*.jpg"))
            for row in range(0, len(samples), 5):
                cols = st.columns(5, gap="small")
                for col, pth in zip(cols, samples[row:row + 5]):
                    with col:
                        st.image(str(pth), use_container_width=True)
                        if st.button("Coba foto ini", key=f"sample_{pth.stem}", use_container_width=True):
                            st.session_state.sample = str(pth)
            file = None
            if st.session_state.get("sample"):
                pth = Path(st.session_state.sample)
                file = io.BytesIO(pth.read_bytes())
                file.name, file.size = f"contoh_{pth.stem}.jpg", pth.stat().st_size
        else:
            st.markdown('<div class="paste-hint">Salin foto terlebih dahulu (dari situs web, aplikasi pesan, atau tangkapan layar), klik kotak di bawah, '
                        'tekan <kbd>Ctrl</kbd>+<kbd>V</kbd>, kemudian tekan <kbd>Enter</kbd>.</div>', unsafe_allow_html=True)
            with st.container(key="pastebox"):
                pasted = st.chat_input("Klik di sini, lalu Ctrl+V untuk menempel foto", key="paste",
                                       accept_file=True, file_type=["jpg", "jpeg", "png", "webp"])
            if pasted is not None and getattr(pasted, "files", None):
                f0 = pasted.files[0]
                st.session_state.pasted = (f0.getvalue(), f0.name or "tempelan.png")
            file = None
            if st.session_state.get("pasted"):
                data, name = st.session_state.pasted
                file = io.BytesIO(data)
                file.name, file.size = name, len(data)

        if file is None:
            st.markdown('<div class="empty"><div class="big">Mulai dengan satu foto</div>'
                        'Krisan, kembang sepatu, gemitir, petunia, atau mawar. Foto close-up memberi hasil terbaik.</div>',
                        unsafe_allow_html=True)
        elif model is not None:
            img = Image.open(file).convert("RGB")
            t0 = time.perf_counter()
            probs = predict(model, img, tta=use_tta, device=DEVICE)
            ms = (time.perf_counter() - t0) * 1000
            top = int(np.argmax(probs))
            label, conf = classes[top], float(probs[top])
            runner = np.sort(probs)[-2]

            key = (getattr(file, "name", "kamera"), file.size if hasattr(file, "size") else len(file.getvalue()))
            if st.session_state.get("last_logged") != key:
                log(getattr(file, "name", "kamera.jpg"), label, conf)
                st.session_state.last_logged = key

            left, right = st.columns([1, 1.15], gap="large")
            with left:
                if show_cam:
                    cam = grad_cam(model, img, top, DEVICE)
                    view = st.segmented_control("Tampilan", ["Foto", "Area fokus"], default="Foto",
                                                label_visibility="collapsed")
                    st.image(overlay_heatmap(img, cam) if view == "Area fokus" else img, use_container_width=True)
                    if view == "Area fokus":
                        st.caption("Area terang = bagian yang paling diperhatikan model.")
                else:
                    st.image(img, use_container_width=True)

            with right:
                c = color_of(label)
                st.markdown(f"""
<div class="verdict" style="--c:{c}">
  <small>Kemungkinan besar ini</small>
  <div class="name">{label}</div>
  <div class="local">{LOCAL_NAME.get(label, '')}</div>
  <div class="meter"><b style="width:{conf*100:.1f}%"></b></div>
  <div class="meter-row"><span>Keyakinan</span><strong>{conf*100:.1f}%</strong></div>
</div>""", unsafe_allow_html=True)

                if conf < threshold:
                    st.markdown(f'<div class="note">Keyakinan di bawah {threshold:.0%}. Foto ini mungkin bukan salah satu '
                                f'dari lima bunga yang dikenali model, atau bunganya terlalu kecil di dalam bingkai.</div>',
                                unsafe_allow_html=True)
                elif conf - runner < 0.15:
                    second = classes[int(np.argsort(probs)[-2])]
                    st.markdown(f'<div class="note">Hasilnya tipis dengan <b>{second}</b>. Coba foto dari sudut lain.</div>',
                                unsafe_allow_html=True)

                st.plotly_chart(prob_chart(probs, classes), use_container_width=True, config={"displayModeBar": False})
                if label in FLOWERS:
                    st.markdown(f'<a class="care-link" href="./?bunga={label}" target="_self" style="--c:{c}">'
                                f'<span>Lihat cara merawat {LOCAL_NAME.get(label, label).lower()}</span>'
                                f'<small>Penyiraman, pemupukan, pemangkasan, dan hama</small></a>',
                                unsafe_allow_html=True)
                st.caption(f"{img.width}×{img.height}px · diproses dalam {ms:.0f} ms" + (" dengan TTA" if use_tta else ""))
                with st.expander("Kapan hasilnya bisa meleset?"):
                    st.markdown("Model dilatih dari ±3.600 foto: foto kebun asli ditambah foto beragam dari iNaturalist, Oxford 102 Flowers, "
                                "dan TensorFlow flower_photos. Pada foto dunia nyata yang belum pernah dilihatnya, model benar sekitar "
                                "2 dari 3 kali. Foto satu tanaman utuh dari jauh, bunga di luar kelima jenis ini, atau foto buram "
                                "bisa ditebak salah, bahkan dengan angka keyakinan tinggi.")


    # ─────────────────────────── Tab 2: banyak foto ───────────────────────────
    with tab_batch:
        files = st.file_uploader("Pilih beberapa foto sekaligus", type=["jpg", "jpeg", "png", "webp"],
                                 accept_multiple_files=True, key="batch")
        if not files:
            st.markdown('<div class="empty"><div class="big">Klasifikasikan satu folder sekaligus</div>'
                        'Pilih banyak foto, lalu unduh hasilnya sebagai CSV.</div>', unsafe_allow_html=True)
        elif model is not None:
            rows, prog = [], st.progress(0.0, text="Memproses foto…")
            for i, f in enumerate(files):
                im = Image.open(f).convert("RGB")
                p = predict(model, im, tta=use_tta, device=DEVICE)
                k = int(np.argmax(p))
                rows.append({"img": ImageOps.fit(im, (320, 320)), "File": f.name, "Prediksi": classes[k], "Keyakinan": float(p[k]),
                             **{c: float(p[j]) for j, c in enumerate(classes)}})
                prog.progress((i + 1) / len(files), text=f"Memproses foto {i+1} dari {len(files)}")
            prog.empty()

            df = pd.DataFrame([{k: v for k, v in r.items() if k != "img"} for r in rows])
            low = (df["Keyakinan"] < threshold).sum()

            s1, s2, s3 = st.columns(3)
            for col, v, l in [(s1, len(df), "foto diproses"),
                              (s2, f"{df['Keyakinan'].mean()*100:.1f}%", "rata-rata keyakinan"),
                              (s3, int(low), f"perlu dicek (< {threshold:.0%})")]:
                col.markdown(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>',
                             unsafe_allow_html=True)
            st.write("")

            g1, g2 = st.columns([1.6, 1], gap="large")
            with g2:
                counts = df["Prediksi"].value_counts()
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.62, sort=False,
                                       marker=dict(colors=[color_of(c) for c in counts.index], line=dict(color=PAPER, width=3)),
                                       textinfo="value", hovertemplate="%{label}: %{value} foto<extra></extra>"))
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                                  font=dict(family="Plus Jakarta Sans", color=INK), legend=dict(orientation="h", y=-.1))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                only_low = st.toggle("Hanya yang perlu dicek")
                out = df.copy()
                out["Keyakinan"] = (out["Keyakinan"] * 100).round(2)
                st.download_button("Unduh hasil (CSV)", out.to_csv(index=False).encode(), "hasil_klasifikasi_bunga.csv",
                                   "text/csv", use_container_width=True)
            with g1:
                shown = [r for r in rows if not only_low or r["Keyakinan"] < threshold]
                if not shown:
                    st.caption("Semua foto di atas batas keyakinan.")
                for start in range(0, len(shown), 4):
                    cols = st.columns(4)
                    for col, r in zip(cols, shown[start:start + 4]):
                        with col:
                            st.image(r["img"], use_container_width=True)
                            flag = " ⚠" if r["Keyakinan"] < threshold else ""
                            st.markdown(f'<span class="chip" style="background:{color_of(r["Prediksi"])}">{r["Prediksi"]}</span>'
                                        f'<div class="thumb-cap">{r["Keyakinan"]*100:.1f}%{flag} · {r["File"][:18]}</div>',
                                        unsafe_allow_html=True)


    # ─────────────────────────── Tab 3: riwayat ───────────────────────────
    with tab_hist:
        h = st.session_state.history
        if not h:
            st.markdown('<div class="empty"><div class="big">Belum ada riwayat</div>'
                        'Setiap foto yang kamu identifikasi di sesi ini akan tercatat di sini.</div>', unsafe_allow_html=True)
        else:
            hdf = pd.DataFrame(h)
            st.dataframe(hdf, use_container_width=True, hide_index=True,
                         column_config={"Keyakinan": st.column_config.ProgressColumn("Keyakinan", format="%.1f%%",
                                                                                    min_value=0, max_value=100)})
            a, b, _ = st.columns([1, 1, 3])
            a.download_button("Unduh riwayat", hdf.to_csv(index=False).encode(), "riwayat_bloomora.csv", "text/csv")
            if b.button("Hapus riwayat"):
                st.session_state.history = []
                st.session_state.pop("last_logged", None)
                st.rerun()


    # ─────────────────────────── Tab 4: tentang model ───────────────────────────
    with tab_about:
        hist = (meta or {}).get("history") or NB_HISTORY
        test_acc = (meta or {}).get("test_acc", NB_TEST_ACC)
        report_df = NB_REPORT
        if (meta or {}).get("report"):
            r = meta["report"]
            report_df = pd.DataFrame({"Kelas": list(r),
                                      "Precision": [round(v["precision"], 2) for v in r.values()],
                                      "Recall": [round(v["recall"], 2) for v in r.values()],
                                      "F1-score": [round(v["f1-score"], 2) for v in r.values()],
                                      "Support": [int(v["support"]) for v in r.values()]})
        ds = (meta or {}).get("dataset_counts") or NB_DATASET
        best_val = max(hist["val_acc"])
        n_params = count_params(model) if model is not None else 16_875_013

        s = st.columns(4)
        for col, v, l in [(s[0], f"{test_acc*100:.1f}%", "akurasi data uji"),
                          (s[1], f"{best_val*100:.1f}%", "akurasi validasi terbaik"),
                          (s[2], f"{len(hist['val_acc'])}", "epoch training"),
                          (s[3], f"{n_params/1e6:.1f} jt", "parameter")]:
            col.markdown(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
        st.write("")

        c1, c2 = st.columns(2, gap="medium")
        c1.plotly_chart(line_chart(hist["train_acc"], hist["val_acc"], "Train", "Validasi", "Akurasi", pct=True),
                        use_container_width=True, config={"displayModeBar": False})
        c2.plotly_chart(line_chart(hist["train_loss"], hist["val_loss"], "Train", "Validasi", "Loss"),
                        use_container_width=True, config={"displayModeBar": False})

        d1, d2 = st.columns([1.3, 1], gap="large")
        with d1:
            st.markdown("### Kinerja per kelas")
            st.dataframe(report_df, hide_index=True, use_container_width=True,
                         column_config={
                             "Precision": st.column_config.NumberColumn(format="%.2f"),
                             "Recall": st.column_config.NumberColumn(format="%.2f"),
                             "F1-score": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                         })
            st.caption(f"Dari {int(report_df['Support'].sum())} foto uji (split 70/15/15, stratified, seed 42).")
        with d2:
            st.markdown("### Isi dataset")
            fig = go.Figure(go.Bar(x=list(ds.values()), y=list(ds.keys()), orientation="h",
                                   marker=dict(color=[color_of(k) for k in ds], cornerradius=6),
                                   text=list(ds.values()), textposition="outside", cliponaxis=False))
            fig.update_layout(height=250, margin=dict(l=0, r=30, t=0, b=0), plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans", color=INK),
                              xaxis=dict(visible=False), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"{sum(ds.values()):,} foto (asli + tambahan), diperkecil ke 128×128 px.".replace(",", "."))

        with st.expander("Arsitektur & konfigurasi training"):
            st.markdown("""
**ImprovedCNN** — 3 blok konvolusi (32 → 64 → 128 filter, kernel 3×3) masing-masing dengan BatchNorm, ReLU, dan MaxPool,
lalu Dense 512 + BatchNorm + Dropout 0,5, dan output 5 kelas.

- Input 128×128 RGB, normalisasi mean/std 0,5
- Augmentasi: RandomResizedCrop (0,8–1,0), flip horizontal, rotasi ±20°, ColorJitter 0,2
- Optimizer AdamW (lr 0,001, weight decay 1e-4), ReduceLROnPlateau (faktor 0,5, patience 3)
- Batch 32, 30 epoch, bobot terbaik dipilih dari akurasi validasi
""")


# ─────────────────────────── Katalog & detail ───────────────────────────
st.markdown(f"""
<style>
.intro {{ font-family:'Fraunces',serif; font-size:2.1rem; line-height:1.15; max-width:22ch; margin:.4rem 0 .5rem; color:{INK}; }}
.lede {{ color:{MUTED}; max-width:56ch; margin-bottom:1.6rem; }}

.strip {{ display:flex; gap:1rem; overflow-x:auto; scroll-snap-type:x mandatory; padding:.2rem .1rem 1.1rem;
  scrollbar-width:thin; scrollbar-color:#C4D0C3 transparent; }}
.strip::-webkit-scrollbar {{ height:6px; }}
.strip::-webkit-scrollbar-thumb {{ background:#C4D0C3; border-radius:6px; }}
.card {{ flex:0 0 250px; scroll-snap-align:start; background:white; border-radius:22px; overflow:hidden;
  text-decoration:none !important; color:{INK} !important; border:1px solid transparent;
  transition:transform .25s ease, border-color .25s ease; }}
.card:hover {{ transform:translateY(-4px); border-color:var(--c); }}
.card:focus-visible {{ outline:2px solid var(--c); outline-offset:3px; }}
.card img {{ width:100%; aspect-ratio:1/1; object-fit:cover; display:block; }}
.card .body {{ padding:.9rem 1.1rem 1.1rem; border-top:4px solid var(--c); }}
.card .nm {{ font-family:'Fraunces',serif; font-size:1.45rem; font-weight:600; line-height:1.1; }}
.card .lt {{ color:{MUTED}; font-style:italic; font-size:.88rem; margin-top:.15rem; }}
.card.sm {{ flex-basis:180px; border-radius:18px; }}
.card.sm .nm {{ font-size:1.15rem; }}
@media (prefers-reduced-motion: reduce) {{ .card {{ transition:none; }} .card:hover {{ transform:none; }} }}

.glance {{ width:100%; border-collapse:collapse; background:white; border-radius:18px; overflow:hidden; font-size:.93rem; }}
.glance th, .glance td {{ text-align:left; padding:.75rem 1rem; border-bottom:1px solid #EEF1ED; }}
.glance th {{ color:{MUTED}; font-weight:500; }}
.glance tr:last-child td {{ border-bottom:none; }}
.glance a {{ color:{INK}; font-weight:600; text-decoration:none; }}
.glance a:hover {{ color:{LEAF}; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:5px 0 5px 0; margin-right:.5rem; }}

.back {{ color:{MUTED} !important; text-decoration:none !important; font-size:.95rem; }}
.back:hover {{ color:{LEAF} !important; }}
.hero-img {{ width:100%; aspect-ratio:1/1; object-fit:cover; border-radius:28px; display:block; }}
.d-name {{ font-family:'Fraunces',serif; font-size:3.6rem; font-weight:600; line-height:1; color:var(--c); margin:.2rem 0 .4rem; }}
.d-latin {{ color:{MUTED}; font-size:1rem; }}
.d-latin i {{ color:{INK}; }}
.d-sum {{ font-size:1.06rem; line-height:1.65; margin:1.1rem 0 1.4rem; max-width:52ch; }}
.facts {{ display:grid; grid-template-columns:1fr 1fr; gap:.6rem; }}
.fact {{ background:white; border-radius:16px; padding:.8rem 1rem; }}
.fact .k {{ color:{MUTED}; font-size:.82rem; }}
.fact .v {{ font-weight:600; margin-top:.1rem; }}

.care {{ margin-top:2.6rem; }}
.care-row {{ display:grid; grid-template-columns:200px 1fr; gap:1.5rem; padding:1.15rem 0; border-top:1px solid {LINE}; }}
.care-row h4 {{ font-family:'Fraunces',serif; font-size:1.2rem; margin:0; color:{INK}; font-weight:600; }}
.care-row p {{ margin:0; line-height:1.7; max-width:68ch; }}
.tipbox {{ margin-top:1.4rem; background:white; border-radius:20px; padding:1.2rem 1.4rem; border-left:5px solid var(--c); }}
.tipbox b {{ display:block; font-family:'Fraunces',serif; font-size:1.05rem; margin-bottom:.25rem; }}
/* suasana halaman detail */
.d-head {{ position:relative; }}
.d-mark {{ position:absolute; right:-20px; top:-50px; width:230px; opacity:.14; pointer-events:none;
  animation: mark-turn 40s linear infinite; }}
@keyframes mark-turn {{ to {{ transform: rotate(360deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .d-mark {{ animation:none; }} }}
.d-mood {{ color:var(--c); font-size:.95rem; font-style:italic; margin-bottom:.2rem; position:relative; }}
.d-name, .d-latin, .d-sum, .facts {{ position:relative; }}
.funfact {{ margin-top:.9rem; border-radius:20px; padding:1.1rem 1.4rem; background:linear-gradient(120deg, white, transparent);
  border:1px dashed var(--c); }}
.funfact b {{ display:block; font-family:'Fraunces',serif; font-size:1.05rem; margin-bottom:.25rem; color:var(--c); }}
.care-link {{ display:flex; flex-direction:column; gap:.1rem; margin-top:.6rem; padding:.9rem 1.2rem; border-radius:16px;
  background:white; border:1px solid {LINE}; text-decoration:none !important; color:{INK} !important;
  transition: border-color .2s ease, transform .2s ease; }}
.care-link:hover {{ border-color:var(--c); transform: translateY(-1px); }}
.care-link span {{ font-weight:600; color:var(--c); }}
.care-link small {{ color:{MUTED}; }}
.flow {{ display:grid; grid-template-columns: repeat(6, 1fr); gap:.6rem; margin:1rem 0 2rem; }}
.flow-step {{ background:white; border-radius:16px; padding:.9rem .9rem 1rem; position:relative; display:flex;
  flex-direction:column; gap:.3rem; }}
.flow-step .n {{ width:24px; height:24px; border-radius:12px 12px 12px 3px; background:{LEAF}; color:white;
  font-size:.8rem; font-weight:600; display:grid; place-items:center; }}
.flow-step b {{ font-family:'Fraunces',serif; font-size:1.02rem; }}
.flow-step span {{ color:{MUTED}; font-size:.84rem; line-height:1.45; flex:1; }}
.flow-step code {{ background:#EEF1ED; color:{LEAF}; border-radius:6px; padding:.1rem .4rem; font-size:.78rem; width:fit-content; }}
.flow-step:not(:last-child)::after {{ content:""; position:absolute; right:-.45rem; top:50%; width:.3rem; height:.3rem;
  border-top:2px solid #B9C6B8; border-right:2px solid #B9C6B8; transform: translateY(-50%) rotate(45deg); }}
@media (max-width: 900px) {{ .flow {{ grid-template-columns: repeat(2, 1fr); }} .flow-step::after {{ display:none; }} }}
.note-soft {{ color:{MUTED}; font-size:.92rem; line-height:1.6; max-width:75ch; margin:.3rem 0 1.6rem; }}
.lessons {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:.8rem; margin-top:.4rem; }}
.lessons > div {{ background:white; border-radius:18px; padding:1.1rem 1.2rem; line-height:1.6; font-size:.93rem; }}
.lessons b {{ display:block; font-family:'Fraunces',serif; font-size:1.08rem; margin-bottom:.35rem; color:{INK}; }}
@media (max-width: 900px) {{ .lessons {{ grid-template-columns: 1fr; }} }}
.grow {{ margin-top:2.6rem; }}
.grow-intro {{ color:{MUTED}; max-width:68ch; margin:.2rem 0 1rem; }}
.grow-grid {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:.7rem; }}
.grow-card {{ display:flex; gap:.9rem; align-items:flex-start; background:white; border-radius:18px; padding:1rem 1.1rem; }}
.grow-card .material-symbols-rounded {{ font-family:'Material Symbols Rounded'; font-weight:normal; font-style:normal;
  font-feature-settings:'liga'; -webkit-font-feature-settings:'liga'; letter-spacing:normal; text-transform:none; white-space:nowrap; font-size:1.5rem; color:var(--c); background:color-mix(in srgb, var(--c) 12%, white);
  border-radius:12px 12px 12px 4px; padding:.35rem; line-height:1; }}
.grow-card small {{ display:block; color:{MUTED}; font-size:.82rem; }}
.grow-card b {{ font-weight:600; line-height:1.45; }}
.grow-low {{ margin:.9rem 0 0; padding:.8rem 1.1rem; border-radius:14px; background:rgba(255,255,255,.6); font-size:.94rem; }}
@media (max-width: 700px) {{ .grow-grid {{ grid-template-columns: 1fr; }} }}
.more-h {{ font-family:'Fraunces',serif; font-size:1.4rem; margin:2.6rem 0 .8rem; }}
@media (max-width: 700px) {{
  .care-row {{ grid-template-columns:1fr; gap:.4rem; }}
  .d-name {{ font-size:2.6rem; }} .intro {{ font-size:1.7rem; }}
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def img_uri(name, size=640):
    return photo_data_uri(name, size)


def card_html(name, small=False):
    f = FLOWERS[name]
    return (f'<a class="card{" sm" if small else ""}" href="?bunga={name}" target="_self" style="--c:{f["color"]}">'
            f'<img src="{img_uri(name, 360)}" alt="Foto {f["local"]}">'
            f'<div class="body"><div class="nm">{f["local"]}</div><div class="lt">{f["latin"]}</div></div></a>')


def render_home():
    st.markdown('<div class="intro">Pilih bunga untuk melihat cara merawatnya</div>'
                '<div class="lede">Lima bunga yang dikenali model Bloomora. Geser ke samping untuk melihat semuanya, '
                'lalu klik salah satu untuk membuka panduan perawatannya.</div>', unsafe_allow_html=True)
    st.markdown('<div class="strip">' + "".join(card_html(n) for n in ORDER) + "</div>", unsafe_allow_html=True)

    st.markdown("### Sekilas kebutuhan tiap bunga")
    rows = "".join(
        f'<tr><td><span class="swatch" style="background:{FLOWERS[n]["color"]}"></span>'
        f'<a href="?bunga={n}" target="_self">{FLOWERS[n]["local"]}</a></td>'
        f'<td>{FLOWERS[n]["facts"]["Cahaya"]}</td><td>{FLOWERS[n]["facts"]["Air"]}</td>'
        f'<td>{FLOWERS[n]["facts"]["Suhu ideal"]}</td></tr>' for n in ORDER)
    st.markdown(f'<table class="glance"><thead><tr><th>Bunga</th><th>Cahaya</th><th>Air</th><th>Suhu ideal</th>'
                f'</tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)


def render_detail(name):
    f = FLOWERS[name]
    t = THEME[name]
    st.markdown(theme_css(name, PAPER), unsafe_allow_html=True)
    st.markdown('<a class="back" href="?" target="_self">‹ Semua bunga</a>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.markdown(f'<img class="hero-img" src="{img_uri(name, 900)}" alt="Foto {f["local"]}">',
                    unsafe_allow_html=True)
    with right:
        facts = "".join(f'<div class="fact"><div class="k">{k}</div><div class="v">{v}</div></div>'
                        for k, v in f["facts"].items())
        mark = base64.b64encode(illustration(name, bare=True).encode()).decode()
        st.markdown(f"""<div class="d-head" style="--c:{f['color']}">
<img class="d-mark" src="data:image/svg+xml;base64,{mark}" alt="">
<div class="d-mood">{t['mood']}</div>
<div class="d-name">{f['local']}</div>
<div class="d-latin"><i>{f['latin']}</i> &nbsp;·&nbsp; {name} &nbsp;·&nbsp; famili {f['family']}</div>
<div class="d-sum">{f['summary']}</div>
<div class="facts">{facts}</div></div>""", unsafe_allow_html=True)

    g = GROW[name]
    icons = {"Ketinggian": "landscape", "Contoh daerah": "location_on", "Media tanam": "grass", "Tanam di": "potted_plant"}
    cards = "".join(
        f'<div class="grow-card"><span class="material-symbols-rounded">{icons[k]}</span>'
        f'<div><small>{k}</small><b>{g[k]}</b></div></div>' for k in icons)
    st.markdown(f"""<div class="grow" style="--c:{f['color']}">
<h2>Cocok ditanam di mana?</h2>
<p class="grow-intro">{g['intro']}</p>
<div class="grow-grid">{cards}</div>
<p class="grow-low"><b>Bagaimana di dataran rendah?</b> {g['lowland']}</p>
</div>""", unsafe_allow_html=True)

    care = "".join(f'<div class="care-row"><h4>{k}</h4><p>{v}</p></div>' for k, v in f["care"].items())
    st.markdown(f'<div class="care" style="--c:{f["color"]}"><h2>Saran perawatan</h2>{care}'
                f'<div class="tipbox"><b>Perlu diketahui</b>{f["tip"]}</div>'
                f'<div class="funfact"><b>Tahukah Anda?</b>{t["fact"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="more-h">Bunga lainnya</div><div class="strip">' +
                "".join(card_html(n, small=True) for n in ORDER if n != name) + "</div>", unsafe_allow_html=True)


def page_catalog():
    sel = st.query_params.get("bunga")
    if sel in FLOWERS:
        render_detail(sel)
    else:
        render_home()




# ─────────────────────────── Cara kerja model ───────────────────────────
ACC_STORY = [  # (nama model, data, akurasi data uji, akurasi foto dunia nyata)
    ("Model awal", "1.400 foto kebun", 98.1, 53.5),
    ("+ Oxford & TF", "±2.000 foto", 93.7, 49.0),
    ("Model final", "±3.600 foto, 50 epoch", 84.8, 69.4),
]


def _first_layer_maps(model, img, n=8):
    """Peta aktivasi lapisan konvolusi pertama: apa yang 'dilihat' model di tahap awal."""
    import torch.nn.functional as F_
    from model import transform_eval
    with torch.no_grad():
        x = transform_eval(img.convert("RGB")).unsqueeze(0)
        a = F_.relu(model.bn1(model.conv1(x)))[0]
    energy = a.mean(dim=(1, 2))
    idx = torch.argsort(energy, descending=True)[:n]
    maps = []
    for i in idx:
        m = a[i].numpy()
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        maps.append(Image.fromarray(np.uint8(m * 255)).resize((160, 160), Image.NEAREST))
    return maps


def page_howto():
    st.markdown("## Cara kerja model")
    st.markdown('<div class="lede">Bloomora memakai jaringan saraf konvolusi (CNN) yang dibangun dan dilatih sendiri. '
                'Berikut perjalanan sebuah foto hingga menjadi tebakan nama bunga.</div>', unsafe_allow_html=True)

    steps = [
        ("Foto masuk", "Diperkecil ke 128×128 piksel lalu dinormalisasi.", "3 × 128 × 128"),
        ("Konvolusi 1", "32 filter 3×3 mengenali tepi dan bercak warna.", "32 × 64 × 64"),
        ("Konvolusi 2", "64 filter mengenali tekstur dan pola kelopak.", "64 × 32 × 32"),
        ("Konvolusi 3", "128 filter mengenali bentuk bagian bunga.", "128 × 16 × 16"),
        ("Lapisan dense", "512 neuron menggabungkan semua ciri; dropout 50% mencegah hafalan.", "512"),
        ("Keputusan", "Softmax mengubah skor menjadi peluang untuk 5 jenis bunga.", "5"),
    ]
    st.markdown('<div class="flow">' + "".join(
        f'<div class="flow-step"><div class="n">{i}</div><b>{t}</b><span>{d}</span><code>{shape}</code></div>'
        for i, (t, d, shape) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)

    model = classes = None
    try:
        if DEFAULT_WEIGHTS.exists():
            stt = DEFAULT_WEIGHTS.stat()
            model, classes, _ = get_model(None, str(DEFAULT_WEIGHTS), f"{stt.st_size}-{stt.st_mtime_ns}")
    except Exception:  # noqa: BLE001
        model = None

    samples = sorted(SAMPLES_DIR.glob("*.jpg"))
    if model is not None and samples:
        st.markdown("### Apa yang diperhatikan model?")
        names = {LOCAL_NAME.get(p.stem.split("_")[0], p.stem): p for p in samples if p.stem.endswith("_1")}
        pick = st.segmented_control("Pilih contoh foto", list(names), default=list(names)[0],
                                    label_visibility="collapsed")
        img = Image.open(names[pick or list(names)[0]]).convert("RGB")
        probs = predict(model, img, tta=True)
        top = int(np.argmax(probs))
        a, b, c3 = st.columns(3, gap="medium")
        a.image(img, caption="Foto asli", use_container_width=True)
        b.image(overlay_heatmap(img, grad_cam(model, img, top)), use_container_width=True,
                caption=f"Area fokus (Grad-CAM) → {classes[top]} {probs[top]*100:.0f}%")
        with c3:
            maps = _first_layer_maps(model, img)
            grid = Image.new("L", (4 * 160, 2 * 160))
            for i, m in enumerate(maps):
                grid.paste(m, ((i % 4) * 160, (i // 4) * 160))
            st.image(grid, use_container_width=True, caption="8 peta aktivasi dari konvolusi pertama")
        st.markdown('<div class="note-soft">Grad-CAM menyorot bagian foto yang paling memengaruhi keputusan. '
                    'Peta aktivasi menunjukkan bahwa lapisan awal masih menangkap hal sederhana seperti tepi, '
                    'terang-gelap, dan bercak warna. Lapisan yang lebih dalam menyusunnya menjadi bentuk kelopak.</div>',
                    unsafe_allow_html=True)

    st.markdown("### Pelajaran dari data")
    fig = go.Figure()
    labels = [f"{n}<br><span style='font-size:11px'>{d}</span>" for n, d, _, _ in ACC_STORY]
    fig.add_trace(go.Bar(name="Data uji (mirip data latih)", x=labels, y=[t for *_, t, _ in ACC_STORY],
                         marker=dict(color="#B9C6B8", cornerradius=6),
                         text=[f"{t:.1f}%" for *_, t, _ in ACC_STORY], textposition="outside"))
    fig.add_trace(go.Bar(name="Foto dunia nyata (iNaturalist)", x=labels, y=[r for *_, r in ACC_STORY],
                         marker=dict(color=LEAF, cornerradius=6),
                         text=[f"{r:.1f}%" for *_, r in ACC_STORY], textposition="outside"))
    fig.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=10, b=0), bargap=.35,
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans", color=INK), yaxis=dict(range=[0, 110], visible=False),
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("""
<div class="lessons">
  <div><b>Akurasi tinggi belum tentu pintar</b>Model awal benar 98% pada data uji, tetapi hanya 53% pada foto dari
  luar dataset. Data uji berasal dari kebun dan kamera yang sama dengan data latih, sehingga model cukup
  menghafal latar dan warna (<i>shortcut learning</i>).</div>
  <div><b>Contohnya mawar putih</b>Krisan di dataset awal hampir semuanya putih, sedangkan mawarnya merah.
  Akibatnya, model menebak setiap bunga putih sebagai krisan. Grad-CAM menunjukkan model justru memperhatikan latar.</div>
  <div><b>Data yang beragam lebih penting</b>Setelah ditambah ±2.200 foto dari iNaturalist, Oxford 102 Flowers,
  dan TensorFlow flower_photos, akurasi data uji turun karena ujiannya lebih sulit, tetapi akurasi pada foto
  dunia nyata naik dari 53% menjadi 69%.</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────── Panduan foto ───────────────────────────
def _frame(good: bool) -> str:
    """Ilustrasi kecil: satu bunga besar memenuhi bingkai vs. banyak bunga kecil dari jauh."""
    if good:
        petals = "".join(
            f'<ellipse cx="40" cy="22" rx="11" ry="17" fill="#C8305F" opacity=".9" transform="rotate({a} 40 40)"/>'
            for a in range(0, 360, 60))
        body = petals + '<circle cx="40" cy="40" r="8" fill="#F3C64B"/>'
    else:
        dots = [(18, 50), (30, 44), (44, 52), (56, 46), (26, 58), (50, 60), (38, 38), (62, 56)]
        body = ('<rect x="10" y="40" width="60" height="30" rx="6" fill="#5E8C6A" opacity=".55"/>'
                + "".join(f'<circle cx="{x}" cy="{y}" r="3.2" fill="#D9A21B"/>' for x, y in dots))
    return (f'<svg viewBox="0 0 80 80" width="64" height="64" aria-hidden="true">'
            f'<rect x="1" y="1" width="78" height="78" rx="12" fill="#F2F5F0" stroke="#DCE3DB"/>{body}</svg>')


PHOTO_GUIDE = f"""
<div class="guide">
  <div class="guide-item">
    {_frame(True)}
    <div><b class="ok">Disarankan</b><span>Foto jarak dekat satu kuntum bunga yang memenuhi bingkai, dengan pencahayaan cukup dan fokus yang tajam.</span></div>
  </div>
  <div class="guide-item">
    {_frame(False)}
    <div><b class="no">Kurang disarankan</b><span>Foto seluruh tanaman atau rangkaian bunga dari jarak jauh, sehingga bunga tampak terlalu kecil untuk dikenali.</span></div>
  </div>
</div>
"""


# ─────────────────────────── Latar: kelopak melayang ───────────────────────────
def _petal_layer() -> str:
    import random as _r
    rng = _r.Random(7)
    cols = list(PETAL.values())
    items = []
    for i in range(14):
        size = rng.randint(14, 30)
        items.append(
            f'<i style="left:{rng.uniform(2, 96):.1f}vw; width:{size}px; height:{size // 2}px; '
            f'background:{cols[i % len(cols)]}; animation-duration:{rng.uniform(22, 38):.1f}s, {rng.uniform(5, 9):.1f}s; '
            f'animation-delay:-{rng.uniform(0, 38):.1f}s, -{rng.uniform(0, 9):.1f}s;"></i>')
    return '<div class="petal-sky" aria-hidden="true">' + "".join(items) + "</div>"


st.markdown(f"""
<style>
.petal-sky {{ position:fixed; inset:0; pointer-events:none; z-index:0; overflow:hidden; }}
.petal-sky i {{ position:absolute; top:-6vh; display:block; border-radius:100% 0 100% 0; opacity:.2;
  animation-name: petal-fall, petal-sway; animation-timing-function: linear, ease-in-out;
  animation-iteration-count: infinite, infinite; animation-direction: normal, alternate; }}
@keyframes petal-fall {{ from {{ transform: translateY(0) rotate(0deg); }} to {{ transform: translateY(112vh) rotate(320deg); }} }}
@keyframes petal-sway {{ from {{ margin-left:-28px; }} to {{ margin-left:28px; }} }}
@media (prefers-reduced-motion: reduce) {{ .petal-sky {{ display:none; }} }}
[data-testid="stMain"], [data-testid="stSidebar"], header {{ position:relative; z-index:1; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.stApp {{ background: radial-gradient(1200px 600px at 100% 0%, #E7EEE4 0%, transparent 60%), {PAPER}; }}

.guide {{ display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin:.2rem 0 .9rem; }}
.guide-item {{ display:flex; gap:.85rem; align-items:center; background:white; border-radius:16px; padding:.75rem .9rem; }}
.guide-item b {{ display:block; font-size:.95rem; margin-bottom:.1rem; }}
.guide-item b.ok {{ color:{LEAF}; }}
.guide-item b.no {{ color:#B8233A; }}
.guide-item span {{ color:{MUTED}; font-size:.88rem; line-height:1.45; }}
@media (max-width: 700px) {{ .guide {{ grid-template-columns:1fr; }} }}
</style>
{_petal_layer()}
""", unsafe_allow_html=True)


def _sprig(side: str) -> str:
    """Gambar garis tangkai berdaun dengan satu bunga, gaya botani minimalis."""
    stem = "#7E9A86"
    if side == "left":
        leaves = [(78, 300, -38, 1), (84, 230, 34, -1), (76, 160, -30, 1), (82, 100, 28, -1)]
        head_y, head_c = 52, PETAL["Rose"]
        path = "M90 520 C 86 420, 70 330, 82 240 S 92 110, 84 60"
    else:
        leaves = [(92, 330, 36, -1), (86, 260, -32, 1), (94, 190, 30, -1), (88, 130, -26, 1)]
        head_y, head_c = 80, PETAL["Marigold"]
        path = "M84 520 C 92 430, 104 350, 94 260 S 82 140, 92 88"
    leaf_svg = "".join(
        f'<path d="M{x} {y} q {d*34} -14 {d*58} {-6 if d>0 else -6} q {-d*30} 18 {-d*58} 6 z" '
        f'fill="none" stroke="{stem}" stroke-width="1.6" transform="rotate({a} {x} {y})"/>'
        f'<path d="M{x} {y} l {d*50} -4" stroke="{stem}" stroke-width="1" opacity=".7" transform="rotate({a} {x} {y})"/>'
        for x, y, a, d in leaves)
    hx = 84 if side == "left" else 92
    petals = "".join(
        f'<ellipse cx="{hx}" cy="{head_y - 15}" rx="7" ry="14" fill="{head_c}" fill-opacity=".22" '
        f'stroke="{head_c}" stroke-width="1.3" transform="rotate({a} {hx} {head_y})"/>' for a in range(0, 360, 60))
    return (f'<svg class="sprig {side}" viewBox="0 0 180 520" aria-hidden="true">'
            f'<path d="{path}" fill="none" stroke="{stem}" stroke-width="2" stroke-linecap="round"/>'
            f'{leaf_svg}{petals}<circle cx="{hx}" cy="{head_y}" r="5" fill="{head_c}" fill-opacity=".6"/></svg>')


st.markdown(_sprig("left") + _sprig("right"), unsafe_allow_html=True)

# ─────────────────────────── Navigasi ───────────────────────────
PAGES = {
    "katalog": st.Page(page_catalog, title="Katalog bunga", url_path="katalog", default=True),
    "kenali": st.Page(page_identify, title="Kenali dari foto", url_path="kenali"),
    "cara": st.Page(page_howto, title="Cara kerja", url_path="cara-kerja"),
}
nav = st.navigation(list(PAGES.values()), position="hidden")

st.markdown("""<style>
.st-key-topnav { gap:.25rem; background:rgba(255,255,255,.55); border:1px solid #DCE3DB; padding:.3rem;
  border-radius:999px; width:fit-content; margin-left:auto; backdrop-filter: blur(6px); flex-wrap:nowrap !important; }
.st-key-topnav [data-testid="stPageLink"] a { border-radius:999px 999px 999px 4px; padding:.3rem 1rem .3rem .8rem;
  transition: background .2s ease; }
.st-key-topnav [data-testid="stPageLink"] a:hover { background:rgba(47,93,70,.07); }
.st-key-topnav [data-testid="stPageLink"] a:focus-visible { outline:2px solid #2F5D46; outline-offset:2px; }
.st-key-topnav p { font-weight:500; color:#4A5A50; }
@media (max-width: 700px) { .st-key-topnav { flex-wrap:wrap !important; border-radius:18px; margin-left:0; } }
.st-key-topnav [data-testid="stIconMaterial"], .st-key-topnav span[data-testid*="Icon"] { color:#7A8C80; }

.paste-hint { color:#66746B; font-size:.92rem; margin:.1rem 0 .5rem; }
.paste-hint kbd { background:white; border:1px solid #DCE3DB; border-bottom-width:2px; border-radius:6px;
  padding:0 .35rem; font-size:.82rem; font-family:inherit; }
.st-key-pastebox [data-testid="stChatInput"] { border:1.5px dashed #B9C6B8; border-radius:18px; background:white; }

/* Hiasan tangkai di kiri & kanan, hanya di layar lebar */
.sprig { position:fixed; bottom:-10px; width:170px; pointer-events:none; z-index:0; opacity:.55;
  transform-origin: 50% 100%; animation: sprig-sway 9s ease-in-out infinite alternate; }
.sprig.left { left:max(8px, calc((100vw - 1120px) / 2 - 210px)); }
.sprig.right { right:max(8px, calc((100vw - 1120px) / 2 - 210px)); animation-duration: 11s; animation-delay:-4s; }
@keyframes sprig-sway { from { transform: rotate(-1.6deg); } to { transform: rotate(1.6deg); } }
@media (max-width: 1360px) { .sprig { display:none; } }
@media (prefers-reduced-motion: reduce) { .sprig { animation:none; } }
</style>""", unsafe_allow_html=True)

b, n = st.columns([1, 1.25], vertical_alignment="center")
b.markdown(
    '<div class="brand"><div class="logo">Bloomora</div><span>Katalog bunga dan cara merawatnya</span></div>'
    '<div class="petals" style="margin-bottom:0">' + "".join(f'<i style="background:{c}"></i>' for c in PETAL.values()) + "</div>",
    unsafe_allow_html=True)
with n.container(horizontal=True, key="topnav"):
    st.page_link(PAGES["katalog"], label="Katalog bunga", icon=":material/local_florist:")
    st.page_link(PAGES["kenali"], label="Kenali dari foto", icon=":material/photo_camera:")
    st.page_link(PAGES["cara"], label="Cara kerja", icon=":material/psychology:")
_active = {"kenali": 2, "cara-kerja": 3}.get(getattr(nav, "url_path", ""), 1)
st.markdown(f"<style>.st-key-topnav > div:nth-child({_active}) a {{ background:white; "
            f"box-shadow:0 1px 3px rgba(30,43,36,.10); }} .st-key-topnav > div:nth-child({_active}) a * "
            f"{{ color:{LEAF} !important; }}</style>", unsafe_allow_html=True)
st.write("")
nav.run()