"""
EcoSort - Streamlit Demo App
-----------------------------
Upload a photo of a piece of waste and the model predicts which
recycling category it belongs to.

Run with:
    streamlit run app.py

Requires:
    - ecosort_model.h5   (produced by train.py)
    - class_indices.json (produced by train.py)
"""

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

# Friendly labels + simple recycling guidance shown to the user
CATEGORY_INFO = {
    "cardboard": ("CARDBOARD", "📦", "Flatten it and toss it in the cardboard stream."),
    "glass": ("GLASS", "🍾", "Rinse it out, keep it away from other materials."),
    "metal": ("METAL", "🥫", "Cans and tins go in the metal stream."),
    "paper": ("PAPER", "📄", "Keep it dry and free of food gunk."),
    "plastic": ("PLASTIC", "🧴", "Check the resin code, give it a rinse."),
    "trash": ("TRASH", "🗑️", "Not recyclable — general waste bin."),
}

BIN_NUMBER = {
    "cardboard": "BIN 01",
    "glass": "BIN 02",
    "metal": "BIN 03",
    "paper": "BIN 04",
    "plastic": "BIN 05",
    "trash": "BIN 00",
}


def inject_industrial_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --charcoal: #23262B;
                --steel: #33383F;
                --steel-light: #3F454E;
                --safety-yellow: #FFC72C;
                --rust: #C1440E;
                --bone: #EDEAE0;
            }

            .stApp {
                background-color: var(--charcoal);
                color: var(--bone);
            }

            /* hazard stripe top bar */
            .hazard-bar {
                height: 14px;
                width: 100%;
                background: repeating-linear-gradient(
                    45deg,
                    var(--safety-yellow),
                    var(--safety-yellow) 16px,
                    #1a1a1a 16px,
                    #1a1a1a 32px
                );
                border-radius: 2px;
                margin-bottom: 22px;
            }

            .plant-header {
                font-family: 'Oswald', sans-serif;
                font-weight: 700;
                font-size: 2.6rem;
                letter-spacing: 1px;
                color: var(--bone);
                text-transform: uppercase;
                margin-bottom: 0;
                line-height: 1.05;
            }

            .plant-subheader {
                font-family: 'Space Mono', monospace;
                color: var(--safety-yellow);
                font-size: 0.85rem;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 4px;
            }

            .plant-desc {
                font-family: 'Space Mono', monospace;
                color: #A7ADB6;
                font-size: 0.92rem;
                margin-top: 6px;
                margin-bottom: 24px;
            }

            /* upload chute panel */
            .chute-panel {
                background-color: var(--steel);
                border: 2px dashed var(--safety-yellow);
                border-radius: 6px;
                padding: 18px 18px 6px 18px;
                margin-bottom: 20px;
            }

            .chute-label {
                font-family: 'Space Mono', monospace;
                color: var(--safety-yellow);
                font-size: 0.78rem;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 8px;
            }

            [data-testid="stFileUploader"] {
                font-family: 'Space Mono', monospace;
            }

            /* the dispatch / sorting ticket result card */
            .ticket {
                background-color: var(--bone);
                color: #1a1a1a;
                border-radius: 4px;
                padding: 22px 26px;
                margin-top: 18px;
                position: relative;
                box-shadow: 0 8px 18px rgba(0,0,0,0.35);
                font-family: 'Space Mono', monospace;
                border-left: 10px solid var(--rust);
            }

            .ticket.recyclable {
                border-left: 10px solid #4C8B2B;
            }

            .ticket-eyebrow {
                font-size: 0.72rem;
                letter-spacing: 3px;
                text-transform: uppercase;
                color: #6b6b6b;
                margin-bottom: 2px;
            }

            .ticket-stamp {
                font-family: 'Oswald', sans-serif;
                font-weight: 700;
                font-size: 2.1rem;
                letter-spacing: 1px;
                text-transform: uppercase;
                display: inline-block;
                transform: rotate(-3deg);
                border: 4px solid #1a1a1a;
                padding: 2px 14px;
                margin: 6px 0 10px 0;
            }

            .ticket-stamp.recyclable {
                color: #2E5B1A;
                border-color: #2E5B1A;
            }

            .ticket-stamp.trash {
                color: var(--rust);
                border-color: var(--rust);
            }

            .ticket-bin {
                font-family: 'Space Mono', monospace;
                font-size: 0.95rem;
                font-weight: 700;
                background: #1a1a1a;
                color: var(--safety-yellow);
                display: inline-block;
                padding: 3px 10px;
                border-radius: 3px;
                letter-spacing: 1px;
                margin-bottom: 10px;
            }

            .ticket-note {
                font-size: 0.88rem;
                color: #333;
                margin-top: 6px;
            }

            .conf-row {
                font-family: 'Space Mono', monospace;
                font-size: 0.82rem;
                color: #1a1a1a;
                display: flex;
                justify-content: space-between;
                margin-top: 14px;
                border-top: 1px dashed #999;
                padding-top: 10px;
            }

            /* class probability bars */
            .prob-label {
                font-family: 'Space Mono', monospace;
                font-size: 0.82rem;
                color: var(--bone);
            }

            footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("ecosort_model.h5")
    with open("class_indices.json") as f:
        idx_to_class = json.load(f)
    # JSON keys come back as strings; convert to int for indexing
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return model, idx_to_class


def predict(image: Image.Image, model, idx_to_class):
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    ranked = sorted(
        [(idx_to_class[i], float(p)) for i, p in enumerate(preds)],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def main():
    st.set_page_config(page_title="EcoSort", page_icon="♻️", layout="centered")
    inject_industrial_css()

    st.markdown('<div class="hazard-bar"></div>', unsafe_allow_html=True)
    st.markdown('<div class="plant-subheader">iGAP Technologies — Sorting Line 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="plant-header">♻️ EcoSort</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="plant-desc">Drop a photo down the chute. The line camera calls out '
        'what it is and which bin it goes in.</div>',
        unsafe_allow_html=True,
    )

    try:
        model, idx_to_class = load_model_and_classes()
    except (OSError, FileNotFoundError):
        st.error(
            "Model files not found. Run `python train.py` first to produce "
            "`ecosort_model.h5` and `class_indices.json` in this folder."
        )
        return

    st.markdown('<div class="chute-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chute-label">📥 Feed Chute</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(" ", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="On the belt", use_container_width=True)

        with st.spinner("Scanning on the line..."):
            ranked = predict(image, model, idx_to_class)

        top_label, top_conf = ranked[0]
        display_name, emoji, guidance = CATEGORY_INFO.get(top_label, (top_label.upper(), "❔", ""))
        bin_id = BIN_NUMBER.get(top_label, "BIN ??")
        is_trash = top_label == "trash"

        ticket_class = "trash" if is_trash else "recyclable"
        verdict_text = "REJECT — NOT RECYCLABLE" if is_trash else "SORTED — RECYCLABLE"

        st.markdown(
            f"""
            <div class="ticket {ticket_class}">
                <div class="ticket-eyebrow">Sorting Ticket #{np.random.randint(1000,9999)}</div>
                <div class="ticket-stamp {ticket_class}">{emoji} {display_name}</div>
                <br>
                <span class="ticket-bin">→ {bin_id}</span>
                <div class="ticket-note"><strong>{verdict_text}</strong><br>{guidance}</div>
                <div class="conf-row">
                    <span>CONFIDENCE</span>
                    <span>{top_conf * 100:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔧 Full line diagnostics (all class probabilities)"):
            for label, conf in ranked:
                name, emoji_i, _ = CATEGORY_INFO.get(label, (label, "", ""))
                st.markdown(f'<div class="prob-label">{emoji_i} {name}: {conf * 100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(min(conf, 1.0))


if __name__ == "__main__":
    main()

