# ============================================================
# HS4 CODE PREDICTION APP
# Word + Character TF-IDF + Calibrated Linear SVM
# ============================================================

import math
import os
import tempfile
import urllib.request
from pathlib import Path

import joblib
import streamlit as st


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HS4 Code Predictor",
    page_icon="📦",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

# Version-specific filename prevents reuse of the old model.
MODEL_FILE = BASE_DIR / "final_hs4_calibrated_svm_v2.pkl"

DEFAULT_MODEL_URL = (
    "https://github.com/Ayesha-Ansa/hs4-code-prediction/"
    "releases/download/v2.0.0/final_hs4_calibrated_svm.pkl"
)

REVIEW_THRESHOLD = 0.50
EXPECTED_CLASSES = 236
MAX_DESCRIPTION_LENGTH = 5000


# ============================================================
# 2. DOWNLOAD AND LOAD THE MODEL
# ============================================================

def get_model_path():
    """Download the version 2 model if it is not already available."""

    if MODEL_FILE.is_file() and MODEL_FILE.stat().st_size > 0:
        return MODEL_FILE

    temporary_path = None

    try:
        request = urllib.request.Request(
            DEFAULT_MODEL_URL,
            headers={"User-Agent": "HS4-Code-Predictor"},
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".pkl",
                dir=str(BASE_DIR),
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    temporary_file.write(chunk)

        if temporary_path.stat().st_size == 0:
            raise ValueError("The downloaded model file is empty.")

        # Only load this project's trusted model release.
        downloaded_model = joblib.load(temporary_path)
        validate_model(downloaded_model)

        os.replace(temporary_path, MODEL_FILE)
        temporary_path = None

        return MODEL_FILE

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def validate_model(candidate):
    """Check the model interface and expected class count."""

    if not hasattr(candidate, "predict_proba"):
        raise TypeError("The model does not support predict_proba().")

    if not hasattr(candidate, "classes_"):
        raise TypeError("The model does not expose its learned classes.")

    if len(candidate.classes_) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} classes, "
            f"but the model contains {len(candidate.classes_)}."
        )


@st.cache_resource
def load_model():
    """Load and cache the version 2 model."""

    model_path = get_model_path()
    loaded_model = joblib.load(model_path)
    validate_model(loaded_model)

    return loaded_model


# ============================================================
# 3. HEADER
# ============================================================

st.title("📦 HS4 Code Predictor")
st.subheader("Product Classification Assistant")

st.write(
    "Enter a product description to explore three candidate "
    "4-digit HS headings."
)

st.caption(
    "Include the product's material, function and construction "
    "where possible. Predictions require human verification."
)


# ============================================================
# 4. LOAD MODEL
# ============================================================

try:
    with st.spinner("Loading classification model..."):
        model = load_model()

except Exception as exc:
    st.error(
        "The classification model could not be loaded. "
        "Please check the deployment configuration."
    )

    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")

    st.stop()


# ============================================================
# 5. PRODUCT INPUT
# ============================================================

with st.form("prediction_form"):
    product_text = st.text_area(
        "Product description",
        placeholder=(
            "Example: Men's knitted trousers made from "
            "100% polyester, with an elastic waistband "
            "and side pockets."
        ),
        height=150,
        max_chars=MAX_DESCRIPTION_LENGTH,
    )

    submitted = st.form_submit_button(
        "Predict HS4 Code",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 6. PREDICTION
# ============================================================

if submitted:
    cleaned_text = product_text.lower().strip()

    if not cleaned_text:
        st.warning("Please enter a product description.")

    else:
        try:
            with st.spinner("Analyzing product description..."):
                probabilities = model.predict_proba([cleaned_text])[0]
                classes = model.classes_

                if len(probabilities) != len(classes):
                    raise ValueError(
                        "Prediction scores do not match the model classes."
                    )

                if not all(
                    math.isfinite(float(score))
                    and 0.0 <= float(score) <= 1.0
                    for score in probabilities
                ):
                    raise ValueError(
                        "The model returned invalid prediction scores."
                    )

                if not math.isclose(
                    float(probabilities.sum()),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        "The prediction scores do not sum to one."
                    )

                top_indices = probabilities.argsort()[::-1][:3]
                top_score = float(probabilities[top_indices[0]])

            # ------------------------------------------------
            # REVIEW WARNING
            # ------------------------------------------------

            if top_score < REVIEW_THRESHOLD:
                st.warning(
                    "⚠️ Uncertain classification — manual review recommended."
                )

                st.write(
                    f"The highest model score is **{top_score:.2%}**, "
                    f"below the **{REVIEW_THRESHOLD:.0%}** review threshold."
                )

                st.info(
                    "The description may be unfamiliar, ambiguous, or "
                    "belong to a heading absent from training. "
                    "The score cannot determine which explanation applies."
                )

                st.subheader("Low-score candidates")

            else:
                st.success("Candidate headings generated.")
                st.subheader("Top-3 HS4 Candidates")

                st.caption(
                    "The score exceeds the review threshold. "
                    "Verify the candidates before using a classification."
                )

            # ------------------------------------------------
            # CANDIDATE RESULTS
            # ------------------------------------------------

            for rank, index in enumerate(top_indices, start=1):
                heading = str(classes[index])
                score = float(probabilities[index])

                st.markdown(f"### {rank}. HS4 **{heading}**")
                st.write(f"Model score: **{score:.2%}**")
                st.progress(max(0.0, min(1.0, score)))

                if rank < len(top_indices):
                    st.divider()

            st.caption(
                "Model scores rank supported headings. They are not "
                "guarantees of correctness. A high score does not prove "
                "the true heading is supported, and a low score does not "
                "prove it was excluded."
            )

        except Exception as exc:
            st.error("The prediction could not be generated.")

            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")


# ============================================================
# 7. MODEL INFORMATION
# ============================================================

with st.expander("About this model"):
    st.write("**Version:** 2.0.0")
    st.write(
        "**Method:** Word + Character TF-IDF "
        "with a calibrated Linear SVM"
    )
    st.write(f"**Supported headings:** {len(model.classes_)}")
    st.write(f"**Review threshold:** {REVIEW_THRESHOLD:.0%}")

    st.markdown("#### Recorded test performance")

    col1, col2 = st.columns(2)
    col1.metric("Top-1 accuracy", "67.37%")
    col2.metric("Top-3 accuracy", "79.53%")

    st.write("**Macro F1:** 0.5294")
    st.write("**Test records:** 2,648")

    st.write(
        "At the 0.50 threshold, Top-1 accuracy was **91.00%** "
        "among the **50.38% of test records** scoring at least "
        "that threshold."
    )

    st.caption(
        "These results describe a filtered, supported-class test split. "
        "They do not establish accuracy across all products or "
        "unsupported headings. Source labels may contain errors, "
        "and related descriptions may occur across dataset splits."
    )

    st.markdown("#### Check heading support")

    heading_to_check = st.text_input(
        "Enter a four-digit heading",
        placeholder="Example: 6103",
        max_chars=4,
        key="heading_support",
    ).strip()

    if heading_to_check:
        if len(heading_to_check) != 4 or not heading_to_check.isdecimal():
            st.caption("Enter exactly four digits.")

        elif heading_to_check in set(map(str, model.classes_)):
            st.info(
                f"HS4 {heading_to_check} is represented in training. "
                "This does not establish that it is correct for your product."
            )

        else:
            st.warning(
                f"HS4 {heading_to_check} is not represented in this model. "
                "The model cannot return it as a candidate."
            )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Portfolio and educational project by Ayesha Ansari. "
    "HS4 headings are not complete country-specific tariff codes. "
    "Verify classifications using authoritative tariff guidance "
    "and qualified review."
)
