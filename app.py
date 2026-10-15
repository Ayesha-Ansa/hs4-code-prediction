# ============================================================
# HS4 CODE PREDICTION — VERSION 3.0.0
# Word + Character TF-IDF + Calibrated Linear SVM
# ============================================================

import hashlib
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

# A new filename prevents reuse of an earlier downloaded model.
MODEL_FILE = BASE_DIR / "hs4_model_v3.pkl"

MODEL_URL = (
    "https://github.com/Ayesha-Ansa/hs4-code-prediction/"
    "releases/download/v3.0.0/final_hs4_calibrated_svm.pkl"
)

MODEL_SHA256 = (
    "e4f905bc09313d20b52e127e88e994b83"
    "c330cc2e6535663eeee7102ff9899eb"
)

EXPECTED_CLASSES = 252
MAX_DESCRIPTION_LENGTH = 5000


# ============================================================
# 2. MODEL DOWNLOAD AND VALIDATION
# ============================================================

def file_checksum(path):
    digest = hashlib.sha256()

    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_model_path():
    """Use a verified local copy or download the version 3 artifact."""

    if MODEL_FILE.is_file():
        if file_checksum(MODEL_FILE) == MODEL_SHA256:
            return MODEL_FILE

    temporary_path = None

    try:
        request = urllib.request.Request(
            MODEL_URL,
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

        if file_checksum(temporary_path) != MODEL_SHA256:
            raise ValueError(
                "The downloaded model does not match the expected release."
            )

        os.replace(temporary_path, MODEL_FILE)
        temporary_path = None

        return MODEL_FILE

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@st.cache_resource
def load_model():
    """Load the verified model and cache it for predictions."""

    model = joblib.load(get_model_path())

    if not hasattr(model, "predict_proba"):
        raise TypeError("The model does not support probability estimates.")

    if not hasattr(model, "classes_"):
        raise TypeError("The model does not expose its learned headings.")

    if len(model.classes_) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_CLASSES} headings, "
            f"but found {len(model.classes_)}."
        )

    return model


# ============================================================
# 3. HEADER AND MODEL LOADING
# ============================================================

st.title("📦 HS4 Code Predictor")
st.subheader("Product Classification Assistant")

st.write(
    "Enter a product description to explore three candidate "
    "four-digit HS headings."
)

st.caption(
    "Include material, function and construction details. "
    "This prototype supports human review."
)

try:
    with st.spinner("Loading classification model..."):
        model = load_model()

except Exception as exc:
    st.error("The classification model could not be loaded.")

    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")

    st.stop()


# ============================================================
# 4. PRODUCT INPUT AND PREDICTION
# ============================================================

with st.form("prediction_form"):
    product_text = st.text_area(
        "Product description",
        placeholder=(
            "Describe one product, including its material, "
            "main function and construction."
        ),
        height=150,
        max_chars=MAX_DESCRIPTION_LENGTH,
    )

    submitted = st.form_submit_button(
        "Show Candidate Headings",
        type="primary",
        use_container_width=True,
    )

if submitted:
    cleaned_text = product_text.lower().strip()

    if not cleaned_text:
        st.warning("Please enter a product description.")

    else:
        try:
            with st.spinner("Analyzing description..."):
                probabilities = model.predict_proba([cleaned_text])[0]
                classes = model.classes_

                if len(probabilities) != len(classes):
                    raise ValueError(
                        "Prediction scores do not match the model headings."
                    )

                if not all(
                    math.isfinite(float(score))
                    and 0.0 <= float(score) <= 1.0
                    for score in probabilities
                ):
                    raise ValueError("The model returned invalid scores.")

                if not math.isclose(
                    float(probabilities.sum()),
                    1.0,
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    raise ValueError("The model scores do not sum to one.")

                top_indices = probabilities.argsort()[::-1][:3]

            st.warning(
                "Manual verification required. These are candidate "
                "headings, and the correct heading may be absent."
            )

            st.subheader("Top-3 Candidate Headings")

            for rank, index in enumerate(top_indices, start=1):
                heading = str(classes[index])
                score = float(probabilities[index])

                st.markdown(f"### {rank}. HS4 **{heading}**")
                st.write(f"Model score: **{score:.2%}**")
                st.progress(max(0.0, min(1.0, score)))

                if rank < len(top_indices):
                    st.divider()

            st.caption(
                "Scores rank the headings learned during training. "
                "They are not precise chances of correctness. "
                "A high score does not prove that the true heading "
                "is supported by this model."
            )

        except Exception as exc:
            st.error("The prediction could not be generated.")

            with st.expander("Technical details"):
                st.code(f"{type(exc).__name__}: {exc}")


# ============================================================
# 5. MODEL INFORMATION
# ============================================================

with st.expander("About this model"):
    st.write("**Version:** 3.0.0")
    st.write(
        "**Method:** Word + Character TF-IDF "
        "with a calibrated Linear SVM"
    )
    st.write(f"**Supported headings:** {len(model.classes_)}")
    st.write("**Training records:** 11,904")
    st.write("**Development holdout records:** 2,976")

    col1, col2 = st.columns(2)
    col1.metric("Top-1 accuracy", "69.83%")
    col2.metric("Top-3 accuracy", "81.59%")

    st.write("**Macro F1:** 0.5877")
    st.write("**Majority-class baseline:** 7.09%")

    st.caption(
        "The holdout was inspected during development and is not "
        "an independent final benchmark. Source labels may contain "
        "errors, and related descriptions may occur across splits. "
        "Performance on unfamiliar products varies."
    )

    st.info(
        "No validated review threshold or unsupported-class detector "
        "is established for this model."
    )

    st.markdown("#### Check heading support")

    heading_to_check = st.text_input(
        "Enter a four-digit heading",
        placeholder="Example: 6204",
        max_chars=4,
    ).strip()

    if heading_to_check:
        if (
            len(heading_to_check) != 4
            or not heading_to_check.isascii()
            or not heading_to_check.isdigit()
        ):
            st.caption("Enter exactly four digits.")

        elif heading_to_check in set(map(str, model.classes_)):
            st.info(
                f"HS4 {heading_to_check} is represented in training. "
                "This does not establish that it is correct for your product."
            )

        else:
            st.warning(
                f"HS4 {heading_to_check} is not represented in this model "
                "and cannot appear among its predictions."
            )


# ============================================================
# 6. FOOTER
# ============================================================

st.divider()

st.caption(
    "Portfolio project by Ayesha Ansari. "
    "HS4 headings are not complete country-specific tariff codes. "
    "Verify classifications using authoritative tariff guidance "
    "and qualified review."
)
