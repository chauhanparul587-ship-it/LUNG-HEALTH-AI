import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Lung Health AI",
    page_icon="🫁",
    layout="centered"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "lung_health_ai_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    st.error(
        "Model file not found. Please keep "
        "'lung_health_ai_model.pkl' in the same "
        "folder as streamlit_app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("🫁 Lung Health AI")

st.write(
    "Enter the information below to see how your "
    "information is classified by our educational "
    "machine learning model."
)

st.info(
    "This application is an educational project. "
    "It cannot diagnose lung cancer or determine "
    "your actual medical risk."
)

st.divider()


# ==========================================
# 4. PATIENT INFORMATION
# ==========================================

st.header("👤 Patient Information")

gender = st.selectbox(
    "Select your gender",
    ["M", "F"]
)

age = st.number_input(
    "Enter your age",
    min_value=1,
    max_value=120,
    value=25
)


# ==========================================
# 5. SYMPTOMS AND LIFESTYLE
# ==========================================

st.header("🩺 Symptoms and Lifestyle")

st.caption(
    "For each question, select the answer that "
    "best matches your information."
)


def yes_no_input(label):
    answer = st.selectbox(
        label,
        ["No", "Yes"],
        key=label
    )

    # Assumes the training dataset uses:
    # 1 = No and 2 = Yes.
    return 1 if answer == "No" else 2


smoking = yes_no_input("Do you smoke?")
yellow_fingers = yes_no_input(
    "Do you have yellow fingers?"
)
anxiety = yes_no_input(
    "Do you experience anxiety?"
)
peer_pressure = yes_no_input(
    "Do you experience peer pressure?"
)
chronic_disease = yes_no_input(
    "Do you have a chronic disease?"
)
fatigue = yes_no_input(
    "Do you experience fatigue?"
)
allergy = yes_no_input(
    "Do you have allergies?"
)
wheezing = yes_no_input(
    "Do you experience wheezing?"
)
alcohol = yes_no_input(
    "Do you consume alcohol?"
)
coughing = yes_no_input(
    "Do you have a cough?"
)
shortness_breath = yes_no_input(
    "Do you experience shortness of breath?"
)
swallowing = yes_no_input(
    "Do you experience swallowing difficulty?"
)
chest_pain = yes_no_input(
    "Do you experience chest pain?"
)


# ==========================================
# 6. PREPARE MODEL INPUT
# ==========================================

input_data = pd.DataFrame(
    [[
        1 if gender == "M" else 0,
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergy,
        wheezing,
        alcohol,
        coughing,
        shortness_breath,
        swallowing,
        chest_pain
    ]],
    columns=[
        "GENDER",
        "AGE",
        "SMOKING",
        "YELLOW_FINGERS",
        "ANXIETY",
        "PEER_PRESSURE",
        "CHRONIC DISEASE",
        "FATIGUE",
        "ALLERGY",
        "WHEEZING",
        "ALCOHOL CONSUMING",
        "COUGHING",
        "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY",
        "CHEST PAIN"
    ]
)


# ==========================================
# 7. ANALYZE INFORMATION
# ==========================================

st.divider()

if st.button(
    "🔍 Analyze My Information",
    use_container_width=True
):

    try:
        # Validate expected feature count.
        expected_count = getattr(
            model,
            "n_features_in_",
            None
        )

        if (
            expected_count is not None
            and input_data.shape[1] != expected_count
        ):
            st.error(
                f"The model expects {expected_count} "
                f"features, but the application supplies "
                f"{input_data.shape[1]}."
            )
            st.stop()

        # Match the model's recorded feature order.
        if hasattr(model, "feature_names_in_"):

            expected_columns = list(
                model.feature_names_in_
            )

            if set(expected_columns) != set(
                input_data.columns
            ):
                st.error(
                    "The application features do not "
                    "match the model's training features."
                )

                st.write(
                    "Expected features:",
                    expected_columns
                )
                st.stop()

            input_data = input_data[
                expected_columns
            ]

        # Generate prediction.
        prediction = model.predict(input_data)[0]

        st.divider()
        st.header("📊 Your Result")

        # ----------------------------------
        # Obtain positive-class probability
        # ----------------------------------

        positive_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = list(model.classes_)

            # The training target is assumed to use
            # YES as the positive class.
            positive_indices = [
                i for i, label in enumerate(classes)
                if str(label).strip().upper() == "YES"
            ]

            if positive_indices:
                positive_probability = float(
                    probabilities[positive_indices[0]]
                )

        result = str(prediction).strip().upper()

        if result in ["YES", "1", "1.0"]:

            st.error(
                "The model has classified your "
                "information in the positive class."
            )

            st.write(
                "Based on the information you entered, "
                "the model predicts the class associated "
                "with lung cancer in its training data."
            )

            st.write(
                "This result does not mean that you "
                "have lung cancer."
            )

        elif result in ["NO", "0", "0.0"]:

            st.success(
                "The model has classified your "
                "information in the negative class."
            )

            st.write(
                "Based on the information you entered, "
                "the model predicts the class not "
                "associated with lung cancer in its "
                "training data."
            )

            st.write(
                "This result does not guarantee that "
                "you are free from lung cancer."
            )

        else:

            st.warning(
                f"The model returned the class: {prediction}"
            )

            st.write(
                "Please verify the target labels used "
                "when training the model."
            )

        # ----------------------------------
        # Display model probability
        # ----------------------------------

        if positive_probability is not None:

            percentage = positive_probability * 100

            st.subheader(
                "📈 Model's Positive-Class Probability"
            )

            st.metric(
                label="Predicted probability",
                value=f"{percentage:.1f}%"
            )

            st.progress(
                max(
                    0.0,
                    min(1.0, positive_probability)
                )
            )

            st.caption(
                "This percentage is the model's estimated "
                "probability for the positive class. It is "
                "not a clinically established percentage "
                "of your actual chance of having cancer."
            )

        else:

            st.info(
                "This model does not expose class "
                "probabilities, so a percentage cannot "
                "be calculated reliably from its prediction."
            )

        # ----------------------------------
        # Suggested next steps
        # ----------------------------------

        st.divider()
        st.subheader("💡 What should you do next?")

        st.write(
            "If you have persistent symptoms or concerns "
            "about your lung health, speak with a qualified "
            "healthcare professional. Do not use this "
            "application to decide whether you need "
            "medical testing or treatment."
        )

        st.warning(
            "dont rely on this model expert this model is developed for educational purposes only. "
        )

        # ----------------------------------
        # Optional: show submitted information
        # ----------------------------------

        with st.expander("View submitted information"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Unable to analyze the information: {e}"
        )

        st.info(
            "Check the model's training features, "
            "feature encoding, and preprocessing."
        )


# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Lung Health AI | MCA Machine Learning Project"
)
