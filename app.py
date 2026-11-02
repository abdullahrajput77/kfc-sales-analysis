import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="KFC Order Type Predictor", page_icon="🍗")
st.title("KFC Order Type Prediction")
st.write("Predict an order type using your trained Decision Tree model.")

@st.cache_resource
def load_artifacts():
    # Load only the model file you created and trust.
    return joblib.load("kfc_decision_tree_model.joblib")

try:
    artifacts = load_artifacts()
    model = artifacts["model"]
    label_encoders = artifacts["label_encoders"]
    target_encoder = artifacts["target_encoder"]
    features = artifacts["features"]
except Exception as e:
    st.error("Could not load the model. Check that kfc_decision_tree_model.joblib is in the same folder as app.py.")
    st.exception(e)
    st.stop()

st.subheader("Enter order details")

inputs = {}
for feature in features:
    if feature == "Quantity":
        inputs[feature] = st.number_input("Quantity", min_value=1, step=1, value=1)
    else:
        encoder = label_encoders.get(feature)
        if encoder is None:
            st.error(f"No saved label encoder was found for {feature}.")
            st.stop()
        options = [str(value) for value in encoder.classes_]
        if not options:
            st.error(f"No options are available for {feature}.")
            st.stop()
        inputs[feature] = st.selectbox(feature.replace("_", " "), options)

if st.button("Predict Order Type", type="primary"):
    try:
        row = {}
        for feature in features:
            value = inputs[feature]
            if feature in label_encoders:
                encoder = label_encoders[feature]
                if value not in encoder.classes_:
                    st.error(f"Unknown value for {feature}.")
                    st.stop()
                row[feature] = int(encoder.transform([value])[0])
            else:
                row[feature] = value

        input_df = pd.DataFrame([row], columns=features)
        prediction = model.predict(input_df)
        predicted_label = target_encoder.inverse_transform(prediction)[0]
        st.success(f"Predicted Order Type: {predicted_label}")
    except Exception as e:
        st.error("Prediction could not be completed.")
        st.exception(e)

st.caption("This app uses your saved KFC Decision Tree model.")
