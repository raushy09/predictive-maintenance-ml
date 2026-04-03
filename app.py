import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline model
model = pickle.load(open("model.pkl", "rb"))

st.title("🔧 Predictive Maintenance System")

st.sidebar.header("Choose Input Method")
option = st.sidebar.radio("Select", ["Manual Input", "Upload CSV"])

# ==============================
# 🔹 Manual Input Section
# ==============================
if option == "Manual Input":
    st.subheader("Enter Machine Parameters")

    # Inputs
    type_input = st.selectbox("Type", ["L", "M", "H"])
    tool_wear = st.number_input("Tool wear [min]")
    temp_diff = st.number_input("Temperature Difference")
    power = st.number_input("Mechanical Power [W]")

    if st.button("Predict"):
        # Create dataframe
        input_data = pd.DataFrame({
            "Type": [type_input],
            "Tool wear [min]": [tool_wear],
            "Temp_diff": [temp_diff],
            "Mechanical Power [W]": [power]
        })

        # Prediction
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)

        # Output
        st.write(f"No Failure Probability: {prob[0][0] * 100:.2f}%")
        st.write(f"Failure Probability: {prob[0][1] * 100:.2f}%")

# ==============================
# 🔹 CSV Upload Section
# ==============================
elif option == "Upload CSV":
    st.subheader("Upload CSV File")

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file is not None:
        # Read file
        data = pd.read_csv(file)

        # Required columns
        required_cols = ["Type", "Tool wear [min]", "Temp_diff", "Mechanical Power [W]"]

        # Check columns
        if not all(col in data.columns for col in required_cols):
            st.error("❌ CSV must contain required columns!")
        else:
            # Keep only required columns
            data = data[required_cols]

            st.write("Preview of Data:", data.head())

            if st.button("Predict on CSV"):
                # Prediction
                predictions = model.predict(data)
                data["Prediction"] = predictions

                st.success("✅ Prediction Completed")

                st.write(data.head())

                # Download button
                st.download_button(
                    "Download Results",
                    data.to_csv(index=False),
                    "output.csv"
                )