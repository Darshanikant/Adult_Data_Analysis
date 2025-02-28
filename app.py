import streamlit as st
import pickle
import numpy as np
import base64

# Load the trained model
model = pickle.load(open(r"C:\Users\sunil\DK\VSCODE\Resume project\Adult Data Analysis\Salary prediction.pkl", "rb"))

# Encoding mappings (Ensure these match the ones used during training)
work_class_mapping = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4, "State-gov": 5, "Without-pay": 6, "Never-worked": 7}
education_mapping = {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3, "Prof-school": 4, "Assoc-acdm": 5, "Assoc-voc": 6, "9th": 7, "7th-8th": 8, "12th": 9, "Masters": 10, "1st-4th": 11, "10th": 12, "Doctorate": 13, "5th-6th": 14, "Preschool": 15}
marital_status_mapping = {"Married-civ-spouse": 0, "Divorced": 1, "Never-married": 2, "Separated": 3, "Widowed": 4, "Married-spouse-absent": 5, "Married-AF-spouse": 6}
occupation_mapping = {"Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9, "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13}
relation_family_mapping = {"Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5}
race_mapping = {"White": 0, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 2, "Other": 3, "Black": 4}
sex_mapping = {"Male": 0, "Female": 1}
country_mapping = {"United-States": 0, "India": 7, "Canada": 4, "Mexico": 20}

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()
    css_code = f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
    </style>
    """
    st.markdown(css_code, unsafe_allow_html=True)

# Set the background image
set_background(r"C:\Users\sunil\DK\VSCODE\Resume project\Adult Data Analysis\Screenshot 2025-02-28 122738.png")  # Make sure this image is in the same directory

# Streamlit UI
#st.set_page_config(page_title="Adult Salary Prediction", page_icon="💰", layout="wide")

st.title("Adult Salary Prediction App 💰")
st.markdown("""Welcome to the **Adult Salary Prediction App**! 

This application predicts whether an individual's income is **greater than 50K** or **less than 50K** based on various features such as age, education, work class, and more. 

To get started, **enter your details in the sidebar** and click **Predict Salary**! 🚀
""")

# Sidebar Inputs
st.sidebar.header("Enter Your Details")
age = st.sidebar.number_input("Age", min_value=17, max_value=90, value=30)
work_class = st.sidebar.selectbox("Work Class", options=list(work_class_mapping.keys()))
education = st.sidebar.selectbox("Education", options=list(education_mapping.keys()))
marital_status = st.sidebar.selectbox("Marital Status", options=list(marital_status_mapping.keys()))
occupation = st.sidebar.selectbox("Occupation", options=list(occupation_mapping.keys()))
relation_family = st.sidebar.selectbox("Relation with Family", options=list(relation_family_mapping.keys()))
race = st.sidebar.selectbox("Race", options=list(race_mapping.keys()))
sex = st.sidebar.selectbox("Sex", options=list(sex_mapping.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
work_hr_week = st.sidebar.number_input("Work Hours per Week", min_value=1, max_value=100, value=40)
country = st.sidebar.selectbox("Country", options=list(country_mapping.keys()))

# Predict Button on Main Page
if st.button("Predict Salary", use_container_width=True):
    input_data = np.array([[
        age, work_class_mapping[work_class], education_mapping[education], 
        marital_status_mapping[marital_status], occupation_mapping[occupation], 
        relation_family_mapping[relation_family], race_mapping[race], 
        sex_mapping[sex], capital_gain, capital_loss, work_hr_week, country_mapping[country]
    ]], dtype=np.float32)
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Predicted Income: **>50K** 💰")
    else:
        st.warning("Predicted Income: **<=50K**")
