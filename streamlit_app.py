import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Initialize session state variables
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None

# Streamlit UI setup
st.title("ASD Prediction System")
input_method = st.sidebar.radio("Input Method", ["Upload CSV", "Fill Form"])

# Data Preprocessing
def preprocess_data(data):
    binary_cols = ['Speech Delay/Language Disorder', 
                   'Learning disorder', 'Genetic_Disorders', 
                   'Global developmental delay/intellectual disability', 
                   'Social/Behavioural Issues', 
                   'Anxiety_disorder',
                    'Depression',
                    'Jaundice'
                    'Family_mem_with_ASD']
    data[binary_cols] = data[binary_cols].apply(lambda x: x.str.capitalize().map({'Yes': 1, 'No': 0}))
    return data

# Model Training
def train_models(X, y):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(), cat_cols)])
    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
    return preprocessor, rf_model, xgb_model, X_test, y_test

# Main Logic
if input_method == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = preprocess_data(data)
        
        X = data.drop(columns=['ASD_traits'])
        y = data['ASD_traits']
        
        st.session_state.preprocessor, st.session_state.rf_model, st.session_state.xgb_model, X_test, y_test = train_models(X, y)
        
        if st.button("Evaluate Random Forest Model"):
            y_pred = st.session_state.rf_model.predict(X_test)
            st.write(f"Accuracy: {st.session_state.rf_model.score(X_test, y_test):.2f}")



elif input_method == "Fill Form":
    user_input = {
        'Qchat_10_Score': st.slider("Q-CHAT-10 Score", 0, 50),
        'A1': st.slider("A1 Score", 0, 10),
        # Add more fields here...
    }
    
    if st.button("Predict"):
        user_df = pd.DataFrame([user_input])
        user_processed = st.session_state.preprocessor.transform(user_df)
        prediction = st.session_state.rf_model.predict(user_processed)
        st.success(f"Prediction: {'ASD' if prediction[0] == 1 else 'Non-ASD'}")

