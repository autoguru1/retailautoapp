# Importing the Libraries
import streamlit as st
import pandas as pd
#import joblib
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import numpy as np

# Load the encoders and model
te_sales_program = load('te_sales_program.pkl')
le_payment_frequency =load('le_payment_frequency.pkl')
te_applicant_state = load('te_applicant_state.pkl')
model = load('model.pkl')

# Functions for encoding
def transform_with_label_encoder(le, series):
    classes = list(le.classes_)
    series = series.apply(lambda x: x if x in classes else 'Other')
    if 'Other' not in classes:
        le.classes_ = np.append(le.classes_, 'Other')
    return le.transform(series)

def transform_with_target_encoder(te, series):
    encoded_series = te.transform(series)
    #st.write(f"Encoded series for {series.name}:", encoded_series.head())
    return encoded_series

# Streamlit app configuration
st.set_page_config(page_title="Auto-Approval Prediction Algorithm", layout="centered")

# Custom CSS to inject for styled title
st.markdown(
    """
    <style>
    .title {
        background-color: #C3231D;  /* Red background */
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the styled title using HTML with inline style for white text
st.markdown('<div class="title"><h1 style="color: #FFFFFF;">Retail Auto-Approval Prediction Algorithm</h1></div>', unsafe_allow_html=True)

# Sidebar for selecting input method
input_method = st.sidebar.radio("Choose input method:", ("Enter Details Manually", "Upload File"))

if input_method == "Enter Details Manually":
    st.subheader("Enter the details to predict auto approval")
    # Input fields for manual entry
    CSM_SCORE = st.number_input("CSM_SCORE", min_value=-1, max_value=1500, value=671, step=1)
    MaxDpdAll = st.number_input("MaxDpdAll", min_value=0, max_value=5000, value=0, step=1)
    CtRepo3 = st.number_input("CtRepo3", min_value=0, max_value=20, value=0, step=1)
    FINANCED_AMT_ORIG = st.number_input("FINANCED_AMT_ORIG", min_value=500.0, max_value=15000000.0, value=54910.52, step=1.0)
    SmFinAmtOri1 = st.number_input("SmFinAmtOri1", min_value=0.0, max_value=35730498.25, value=0.0, step=1.0)
    SmFinAmtOriOpen = st.number_input("SmFinAmtOriOpen", min_value=0.0, max_value=50730815.15, value=0.0, step=1.0)
    CtNote60 = st.number_input("CtNote60", min_value=0, max_value=500, value=1, step=1)
    PAYMENT_FREQUENCY = st.selectbox("PAYMENT_FREQUENCY", ['Annual', 'Monthly', 'Quarterly', 'SemiAnnual', 'Other'])
    SALES_PROGRAM_GROUP = st.selectbox("SALES_PROGRAM_GROUP", ['110', '120', '130', '140', '150', '1500', '1501', '1600', '1601', '160A', '310', '311', '315', '500', '510', '6000', '6001',\
                                    '6002', '6021', '6021A', '6021CA', '6021H', '6021P', '6021T', '6021TI', '6021Z', '6022EW', '6023', '6023A', '6023B', '6068S', '6068Z', '6070', '6070P',\
                                    '6070ST', '6070TI', '6071', '6076CQ', '6076DC', '6077DC', '6078', '6078FP', '6078H', '6078WP', '6079TB', '6105', '6108', '6108T', '6109C', '6109CF',\
                                    '6109CI', '6109D', '6109F', '6109H', '6109HF', '6109HI', '6109M', '6109MF', '6109MI', '6109N', '6109NC', '6109RP', '6109TA', '6109TC', '6109TM', \
                                    '6109TN', '6109U', '6109X', '6110A', '6110TS', '6150D', '6150FV', '6150G', '6150K', '6150L', '6150M', '6150T', '6150U', '6150WT', '6150X', '6151C', \
                                    '6198', '6198T', '6214', '6215', '6227ER', '6227ES', '6227EW', '6227IA', '6228ER', '6228ES', '6228EW', '6237', '6237A', '6238', '6239', '6239A', \
                                    '6239B', '6239C', '6240', '6240A', '6240HU', '6241', '6241A', '6241B', '6241C', '6242', '6243', '6244', '6245', '6245A', '6245B', '6245C', '6246',\
                                    '6247', '6249', '6260', '6260M', '6261', '6261A', '6261B', '6261C', '6261M', '6261MD', '6262B', '6262C', '6262D', '6263', '6263A', '6263B', '6263C',\
                                    '6263S', '6264', '6264A', '6264B', '6264C', '6264D', '6265', '6265A', '6265B', '6266', '6267', '6271', '6328', '6506', '6510', '6555', '6598', '6670T',\
                                    '6680T', '6683T', '6683Z', '6686', '6686T', '6686Z', '6690T', '6703N', '6703T', '6704N', '6704T', '6705N', '6705T', '6711', '6787T', '6789T', '6800', \
                                    '6855', '6865', '6867', '6868T', '6869T', '6870', '6870T', '6871', '6890T', '6899N', '6899T', '6899TD', '6899TI', '6899Z', '6900', '6901', '6902', \
                                    '6967Z', '6970Z', '6971Z', '717', '718', '9000', '9001', '9005', '9005C', '9005CF', '9005I', '9005TB', '9005U', '9005UD', '9005UW', '9005WA', '9006',\
                                    '9006UD', '9007T', '9007UD', '9008TC', '9008UD', '9009U', '9009UW', '9010T', '9011TC', '9021BA', '9021C', '9021CF', '9021I', '9021S', '9021TB', '9021TC',\
                                    '9021TS', '9023C', '9040BA', '9040C', '9040CF', '9040I', '9040TB', '9040TC', '9040TM', '9040TN', '9040TS', '9040U', '9040UP', '9041U', '9047TC',\
                                    '9080BA', '9080C', '9080CH', '9081DW', '9081T', '9081U', '9081UW', '9081W', '9081WA', '9081WH', '9082TW', '9082U', '9082UW', '9083UW', '9084UW', \
                                    '9085TW', '9085UW', '9086TW', '9086UW', '9087TW', '9087UW', '9088UW', '9090BA', '9090UP', '9092W', '9093UW', '9094UW', '9098LR', '9099LR', '9100U',\
                                    '9100UW', '9101C', '9101W', '9102C', '9102W', '9103UW', '9110', '9118TW', '9138', '9138T', '9140C', '9140U', '9140UW', '9188', '9427ER', '9427ES',\
                                    '9427EW', '9427IA', '9427T', '9428ER', '9428ES', '9428EW', '9501', '9501C', '9501U', '9502', '9502TC', '9503', '9514', '9514A', '9520', '9521', \
                                    '9522', '9629', '977', '978', '9900', '995', '9970Z', 'Other'])                                              
    APPLICANT_STATE = st.selectbox("APPLICANT_STATE", ['AB', 'AK', 'AL', 'AR', 'AZ', 'BC', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', \
                                    'MB', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NB', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NL', 'NM', 'NS', 'NT', 'NV', 'NY', 'OH', 'OK', 'ON', 'OR', 'PA',\
                                    'PE', 'PR', 'QC', 'RI', 'SC', 'SD', 'SK', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY', 'YT', 'Other'])
    
    # Predict button
    if st.button("Predict"):
        # Create a DataFrame for the input
        df_input = pd.DataFrame({
            'CSM_SCORE': [CSM_SCORE],
            'MaxDpdAll': [MaxDpdAll],
            'CtRepo3': [CtRepo3],
            'FINANCED_AMT_ORIG': [FINANCED_AMT_ORIG],
            'SmFinAmtOri1': [SmFinAmtOri1],
            'SmFinAmtOriOpen': [SmFinAmtOriOpen],
            'CtNote60': [CtNote60],
            'PAYMENT_FREQUENCY': [PAYMENT_FREQUENCY],
            'SALES_PROGRAM_GROUP': [SALES_PROGRAM_GROUP],
            'APPLICANT_STATE': [APPLICANT_STATE]
        })

        # Encode the input
        df_input['Encoded_Payment_Frequency'] = transform_with_label_encoder(le_payment_frequency, df_input['PAYMENT_FREQUENCY'])
        df_input['Encoded_Sales_Program'] = transform_with_target_encoder(te_sales_program, df_input['SALES_PROGRAM_GROUP'])
        df_input['Encoded_Applicant_State'] = transform_with_target_encoder(te_applicant_state, df_input['APPLICANT_STATE'])

        X = df_input[['CSM_SCORE', 'MaxDpdAll', 'CtRepo3', 'FINANCED_AMT_ORIG', 'SmFinAmtOri1', 'SmFinAmtOriOpen', 'CtNote60', 'Encoded_Payment_Frequency', 'Encoded_Sales_Program', 'Encoded_Applicant_State']]
        
        # Make prediction and determine the class based on the probability
        pred_proba = model.predict_proba(X)[:, 1]
        pred_class = 1 if pred_proba[0] > 0.8 else 0
        class_label = "AutoApprove" if pred_class == 1 else "ManualAssess"
        
        # Display the prediction result
        st.success(f"Prediction: {class_label}")
        st.info(f"Probability: {pred_proba[0]:.3f}")

elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Check the file extension and load to dataframe
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            
        # Transform SALES_PROGRAM_GROUP to string type
        df['SALES_PROGRAM_GROUP'] = df['SALES_PROGRAM_GROUP'].astype(str)    

        # Show the uploaded data
        st.write("Data Preview:", df.head())

        # Encode the input
        df['Encoded_Payment_Frequency'] = transform_with_label_encoder(le_payment_frequency, df['PAYMENT_FREQUENCY'])
        df['Encoded_Sales_Program'] = transform_with_target_encoder(te_sales_program, df['SALES_PROGRAM_GROUP'])        
        df['Encoded_Applicant_State'] = transform_with_target_encoder(te_applicant_state, df['APPLICANT_STATE'])

        # Debugging information
        #st.write("Encoded Sales Program Preview:", df[['SALES_PROGRAM_GROUP', 'Encoded_Sales_Program']].head())
        #st.write("Encoded Applicant State Preview:", df[['APPLICANT_STATE', 'Encoded_Applicant_State']].head())

        # Prepare features
        features = df[['CSM_SCORE', 'MaxDpdAll', 'CtRepo3', 'FINANCED_AMT_ORIG', 'SmFinAmtOri1', 'SmFinAmtOriOpen', 'CtNote60', 'Encoded_Payment_Frequency', 'Encoded_Sales_Program', 'Encoded_Applicant_State']]
        
        # Make predictions
        predictions = model.predict_proba(features)[:, 1]
        df['Probability'] = predictions
        df['Prediction'] = df['Probability'].apply(lambda x: 'AutoApprove' if x > 0.8 else 'ManualAssess')

        # Display predictions
        st.write("Predictions:", df[['CSM_SCORE', 'Probability', 'Prediction']])

        # Convert DataFrame to CSV for downloading
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Predictions as CSV", data=csv, file_name='batch_predictions.csv', mime='text/csv')
