import streamlit as st
import requests
import pandas as pd
import io
import json

st.title('Lemonaade Insurance Company: Insurance Cross-Sell Prediction App')
st.write('This app is used to predict if a current health policy customer will be interested in vehicle insurance bundling')
endpoint = 'http://server:8000/predict'

test_csv = st.file_uploader('', type=['csv'], accept_multiple_files=False)

if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())

    # Convert dataframe to BytesIO object (for parsing as file into FastAPI later)
    test_bytes_obj = io.BytesIO()
    # write to BytesIO buffer
    test_df.to_csv(test_bytes_obj, index=False)  
    # Reset pointer to avoid EmptyDataError
    test_bytes_obj.seek(0) 

    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    # Upon click of button
    if st.button('Start Prediction'):
        if len(test_df) == 0:
            # handle case with no image
            st.write("Please upload a valid test dataset!")  
        else:
            with st.spinner('Prediction in Progress. Please Wait...'):
                output = requests.post(endpoint, 
                                       files=files,
                                       timeout=8000)
            st.success('Success! Click Download button below to get prediction results (in JSON format)')
            st.download_button(
                label='Download',
                data=json.dumps(output.json()), # Download as JSON file object
                file_name='ml_prediction_results.json'
            )
