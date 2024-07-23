import streamlit as st
import pandas as pd
import pyodbc
from datetime import date
import openai
from sentence_transformers import SentenceTransformer, util
from elasticsearch import Elasticsearch
import numpy as np
import os
import torch
from langchain_community.chat_models import ChatOpenAI
from pathlib import Path

# Initialize the Elasticsearch client
es = Elasticsearch(
    hosts=["http://10.105.107.123:9200"],  # Elasticsearch host
    http_auth=('your_username', 'your_password'),   # Elasticsearch username and password
    timeout=60
)

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure your API key is set in the environment variables

# Initialize Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

MAX_TOKENS = 5000  # Adjust this value as needed to stay within token limits
PROMPT_TEMPLATE = """
Given the following patient profile: gender, age (if below 65 years old - do not mention about age), diagnosis, operation, 
BMI (if < 23 do not mention about obesity), vital signs, labs, ASA (American Society of Anesthesiologist Classification)and estimated time of operation, alert for any clinical concerns, abnormal vital signs, abnormal labs data (especially diabetes if fasting blood sugar over 100) and assess the risk for the upcoming operation:

{context}

---

Based on the above context, what are alerts and the risk for the operation?
"""

# Define the connection details
denodo_driver_path = r""
denodo_host = ""
denodo_database = ""
denodo_username = ""
denodo_password = ""

def truncate_text(text, max_tokens):
    words = text.split()
    truncated_words = words[:max_tokens]
    return ' '.join(truncated_words)

def filter_vital_signs(df):
    # Define the normal ranges for each vital sign
    conditions = {
        8: lambda x: x > 140,           # Systolic BP > 140
        7: lambda x: x > 90,            # Diastolic BP > 90
        9: lambda x: x > 150 or x < 50, # PR > 150 or PR < 50
        10: lambda x: x > 30 or x < 3,  # RR > 30 or RR < 3
        11: lambda x: x > 37.5 or x < 35, # T > 37.5 or T < 36
        69: lambda x: x < 95            # SaO2 < 95%
    }

    # Filter the DataFrame based on the conditions
    df_filtered = df[df.apply(lambda row: conditions.get(row['obs_item_dr'], lambda x: False)(float(row['obs_value'])), axis=1)]

    # Calculate BMI if both height and weight are present
    if '90' in df['obs_item_dr'].values and '87' in df['obs_item_dr'].values:
        weight = df.loc[df['obs_item_dr'] == '90', 'obs_value'].values[0]
        height = df.loc[df['obs_item_dr'] == '87', 'obs_value'].values[0]
        bmi = float(weight) / ((float(height) / 100) ** 2)
        df_filtered = df_filtered.append({'paadm_admno': df['paadm_admno'].values[0], 'itm_desc': 'BMI', 'obs_value': bmi, 'obs_date': df['obs_date'].values[0], 'obs_time': df['obs_time'].values[0], 'obs_item_dr': 'BMI'}, ignore_index=True)
        return df_filtered, bmi
    
    return df_filtered, None

def filter_lab_data(df):
    lab_conditions = {
        'C0090': (136, 145),  # Na 136-145
        'C0100': (3.5, 5.1),  # K 3.5-5.1
        'C0110': (98, 107),   # Cl 98-107
        'C0805': (22, 29),    # TCO2 22-29
        'C0070': (0.56, 1.01),# Cr 0.56-1.01
        'C0530': (8.4, 25.7), # BUN 8.4-25.7 for male
        'A0005': (4.00, 10.00),# WBC 4.00-10.00
        'C0050': (6.4, 8.3),  # TP 6.4-8.3
        'C0060': (3.5, 5.2),  # alb 3.5-5.2
        'C0065': (2.1, 3.7),  # glob 2.1-3.7
        'C0000': (0.2, 1.2),  # T bili 0.2-1.2
        'C0005': (None, 0.5), # D bili > 0.5
        'C0040': (5, 35),     # SGOT 5-35
        'C0030': (None, 45),  # SGPT > 45
        'C0010': (40, 150),   # ALP 40-150
        'C0180': (70, 99),    # BS 70-99
        'C0162': (None, 40),  # VLDL > 40
        'C0141': (40,None),   # HDL < 40
        'C0158': (None, 130), # LDL > 130
        'C0131': (None, 200), # Chole > 200
        'C0142': (None,150)   # TG > 150
    }

    def is_abnormal(row):
        range_vals = lab_conditions.get(row['cttc_cde'])
        if range_vals is None:
            return False
        min_val, max_val = range_vals
        try:
            value = float(row['tst_dta'])
            return (min_val is not None and value < min_val) or (max_val is not None and value > max_val)
        except ValueError:
            return False

    df_filtered = df[df.apply(is_abnormal, axis=1)]
    return df_filtered

def index_md_files(md_dir):
    documents = []
    filenames = []
    for md_file in Path(md_dir).glob("*.md"):
        with open(md_file, "r", encoding='utf-8', errors='ignore') as file:
            content = file.read()
            documents.append(content)
            filenames.append(md_file.name)
    return documents, filenames

def semantic_search(query, documents, filenames, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(documents, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    return [(documents[idx], filenames[idx], cos_scores[idx].item()) for idx in top_results[1]]

def main():
    # Set up the page title and layout
    st.set_page_config(page_title="DHV AI Smart Trigger", layout="wide")
    st.title("DHV AI Preoperative Risk Assessment")

    # Load and index MD files
    md_dir = "data/books/"
    documents, filenames = index_md_files(md_dir)

    # User input for 'en'
    user_en = st.text_input("โปรดใส่ 'en' ผู้ป่วย:", "")

    if user_en:
        try:
            # Connect to the Denodo server
            conn = pyodbc.connect(
                r"DRIVER={DenodoODBC Unicode(x64)};"
                f"Server={denodo_host};"
                f"Database={denodo_database};"
                f"UID={denodo_username};"
                f"PWD={denodo_password};"
            )

            # Create a cursor object
            cursor = conn.cursor()

            # First SQL query for vital signs
            query1 = """
            SELECT 
              paadm_admno,
              itm_desc,
              obs_value,
              obs_date,
              obs_time,
              obs_item_dr
            FROM mr_observations
            INNER JOIN mr_adm ON mradm_rowid = obs_parref
            INNER JOIN mrc_observationitem ON itm_rowid = obs_item_dr
            INNER JOIN pa_adm ON paadm_rowid = mradm_adm_dr
            WHERE obs_item_dr IN (7, 8, 9, 10, 11, 69, 87, 90) AND paadm_admno = ?
            ;
            """
            # Execute the first query
            cursor.execute(query1, user_en)

            # Fetch the first set of results
            results1 = cursor.fetchall()

            # Check the results and columns for the first query
            if results1:
                columns1 = [column[0] for column in cursor.description]
                results1 = [list(row) for row in results1]

                # Create a DataFrame from the first set of results
                df1 = pd.DataFrame(results1, columns=columns1)

                # Filter the vital signs DataFrame based on abnormal values and calculate BMI
                df1_filtered, bmi = filter_vital_signs(df1)

                # Display the filtered vital signs DataFrame
                if not df1_filtered.empty:
                    st.write("Abnormal Vital Signs Data:")
                    st.write(df1_filtered)
                else:
                    st.write("Vital signs are within normal limits.")
            else:
                df1 = pd.DataFrame()
                df1_filtered = pd.DataFrame()  # Ensure df1_filtered is defined even if no results
                bmi = None

            # Second SQL query for patient information
            query2 = """
            SELECT 
              vw_operating.hn,
              vw_operating.en, 
              vw_operating.bookingorlocationcode,
              vw_operating.icd10,
              vw_operating.icd9,
              vw_operating.anmet_code,
              vw_operating.operation_date,
              vw_operating.rbop_estimatedtime,
              vw_operating.asa,
              pa_patmas.papmi_no,
              pa_patmas.papmi_dob,
              pa_patmas.papmi_sex_dr,
              pa_patmas.papmi_allergy,
              vw_patient_underlying.presi_desc
            FROM vw_operating
            INNER JOIN pa_patmas
              ON vw_operating.hn = pa_patmas.papmi_no
            LEFT JOIN vw_patient_underlying
              ON vw_operating.hn = vw_patient_underlying.hn
              AND vw_operating.en = vw_patient_underlying.en
            WHERE vw_operating.en = ?
            ;
            """
            # Execute the second query
            cursor.execute(query2, (user_en,))

            # Fetch the second set of results
            results2 = cursor.fetchall()

            # Check the results and columns for the second query
            if results2:
                columns2 = [column[0] for column in cursor.description]
                results2 = [list(row) for row in results2]

                # Create a DataFrame from the second set of results
                df2 = pd.DataFrame(results2, columns=columns2)

                if df2.empty:
                    st.warning("ไม่พบ'en' ที่ใส่")
                    return

                # Convert the 'operation_date' and 'papmi_dob' columns to datetime
                df2['operation_date'] = pd.to_datetime(df2['operation_date'])
                df2['papmi_dob'] = pd.to_datetime(df2['papmi_dob'])

                # Convert 'asa' and 'rbop_estimatedtime' to numeric types
                df2['asa'] = pd.to_numeric(df2['asa'], errors='coerce')
                df2['rbop_estimatedtime'] = pd.to_numeric(df2['rbop_estimatedtime'], errors='coerce')

                # Calculate the age
                df2['age'] = (date.today() - df2['papmi_dob'].dt.date).apply(lambda x: x.days // 365)

                # Check if the patient is under 15 years old
                if df2['age'].iloc[0] < 15:
                    st.warning("ผู้ป่วยเด็ก โปรดปรึกษากุมารแพทย์")
                    return

                # Alert if asa >= 3, age >= 65, or rbop_estimatedtime >= 120
                if df2['asa'].iloc[0] >= 3 or df2['age'].iloc[0] >= 65 or df2['rbop_estimatedtime'].iloc[0] >= 120:
                    if df2['asa'].iloc[0] >= 3:
                        st.warning("Risk for this patient: ASA class >= 3")
                    if df2['age'].iloc[0] >= 65:
                        st.warning("Risk for this patient: age >= 65")
                    if df2['rbop_estimatedtime'].iloc[0] >= 120:
                        st.warning("Risk for this patient: estimated operating time >= 120 minutes")

                # Concatenate non-empty fields for 'ptinfo'
                df2.fillna('', inplace=True)  # Replace NaN with empty string
                #st.write(df2)
            else:
                st.warning("ไม่พบ 'en' ที่ใส่")
                return

            # Third SQL query for lab data
            query3 = """
            SELECT epi_no, dte_of_req, ctts_cde, ctts_nme, tst_dta, cttc_cde, cttc_des
            FROM bv_tcl01_vw_labord
            WHERE epi_no = ? AND dte_of_req BETWEEN '2021-01-01' AND CURRENT_DATE
            ;
            """
            # Execute the third query
            cursor.execute(query3, (user_en,))

            # Fetch the third set of results
            results3 = cursor.fetchall()

            # Check the results and columns for the third query
            if results3:
                columns3 = [column[0] for column in cursor.description]
                results3 = [list(row) for row in results3]

                # Create a DataFrame from the third set of results
                df3 = pd.DataFrame(results3, columns=columns3)

                # Filter the lab data based on abnormal values
                df3_filtered = filter_lab_data(df3)

                # Display the filtered lab data DataFrame
                if not df3_filtered.empty:
                    st.write("Abnormal Lab Data:")
                    st.write(df3_filtered)
                else:
                    st.write("Lab results are within normal limits.")
            else:
                df3 = pd.DataFrame()
                df3_filtered = pd.DataFrame()

            # Construct the patient information text
            gender_dict = {
                '1': "Male",
                '2': "Female",
                '3': "Unknown",
                '4': "Transgender",
                '5': "Not Stated/Inadequately Described",
                '6': "Other",
                '7': "Not Specified",
                '8': "Intersex",
                '9': "Indeterminate",
                '10': "Another Term"
            }
            gender = gender_dict.get(str(df2['papmi_sex_dr'].iloc[0]), "Unknown")
            age = df2['age'].iloc[0]
            icd10 = df2['icd10'].iloc[0]
            icd9 = df2['icd9'].iloc[0]
            underlying_disease = df2['presi_desc'].iloc[0]
            asa = df2['asa'].iloc[0]
            rbop_estimatedtime = df2['rbop_estimatedtime'].iloc[0]

            # Construct the patient information text with BMI
            if bmi is not None:
                ptinfo_text = f"{gender} aged (when operated {age} yr), BMI - {bmi}, with diagnosis of - {icd10}, underlying disease - {underlying_disease}, history of allergy - {drugallergy} , ASA - {asa}, estimated time of operation - {rbop_estimatedtime} and to undergo: {icd9}"
            else:
                ptinfo_text = f"{gender} aged (when operated {age} yr), with diagnosis of - {icd10}, underlying disease - {underlying_disease}, history of allergy - {drugallergy}, ASA - {asa}, estimated time of operation - {rbop_estimatedtime} and to undergo: {icd9}"

            # Construct the vital signs text
            vital_signs = {
                '8': "BP systolic",
                '7': "BP diastolic",
                '9': "PR",
                '10': "RR",
                '69': "SpO2",
                '11': "T"
            }
            vitals_text = ", ".join([f"{vital_signs[str(row['obs_item_dr'])]}: {row['obs_value']}" for _, row in df1_filtered.iterrows() if str(row['obs_item_dr']) in vital_signs])

            # Construct the lab data text
            lab_conditions = {
                'A0005': "WBC",
                'C0170': "BS",
                'C0180': "FBS",
                'C0070': "Cr",
                'C0100': "K"
            }
            labs_text = ", ".join([f"{lab_conditions[row['cttc_cde']]}: {row['tst_dta']}" for _, row in df3_filtered.iterrows() if row['cttc_cde'] in lab_conditions])

            query_text = f"{ptinfo_text}  having abnormal vital signs: - {vitals_text}  and abnormal labs: - {labs_text}"

            # Truncate query_text to ensure it does not exceed token limits
            query_text = truncate_text(query_text, MAX_TOKENS)

            st.write("Patient Information for Validation:")
            st.write(query_text)

            # Perform the semantic search using Sentence Transformers
            semantic_results = semantic_search(query_text, documents, filenames, top_k=3)
            if not semantic_results:
                st.write("Minimal risk/ไม่สามารถวิเคราะห์ด้วยข้อมูลขณะนี้ได้ โปรดปรึกษาแพทย์เจ้าของไข้")
                return

            context_text = "\n\n---\n\n".join([doc[0] for doc in semantic_results])
            
            # Replace the prompt template handling (assuming we don't have langchain_community)
            prompt = PROMPT_TEMPLATE.format(context=context_text)

            # Generate response from ChatOpenAI (assuming this is correctly set up)
            model = ChatOpenAI()
            response_text = model.predict(prompt)

            # Extract sources from the results
            sources = [doc[1] for doc in semantic_results]

            # Format the response text with red color
            formatted_response = f"<span style='color:red'>{response_text}</span>"

            # Format the sources with blue color
            formatted_sources = "<br>".join([f"<span style='color:blue'>{source}</span>" for source in sources if source])

            # Combine the formatted response text and sources
            complete_formatted_response = f"{formatted_response}<br>Sources:<br>{formatted_sources}"

            # Display the formatted response in Streamlit
            st.write(complete_formatted_response, unsafe_allow_html=True)

        except pyodbc.Error as e:
            st.error(f"Error connecting to the database: {e}")
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except Exception as ex:
            st.error(f"An unexpected error occurred: {ex}")
    else:
        st.info("Please enter an 'en' value to start the query.")

if __name__ == "__main__":
    main()