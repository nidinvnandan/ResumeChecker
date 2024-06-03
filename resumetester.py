import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
import nltk
import re
from nltk.corpus import stopwords
import time
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import json
import csv
import zipfile
from langchain.tools import BaseTool
from typing import List, Dict, Any
from langchain.agents import initialize_agent,AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import base64
import csv
import os
import time
from ast import literal_eval

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
api_key = os.getenv('GOOGLE_API')

def unzip_folder(uploaded_file):
    # Create a temporary directory to extract the contents
    extraction_dir = "./temp_extracted_folder"
    os.makedirs(extraction_dir, exist_ok=True)

    # Extract the contents of the uploaded zip file
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    return extraction_dir
def get_repsonse(input,text,jd):
    genai.configure(api_key=api_key)
    model=genai.GenerativeModel('models/gemini-pro')
    response=model.generate_content([input,text,jd])
    time.sleep(8) 
    return response.text
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenize and remove stop words
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Stemming or lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)
llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=api_key,convert_system_message_to_human=True)

prompt =  """
You are a experienced Human Resource Manager capable of distingusing the resume of candidates based on job description.
when you get the resume:{text} then you hace to extract the educational details,skills,experience and then compare it with the skills required,educational qualification reponsibilities and give a matching score based on this.You may get resume which is not at all related to the job description you have to identify it very carefully
for eg: you may get a resume of a data analyst but the job description will be for flutter developer if this is the situation don't give good matching score keep it low.
Please provide the response in one single string having the structure below, and also display the name and mail id of the applicant from the resume. If available, include the contact number as well.
resume: {text}
job description: {job_description}
{"name":, "matching_score": "%", "mail_id":, contact_no:}"""

document_info = {}
st.title("Application Tracking System(ATS) Using LLM")
job_description = st.text_area("Enter the job description", height=200)
threshold = st.slider("Select the threshold for matching score", min_value=0.0, max_value=100.0, value=60.0, step=0.1)    
if st.button("Submit"):
        # Save the job description to a variable for later use
        job_description = preprocess_text(job_description)
uploaded_file = st.file_uploader("Upload a ZIP file containing the folder", type="zip")



if uploaded_file is not None:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                

                # Process each file in the zip file
                with zip_ref.open(file_name, "r") as file:
                    if file_name.endswith(".pdf"):
                        pdf_reader = pdf.PdfReader(file)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text()
                        


                # Perform further processing or analysis here
                # For example, you can pass the preprocessed_text to your model for analysis

                # Process the page text using your model
                        response = get_repsonse(prompt, text, job_description)
                        
                        response_parts = response.strip().split('\n')
                        for part in response_parts:
                            try:
                                # Parse the response part into a dictionary
                                response_dict = json.loads(part)

                                # Store the information in the dictionary
                                if file_name not in document_info:
                                    document_info[file_name] = []
                                document_info[file_name].append({
                                    'name': response_dict['name'],
                                    'matching_score': response_dict['matching_score'],
                                    'mail_id': response_dict['mail_id']
                                    
                                })
                            except json.JSONDecodeError as e:
                                print(f"Failed to parse response part for {file_name}: {e}")

if document_info:
    for document, info_list in document_info.items():
        # Check if info_list is a list with exactly one item
        if isinstance(info_list, list) and len(info_list) == 1 and isinstance(info_list[0], dict):
            info = info_list[0]  # Get the dictionary from the list
            

appended_list = []
if document_info:
    for key, value_list in document_info.items():
        for value_dict in value_list:
            appended_list.append(value_dict)
for item in appended_list:
    try:
        item['matching_score'] = float(item['matching_score'].rstrip('%'))
    except ValueError:
        item['matching_score'] = 0


sorted_documents = sorted(appended_list, key=lambda x: x['matching_score'], reverse=True)

# Create a new list and append the dictionaries in sorted order
new_list = []
for doc in sorted_documents:
    new_list.append(doc)

# Print the new list


# Convert the string to a list of dictionaries
output_list = new_list

# CSV file path
csv_file = 'output.csv'
if os.path.exists(csv_file):
    os.remove(csv_file) 


if not os.path.exists(csv_file):
    # Ask the user for the threshold

    with open(csv_file, mode='w', newline='') as file:
        if output_list:
            writer = csv.DictWriter(file, fieldnames=output_list[0].keys())
            writer.writeheader()
            for row in output_list:
                if float(row["matching_score"]) > threshold:
                    writer.writerow(row)


# Provide a download link for the file
st.markdown(f"Download the CSV file [here](data:text/csv;base64,{base64.b64encode(open(csv_file, 'rb').read()).decode()})")
