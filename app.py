import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from glob import glob

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmUYQdImYbjPJesYFoMHVEfibp5l1CKBc"  # Replace with your Gemini API key


# Initialize vector database and create if not exists
def initialize_vector_db():
    db_path = "safety_incidents_faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Sample incident data - in a real app, you would load from your actual incident reports
    sample_incidents = [
        "2023-05-15: Worker injured due to lack of helmet in furnace area. Action: Conducted safety training.",
        "2023-06-22: Gas leak detected near boiler room. Action: Replaced faulty valves and improved ventilation.",
        "2023-07-10: High temperature alarm triggered. Action: Performed maintenance on cooling system.",
        "2023-08-05: Noise level exceeded limits in assembly line. Action: Provided ear protection and installed sound dampeners.",
        "2023-09-18: Unauthorized entry in restricted zone. Action: Improved access control and signage."
    ]
    
    # Create sample incident files if they don't exist
    if not os.path.exists("incidents"):
        os.makedirs("incidents")
        for i, incident in enumerate(sample_incidents):
            with open(f"incidents/incident_{i+1}.txt", "w") as f:
                f.write(incident)
    
    # Load all incident files
    incident_files = glob("incidents/*.txt")
    all_documents = []
    
    for file in incident_files:
        loader = TextLoader(file)
        all_documents.extend(loader.load())
    
    # Check if FAISS index exists
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(all_documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(db_path)
    
    return vector_store, embeddings

# Initialize RAG system
def initialize_rag():
    vector_store, embeddings = initialize_vector_db()
    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa_chain

# Dummy functions to simulate agent behavior
def analyze_video_feed():
    return ["No helmet detected near Furnace 3", "Unauthorized entry detected in Zone B"]

def check_sensor_data(sensor_df):
    alerts = []
    if sensor_df['gas_level'].iloc[-1] > 300:
        alerts.append("High gas level detected")
    if sensor_df['temperature'].iloc[-1] > 80:
        alerts.append("High temperature in Boiler Room")
    if sensor_df['noise_level'].iloc[-1] > 85:
        alerts.append("Noise level exceeds safety threshold")
    return alerts

def generate_prevention_checklist():
    return ["Wear helmet and safety gear", "Check gas detector calibration", "Inspect fire extinguishers"]

def generate_compliance_report():
    return "Safety compliance is at 92% this month. Helmet violations decreased by 15%."

# Initialize RAG system
qa_chain = initialize_rag()

def retrieve_similar_incidents(query):
    result = qa_chain.run(query)
    return [result]

# Streamlit UI
st.set_page_config(page_title="AI-Powered Safety Monitoring", layout="wide")
st.title("AI-Powered Industrial Safety Monitoring")

st.sidebar.header("Control Panel")
shift_start = st.sidebar.time_input("Shift Start Time", value=datetime.time(8, 0))
selected_area = st.sidebar.selectbox("Select Area", ["Furnace", "Boiler Room", "Assembly Line"])

st.header("Real-time Hazard Alerts")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Video Surveillance Alerts")
    video_alerts = analyze_video_feed()
    for alert in video_alerts:
        st.error(f"📹 {alert}")

with col2:
    st.subheader("Sensor Alerts")
    sensor_df = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='T'),
        'gas_level': np.random.randint(250, 350, 10),
        'temperature': np.random.randint(60, 90, 10),
        'noise_level': np.random.randint(70, 95, 10)
    })
    sensor_alerts = check_sensor_data(sensor_df)
    for alert in sensor_alerts:
        st.error(f"📊 {alert}")

st.header("Prevention Checklist")
checklist = generate_prevention_checklist()
for item in checklist:
    st.checkbox(item, value=False)

st.header("Historical Incident Analysis (RAG)")
user_query = st.text_input("Describe current risk or incident", value="helmet violation near furnace")
if user_query:
    with st.spinner("Searching similar historical incidents..."):
        rag_results = retrieve_similar_incidents(user_query)
        st.subheader("Similar Past Incidents & Resolutions")
        for res in rag_results:
            st.info(res)

st.header("Safety Compliance Report")
report = generate_compliance_report()
st.success(report)

st.header("Sensor Readings (Last 10 min)")
st.dataframe(sensor_df.set_index('timestamp'))

st.header("Add New Incident")
with st.form("incident_form"):
    incident_date = st.date_input("Incident Date")
    incident_desc = st.text_area("Incident Description")
    action_taken = st.text_area("Action Taken")
    
    if st.form_submit_button("Add to Knowledge Base"):
        # Save new incident to file
        new_incident = f"{incident_date}: {incident_desc} Action: {action_taken}"
        incident_count = len(glob("incidents/*.txt")) + 1
        with open(f"incidents/incident_{incident_count}.txt", "w") as f:
            f.write(new_incident)
        
        # Update the vector database
        loader = TextLoader(f"incidents/incident_{incident_count}.txt")
        new_doc = loader.load()
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db_path = "safety_incidents_faiss_index"
        
        if os.path.exists(db_path):
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            vector_store.add_documents(new_doc)
            vector_store.save_local(db_path)
        else:
            initialize_vector_db()
        
        st.success("Incident added to knowledge base!")
