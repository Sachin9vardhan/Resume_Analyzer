import streamlit as st
import pandas as pd
import base64
import time
import os
import datetime
import re
import fitz  # PyMuPDF
import joblib
import pdfplumber 
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from fuzzywuzzy import process
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load the pre-trained job role prediction model
job_role_model = joblib.load("job_role_model.pkl")

# Load the pre-trained vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")  

# Load the label encoder
encoder = joblib.load("label_encoder.pkl")  

# Load skill dataset
skills_df = pd.read_csv("expanded_skills_dataset.csv")

if "Skills" in skills_df.columns:
    skill_list = list(skills_df["Skills"].dropna().str.lower().str.strip())
else:
    st.error("⚠️ The skill dataset is missing a 'Skills' column. Please check the file.")
    skill_list = []

# display PDF
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>', unsafe_allow_html=True)


# preprocess text for model prediction
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(words)

# extract text from PDF
def pdf_reader(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text.strip()

# extract email
def extract_email(text):
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return email_match.group(0) if email_match else "Unknown"

# extract name from email
def extract_name_from_email(email):
    if email and email != "Unknown":
        name_part = email.split("@")[0]
        name_part = re.sub(r"[^a-zA-Z]", " ", name_part).title()
        return name_part.split()[0]  # Return the first word as the likely name
    return "Unknown"

# common job titles to filter out for name recognition
job_titles = [
    "data scientist", "software engineer", "machine learning engineer",
    "business analyst", "full stack developer", "product manager",
    "graphic designer", "ux designer", "technical writer"
]

# extract name from resume text
def extract_name_from_resume(resume_text):
    lines = resume_text.strip().split("\n")[:5]  # Check only first 5 lines (header section)
    
    for line in lines:
        line = line.strip()
        if len(line.split()) == 2:  # Likely a full name (First Last)
            words = line.split()
            first, last = words[0], words[1]
            
            # Ensure it’s not a job title
            if line.lower() not in job_titles and re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", line):
                return first + " " + last  # Return the formatted name
    
    return "Unknown"  # If no valid name is found

# extract name from email
def extract_name_from_email(email):
    if email and email != "Unknown":
        name_part = email.split("@")[0]  # Extract text before @
        name_part = re.sub(r"[^a-zA-Z]", " ", name_part).title()  # Remove numbers & special chars
        return name_part.split()[0]  # First word as name
    return "Unknown"


# extract skills
def extract_skills(resume_text):
    resume_words = set(re.findall(r'\b[a-zA-Z]+\b', resume_text.lower()))
    matched_skills = resume_words.intersection(set(skill_list))
    return list(matched_skills)

# predict job role using trained model
def predict_job_role(user_skills):
    user_skills = clean_text(user_skills)  # Preprocess input
    user_vector = vectorizer.transform([user_skills])  # Convert to numerical format
    predicted_label = job_role_model.predict(user_vector)[0]  # Get numerical prediction
    predicted_role = encoder.inverse_transform([predicted_label])[0]  # Convert to job role
    return predicted_role  # Return actual job role name

# Load job skills dataset
job_skills_df = pd.read_csv("job_skills_dataset.csv")

def recommend_skills(job_role, extracted_skills):
    if not job_role or job_role == "Not Identified":
        return []

    # Convert extracted skills to lowercase set
    extracted_skills = set(skill.lower() for skill in extracted_skills)

    # Ensure "Job Role" column is of string type
    job_skills_df["Job Role"] = job_skills_df["Job Role"].astype(str).str.lower()
    
    # Convert job_role to lowercase
    job_role = str(job_role).lower()

    # Find required skills for the predicted job role
    job_role_skills = job_skills_df[job_skills_df["Job Role"] == job_role]
    
    if job_role_skills.empty:
        return []  # No data for this job role

    required_skills = set(str(job_role_skills["Skills"].iloc[0]).lower().split(", "))

    # Determine missing skills
    missing_skills = required_skills - extracted_skills

    return list(missing_skills)



# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", page_icon="Logo/logo.jpg")

def run():
    img = Image.open("logo.jpg")
    st.image(img)
    st.title("AI Resume Analyzer")
    
    st.sidebar.markdown("# Choose User")
    activities = ["User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == "User":
        st.markdown("<h5>Upload your resume, and get smart recommendations</h5>", unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        
        if pdf_file is not None:
            with st.spinner("Uploading your Resume..."):
                time.sleep(2)

            save_path = os.path.join("/tmp", pdf_file.name)
            with open(save_path, "wb") as f:
                f.write(pdf_file.getbuffer())


            show_pdf(save_path)
            resume_text = pdf_reader(save_path)

            if not resume_text:
                st.error("⚠️ Could not extract text from PDF. Try another file.")
                return
            
            
            st.header("**Resume Analysis**")
            # Extract User Info
            email = extract_email(resume_text)
            name = extract_name_from_resume(resume_text)

            # If resume name extraction fails, try email-based extraction
            if name == "Unknown":
                name = extract_name_from_email(email)
            
            skills = extract_skills(resume_text)  # Extract skills here

            st.success(f'Hello {name} 🎉')            
            st.subheader("**Your Basic Info**")
            st.text(f'Email: {email}')
            st.text(f'Resume pages: {len(resume_text) // 2000 + 1}')
            
            # Show Extracted Skills
            st.subheader("Your Extracted Skills")
            if skills:
                skills_html = " ".join([
                    f'<span style="background-color:#586e26;color:white;padding:5px 10px;margin:5px;border-radius:5px;">{skill}</span>' 
                    for skill in skills
                ])
                st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
            else:
                st.warning("No skills extracted. Try another resume.")

            # predicted job                
            job_role = predict_job_role(resume_text)
            
            st.subheader("**Predicted Job Role**")
            st.success(job_role)
            


            # Recommend skills
            recommended_skills = recommend_skills(job_role, skills)
            st.subheader("**Recommended Skills to Improve Your Resume**")
            if recommended_skills:
                recommended_html = " ".join([
                    f'<span style="background-color:#ffcc00;color:black;padding:5px 10px;margin:5px;border-radius:5px;">{skill}</span>' 
                    for skill in recommended_skills
                ])
                st.markdown(f"<div>{recommended_html}</div>", unsafe_allow_html=True)
            else:
                st.success("You have most of the required skills for this role!")
            
            # Save user data to CSV
            # Save user data to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            df = pd.DataFrame([{ 
                "Name": name, 
                "Email": email, 
                "Applied Job Role": job_role if job_role else "Not Identified", 
                "Timestamp": timestamp,  
                "Skills": ", ".join(skills)  
            }])

            try:
                existing_df = pd.read_csv("user_data.csv")
                df = pd.concat([existing_df, df], ignore_index=True)
            except FileNotFoundError:
                pass

            df.to_csv("user_data.csv", index=False)
    else:
        st.success("Welcome to Admin Side")
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type="password")

        if st.button("Login"):
            if ad_user == "a" and ad_password == "a":
                st.success("Welcome!")
                try:
                    df = pd.read_csv("user_data.csv")
                    st.dataframe(df)
                
                    # Group data by Applied Job Role
                    job_counts = df["Applied Job Role"].value_counts()

                    # Plot Pie Chart
                    fig, ax = plt.subplots()
                    ax.pie(job_counts, labels=job_counts.index, autopct="%1.1f%%", startangle=90, colors=plt.cm.Paired.colors)
                    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

                    st.subheader("📊 Number of Resumes per Job Applied")
                    st.pyplot(fig)

                    # Job Role Distribution Bar Chart
                    st.subheader("📊 Number of Resumes per Job Applied")
                    job_counts = df["Applied Job Role"].value_counts()
    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=job_counts.index, y=job_counts.values, ax=ax, palette="viridis")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                    ax.set_ylabel("Number of Resumes")
                    ax.set_title("Resumes per Job Role")
                    st.pyplot(fig)

                    # Job Trends Over Time
                   
                    try:
                        df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str).str.replace("_", " "), errors="coerce")
                        df["Date"] = df["Timestamp"].dt.date  # Extract only the date part
                    except Exception as e:
                        st.error(f"⚠️ Error parsing timestamps: {str(e)}")

    
                    st.subheader("📈 Resume Submissions Over Time")
                    time_series = df.groupby("Date").size()

                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.lineplot(x=time_series.index, y=time_series.values, ax=ax, marker="o", color="b")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Number of Submissions")
                    ax.set_title("Resume Submissions Trend")
                    st.pyplot(fig)
                    
                except FileNotFoundError:
                    st.error("No user data found!")
            else:
                st.error("Wrong Username or Password")
    
    

run()   

