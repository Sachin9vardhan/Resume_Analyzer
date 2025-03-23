import streamlit as st
import pandas as pd
import base64
import time
import datetime
import re
import fitz  # PyMuPDF
import joblib
import pdfplumber
import nltk
import matplotlib.pyplot as plt
from PIL import Image
from fuzzywuzzy import process
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load trained model and vectorizer
model = joblib.load("resume_scoring_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load skill dataset
skills_df = pd.read_csv("skills_dataset.csv")
skill_list = list(set([skill.strip().lower() for sublist in skills_df["label"].apply(lambda x: x.split(",")) for skill in sublist]))

# Function to display PDF
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>', unsafe_allow_html=True)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 1]
    return " ".join(words)

# Function to extract text from PDF
def pdf_reader(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text.strip()

# Function to predict resume score
def predict_resume_score(resume_text):
    cleaned_text = preprocess_text(resume_text)
    features = vectorizer.transform([cleaned_text]).toarray()
    return model.predict(features)[0]

# Function to extract email
def extract_email(text):
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return email_match.group(0) if email_match else "Unknown"

# Function to extract name from email
def extract_name_from_email(email):
    if email and email != "Unknown":
        name_part = email.split("@")[0]
        name_part = re.sub(r"[^a-zA-Z]", " ", name_part).title()
        return name_part.split()[0]  # Return the first word as the likely name
    return "Unknown"

# List of common job titles to filter out
job_titles = [
    "data scientist", "software engineer", "machine learning engineer",
    "business analyst", "full stack developer", "product manager",
    "graphic designer", "ux designer", "technical writer"
]

# Function to extract name from resume text
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



# function to extract years of experience
def extract_experience(resume_text):
    current_year = datetime.datetime.now().year  # Get current year
    total_experience = 0

    # Regex patterns to extract experience from date ranges
    experience_patterns = [
        r"(\b\d{4})\s*[-–]\s*(\b\d{4}|\b(?:present|current))",  # e.g., "2005 - 2010" or "2006 - current"
        r"(\b[A-Za-z]+\s\d{4})\s*[-–]\s*(\b[A-Za-z]+\s\d{4}|\b(?:present|current))"  # e.g., "May 1994 – May 2005"
    ]

    for pattern in experience_patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        for match in matches:
            start_year = extract_year(match[0])  # Convert to year
            end_year = extract_year(match[1]) if "present" not in match[1].lower() and "current" not in match[1].lower() else current_year

            if start_year and end_year and start_year < end_year:  # Ensure valid range
                total_experience += (end_year - start_year)

    return total_experience

def extract_year(text):
    """Helper function to extract year from 'May 1994' or '1994' format."""
    match = re.search(r"\b\d{4}\b", text)
    return int(match.group(0)) if match else None


# Function to extract skills
def extract_skills(resume_text):
    resume_text = resume_text.lower()
    extracted_skills = set()
    for skill in skill_list:
        match_score = process.extractOne(skill, resume_text.split(), score_cutoff=75)
        if match_score:
            extracted_skills.add(skill)
    return list(extracted_skills)

# Load job-skills dataset
job_skills_df = pd.read_csv("job_skills_dataset.csv")

# Function to determine the most relevant job role
def determine_job_role(extracted_skills):
    if job_skills_df.empty or "Job Role" not in job_skills_df.columns or "Required Skills" not in job_skills_df.columns:
        st.error("⚠️ Missing necessary columns in job_skills_dataset.csv.")
        return "Not Identified"

    extracted_skills = set([skill.lower().strip() for skill in extracted_skills])  # Standardize skills

    best_match = None
    highest_score = 0

    for _, row in job_skills_df.iterrows():
        job_role = row["Job Role"]
        required_skills = set(str(row["Required Skills"]).lower().strip().split(", "))

        # Fuzzy Matching to find relevant skills
        match_scores = [process.extractOne(skill, required_skills, score_cutoff=60) for skill in extracted_skills]
        matched_skills = [match[0] for match in match_scores if match]  # Extract matched skills

        overlap = len(matched_skills)  # Count the number of matched skills

        if overlap > highest_score:
            highest_score = overlap
            best_match = job_role

    return best_match if best_match else "Not Identified"

# Function to recommend additional skills
def recommend_skills(job_role, extracted_skills):
    if not job_role or job_role == "Not Identified":
        return []  # Return empty list if job role is not found

    # Filter job_skills_df for the matching job role
    matched_rows = job_skills_df[job_skills_df["Job Role"].str.lower().str.strip() == job_role.lower().strip()]

    if matched_rows.empty:
        return []  # Return empty list if no matching job role is found

    required_skills = set(matched_rows["Required Skills"].values[0].split(", "))  # Extract skills
    missing_skills = required_skills - set(extracted_skills)

    return list(missing_skills)

# Streamlit UI
st.set_page_config(page_title="AI Resume Analyzer", page_icon="Logo/logo.jpg")

def run():
    img = Image.open("Logo/logo.jpg")
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

            save_path = f"./Uploaded_Resumes/{pdf_file.name}"
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

            # Extract years of experience
            experience = extract_experience(resume_text)
            st.subheader("**Years of Experience**")
            st.text(f"{experience} years")


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

            # Determine job role (Moved Here!)
            job_role = determine_job_role(skills)
            st.subheader("**Identified Job Role**")
            if job_role:
                job_role_html = f'<span style="background-color:#27822b;color:white;padding:5px 10px;margin:5px;border-radius:5px;">{job_role}</span>'
                st.markdown(f"<div>{job_role_html}</div>", unsafe_allow_html=True)
            else:
                st.warning("Job role not identified. Try another resume.")


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

            # Predict Resume Score
            score = predict_resume_score(resume_text)
            st.subheader("**Predicted Resume Score**")
            st.progress(score / 100)
            st.success(f"🎯 Your Resume Score: {round(score, 2)}")
            
            # Save user data to CSV
            # Save user data to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            df = pd.DataFrame([{ 
                "Name": name, 
                "Email": email, 
                "Resume Score": score, 
                "Applied Job Role": job_role if job_role else "Not Identified", 
                "Experience": experience,  # Add extracted experience
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

            except FileNotFoundError:
                st.error("No user data found!")
        else:
            st.error("Wrong Username or Password")

run()
