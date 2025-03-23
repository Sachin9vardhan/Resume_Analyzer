import pandas as pd

# Sample resume data
data = {
    "Name": ["Alice", "Bob", "Eve"],
    "resume_text": [
        "Experienced Python developer with expertise in NLP and Machine Learning.",
        "Java backend developer skilled in Spring Boot and Microservices.",
        "Data scientist proficient in SQL, Python, and statistical analysis."
    ],
    "Experience": ["3 years", "5 years", "2 years"],
    "Education": ["B.Tech in CS", "M.Tech in CS", "B.Sc in IT"],
    "score": [85, 90, 78]  # Ensure this column exists!
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Save DataFrame as CSV
df.to_csv("resumes.csv", index=False)

print("✅ resumes.csv created successfully with 'score' column!")
