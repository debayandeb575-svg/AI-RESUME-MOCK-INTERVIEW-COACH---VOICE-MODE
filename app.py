import os
import streamlit as st
import pypdf
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI Client
# Fetches the OPENAI_API_KEY environment variable automatically
client = OpenAI()

st.set_page_config(page_title="AI Resume & Interview Coach", page_icon="💼", layout="wide")
st.title("💼 AI Resume & Mock Interview Coach")

# --- Helper Functions ---
def extract_text_from_pdf(uploaded_file):
    """Parses text from a uploaded PDF file."""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_resume(resume_text, job_description):
    """Uses LLM to provide tailored feedback based on the job description."""
    system_prompt = (
        "You are an expert technical recruiter. Analyze the candidate's resume against the provided job description. "
        "Avoid generic advice. Provide specific feedback under three headings: "
        "1. Critical Gaps (Missing skills/keywords), "
        "2. Tailoring Suggestions (How to reword specific bullet points), "
        "3. Match Score (0-100%) with justification."
    )
    user_content = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_interview_question(resume_text, job_description):
    """Generates a tailored interview question based on the profile and role."""
    system_prompt = (
        "Based on the user's resume and target job description, generate ONE highly relevant, situational, "
        "or technical interview question they are likely to encounter. Do not include introductory text, just the question."
    )
    user_content = f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{job_description}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

def evaluate_answer(question, answer_text):
    """Evaluates the mock interview answer performance."""
    system_prompt = (
        "You are an interview coach. Evaluate the candidate's answer to the given question. "
        "Provide constructive feedback on what they did well, what they missed, and a model answer variant."
    )
    user_content = f"QUESTION: {question}\nCANDIDATE ANSWER: {answer_text}"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

# --- Streamlit UI Tabs ---
tab1, tab2 = st.tabs(["📄 Resume Tailor Engine", "🎙️ Mock Interview Coach"])

# Keep resume text stored across tabs using session state
if 'resume_text' not in st.session_state:
    st.session_state['resume_text'] = ""
if 'job_desc' not in st.session_state:
    st.session_state['job_desc'] = ""

with tab1:
    st.header("Tailor Your Resume")
    uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    job_desc_input = st.text_area("Paste the Target Job Description here", height=200)
    
    if st.button("Analyze & Tailor"):
        if uploaded_file and job_desc_input:
            with st.spinner("Analyzing resume against the job description..."):
                extracted_text = extract_text_from_pdf(uploaded_file)
                st.session_state['resume_text'] = extracted_text
                st.session_state['job_desc'] = job_desc_input
                
                feedback = analyze_resume(extracted_text, job_desc_input)
                st.subheader("💡 Tailored Feedback")
                st.markdown(feedback)
        else:
            st.warning("Please upload a PDF resume and paste a job description first.")

with tab2:
    st.header("Voice Mock Interview Practice")
    
    if not st.session_state['resume_text'] or not st.session_state['job_desc']:
        st.info("⚠️ Please complete the Resume Tailor step first to generate context-aware questions.")
    else:
        if st.button("Generate Interview Question"):
            with st.spinner("Generating target question..."):
                st.session_state['current_question'] = generate_interview_question(
                    st.session_state['resume_text'], st.session_state['job_desc']
                )
        
        if 'current_question' in st.session_state:
            st.info(f"**Question:** {st.session_state['current_question']}")
            
            # Text area input for hackathon simplicity (can swap with audio processing later)
            user_answer = st.text_area("Type your answer here (or paste speech-to-text transcript):", height=150)
            
            if st.button("Submit Answer for Evaluation"):
                if user_answer:
                    with st.spinner("Evaluating your response..."):
                        evaluation = evaluate_answer(st.session_state['current_question'], user_answer)
                        st.subheader("📊 Performance Feedback")
                        st.markdown(evaluation)
                else:
                    st.warning("Please enter an answer before submitting.")