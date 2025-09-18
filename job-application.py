"""
Streamlit app: Job Response Generator
Generates tailored job-application replies from Resume + Cover Letter + Job Description URL/text

Usage:
This app extracts text from uploaded resume/cover letter (PDF/DOCX), fetches the job description if a URL is provided (or accepts pasted job text), builds a safe prompt following the user's reusable prompt template, and sends it to OpenAI. It returns a short, natural-sounding reply you can copy to applications.
Important: This template strictly uses only the text provided in the uploaded resume and cover letter plus the job description — it will not invent or add information not present in those inputs.
"""

import os
import warnings
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from pdfminer.high_level import extract_text as extract_pdf_text
import docx

# Suppress warnings
warnings.filterwarnings("ignore")

def load_api_key():
    """Load OpenAI API key from Streamlit secrets or local backup."""
    # Priority 1: Streamlit secrets (for Streamlit Community Cloud deployment)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            key = st.secrets["OPENAI_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass
    
    # Priority 2: Environment variables (fallback)
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    
    # Priority 3: Local file backup (for development)
    try:
        with open("OPENAI_API_KEY.txt", "r") as f:
            key = f.read().strip()
            if key:
                return key
    except FileNotFoundError:
        pass

    st.error("❌ OpenAI API key not found in environment variables.")
    return None

if __name__ == "__main__":
    api_key = load_api_key()
    if api_key:
        # Initialize OpenAI client
        openai = OpenAI(api_key=api_key)

def extract_text_from_pdf(file_bytes):
    try:
        text = extract_pdf_text(BytesIO(file_bytes))
        return text
    except Exception as e:
        return ""

def extract_text_from_docx(file_bytes):
    try:
        doc = docx.Document(BytesIO(file_bytes))
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    except Exception as e:
        return ""

def fetch_job_description(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Try common tags
        texts = []
        for tag in soup.find_all(['h1','h2','h3','p','li']):
            if tag.get_text(strip=True):
                texts.append(tag.get_text(strip=True))
        return '\n'.join(texts)
    except Exception as e:
        return ''

def get_language_instructions(language):
    """Get detailed language-specific instructions for better native output."""
    language_guides = {
        "Dutch": {
            "style": "Use formal but approachable Dutch. Avoid overly complex sentence structures. Use 'u' for formal address. Include typical Dutch business expressions like 'graag', 'bijzonder', 'uitstekend'.",
            "cultural": "Dutch business culture values directness and efficiency. Be straightforward but polite. Mention specific achievements with confidence.",
            "examples": "Good phrases: 'Ik ben bijzonder geïnteresseerd in...', 'Mijn ervaring met... maakt mij een sterke kandidaat', 'Ik zou graag de kans krijgen om...'"
        },
        "French": {
            "style": "Use formal French with proper business etiquette. Use 'vous' form throughout. Include sophisticated vocabulary and proper French business expressions.",
            "cultural": "French business culture appreciates elegance and sophistication. Use refined language and show appreciation for the company's values.",
            "examples": "Good phrases: 'Je suis particulièrement intéressé(e) par...', 'Mon expérience dans... me permet de...', 'J'aimerais avoir l'opportunité de...'"
        },
        "German": {
            "style": "Use formal German (Sie form). German business communication is precise and structured. Use compound words appropriately and maintain professional tone.",
            "cultural": "German business culture values precision, reliability, and thoroughness. Be specific about qualifications and achievements.",
            "examples": "Good phrases: 'Ich bin besonders interessiert an...', 'Meine Erfahrung in... qualifiziert mich für...', 'Ich würde mich freuen, die Gelegenheit zu bekommen...'"
        },
        "Spanish": {
            "style": "Use formal Spanish with usted form. Include appropriate business vocabulary and maintain professional but warm tone typical of Spanish business culture.",
            "cultural": "Spanish business culture values personal relationships and enthusiasm. Show genuine interest and passion for the role.",
            "examples": "Good phrases: 'Estoy especialmente interesado/a en...', 'Mi experiencia en... me convierte en un candidato ideal', 'Me encantaría tener la oportunidad de...'"
        },
        "Italian": {
            "style": "Use formal Italian with Lei form. Italian business communication balances professionalism with warmth and personal touch.",
            "cultural": "Italian business culture appreciates passion, creativity, and personal connection. Show enthusiasm while maintaining professionalism.",
            "examples": "Good phrases: 'Sono particolarmente interessato/a a...', 'La mia esperienza in... mi rende un candidato ideale', 'Mi piacerebbe avere l'opportunità di...'"
        }
    }
    
    return language_guides.get(language, {})

def build_prompt(resume_text, cover_text, job_text, additional_notes="", tone=None, language="English", max_length=350):
    """
    Build the system + user prompt following the reusable prompt spec.
    IMPORTANT: Only use information present in resume_text, cover_text, user-provided notes, and job description.
    Do NOT invent facts.
    """
    if tone is None:
        tone = []

    tone = ", ".join(tone) if tone else "natural, confident"

    # Get language-specific instructions
    lang_guide = get_language_instructions(language)

    # Build language-specific instructions
    if language != "English" and lang_guide:
        language_instruction = f"""
        CRITICAL: Write the response in {language}. Follow these specific guidelines for native-quality output:

        Style: {lang_guide.get('style', '')}
        Cultural Context: {lang_guide.get('cultural', '')}
        Example Phrases: {lang_guide.get('examples', '')}

        IMPORTANT:
        - Use only native-level {language} expressions
        - Ensure the text sounds natural to a native {language} speaker
        - Use appropriate formal/informal register for business context
        - Avoid literal translations from English
        """
    else:
        language_instruction = ""

    system_instructions = (
        "You are a personal assistant that writes professional job application cover letters and responses."
        "Always write in first person as the job applicant expressing interest in the position."
        "Begin by expressing enthusiasm for the specific role and company you're applying to."
        "Only use information from the resume, cover letter, user-provided notes, and job description."
        "Do not invent details. If a skill or experience is missing, emphasize transferable skills instead."
        "Explicitly connect your skills, projects, or achievements to the requirements in the job description, "
        "and use keywords from the posting where appropriate."
        "Focus on how you can deliver value to the company, not just on listing skills."
        "Make sure to bring variety in sentences, not just starting with 'My', 'I', or 'Me'."
        "Target length: 200–250 words (3–4 short paragraphs)."
        "Each sentence should be under 20 words for readability."
        "If measurable results are present in the resume/cover letter, include one strong example."
        "Always close with a confident, positive line about contributing to the role or team."
        f"Keep tone: {tone}. {language_instruction}"
    )

    user_content = (
        f"Resume:\n{resume_text}\n\n"
        f"Cover Letter:\n{cover_text}\n\n"
        f"Job Description:\n{job_text}\n\n"
        f"Additional Notes (user-provided):\n{additional_notes}\n\n"
        f"Tone:\n{tone}\n\n"
        f"Language:\n{language}\n\n"
        "Instructions: Using only the information above, produce a cohesive reply tailored to the job description."
        "Make it sound natural, confident, and professional — not like an AI. Avoid complex language use like 'prowess', 'honed', or 'renowned'."
        "Show enthusiasm for the role, align your skills with requirements, and highlight how you will add value to the company."
        "Do not add any claims not supported by the provided materials."
        "If no direct match is found, highlight transferable skills starting with your study background or related projects."
        "This should be a job application cover letter, not a response from the employer."
    )

    return system_instructions, user_content

# Page config
st.set_page_config(page_title="Job Response Generator", layout="wide")

st.title("Job Response Generator — Streamlined replies from your resume + cover letter")
st.markdown("Upload your resume and cover letter, paste or provide a job URL, type the question or message you want to answer, and get a short, natural reply you can send.")

# Upload resume and cover letter
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf','docx'])
    cover_file = st.file_uploader("Upload Cover Letter (PDF or DOCX)", type=['pdf','docx'])
with col2:
    job_url = st.text_input("Job URL (optional)")
    job_text = st.text_area("Or paste job description text here (optional)")

# Tone and language selection
col1, col2 = st.columns(2)
with col1:
    # Tone selection
    tone = st.multiselect("Tone", ["Natural", "Professional", "Confident", "Conversational", "Enthusiastic"])
with col2:
    # Language selection
    language = st.selectbox("Language", ["English", "Dutch", "French", "German", "Spanish", "Italian"])

# Additional user-provided notes at the bottom
additional_notes = st.text_input("Manual notes (optional)", placeholder="Add any short notes you want considered, e.g., 'experience with WordPress'.")

# Generate reply
if st.button("Generate reply"):
    resume_text = ""
    cover_text = ""
    if resume_file:
        bytes_data = resume_file.read()
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(bytes_data)
        else:
            resume_text = extract_text_from_docx(bytes_data)
    if cover_file:
        bytes_data = cover_file.read()
        if cover_file.type == "application/pdf":
            cover_text = extract_text_from_pdf(bytes_data)
        else:
            cover_text = extract_text_from_docx(bytes_data)

    # Job description
    if job_url and not job_text:
        job_text = fetch_job_description(job_url)

    # Basic validation
    if not (resume_text or cover_text):
        st.error("Please upload at least your resume or cover letter.")
    elif not (job_text or job_url):
        st.warning("No job description provided so the reply will be generic based only on your resume/cover letter.")

    system_instructions, user_content = build_prompt(resume_text, cover_text, job_text, additional_notes=additional_notes, tone=tone, language=language)

    # Show the built prompt for transparency
    with st.expander("Preview prompt sent to the model (for transparency)"):
        st.markdown("**System instructions:**")
        st.text(system_instructions)
        st.markdown("**User content:**")
        st.text(user_content[:2000] + ("..." if len(user_content)>2000 else ""))

    if not openai.api_key:
        st.error("OpenAI API key missing. Please check your API key file or environment variable.")
    else:
        try:
            st.info("Generating... this can take a few seconds")
            # Adjust temperature based on language for better native output
            temperature = 0.3 if language == "English" else 0.4
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=600,
            )
            output = response.choices[0].message.content.strip()
            st.success("Reply generated")
            st.text_area("Generated reply", value=output, height=300, key="generated_reply")
            
        except Exception as e:
            st.error(f"Error while generating: {e}")

st.markdown("---")
st.caption("Template strictly uses only uploaded resume & cover letter and provided job description. Do not claim skills not present in your documents.")
