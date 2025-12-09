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
from main_backend import extract_text_from_pdf, extract_text_from_docx, fetch_job_description, build_prompt, load_api_key

# Page config
st.set_page_config(page_title="Job Response Generator", layout="wide")

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize OpenAI client
api_key = load_api_key()
if api_key:
    openai = OpenAI(api_key=api_key)
else:
    openai = None

st.title("Job Response Generator — Streamlined replies from your resume + cover letter")
st.markdown("Upload your resume and cover letter, paste or provide a job URL, type the question or message you want to answer, and get a short, natural reply you can send.")

# Upload resume and cover letter
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=['pdf','docx'])
    cover_file = st.file_uploader("Upload previous Cover Letter (PDF or DOCX) (optional)", type=['pdf','docx'])
with col2:
    job_url = st.text_input("Job URL (optional)")
    job_text = st.text_area("Or paste job description text here (optional)")


# Additional user-provided notes at the bottom
additional_notes = st.text_input("Manual notes (optional)", placeholder="Add any short notes you want considered, e.g., 'experience with WordPress'.")

# Tone, language, and formality selection
col1, col2, col3 = st.columns(3)
with col1:
    # Tone selection
    tone = st.multiselect("Tone", ["Natural", "Professional", "Confident", "Conversational", "Enthusiastic"])
with col2:
    # Language selection
    language = st.selectbox("Language", ["English", "Dutch", "French", "German", "Spanish", "Italian"])
with col3:
    # Formal/informal selection (only show for languages that have this distinction)
    formality_options = {
        "English": None,  # English doesn't have formal/informal distinction
        "Dutch": ["Formal (u)", "Informal (je)"],
        "French": ["Formal (vous)", "Informal (tu)"],
        "German": ["Formal (Sie)", "Informal (du)"],
        "Spanish": ["Formal (usted)", "Informal (tú)"],
        "Italian": ["Formal (Lei)", "Informal (tu)"]
    }
    
    if formality_options[language]:
        formality = st.selectbox("Formality", formality_options[language], key=f"formality_{language}")
    else:
        formality = None
        st.selectbox("Formality", ["Standard business tone"], disabled=True, key="formality_disabled")

# Store generated content in session state
if 'last_generated_content' not in st.session_state:
    st.session_state.last_generated_content = ""

# Generate button - change text based on whether content exists
button_text = "Re-generate response" if st.session_state.last_generated_content else "Generate response"
generate_clicked = st.button(button_text, key="generate_btn", use_container_width=False)

# Check for regenerate button from bottom (will be defined later but we check here)
regenerate_clicked = st.session_state.get('regenerate_clicked', False)
if 'regenerate_clicked' in st.session_state:
    del st.session_state['regenerate_clicked']

# Process files and generate content (works for both buttons)
if generate_clicked or regenerate_clicked:
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
        st.stop()
    elif not (job_text or job_url):
        st.warning("No job description provided so the reply will be generic based only on your resume/cover letter.")

    system_instructions, user_content = build_prompt(resume_text, cover_text, job_text, additional_notes=additional_notes, tone=tone, language=language, formality=formality)

    # Show the built prompt for transparency
    with st.expander("Preview prompt sent to the model (for transparency)"):
        st.markdown("**System instructions:**")
        st.text(system_instructions)
        st.markdown("**User content:**")
        st.text(user_content[:2000] + ("..." if len(user_content)>2000 else ""))

    if not openai or not openai.api_key:
        st.error("OpenAI API key missing. Please check your API key file or environment variable.")
    else:
        try:
            # Generate new response
            st.info("Generating...")
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
            
            # Store the generated content
            if output:
                st.session_state.last_generated_content = output
            
        except Exception as e:
            st.error(f"Error while generating: {e}")

# Show last generated content if available
# Note: Streamlit reruns on widget changes, but we only process/generate when button is clicked
if st.session_state.get('last_generated_content'):
    st.markdown("### Generated Reply")
    # Use columns to make text area narrower (70% width), left-aligned
    col_left, col_right = st.columns([0.7, 0.3])
    with col_left:
        st.text_area(
            "Generated reply",
            value=st.session_state.last_generated_content,
            height=400,
            key="generated_reply_display",
            disabled=False,
            label_visibility="collapsed"
        )
        st.caption("💡 Select text and copy (Ctrl+C / Cmd+C)")
    
    # User-friendly note about checking details
    st.info("📝 **Before sending:** Please review the generated cover letter and update:\n"
            "• The greeting (replace (#) with the hiring manager's name if available in the pasted job description)\n"
            "• Any specific questions or requirements mentioned in the job description (e.g., salary expectations, availability, project duration)")
    
    # Re-generate button below the output
    if st.button("Re-generate response", key="regenerate_btn_bottom", use_container_width=False):
        st.session_state['regenerate_clicked'] = True
        st.rerun()
