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
    """Load OpenAI API key from environment variables or local file."""
    # Priority 1: Environment variables
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    
    # Priority 2: Local file backup (for development)
    try:
        with open("OPENAI_API_KEY.txt", "r") as f:
            key = f.read().strip()
            if key:
                return key
    except FileNotFoundError:
        pass

    st.error("❌ OpenAI API key not found in environment variables.")
    return None

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

def get_greeting_format(language, formality=None):
    """Get appropriate greeting format with placeholder for the language and formality level."""
    greeting_formats = {
        "English": {
            "formal": "Dear (#)",
            "informal": "Dear (#)"
        },
        "Dutch": {
            "formal": "Geachte (#)",
            "informal": "Beste (#)"
        },
        "French": {
            "formal": "Madame, Monsieur",
            "informal": "Bonjour (#)"
        },
        "German": {
            "formal": "Sehr geehrte/r (#)",
            "informal": "Hallo (#)"
        },
        "Spanish": {
            "formal": "Estimado/a (#)",
            "informal": "Hola (#)"
        },
        "Italian": {
            "formal": "Gentile (#)",
            "informal": "Salve (#)"
        }
    }
    
    base_format = greeting_formats.get(language, {"formal": "Dear (#)", "informal": "Dear (#)"})
    
    if formality and "formal" in formality.lower():
        return base_format.get("formal", "Dear (#)")
    elif formality and "informal" in formality.lower():
        return base_format.get("informal", "Dear (#)")
    else:
        # Default to formal for business contexts
        return base_format.get("formal", "Dear (#)")

def get_language_instructions(language, formality=None):
    """Get detailed language-specific instructions for better native output."""
    language_guides = {
        "Dutch": {
            "formal": {
                "style": "Use natural, business-appropriate Dutch with 'u' for formal address. Avoid overly formal constructions like 'ik ben goed uitgerust om de taak' - use more natural expressions like 'ik heb ervaring met' or 'ik zou graag bijdragen aan'. Include typical Dutch business expressions like 'graag', 'bijzonder', 'uitstekend'.",
                "cultural": "Dutch business culture values directness and efficiency. Be straightforward but polite. Use natural, conversational tone even in formal contexts.",
                "avoid": "Avoid overly formal or literal translations from English. Don't use constructions like 'goed uitgerust zijn om' - this sounds unnatural in Dutch."
            },
            "informal": {
                "style": "Use natural, friendly Dutch with 'je' for informal address. Keep it professional but approachable. Use common Dutch expressions and natural sentence structures.",
                "cultural": "Dutch informal business communication is still professional but more relaxed and personal.",
                "avoid": "Avoid overly casual expressions that might be unprofessional in a business context."
            }
        },
        "French": {
            "formal": {
                "style": "Use formal French with proper business etiquette. Use 'vous' form throughout. Include sophisticated vocabulary and proper French business expressions.",
                "cultural": "French business culture appreciates elegance and sophistication. Use refined language and show appreciation for the company's values.",
                "avoid": "Avoid overly complex sentence structures that might sound unnatural."
            },
            "informal": {
                "style": "Use 'tu' form but maintain professional tone. French informal business communication still requires elegance.",
                "cultural": "French informal business communication maintains a certain level of sophistication.",
                "avoid": "Avoid overly casual expressions that might be inappropriate in business contexts."
            }
        },
        "German": {
            "formal": {
                "style": "Use formal German (Sie form). German business communication is precise and structured. Use compound words appropriately and maintain professional tone.",
                "cultural": "German business culture values precision, reliability, and thoroughness. Be specific about qualifications and achievements.",
                "avoid": "Avoid overly complex compound words that might be hard to read."
            },
            "informal": {
                "style": "Use 'du' form but maintain professional structure and precision typical of German business communication.",
                "cultural": "German informal business communication still values clarity and precision.",
                "avoid": "Avoid overly casual expressions that might seem unprofessional."
            }
        },
        "Spanish": {
            "formal": {
                "style": "Use formal Spanish with usted form. Include appropriate business vocabulary and maintain professional but warm tone typical of Spanish business culture.",
                "cultural": "Spanish business culture values personal relationships and enthusiasm. Show genuine interest and passion for the role.",
                "avoid": "Avoid overly formal expressions that might sound cold or distant."
            },
            "informal": {
                "style": "Use 'tú' form but maintain professional warmth and enthusiasm typical of Spanish business culture.",
                "cultural": "Spanish informal business communication still emphasizes personal connection and enthusiasm.",
                "avoid": "Avoid overly casual expressions that might be inappropriate in business contexts."
            }
        },
        "Italian": {
            "formal": {
                "style": "Use formal Italian with Lei form. Italian business communication balances professionalism with warmth and personal touch.",
                "cultural": "Italian business culture appreciates passion, creativity, and personal connection. Show enthusiasm while maintaining professionalism.",
                "avoid": "Avoid overly formal expressions that might sound cold or distant."
            },
            "informal": {
                "style": "Use 'tu' form but maintain the warmth and personal touch typical of Italian business communication.",
                "cultural": "Italian informal business communication still emphasizes passion and personal connection.",
                "avoid": "Avoid overly casual expressions that might be inappropriate in business contexts."
            }
        }
    }
    
    base_guide = language_guides.get(language, {})
    if not base_guide:
        return {}
    
    # Return the appropriate formality level guide
    if formality and "formal" in formality.lower():
        return base_guide.get("formal", {})
    elif formality and "informal" in formality.lower():
        return base_guide.get("informal", {})
    else:
        # Default to formal for business contexts
        return base_guide.get("formal", {})

def build_prompt(resume_text, cover_text, job_text, additional_notes="", tone=None, language="English", formality=None, max_length=350):
    """
    Build the system + user prompt following the reusable prompt spec.
    IMPORTANT: Only use information present in resume_text, cover_text, user-provided notes, and job description.
    Do NOT invent facts.
    """
    if tone is None:
        tone = []

    tone = ", ".join(tone) if tone else "natural, confident"

    # Get language-specific instructions
    lang_guide = get_language_instructions(language, formality)

    # Build language-specific instructions
    if language != "English" and lang_guide:
        formality_desc = formality if formality else 'standard business tone'
        language_instruction = f"""
        LANGUAGE: Write in {language} using {formality_desc}.
        
        FORMALITY LEVEL: {formality_desc.upper()}. You MUST use the appropriate formality level throughout the entire letter - in both the greeting AND the body content.
        
        STYLE: {lang_guide.get('style', '')}
        CULTURAL CONTEXT: {lang_guide.get('cultural', '')}
        AVOID: {lang_guide.get('avoid', '')}
        """
    else:
        language_instruction = ""

    greeting_format = get_greeting_format(language, formality)
    system_instructions = (
        "You are a personal assistant that writes professional job application cover letters and responses. "
        "Always write in first person as the job applicant expressing interest in the position. "
        "Only use information from the resume, cover letter, user-provided notes, and job description. "
        "Do not invent details. If a skill or experience is missing, emphasize transferable skills instead. "
        "Explicitly connect your skills, projects, or achievements to the requirements in the job description, "
        "and use keywords from the posting where appropriate. "
        "Focus on how you can deliver value to the company, not just on listing skills. "
        "Make sure to bring variety in sentences, not just starting with 'My', 'I', or 'Me'. "
        "Each sentence should be under 20 words for readability. "
        "If measurable results are present in the resume/cover letter, include one strong example. "
        "MANDATORY GREETING FORMAT: The cover letter MUST begin with the exact format provided: '{greeting_format}' where (#) is a placeholder. "
        "Format: Write a complete professional cover letter with: "
        f"1) A proper introduction/greeting ('{get_greeting_format(language, formality)}' where (#) is a placeholder for the recipient's name - use this exact format), "
        "2) Exactly 3 body paragraphs, each containing 40–55 words, expressing enthusiasm for the role and connecting your skills and qualifications to the job requirements, "
        "3) Followed by a short closing paragraph with a confident, positive one-liner sentence about contributing to the role or team, "
        "4) An official closing (e.g., 'Sincerely' or appropriate closing for the language) followed by your full name and email address (extract from resume), then on a new line add: [LinkedIn](#) | [Github](#) | [Website](#) as placeholders. "
        "CRITICAL: Do not repeat information across paragraphs. The closing paragraph (step 3) must be distinct from the third body paragraph (step 2) - avoid repeating phrases like 'I look forward to' or similar expressions in both. "
        f"Keep tone: {tone}. {language_instruction}"
    )

    user_content = (
        f"Resume:\n{resume_text}\n\n"
        f"Cover Letter:\n{cover_text}\n\n"
        f"Job Description:\n{job_text}\n\n"
        f"Additional Notes (user-provided):\n{additional_notes}\n\n"
        f"Language:\n{language}\n"
        f"Formality:\n{formality if formality else 'Standard business tone'}\n\n"
        "Instructions: Using only the information above, produce a complete professional cover letter tailored to the job description. "
        "Make it sound natural and professional — not like an AI. Avoid complex language use like 'prowess', 'honed', or 'renowned'. "
        "Do not add any claims not supported by the provided materials. "
        "If no direct match is found, highlight transferable skills starting with your study background or related projects. "
        "This should be a job application cover letter, not a response from the employer. "
        "Structure requirements: "
        f"1) Start with a proper greeting ('{get_greeting_format(language, formality)}' where (#) is a placeholder for the recipient's name - use this exact format), "
        "2) Write exactly 3 body paragraphs of 40-55 words each, expressing enthusiasm for the role and connecting your skills and qualifications to the job requirements, "
        "3) Followed by a short closing paragraph with a confident, positive one-liner sentence about contributing to the role or team, "
        "4) End with an official closing (e.g., 'Sincerely' or appropriate closing for the language) followed by your full name and email address (extract from resume), then on a new line add: [LinkedIn](#) | [Github](#) | [Website](#) as placeholders. "
        "CRITICAL: The closing paragraph (step 3) must be distinct from the third body paragraph (step 2). Do not repeat phrases like 'I look forward to contributing' or similar expressions in both paragraphs. "
    )

    return system_instructions, user_content
