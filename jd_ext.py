import os
import re
import json
import spacy
from openai import OpenAI
import streamlit as st
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

class JobDescriptionParameterExtractor:
    def __init__(self):
        """
        Initialize the job description parameter extractor
        """

        # Load spaCy model for advanced NLP processing
        self.nlp = spacy.load('en_core_web_sm')

        # Predefined parameter structure
        self.parameter_structure = {
            "Customer Details": {
                "customer_name": ["client", "customer", "organization", "company"],
                "hiring_manager": ["hiring manager", "recruiter", "hr", "recruiter name"]
            },
            "Position Details": {
                "engagement_type": ["full-time", "contract", "part-time", "freelance"],
                "position_status": ["new", "replacement", "expansion"],
                "account_manager": ["account manager", "lead", "spoc"],
                "grade_bands": ["grade", "band", "level", "competency"],
                "education_requirement": ["education", "degree", "qualification"],
                "ctc_range": ["salary", "ctc", "compensation", "package"]
            },
            "Technical Skills": {
                "mandatory_skills": ["required skills", "must have", "mandatory"],
                "good_to_have_skills": ["good to have", "nice to have", "additional skills"]
            },
            "Business Unit Details": {
                "work_type": ["project type", "work type"],
                "team_size": ["team size", "team composition"],
                "project_type": ["in-house", "customer project", "outsourced"]
            },
            "Roles & Responsibilities": {
                "cultural_expectations": ["culture", "values", "expectations"],
                "key_roles": ["key roles", "primary responsibilities"],
                "reporting_structure": ["reporting", "hierarchy"],
                "non_technical_skills": ["soft skills", "interpersonal skills"],
                "screening_questions": ["screening", "interview questions"]
            },
            "Other Details": {
                "location": ["location", "preferred location"],
                "work_timings": ["work hours", "timings"],
                "notice_period": ["notice period"],
                "travel_requirements": ["travel", "travel expectation"],
                "job_changes": ["job changes", "career progression"]
            }
        }

    def preprocess_text(self, text: str) -> str:
        """
        Advanced text preprocessing

        Args:
            text (str): Raw job description text

        Returns:
            str: Preprocessed text
        """
        # Remove special characters and normalize
        text = re.sub(r'[^a-zA-Z0-9\s,.]', '', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()

        return text

    def extract_parameters_with_openai(self, job_description: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Extract parameters using OpenAI's advanced language model

        Args:
            job_description (str): Preprocessed job description

        Returns:
            Dict of extracted parameters
        """
        # Construct detailed prompt for parameter extraction
        prompt = f"""
        Analyze the following job description and extract parameters precisely:

        Job Description:
        {job_description}

        EXTRACTION GUIDELINES:
        1. Use the following parameter structure strictly
        2. Extract responses for each parameter
        3. Be comprehensive and context-aware
        4. Output format must be a JSON matching the given structure
        5. If some details are missing, use an empty list
        6. Make sure, nothing extra from the job description is included

        PARAMETER STRUCTURE:
        {json.dumps(self.parameter_structure, indent=2)}

        OUTPUT FORMAT:
        {{
            "Customer Details": {{
                "customer_name": ["extracted values"],
                "hiring_manager": ["extracted values"]
            }},
            ...
        }}
        """

        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert job description parameter extraction assistant, well-organized information and outputs only valid JSON objects."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.2,  # Low temperature for precision
                response_format={"type": "json_object"}
            )

            # Parse and return extracted data
            extracted_params = json.loads(
                chat_completion.choices[0].message.content)
            return extracted_params

        except Exception as e:
            st.error(f"OpenAI Extraction Error: {e}")
            return {}

    def nlp_parameter_validation(self, job_description: str, extracted_params: Dict) -> Dict:
        """
        Validate and enhance parameter extraction using spaCy NLP

        Args:
            job_description (str): Original job description
            extracted_params (Dict): Parameters extracted by OpenAI

        Returns:
            Validated and enhanced parameters
        """
        # Process job description with spaCy
        doc = self.nlp(job_description)

        # Additional NLP-based validation and extraction
        for category, parameters in extracted_params.items():
            for param, values in parameters.items():
                # Use Named Entity Recognition for validation
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'GPE', 'PERSON']:
                        # Add additional context-based extraction
                        if any(keyword in ent.text.lower() for keyword in self.parameter_structure.get(category, {}).get(param, [])):
                            values.append(ent.text)

        return extracted_params

    def format_extracted_parameters(self, extracted_params: Dict) -> str:
        """
        Format extracted parameters for display

        Args:
            extracted_params (Dict): Extracted job description parameters

        Returns:
            Formatted parameter output
        """
        output = ["# Extracted Parameters\n"]

        for category, parameters in extracted_params.items():
            output.append(f"## {category}")

            for param, values in parameters.items():
                # Remove duplicates and format
                unique_values = list(set(values))
                if unique_values:
                    output.append(
                        f"**{param.replace('_', ' ').title()}**: {', '.join(unique_values)}")

            output.append("")

        return "\n".join(output)


def main():
    st.title("🔍 Job Description Parameter Extractor")

    # Job Description input
    job_description = st.text_area("Paste Job Description", height=300)

    if st.button("🚀 Extract Parameters"):
        if not job_description:
            st.warning("Please paste a job description")
            return

        try:
            # Initialize extractor
            extractor = JobDescriptionParameterExtractor()

            preprocessed_jd = extractor.preprocess_text(job_description)

            extracted_params = extractor.extract_parameters_with_openai(
                preprocessed_jd)

            # Validate and enhance with NLP
            validated_params = extractor.nlp_parameter_validation(
                job_description, extracted_params)

            # Format and display results
            formatted_output = extractor.format_extracted_parameters(
                validated_params)

            st.markdown(formatted_output)

        except Exception as e:
            st.error(f"Extraction Error: {e}")


if __name__ == "__main__":
    main()
