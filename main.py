import os
import re
import json
import spacy
from openai import OpenAI
import streamlit as st
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


class JobDescriptionParameterExtractor:
    def __init__(self):
        """Initialize the job description parameter extractor"""
        self.nlp = spacy.load('en_core_web_sm')

        # Enhanced parameter structure with more specific categorization
        self.parameter_structure = {
            "Position Details": {
                "role_title": ["position", "role", "title", "designation", "specialist", "lead", "engineer"],
                "experience_required": ["years of experience", "work experience", "experience in", "experience with",
                                        "experience using", "familiarity", "knowledge", "understanding"],
                "education_requirement": ["education", "degree", "qualification", "academic", "BE", "ME", "MCA", "B.Tech", "M.Tech"],
                "engagement_type": ["full-time", "contract", "part-time", "permanent", "temporary"],
                "grade_level": ["grade", "band", "level", "position level", "lead", "specialist", "senior"]
            },
            "Technical Requirements": {
                "mandatory_skills": ["required skills", "must have", "mandatory skills", "essential skills",
                                     "solid understanding", "core", "in-depth knowledge"],
                "technical_skills": ["technical", "programming", "technologies", "tools", "frameworks",
                                     "development", "platforms", "software"],
                "good_to_have_skills": ["good to have", "nice to have", "preferred", "desirable", "plus",
                                        "additionally", "bonus"]
            },
            "Job Responsibilities": {
                "primary_responsibilities": ["responsibilities", "duties", "tasks", "what you will do",
                                             "will be responsible for", "key responsibilities"],
                "team_responsibilities": ["team work", "collaboration", "work with teams", "cross-functional",
                                          "coordinate with", "work closely"],
                "project_responsibilities": ["project management", "deliverables", "ownership", "development",
                                             "maintain", "ensure", "quality"]
            },
            "Soft Skills": {
                "communication_skills": ["communication", "verbal", "written", "presentation", "stakeholder"],
                "interpersonal_skills": ["interpersonal", "team work", "collaboration", "working with others"],
                "other_competencies": ["analytical", "problem solving", "leadership", "passionate",
                                       "quality focused", "architecture", "design thinking"]
            },
            "Work Details": {
                "location": ["location", "place", "city", "office", "based in"],
                "work_mode": ["work mode", "remote", "hybrid", "onsite", "office", "work from"],
                "work_hours": ["working hours", "shift", "timing", "schedule", "GMT", "IST"],
                "notice_period": ["notice period", "joining time", "availability"]
            },
            "Additional Information": {
                "company_info": ["company", "organization", "firm", "business", "about us"],
                "department": ["department", "team", "division", "unit", "function"],
                "benefits": ["benefits", "perks", "compensation", "salary", "package", "offerings"],
                "travel_requirements": ["travel", "mobility", "relocation", "movement"]
            }
        }

    def extract_parameters_with_openai(self, job_description: str) -> Dict[str, Dict[str, List[str]]]:
        """Extract parameters using OpenAI's language model with enhanced prompting"""
        prompt = f"""
        Analyze this job description and provide a detailed categorization following these strict rules:

        1. Split all information into appropriate categories and subcategories
        2. Format each point as a complete, clear statement
        3. Separate distinct requirements even if they appear together in the text
        4. Include implied skills and requirements
        5. Remove duplicates but preserve different aspects of similar items
        6. Ensure consistent formatting across all entries
        7. Break down compound requirements into individual points
        8. Categorize both explicit and implicit requirements
        9. Use professional language and complete sentences
        10. Maintain hierarchical organization of information
        
        Job Description:
        {job_description}

        Use this parameter structure for categorization:
        {json.dumps(self.parameter_structure, indent=2)}

        Format the response as a JSON object where:
        - Each category contains relevant subcategories
        - Each point is a complete, well-formed statement
        - Related items are grouped together
        - Information is not duplicated but is comprehensive
        - If some details are missing, use an empty list
        - Make sure, nothing extra from the job description is included
        - Dont add any Skills which are not provided in the Job Description by yourselves.

        Example:
        - Designation: "frontend developer with there years of exp in react or vue.js"
        - Skills: ["React.js", "Vue.js"]
        - "role_title": "Frontend Developer"
        - "experience_required": "3 years"

        """

        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing job descriptions and extracting detailed, well-organized information and outputs only valid JSON objects"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=4000,
                temperature=0.8,
                response_format={"type": "json_object"}
            )

            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI Extraction Error: {e}")
            return {}

    def format_output(self, extracted_params: Dict) -> str:
        """Format the extracted parameters in the desired output format"""
        output = ["# Extracted Job Description Parameters\n"]

        for category, parameters in extracted_params.items():
            if parameters and any(parameters.values()):
                output.append(f"## {category}")

                for param, values in parameters.items():
                    if values:
                        values_list = self.ensure_list(values)
                        if values_list:
                            # Format parameter name
                            output.append(
                                f"**{param.replace('_', ' ').title()}**:")
                            # Add bullet points for values
                            for value in values_list:
                                output.append(f"- {value.strip()}")
                            output.append("")  # Add spacing between parameters

                output.append("")  # Add spacing between categories

        return "\n".join(output)

    def ensure_list(self, value: Union[str, List[str], None]) -> List[str]:
        """Convert a value to a list if it isn't already"""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return list(value)


def main():
    st.title("🎯 Job Description Parameter Extractor")

    job_description = st.text_area("Paste Job Description", height=300)

    if st.button("Extract Parameters"):
        if not job_description:
            st.warning("Please paste a job description")
            return

        try:
            with st.spinner("Analyzing job description..."):
                extractor = JobDescriptionParameterExtractor()
                extracted_params = extractor.extract_parameters_with_openai(
                    job_description)
                with open('db.json', 'w') as f:
                    json.dump(extracted_params, f)
                # formatted_output = extractor.format_output(extracted_params)
                # st.markdown(formatted_output)

                st.success("Parameters extracted successfully!✨")

                # if st.button("Proceed with Conversation"):
                #     # open new page named 'conversation_new.py' inside of pages folder
                #     st.session_state.page = 'conversation_new'
                #     # st.session_state.page = 'conversation'
                #     # st.rerun()

        except Exception as e:
            st.error(f"Extraction Error: {str(e)}")


if __name__ == "__main__":
    main()
