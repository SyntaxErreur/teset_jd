import os
import json
import streamlit as st
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


class JobDescriptionAnalyzer:
    def __init__(self):
        self.parameter_structure = {
            "Position Details": ["role_title", "experience_required", "education_requirement", "engagement_type", "grade_level"],
            "Technical Requirements": ["mandatory_skills", "technical_skills", "good_to_have_skills"],
            "Job Responsibilities": ["primary_responsibilities", "team_responsibilities", "project_responsibilities"],
            "Soft Skills": ["communication_skills", "interpersonal_skills", "other_competencies"],
            "Work Details": ["location", "work_mode", "work_hours", "notice_period"],
            "Additional Information": ["company_info", "department", "benefits", "travel_requirements"]
        }

    def analyze_job_description(self, jd_text: str) -> Dict:
        """Analyze job description using GPT-4 to extract structured information."""
        system_prompt = """You are an expert job description analyzer. Extract structured information from the provided job description.
        Return only a JSON object with the extracted information organized according to the given categories and parameters.
        If information is not found, include the parameter with an empty list."""

        analysis_prompt = f"""Analyze this job description and extract information for these categories:
        {json.dumps(self.parameter_structure, indent=2)}

        Job Description:
        {jd_text}

        Return a JSON object where each category is a dictionary containing parameters as keys with list values.
        Example format:
        {{
            "Position Details": {{
                "role_title": ["Software Engineer"],
                "experience_required": ["3-5 years"]
            }},
            ...
        }}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.0
            )

            analyzed_data = json.loads(response.choices[0].message.content)
            # Initialize any missing categories and parameters
            for category, params in self.parameter_structure.items():
                if category not in analyzed_data:
                    analyzed_data[category] = {}
                for param in params:
                    if param not in analyzed_data[category]:
                        analyzed_data[category][param] = []
            return analyzed_data
        except Exception as e:
            st.error(f"Error analyzing job description: {str(e)}")
            # Return properly structured empty data
            return {category: {param: [] for param in params}
                    for category, params in self.parameter_structure.items()}


class QuestionGenerator:
    def generate_contextual_question(self, param: str, role_context: str, category: str, collected_data: Dict) -> str:
        """Generate a contextual question based on the parameter and existing data."""
        system_prompt = """You are an expert technical recruiter specialized in improving job descriptions.
        Generate a specific, contextual question to gather missing information.
        Consider the role context, category, and already collected information."""

        context_prompt = f"""Given:
        - Role: {role_context}
        - Category: {category}
        - Parameter: {param}
        - Collected Data: {json.dumps(collected_data, indent=2)}

        Generate a natural, conversational question to gather information about {param.replace('_', ' ')}.
        Requirements:
        - Be specific to the role and context
        - Consider existing information
        - Keep it clear and professional
        - For skills/responsibilities, encourage detailed responses
        - Maximum 25 words"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could you please provide information about {param.replace('_', ' ')}?"


class ResponseValidator:
    def validate_response(self, response: str, param: str, role_context: str, category: str, collected_data: Dict) -> Tuple[bool, Optional[str], List[str]]:
        """Validate user response using GPT-4."""
        if response.lower() in ['skip', 'no', 'none', 'na', 'next']:
            return True, None, ['Not specified']

        system_prompt = """You are an expert job description validator.
        Evaluate if the response is appropriate and meaningful for the given parameter.
        Consider the role context and existing information."""

        validation_prompt = f"""Validate this response:
        Role Context: {role_context}
        Category: {category}
        Parameter: {param}
        Response: {response}
        Existing Data: {json.dumps(collected_data, indent=2)}

        Requirements:
        1. Check relevance and appropriateness
        2. Verify alignment with role and industry standards
        3. Check completeness and clarity
        4. Ensure response follows common HR practices

        Return exactly one of these formats:
        If valid: VALID|[list,of,processed,items]
        If invalid: INVALID|detailed_feedback"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.3
            )

            result = response.choices[0].message.content
            status, content = result.split('|', 1)

            if status == 'VALID':
                return True, None, json.loads(content)
            else:
                return False, content, []
        except Exception as e:
            return False, str(e), []


def main():
    st.set_page_config(page_title="JD Enhancement Assistant", layout="wide")
    st.title("🎯 Advanced Job Description Enhancement Assistant")
    st.markdown(
        "*Transform your job descriptions into comprehensive and effective listings*")

    # Initialize session state
    if "state" not in st.session_state:
        st.session_state.state = {
            "initialized": False,
            "jd_analyzed": False,
            "current_param_index": 0,
            "collected_data": None,
            "missing_params": [],
            "conversation_history": []
        }

    # Initialize components
    analyzer = JobDescriptionAnalyzer()
    question_gen = QuestionGenerator()
    validator = ResponseValidator()

    # Job Description Input
    if not st.session_state.state["jd_analyzed"]:
        jd_text = st.text_area("📝 Enter the Job Description", height=200)
        if st.button("Analyze Job Description"):
            if jd_text:
                with st.spinner("Analyzing job description..."):
                    # Analyze JD
                    analyzed_data = analyzer.analyze_job_description(jd_text)
                    st.session_state.state["collected_data"] = analyzed_data

                    # Identify missing parameters
                    missing_params = []
                    for category, params in analyzer.parameter_structure.items():
                        for param in params:
                            if not analyzed_data[category][param]:  # Changed this line
                                missing_params.append((category, param))

                    st.session_state.state["missing_params"] = missing_params
                    st.session_state.state["jd_analyzed"] = True
                    st.rerun()
            else:
                st.warning("Please enter a job description first.")

    if st.session_state.state["jd_analyzed"]:
        # Display conversation history
        for message in st.session_state.state["conversation_history"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Handle current parameter
        if st.session_state.state["current_param_index"] < len(st.session_state.state["missing_params"]):
            category, param = st.session_state.state["missing_params"][
                st.session_state.state["current_param_index"]]

            # Generate and display question if not already asked
            if not st.session_state.state["conversation_history"] or \
               st.session_state.state["conversation_history"][-1]["role"] == "user":
                question = question_gen.generate_contextual_question(
                    param,
                    st.session_state.state["collected_data"].get(
                        "Position Details", {}).get("role_title", ["Unknown Role"])[0],
                    category,
                    st.session_state.state["collected_data"]
                )
                st.session_state.state["conversation_history"].append(
                    {"role": "assistant", "content": question})
                st.rerun()

            # Handle user response
            if user_response := st.chat_input("Your response..."):
                st.session_state.state["conversation_history"].append(
                    {"role": "user", "content": user_response})

                # Validate response
                is_valid, feedback, processed = validator.validate_response(
                    user_response,
                    param,
                    st.session_state.state["collected_data"].get(
                        "Position Details", {}).get("role_title", ["Unknown Role"])[0],
                    category,
                    st.session_state.state["collected_data"]
                )

                if is_valid:
                    # Update collected data
                    if category not in st.session_state.state["collected_data"]:
                        st.session_state.state["collected_data"][category] = {}
                    st.session_state.state["collected_data"][category][param] = processed

                    # Move to next parameter
                    st.session_state.state["current_param_index"] += 1
                else:
                    st.session_state.state["conversation_history"].append(
                        {"role": "assistant",
                            "content": f"⚠️ {feedback}\nPlease try again."}
                    )
                st.rerun()

        # Show completion message and final JD
        elif len(st.session_state.state["missing_params"]) > 0:
            st.success("🎉 Job Description Enhancement Complete!")

            # Generate final JD
            system_prompt = """You are an expert job description writer.
            Create a professional, well-formatted job description using the provided structured data."""

            final_jd_prompt = f"""Create a comprehensive job description using this data:
            {json.dumps(st.session_state.state["collected_data"], indent=2)}

            Requirements:
            - Professional and engaging tone
            - Clear structure with appropriate headings
            - Highlight key requirements and responsibilities
            - Include all provided information
            - Use markdown formatting"""

            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": final_jd_prompt}
                    ],
                    temperature=0.7
                )

                st.markdown("## 📋 Enhanced Job Description")
                st.markdown(response.choices[0].message.content)

                # Add download button for the enhanced JD
                st.download_button(
                    label="Download Enhanced JD",
                    data=response.choices[0].message.content,
                    file_name="enhanced_job_description.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"Error generating final job description: {str(e)}")


if __name__ == "__main__":
    main()
