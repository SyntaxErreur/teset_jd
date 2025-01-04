import os
import json
import streamlit as st
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

class ConversationalParameterCollector:
    def __init__(self):
        self.parameter_structure = {
            "Position Details": {
                "role_title": [],
                "experience_required": [],
                "education_requirement": [],
                "engagement_type": [],
                "grade_level": []
            },
            "Technical Requirements": {
                "mandatory_skills": [],
                "technical_skills": [],
                "good_to_have_skills": []
            },
            "Job Responsibilities": {
                "primary_responsibilities": [],
                "team_responsibilities": [],
                "project_responsibilities": []
            },
            "Soft Skills": {
                "communication_skills": [],
                "interpersonal_skills": [],
                "other_competencies": []
            },
            "Work Details": {
                "location": [],
                "work_mode": [],
                "work_hours": [],
                "notice_period": []
            },
            "Additional Information": {
                "company_info": [],
                "department": [],
                "benefits": [],
                "travel_requirements": []
            }
        }

    def get_missing_parameters(self, extracted_data: Dict) -> List[tuple]:
        """Identify missing parameters from the extracted data"""
        missing_params = []
        for category, subcategories in self.parameter_structure.items():
            for param in subcategories.keys():
                if not extracted_data.get(category, {}).get(param, []):
                    missing_params.append((category, param))
        return missing_params

    def generate_conversation(self, param: str) -> str:
        """Generate a conversational prompt for collecting missing parameters"""
        prompt = f"""
        Based on the parameter '{param.replace('_', ' ')}', generate a natural, conversational question to ask the candidate.
        Make it sound friendly and casual, as if you're having a conversation.

        example: "Say, the field "experience_required" is empty like no data [], then ask the company what exactly they need for the new position. "How much Experience do you expect from the candidate ?"
        """
        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert recruiter who makes make detailed and complete job descriptions. Asking companies what exactly they need for the new position.  Generate only the question, no additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            st.error(f"Error generating conversation: {e}")
            return f"Could you please tell me about your {param.replace('_', ' ')}?"

    def process_response(self, response: str, param: str) -> List[str]:
        """Process user response and extract relevant information"""
        prompt = f"""
        Extract key points from this response about {param.replace('_', ' ')}.
        Convert the response into a list of clear, professional statements.
        Response: {response}
        """

        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting and formatting key information from conversational responses. Return only a JSON array of strings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            return json.loads(chat_completion.choices[0].message.content).get("points", [])

        except Exception as e:
            st.error(f"Error processing response: {e}")
            return [response]

def main():
    st.title("🗣️ Complete Missing Information")

    # Load existing data
    try:
        with open('db.json', 'r') as f:
            extracted_data = json.load(f)
    except FileNotFoundError:
        st.error("Please extract job parameters first!")
        return

    collector = ConversationalParameterCollector()
    missing_params = collector.get_missing_parameters(extracted_data)

    if not missing_params:
        st.success("All parameters are already collected! 🎉")
        return

    if 'current_param_index' not in st.session_state:
        st.session_state.current_param_index = 0

    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    if st.session_state.current_param_index < len(missing_params):
        category, param = missing_params[st.session_state.current_param_index]
        
        st.subheader(f"Category: {category}")
        question = collector.generate_conversation(param)
        st.write(question)

        user_response = st.text_area("Your response:", key=f"response_{param}")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Submit"):
                if user_response:
                    processed_response = collector.process_response(user_response, param)
                    if not category in extracted_data:
                        extracted_data[category] = {}
                    extracted_data[category][param] = processed_response
                    
                    # Save updated data
                    with open('db.json', 'w') as f:
                        json.dump(extracted_data, f)
                    
                    st.session_state.current_param_index += 1
                    st.rerun()
                else:
                    st.warning("Please provide a response.")

        progress = st.session_state.current_param_index / len(missing_params)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.current_param_index}/{len(missing_params)} parameters collected")

    else:
        st.success("All missing information has been collected! 🎉")
        st.json(extracted_data)

if __name__ == "__main__":
    main()