import os
import json
import streamlit as st
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

class ChatParameterCollector:
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
    
    def is_skip_response(self, response: str) -> bool:
        """Check if the response indicates a desire to skip"""
        skip_phrases = [
            "skip", "don't want to share", "dont want to share",
            "do not want to share", "i don't wish to share",
            "prefer not to share", "rather not say", "SKIP"
        ]
        return any(phrase in response.lower() for phrase in skip_phrases)

    def get_missing_parameters(self, extracted_data: Dict) -> List[tuple]:
        missing_params = []
        for category, subcategories in self.parameter_structure.items():
            for param in subcategories.keys():
                if not extracted_data.get(category, {}).get(param, []):
                    missing_params.append((category, param))
        return missing_params

    def validate_response(self, response: str, param: str) -> bool:
        if self.is_skip_response(response):
            return True, ""

        prompt = f"""
        Validate if this response makes sense for the parameter '{param.replace('_', ' ')}':
        Response: {response}
        
        Return a JSON object with:
        1. "is_valid": boolean indicating if response is valid and makes sense
        2. "reason": explanation if invalid
        """

        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at validating job description parameters. Be strict but reasonable in validation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            result = json.loads(chat_completion.choices[0].message.content)
            return result["is_valid"], result.get("reason", "")
        except Exception as e:
            return False, str(e)

    def process_response(self, response: str, param: str) -> List[str]:
        """Process user response into structured format"""
        if self.is_skip_response(response):
            return ["-----"]

        prompt = f"""
        Extract key points from this response about {param.replace('_', ' ')}.
        Convert into clear, professional statements.

        Example: "The candidate must have BSc or BS in Computer Science / Information Technology or equivalent.", "Bsc CS"
        Sample Response: "{{
                "points": [
                    "Bachelor of Science in Computer Science is the required education for this position.",
                    "A degree in Information Technology is also acceptable."
                ]
            }}"
        Response: {response.replace("{", "{{").replace("}", "}}")}
        """

        try:
            chat_completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting and formatting key information. Return only a JSON array of strings."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            # print(chat_completion.choices[0].message.content)
            return json.loads(chat_completion.choices[0].message.content).get("points", [])
        except Exception as e:
            return [response]

def main():
    st.title("🗣️ Job Description Completion Chat")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_param_index" not in st.session_state:
        st.session_state.current_param_index = 0
    if "can_proceed" not in st.session_state:
        st.session_state.can_proceed = False
    if "asked_to_proceed" not in st.session_state:
        st.session_state.asked_to_proceed = False
    if "current_category" not in st.session_state:
        st.session_state.current_category = None
    if "current_param" not in st.session_state:
        st.session_state.current_param = None
    if "waiting_for_retry" not in st.session_state:
        st.session_state.waiting_for_retry = False
    if "extracted_data" not in st.session_state:
        try:
            with open('db.json', 'r') as f:
                st.session_state.extracted_data = json.load(f)
        except FileNotFoundError:
            st.error("Please extract job parameters first!")
            return

    collector = ChatParameterCollector()
    missing_params = collector.get_missing_parameters(st.session_state.extracted_data)

    if not missing_params:
        st.success("All parameters are already collected! 🎉")
        st.json(st.session_state.extracted_data)
        return

    # Display progress if we're collecting parameters
    if st.session_state.can_proceed:
        progress = st.session_state.current_param_index / len(missing_params)
        st.progress(progress)
        st.write(f"Progress: {st.session_state.current_param_index}/{len(missing_params)} parameters collected")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Initial consent message if not asked yet
    if not st.session_state.asked_to_proceed and not st.session_state.messages:
        initial_msg = "I've identified some missing information in the job description. Shall we proceed with collecting it?"
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})
        st.session_state.asked_to_proceed = True
        with st.chat_message("assistant"):
            st.markdown(initial_msg)

    # Handle user input
    if prompt := st.chat_input("Your response..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # If we haven't gotten consent yet, process the response for consent
        if not st.session_state.can_proceed:
            consent_prompt = f"""
            Determine if this response indicates willingness to proceed:
            Response: {prompt}
            Return JSON: {{"is_positive": boolean}}
            """
            try:
                consent_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": consent_prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )
                is_positive = json.loads(consent_response.choices[0].message.content)["is_positive"]
                
                with st.chat_message("assistant"):
                    if is_positive:
                        category, param = missing_params[0]
                        st.session_state.current_category = category
                        st.session_state.current_param = param
                        response = f"Great! Let's begin. First, could you tell me about the {param.replace('_', ' ')}?"
                        st.session_state.can_proceed = True
                    else:
                        response = "I understand. Would you like to proceed with collecting the missing information?"
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

            except Exception as e:
                st.error(f"Error processing response: {e}")
                return

        # If we have consent, proceed with parameter collection
        elif st.session_state.current_param_index < len(missing_params):
            # Get current parameter we're working with
            current_category = st.session_state.current_category
            current_param = st.session_state.current_param
            
            # Validate response against the current parameter
            is_valid, reason = collector.validate_response(prompt, current_param)
            
            with st.chat_message("assistant"):
                print('The response is :', is_valid, '-', reason)
                if is_valid:
                    # Process and save valid response
                    processed_response = collector.process_response(prompt, current_param)

                    # Update JSON
                    if current_category not in st.session_state.extracted_data:
                        st.session_state.extracted_data[current_category] = {}
                    st.session_state.extracted_data[current_category][current_param] = processed_response
                    
                    # Save to file
                    with open('db.json', 'w') as f:
                        json.dump(st.session_state.extracted_data, f)
                    
                    # Move to next parameter
                    st.session_state.current_param_index += 1
                    st.session_state.waiting_for_retry = False
                    
                    if st.session_state.current_param_index < len(missing_params):
                        next_category, next_param = missing_params[st.session_state.current_param_index]
                        st.session_state.current_category = next_category
                        st.session_state.current_param = next_param
                        response = f"Perfect! Now, could you tell me about the {next_param.replace('_', ' ')}?"
                    else:
                        response = "Excellent! We've collected all the missing information! 🎉"
                else:
                    # Stay on the same parameter and prompt the user to try again
                    st.session_state.waiting_for_retry = True
                    response = f"I'm not sure that response quite fits what we need for {current_param.replace('_', ' ')}. {reason} Could you please provide more specific information?"

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

    if st.session_state.current_param_index >= len(missing_params) and st.session_state.can_proceed:
        st.success("All missing information has been collected! 🎉")
        # st.json(st.session_state.extracted_data)

if __name__ == "__main__":
    main()