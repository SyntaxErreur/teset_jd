from typing import Dict, Tuple
import os
import json
import streamlit as st
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
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
        """Analyze job description using gpt-4o-mini to extract structured information."""
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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.5
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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could you please provide information about {param.replace('_', ' ')}?"

class Category(str, Enum):
    answered = "answered"  # The user answered the bot's question
    follow_up = "follow_up"  # User provided follow-up feedback
    clarification = "clarification"  # User asked a clarifying question
    off_topic = "off_topic"  # User went off-topic or irrelevant
    partially_answered = "partially_answered"  # User partially answered the bot's question


class ValidationResponse(BaseModel):
    category: Category  # One of the 4 categories
    explanation: str  # Explanation for why the category was triggered

def validate_response(user_input: str, conversation_history: str) -> ValidationResponse:
    """
    Validates the user's response using OpenAI's structured response format.

    Args:
        user_input (str): The user's input.
        conversation_history (str): The conversation history as a single string.

    Returns:
        ValidationResponse: A parsed response containing `category` and `explanation`.
    """


    def handle_answered(parsed_response):
        """Handle the 'answered' category."""
        print("Handle the 'answered' category.")
        answered_prompt = """
        The user has provided an answer to the question. Validate if the response fully addresses the query or if it could be improved from a Hiring Manager's Perspective.
        Analyze the user's answer critically, considering the following:
        - Is the response well-structured and clear?
        - Does it provide specific, actionable details?
        - Is it aligned with what a hiring manager would expect for this context?

        If the response is valid, confirm it explicitly.
        If the response is vague or poorly structured, provide constructive feedback with specific suggestions for improvement, Also give a very good example for the user to understand better.

        Context:
        {conversation_history}
        User's Answer:
        {user_input}

        Your output must strictly adhere to this format:
        {{
            "category": "answered" | "partially_answered",
            "explanation": "Provide detailed reasoning here, explaining whether the response is fully valid or partially valid, and include actionable feedback if necessary."
        }}
        """
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": answered_prompt.format(conversation_history=conversation_history, user_input=user_input)},
                    {"role": "user", "content": "Validate and provide feedback on the above response."}
                ],
                temperature=0.7
            )
            bot_reply = json.loads(completion.choices[0].message.content)
            print("Bot reply =========", bot_reply)
            if bot_reply.get("category") in ["answered", "partially_answered"]:
                return ValidationResponse(
                    category=bot_reply["category"],
                    explanation=bot_reply["explanation"]
                )
            else:
                raise ValueError("Unexpected category in bot reply.")

        except Exception as e:
            raise ValueError(f"Error generating answered response validation: {str(e)}")

    def handle_follow_up(parsed_response):
        """Handle the 'follow_up' category."""
        print("Handle the 'follow_up' category.")
        follow_up_prompt = """
        The user has provided feedback on the bot's previous suggestion. Depending on the user's validation:
        - If the user accepts the suggestion (Yes), generate a new response based on the most recent bot suggestion from the conversation history.
        - If the user rejects the suggestion (No), use the user's most recent input as the final response.

        Context:
        {conversation_history}
        User Validation:
        {user_input}

        Your output must strictly adhere to this format:
        {{
            "category": "follow_up",
            "explanation": "Provide a clear explanation of the user's validation and the resulting action (Yes or No).",
            "response": "Provide the generated or confirmed response based on the user's validation."
        }}
        """
        try:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": follow_up_prompt.format(
                            conversation_history=conversation_history, user_input=user_input
                        ),
                    },
                    {
                        "role": "user",
                        "content": "Process the user's validation and generate the appropriate response.",
                    },
                ],
                temperature=0.7,
            )

            # Log raw API response for debugging
            raw_reply = completion.choices[0].message.content
            print("Raw API Reply:", raw_reply)

            # Attempt to parse the response
            try:
                bot_reply = json.loads(raw_reply)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing JSON response: {raw_reply}")

            # Validate and return response
            if bot_reply.get("category") == "follow_up":
                return ValidationResponse(
                    category="answered",
                    explanation=bot_reply["response"]
                )
            else:
                raise ValueError(f"Unexpected category in bot reply: {bot_reply}")

        except Exception as e:
            raise ValueError(f"Error generating follow-up response: {str(e)}")

    


    def handle_clarification(parsed_response):
        print("Handle the 'clarification' category.")
        """Handle the 'clarification' category."""
        clarification_prompt = """
        The user has asked a clarifying question about the previous question or context.
        Your task is to respond to their doubt as a knowledgeable hiring manager with a solid and well-structured explanation.
        Ensure your response provides helpful context and resolves the user's doubt effectively.

        Context:
        {conversation_history}
        User's Question:
        {user_input}

        Your output must strictly adhere to this format:
        {{
            "category": "clarification",
            "explanation": "Provide a clear and concise explanation addressing the user's clarifying question."
        }}
        """
        try:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": clarification_prompt.format(conversation_history=conversation_history, user_input=user_input)},
                    {"role": "user", "content": "Respond to the user's clarifying question and provide a detailed explanation."}
                ],
                temperature=0.7
            )
            
            # Debug raw API response
            raw_reply = completion.choices[0].message.content
            print("Raw API Reply:", raw_reply)

            # Parse the response
            bot_reply = json.loads(raw_reply)

            # Validate the category in the parsed response
            if bot_reply.get("category") == "clarification":
                return ValidationResponse(
                    category=bot_reply["category"],
                    explanation=bot_reply["explanation"]
                )
            else:
                raise ValueError("Unexpected category in bot reply.")

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error generating clarification response: {str(e)}")


    def handle_off_topic(parsed_response):
        print("Handle the 'off_topic' category.")
        off_topic_prompt = """
        The user has provided input that is unrelated to the current conversation or the bot's scope.
        Generate a polite and professional response to inform the user that their input is outside the bot's capabilities, while maintaining a helpful and friendly tone.
        Provide this response in the context of the following conversation history:
        {conversation_history}
        """
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": off_topic_prompt.format(conversation_history=conversation_history)},
                    {"role": "user", "content": "User Input: {user_input}"}
                ],
                temperature=0.7
            )
            bot_reply = completion.choices[0].message.content
            parsed_response["explanation"] = bot_reply
            return ValidationResponse(**parsed_response)
        except Exception as e:
            raise ValueError(f"Error generating off-topic response: {str(e)}")

    categoryhandaling_prompt = """
    You are an expert assistant that categorizes user responses into one of the following categories:
    - answered: The user has provided a definitive answer to the bot's question, addressing the bot's inquiry directly or indirectly.
    - clarification: The user asked a clarifying question about the bot's question and needs some more context on the question.
    - follow_up: The user is providing feedback on a suggestion given by the bot, explicitly accepting or rejecting it, look at the conversation and determine if the user is building upon or responding to prior feedback.
    - off_topic: The user input is unrelated to the bot's content or context.

    Rules:
    1. If the user is refining, clarifying, or revising their previous answer, categorize it as follow_up.
    2. If the user is directly answering the bot's question, categorize it as answered.
    3. Always use the conversation history to determine if the user is building upon or responding to prior feedback.

    Based on the user's response and the conversation history, assign a category and explain your reasoning.
    Your response must strictly adhere to this format:
    {
        "category": "answered" | "follow_up" | "clarification" | "off_topic",
        "explanation": "string"
    }
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": categoryhandaling_prompt},
                {"role": "user", "content": f"Conversation History: {conversation_history}\n\nUser Input: {user_input}"}
            ],
            temperature=0.5
        )

        print("==========",completion.choices[0].message)

        parsed_response = json.loads(completion.choices[0].message.content)
        category = parsed_response.get("category")

        if category == "answered":
            return handle_answered(parsed_response)
        elif category == "follow_up":
            return handle_follow_up(parsed_response)
        elif category == "clarification":
            print("Clarification categorry" )
            return handle_clarification(parsed_response)
        elif category == "off_topic":
            return handle_off_topic(parsed_response)
        else:
            raise ValueError("Unexpected category in parsed response.")

    except Exception as e:
        raise ValueError(f"Error during validation: {str(e)}")


def main():
    st.set_page_config(page_title="JD Enhancement Assistant", layout="wide")
    st.title("Job Description Assistant")
    st.markdown(
        "Transform your job descriptions into comprehensive and effective listings for Perfect Candidates! 🚀")

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

                # Prepare the conversation history as a string
                conversation_history_str = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.state["conversation_history"]]
                )

                # Validate response
                try:
                    validation_result = validate_response(
                        user_response,
                        conversation_history_str
                    )

                    print("Validation Result: ", validation_result)

                    if validation_result.category == Category.answered:
                        # Update collected data
                        if category not in st.session_state.state["collected_data"]:
                            st.session_state.state["collected_data"][category] = {}
                        st.session_state.state["collected_data"][category][param] = user_response  # Store user input

                        # Move to next parameter
                        st.session_state.state["current_param_index"] += 1
                    elif validation_result.category == Category.partially_answered:
                        st.session_state.state["conversation_history"].append(
                            {"role": "assistant", "content": validation_result.explanation}
                        )
                    else:
                        st.session_state.state["conversation_history"].append(
                            {"role": "assistant", "content": validation_result.explanation}
                        )
                except Exception as e:
                    print("Error in validation: ", e)
                    st.error(f"Error during validation: {str(e)}")
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
                    model="gpt-4o-mini",
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
