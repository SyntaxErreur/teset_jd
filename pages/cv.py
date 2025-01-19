from typing import List, Dict
from typing import Dict, Tuple
import os
import json
import streamlit as st
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
import logging
from fpdf import FPDF
import markdown2
import datetime

st.set_page_config(page_title="JD Enhancement Assistant", layout="wide")

# # Configure logging
# logging.basicConfig(level=printf, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


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
        """Analyze job description using GPT to extract structured information."""
        system_prompt = """You are an expert job description analyzer. Extract structured information from the provided job description.
        Return only a JSON object with the extracted information organized according to the given categories and parameters.
        If information is not found, include the parameter with an empty list."""

        analysis_prompt = f"""Analyze this job description and extract information for these categories:
        {json.dumps(self.parameter_structure, indent=2)}

        Job Description:
        {jd_text}

        Return a JSON object where each category is a dictionary containing parameters as keys with list values."""

        try:
            # Make API request to OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.5
            )

            # Log raw response for debugging
            raw_response = response.choices[0].message.content
            print("Raw response:", raw_response)

            # Preprocess response to extract JSON (handle triple backticks and extra formatting)
            if raw_response.startswith("```json"):
                raw_response = raw_response.strip(
                    "```json").strip("```").strip()

            # Parse the JSON content
            analyzed_data = json.loads(raw_response)

            # Ensure all required categories and parameters are present
            for category, params in self.parameter_structure.items():
                if category not in analyzed_data:
                    analyzed_data[category] = {}
                for param in params:
                    if param not in analyzed_data[category]:
                        analyzed_data[category][param] = []

            return analyzed_data

        except json.JSONDecodeError as e:
            st.error("Error parsing the analysis result. Please try again.")
            print(f"JSON Decode Error: {e}")
            print("Response content:", raw_response)
            return {category: {param: [] for param in params} for category, params in self.parameter_structure.items()}
        except Exception as e:
            st.error(
                "Error analyzing job description. Please check the job description and try again.")
            print(f"Unexpected Error: {e}")
            return {category: {param: [] for param in params} for category, params in self.parameter_structure.items()}


class FileManager:
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        if not isinstance(filepath, (str, os.PathLike)):
            raise TypeError(
                f"Expected filepath to be a string or os.PathLike, got {type(filepath).__name__} instead.")
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                return json.load(file)
        return {"Position Details": {}, "Technical Requirements": {}, "Job Responsibilities": {}, "Soft Skills": {}, "Work Details": {}, "Additional Information": {}}

    @staticmethod
    def save_json(filepath: str, data: Dict):
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def save_questions(file_path: str, questions_data: dict):
        """Save questions to JSON file with logging and error handling."""
        try:
            # Load existing data if file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    existing_data = json.load(file)

                # Ensure the structure is valid
                if not isinstance(existing_data, dict) or "all_questions" not in existing_data:
                    existing_data = {"all_questions": []}
            else:
                # Initialize new structure if file doesn't exist
                existing_data = {"all_questions": []}

            # Append the new questions data
            existing_data["all_questions"].append(questions_data)

            # Save the updated data
            with open(file_path, 'w') as file:
                json.dump(existing_data, file, indent=4)

            print(f"Questions saved successfully to {file_path}")

        except Exception as e:
            print(f"Error saving questions to {file_path}: {e}")


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
    # User partially answered the bot's question
    partially_answered = "partially_answered"


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
        The user has provided an answer to the question. Your task as a Job Description Enhancer or Builder is to:
            1. Evaluate whether the user's response effectively addresses the query, considering the following:
                - Is the response relevant to the question in the context of enhancing a job description?
                - Does it address the key intent of the query, such as providing details, examples, or specifications that can improve the job description?
                - Does it require minor refinements (e.g., spelling corrections, phrasing adjustments) to meet professional expectations, while still being a valid response?
            
            2. You are an expert assistant that categorizes user responses into one of the following categories:
                - **answered**: The user's response directly or sufficiently addresses the question. Examples include:
                    - Providing relevant context or specific details related to the job description.
                    - Listing key points or valid examples that align with the question's intent, even if the response is concise.

                - **partially_answered**: The user's response is vague, incomplete, or misaligned with the question. Examples include:
                    - The response addresses only one part of the question or lacks clarity.
                    - It provides irrelevant or unnecessary information that does not improve the job description.

        **Important Note**: Your evaluation and feedback must always focus on enhancing the job description, not on assessing a candidate's personal availability, status, or experience. Avoid referencing individual circumstances or personal context. 

        If the response is vague or poorly structured, provide constructive feedback with specific suggestions for improvement. Also, give a clear and concise example for the user to understand better.

        Ensure the tone is polite, formal, and professional, avoiding harsh or dismissive language. Do not restate the user's input in the feedback.

        Context:
        {conversation_history}
        User's Answer:
        {user_input}

        -----------------------
        Your output must strictly adhere to this format for answered:
        {{
            "category": "answered",
            "explanation": "refined_output - refine it into a concise and professional phrase suitable for direct use in the job description. NO EXPLANATION TO BE PROVIDED JUST the REFINED OUTPUT. Example
            - User Input: "5 years"
            - Refined Output: "5 years of experience in `whatever field the discussion is on` "        
        }}

        Your output must strictly adhere to this format for partially_answered:
        {{
            "category": "partially_answered",
            "explanation": "Make sure you are addresing the user in the first perspective since this will be used as a response to the user. Provide detailed reasoning here, explaining why the response is partially valid, exclude any spelling error and include actionable feedback with a valid Example if necessary end with asking would you like to revise your response.Keep the tone polite and professional."
        }}
        """
        #     3. Handle the response as follows:
        # - If the response is valid:
        #     - Accept it and proceed to ask the next question immediately, without acknowledgment.
        #     - If the response requires minor corrections (e.g., spelling, shortness, phrasing adjustments), store a refined version in the backend silently for professional alignment.

        # - If the response is partially valid or vague:
        #     - Provide constructive feedback explaining why the response is partially correct or misaligned.
        #     - Offer actionable suggestions to improve the response, including a c lear example.
        #     - Conclude by asking politely if the user would like to refine their response.
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": answered_prompt.format(
                        conversation_history=conversation_history, user_input=user_input)},
                    {"role": "user", "content": "Validate and provide feedback on the above response."}
                ],
                temperature=0.7
            )
            bot_reply = json.loads(completion.choices[0].message.content)

            print(" ============== Bot reply =========")
            print(bot_reply)
            print(" ==========================")

            if bot_reply.get("category") in ["answered", "partially_answered"]:
                return ValidationResponse(
                    category=bot_reply["category"],
                    explanation=bot_reply["explanation"]
                )
            else:
                raise ValueError("Unexpected category in bot reply.")

        except Exception as e:
            raise ValueError(
                f"Error generating answered response validation: {str(e)}")

    def handle_follow_up(parsed_response):
        """Handle the 'follow_up' category."""
        print("Handle the 'follow_up' category.")
        follow_up_prompt = """
        The user has provided feedback on the bot's previous suggestion. Depending on the user's validation:
        - If the user accepts the suggestion (Yes), generate a new response based on the most recent bot suggestion from the conversation history which will be inserted in the Job description directly.
        - If the user rejects the suggestion (No), use the user's most recent input as the final response.

        Context:
        {conversation_history}
        User Validation:
        {user_input}

        Your output must strictly adhere to this format:
        {{
            "category": "follow_up",
            "explanation": "Provide a clear explanation of the user's validation and the resulting action (Yes or No).",
            "response": "Provide the generated or confirmed response based on the user's validation which will be used directly in the Job description."
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
                raise ValueError(
                    f"Unexpected category in bot reply: {bot_reply}")

        except Exception as e:
            raise ValueError(f"Error generating follow-up response: {str(e)}")

    def handle_clarification(parsed_response):
        print("Handle the 'clarification' category.")
        """Handle the 'clarification' category."""
        clarification_prompt = """
        The user has asked a clarifying question about the previous question or context.
        Your task is to respond to their doubt as a knowledgeable hiring manager with a solid and well-structured explanation.
        Ensure your response provides helpful context and resolves the user's doubt effectively also give good example and ask the question again.

        Context:
        {conversation_history}
        User's Question:
        {user_input}

        Your output must strictly adhere to this format:
        {{
            "category": "clarification",
            "explanation": "Provide a clear and concise explanation addressing the user's clarifying question. The Formatting should be clear and easy to understand with spaces and markdown output." 
        }}
        """
        try:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": clarification_prompt.format(
                        conversation_history=conversation_history, user_input=user_input)},
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
            raise ValueError(
                f"Error generating clarification response: {str(e)}")

    def handle_off_topic(parsed_response):
        print("Handle the 'off_topic' category.")
        off_topic_prompt = """
        The user has provided input that is unrelated to the current conversation or the bot's scope. 
        Your task is to:
            1. Classify the sentiment of the user's response as:
            - Respectful
            - Neutral
            - Harmful
            2. Generate a response based on the sentiment and ensure the starting phrase is appropriate
            - If the response is respectful or neutral:
                - Start with one of the following:
                    - "I can only answer questions related to the current discussion."
                    - "Let’s keep the conversation focused on the current discussion."
                - Follow it with a polite re-asking of the original question or clarification request.
            - If the sentiment is harmful:
                - Start with one of the following:
                    - "Let’s keep the conversation respectful and focused on the current discussion."
                    - "I cannot engage in harmful or inappropriate discussions."
                - Politely re-ask the original question to redirect the user back to the discussion topic.
            3. Remove any unnecessary lines or repetitions in the response. For example:
            - Avoid repeating "Let’s keep the conversation focused on the current discussion." after rephrasing the question.

        Generate a concise, polite, and professional response that:
        1. Acknowledges the user's input.
        2. Politely informs the user that their response is unrelated to the current topic.
        3. Redirects the user to focus on the question at hand by re-asking it in a clear manner.

        Ensure the tone is polite, formal, and non-harsh while keeping the response brief.Avoid using generic phrases like "Thank you for your input" at the start.
        Provide this response in the context of the following conversation history:
        {conversation_history}
        """
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": off_topic_prompt.format(
                        conversation_history=conversation_history)},
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
    - answered:The user has provided a definitive answer to the bot's question, addressing the bot's inquiry directly or indirectly. Examples include:
            - The response addresses the main intent of the question and provides sufficient detail.
            - The response may lack minor details but is valid and acceptable as a standalone answer.
            - If the response indirectly addresses the question but is relevant and logical, categorize it as answered.
        Rules for answered:
            - If the response satisfies the question's main intent, even if indirectly or with minor omissions, categorize it as answered.
    - clarification: The user asked a clarifying question about the bot's question or needs some more context on the question for example user ask : Give me an example.
    - follow_up: The user is providing feedback on a suggestion given by the bot, explicitly accepting or rejecting it, look at the conversation and determine if the user is building upon or responding to prior feedback.
    - off_topic: The user input is unrelated to the bot's content or context. Examples include responses that:
                    - Do not address or build upon the current question or discussion.
                    - Shift the topic away from the question posed by the bot.
                    - Do not contain any relevant content related to the ongoing conversation.

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

        parsed_response = json.loads(completion.choices[0].message.content)
        category = parsed_response.get("category")

        if category == "answered":
            return handle_answered(parsed_response)
        elif category == "follow_up":
            return handle_follow_up(parsed_response)
        elif category == "clarification":
            print("Clarification categorry")
            return handle_clarification(parsed_response)
        elif category == "off_topic":
            return handle_off_topic(parsed_response)
        else:
            raise ValueError("Unexpected category in parsed response.")

    except Exception as e:
        raise ValueError(f"Error during validation: {str(e)}")


def main():
    st.title("Job Description Assistant")
    st.markdown(
        "Transform your job descriptions into comprehensive and effective listings for Perfect Candidates! 🚀")

    # File path for JSON storage
    global db_path
    db_path = "db.json"

    # Load or initialize JSON data
    try:
        db_data = FileManager.load_json(db_path)
        # Ensure the file is not empty
        if not db_data or all(not params for params in db_data.values()):
            db_data = {
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
            FileManager.save_json(db_path, db_data)
    except Exception as e:
        st.error("Failed to load or initialize the database.")
        print(f"Error initializing db.json: {e}")
        return

    # Initialize session state
    if "state" not in st.session_state:
        st.session_state.state = {
            "jd_analyzed": False,
            "current_param_index": 0,
            "conversation_history": [],
            "missing_params": []
        }

    # Initialize components
    analyzer = JobDescriptionAnalyzer()
    question_gen = QuestionGenerator()

    # Job Description Input and Analysis
    if not st.session_state.state["jd_analyzed"]:
        jd_text = st.text_area("📝 Enter the Job Description", height=200)
        if st.button("Analyze Job Description"):
            if jd_text.strip():
                with st.spinner("Analyzing job description..."):
                    try:
                        # Analyze JD
                        analyzed_data = analyzer.analyze_job_description(
                            jd_text)
                        if analyzed_data:
                            db_data.update(analyzed_data)
                            FileManager.save_json(db_path, db_data)

                            # Identify missing parameters
                            missing_params = [
                                (category, param)
                                for category, params in db_data.items()
                                for param, values in params.items()
                                if not values
                            ]
                            st.session_state.state["missing_params"] = missing_params
                            st.session_state.state["jd_analyzed"] = True

                            # Log missing parameters
                            print("Missing Parameters Identified:")
                            for category, param in missing_params:
                                print(f"- {category}: {param}")

                            # Display missing parameters
                            if missing_params:
                                st.markdown(
                                    "### Missing Parameters Identified:")
                                for category, param in missing_params:
                                    st.markdown(f"- **{category}**: {param}")
                            else:
                                st.success(
                                    "No missing parameters! Ready to generate the JD.")

                            st.rerun()
                        else:
                            st.error(
                                "Analysis failed. Please check the job description and try again.")
                    except Exception as e:
                        st.error(
                            "Error analyzing job description. Please try again.")
                        logging.error(f"Error during JD analysis: {e}")
            else:
                st.warning("Please enter a valid job description.")

    # Handle Missing Parameters
    if st.session_state.state["jd_analyzed"]:
        st.subheader("🗨️ Conversation History")
        for message in st.session_state.state["conversation_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        if st.session_state.state["missing_params"]:
            category, param = st.session_state.state["missing_params"][
                st.session_state.state["current_param_index"]]
            role_context = db_data["Position Details"].get("role_title", [])
            role_context = role_context[0] if role_context else "Unknown Role"

            # Generate question and display if not already asked
            if not st.session_state.state["conversation_history"] or \
               st.session_state.state["conversation_history"][-1]["role"] == "user":
                question = question_gen.generate_contextual_question(
                    param, role_context, category, db_data)
                st.session_state.state["conversation_history"].append(
                    {"role": "assistant", "content": question}
                )
                st.rerun()

            # User Input Handling
            user_response = st.chat_input("Your response...")
            if user_response:
                st.session_state.state["conversation_history"].append(
                    {"role": "user", "content": user_response}
                )
                conversation_history_str = "\n".join(
                    [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.state["conversation_history"]]
                )
                try:
                    validation_result = validate_response(
                        user_response, conversation_history_str)

                    if validation_result.category == Category.answered:
                        # Update database with bot explanation and save to file
                        db_data[category][param] = [
                            validation_result.explanation]
                        FileManager.save_json(db_path, db_data)
                        st.session_state.state["current_param_index"] += 1
                        if st.session_state.state["current_param_index"] >= len(st.session_state.state["missing_params"]):
                            st.session_state.state["missing_params"] = [
                                (category, param)
                                for category, params in db_data.items()
                                for param, values in params.items()
                                if not values
                            ]
                            st.session_state.state["current_param_index"] = 0
                        st.success(f"Response for {param} saved successfully!")
                    elif validation_result.category == Category.partially_answered:
                        st.session_state.state["conversation_history"].append(
                            {"role": "assistant",
                                "content": validation_result.explanation}
                        )
                    else:
                        st.session_state.state["conversation_history"].append(
                            {"role": "assistant",
                                "content": validation_result.explanation}
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during validation: {str(e)}")
                    print(f"Validation error: {e}")

        else:
            st.success("All parameters have been answered.")
            if st.button("Generate Final Job Description"):
                generate_final_jd(db_data)


def generate_final_jd(db_data):
    """Generate the final job description."""
    with st.spinner("Generating Job Description..."):
        system_prompt = """You are an expert job description writer.
        Create a professional, well-formatted job description using the provided structured data."""

        final_jd_prompt = f"""Create a comprehensive job description using this data:
        {json.dumps(db_data, indent=2)}

        Requirements:
        - Professional and engaging tone
        - Clear structure with appropriate headings
        - Highlight key requirements and responsibilities
        - Include all provided information
        - Use markdown formatting"""

        try:
            # Generate Job Description
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_jd_prompt}
                ],
                temperature=0.7
            )
            final_jd = response.choices[0].message.content

            # Display Job Description
            st.markdown("## 📋 Enhanced Job Description")
            st.markdown(final_jd, unsafe_allow_html=True)

            # Download Button
            st.download_button(
                label="Download Enhanced JD",
                data=final_jd,
                file_name="enhanced_job_description.md",
                mime="text/markdown"
            )

            # Add a separator
            st.markdown("---")

            # Generate Initial Questions
            st.markdown("## 🎯 Interview Questions")

            with st.spinner("Generating interview questions..."):
                # Initialize questions in session state if not present
                if 'questions' not in st.session_state:
                    st.session_state.questions = []
                    initial_questions = generate_questions(db_path, [])
                    if initial_questions:
                        st.session_state.questions.extend(initial_questions)
                        # Store questions in question.json

                # Display Questions
                if st.session_state.questions:
                    for i, question in enumerate(st.session_state.questions, 1):
                        st.markdown(f"**Q{i}.** {question}")
                        # st.rerun()

                # col1 = st.columns([1, 4])
                # with col1:
                #     if st.button("Generate More Questions"):
                #         with st.spinner("Generating additional questions..."):
                #             new_questions = generate_questions(
                #                 db_path, st.session_state.questions)
                #             st.session_state.questions.extend(
                #                 new_questions)


        except Exception as e:
            st.error(f"Error generating final job description: {str(e)}")
            print(f"JD generation error: {e}")


def generate_questions(db_path: str, asked_questions: List[str] = []) -> List[str]:
    """Generate role-specific interview questions."""
    try:
        db_data = FileManager.load_json(db_path)
        system_prompt = f"""
        You are a professional recruiter and industry-level interview expert tasked with creating role-specific interview questions tailored to a fully enhanced job description. Your goal is to design questions that assess candidates effectively for the specified role.

        Instructions:
    You are a professional recruiter and industry-level interview expert tasked with creating role-specific interview questions tailored to a fully enhanced job description. Your goal is to design questions that assess candidates effectively for the specified role. 

        1. Analyze the Role Context:
        - Review the `role_title`, `grade_level`, and `experience_required` to tailor the difficulty and scope of the questions.

        2. Evaluate the Fully Enhanced Job Description:
        - Use the following key sections from the job description:
            - Technical Requirements: `mandatory_skills`, `technical_skills`, `good_to_have_skills`
            - Job Responsibilities: `primary_responsibilities`, `team_responsibilities`, `project_responsibilities`
            - Soft Skills: `communication_skills`, `interpersonal_skills`, `other_competencies`

        3.Generate 10 Professional Questions:
        - Frame highly professional and formal interview questions focusing on the following areas:
            - Technical Skills: Evaluate the candidate’s expertise in the tools, technologies, and methodologies listed in the job description.
            - Job Responsibilities: Assess the candidate's ability to handle responsibilities similar to those described in the role.
            - Soft Skills: Include questions to evaluate communication, problem-solving, leadership, and collaboration abilities.

        4.Formatting:
        - Each question must be specific, clear, and directly relevant to the job description.
        - Use industry-standard phrasing suitable for formal interviews.

        5.Output:
        - Generate a total of 10 questions.
        - End the output with this message:
            _"Would you like more questions tailored to this role? Let me know, and I can provide additional questions as needed."

        Context:
        This task is based on the following fully enhanced job description data:
        {json.dumps(db_data, indent=2)}

        Important Notes:
        - Only use the data provided in the `enhanced_job_description` to frame questions. Do not introduce topics outside the listed skills, responsibilities, and soft skills.
        - Ensure the tone is professional and reflective of a seasoned hiring manager.

        Your output must strictly adhere to the following format:
        1. Question 1: [Your question here]
        2. Question 2: [Your question here]
        ...
        10. Question 10: [Your question here]

        End with this message:
        "Would you like more questions tailored to this role? Let me know, and I can provide additional questions as needed."
        """
        print("DB Data being passed to OpenAI API:")
        print(json.dumps(db_data, indent=2))

        print("Asked Questions:")
        print(asked_questions)

        # Generate questions using GPT-4o-mini

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate 10 interview questions based on the enhanced job description."}
            ],
            temperature=0.7
        )
        questions_text = response.choices[0].message.content

        # Parse the questions into a list
        questions = [
            q.split(". ", 1)[1] if ". " in q else q
            for q in questions_text.split("\n")
            if q.strip() and any(q.lower().startswith(str(i)) for i in range(1, 11))
        ]

        # Prepare metadata for JSON
        questions_data = {
            "questions": questions[:10],  # Ensure exactly 10 questions
            "timestamp": datetime.datetime.now().isoformat(),
            "role_title": db_data.get("Position Details", {}).get("role_title", ["Unknown Role"])[0]
        }

         # Log the prepared data
        print(f"Prepared Questions Data: {json.dumps(questions_data, indent=4)}")

        # Save questions to JSON file
        try:
            FileManager.save_questions("questions.json", questions_data)
        except Exception as save_error:
            print(f"Error while saving questions: {save_error}")

        print( "questionss saved as ---", questions_data)


        return questions[:10]  # Return exactly 10 questions for display

    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []


if __name__ == "__main__":
    main()
