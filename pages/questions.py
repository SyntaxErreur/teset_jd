import streamlit as st
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


# FileManager Class: Handles JSON file operations
class FileManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_json(self):
        """Load JSON data from a file."""
        try:
            with open(self.file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            st.error(
                f"The file `{self.file_path}` was not found. Please make sure it exists.")
            return None
        except json.JSONDecodeError:
            st.error(
                f"Error decoding `{self.file_path}`. Ensure the file contains valid JSON.")
            return None

    def save_json(self, data):
        """Save JSON data to a file."""
        try:
            with open(self.file_path, 'w') as file:
                json.dump(data, file, indent=4)
            st.success(f"Data successfully saved to `{self.file_path}`.")
        except Exception as e:
            st.error(f"Error saving data to `{self.file_path}`: {e}")

# ConversationManager Class: Handles conversation flow and OpenAI integration


class ConversationManager:
    def __init__(self, questions):
        self.questions = questions

    def generate_question_text(self):
        """Generate a formatted string containing all the questions."""
        return "\n\n".join([f"**{i+1}.** {q}" for i, q in enumerate(self.questions)])

    def handle_response(self, user_input):
        """
        Process the user's input dynamically with GPT to detect intent and handle requests.
        """
        print("Received user input:", user_input)

        # Use GPT to classify the intent and decide the next action
        intent_prompt = f"""
        You are an intelligent assistant for job interview preparation. Analyze the user's input to determine their intent.

        - If the user wants to change a specific question, identify the question numbers and return an action to replace it.
        - If the user wants to change all questions, specify that the action is to replace all questions.
        - If the user wants additional questions, specify that the action is to provide more questions without replacing the existing ones.
        - If the user wants answers to all the questions, specify that the action is to provide answers to all the questions.
        - If the user is providing feedback or asking for clarification, suggest the appropriate response.
        - If the input is unclear or unrelated, provide a polite request for clarification.

        User Input: "{user_input}"

        Respond in this JSON format:
        {{
            "action": "change_one" | "change_all" | "add_more" | "answer_all" | "clarify" | "other",
            "details": {{
                "question_number": <number>,  # Required if action is 'change_one'
                "explanation": "<Optional: explanation or clarification for the user>"
            }}
        }}
        """

        try:
            # Call OpenAI to determine intent
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": intent_prompt}
                ],
                temperature=0.7
            )
            parsed_response = json.loads(response.choices[0].message.content)
            print("GPT Intent Analysis:", parsed_response)

            # Extract action and details
            action = parsed_response.get("action")
            details = parsed_response.get("details", {})
            explanation = details.get("explanation", "")

            # Handle actions based on GPT's intent classification
            if action == "change_one":
                question_number = details.get("question_number")
                if question_number and 1 <= question_number <= len(self.questions):
                    old_question = self.questions[question_number - 1]
                    new_question = self.regenerate_questions(1)[0]
                    self.questions[question_number - 1] = new_question

                    # Synchronize with session state
                    st.session_state.questions = self.questions
                    print(
                        f"Replaced Question {question_number}: {old_question} -> {new_question}")
                    return (
                        f"Question {question_number} has been replaced:\n\n{new_question}\n\n"
                        "Thank you for your responses! If you have more to add, feel free to do so."
                    )
                else:
                    return f"Invalid question number. Please choose a number between 1 and {len(self.questions)}."

            elif action == "change_all":
                new_questions = self.regenerate_questions()
                # self.questions = new_questions

                # Synchronize with session state
                st.session_state.questions = new_questions
                print("All questions replaced:", new_questions)
                return ("All questions have been replaced. Here are the new questions:\n\n" + self.generate_question_text() +
                        "\n\nThank you for your responses! If you have more to add, feel free to do so.")
            
            elif action == "add_more":
                # Generate new questions without losing the previous ones
                new_questions = self.regenerate_questions()

                # Append new questions to the existing list
                # self.questions.extend(new_questions) - works with the previous questions not updated
                
                # Synchronize with session state
                st.session_state.questions.extend(new_questions)# Update the state with the new list
                print("Updated Questions (after adding more):", json.dumps(st.session_state.questions, indent=2))

                # Create a response with all questions (old + new)
                response_text = (
                    "Here are your updated questions (including new ones):\n\n"
                    + "\n\n".join([f"**{i+1}.** {q}" for i, q in enumerate(st.session_state.questions)])
                )
                return response_text

            elif action == "answer_all":
                # Use GPT to answer all the questions
                try:
                    answers_prompt = f"""
                    You are an expert assistant for job interview preparation. Answer the following questions clearly and professionally:

                    Questions:
                    {json.dumps(st.session_state.questions, indent=2)}

                    Provide answers in the following format:
                    Question 1: [Question here]
                    Answer 1: [Answer here]

                    Question 2: [Question here]
                    Answer 2: [Answer here]

                    Continue for all questions.
                    """
                    # Call GPT to generate answers
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": answers_prompt}
                        ],
                        temperature=0.7
                    )
                    answers_text = response.choices[0].message.content

                    # Return the generated answers
                    return f"Here are the answers to your questions:\n\n{answers_text}"

                except Exception as e:
                    print("Error during answering questions:", e)
                    return "Sorry, there was an error generating answers to the questions. Please try again."


            elif action == "clarify":
                return explanation or "Could you please clarify your request?"

            else:
                return "I'm sorry, I couldn't understand your request. Could you please rephrase?"
        except json.JSONDecodeError as json_err:
            print("JSON decoding error:", json_err)
            return "There was an error understanding the response. Please try again."
        except Exception as e:
            print("Error during intent detection or question handling:", e)
            return "Sorry, there was an error processing your request. Please try again."


    def regenerate_questions(self, num_questions=10):
        try:
            # Fetch the role title dynamically
            role_title = st.session_state.get("role_title", "Unknown Role")

            # Use the updated questions from session state
            updated_questions =  st.session_state.questions

            system_prompt = f"""
            Generate {num_questions} professional interview questions tailored to the role: {role_title}.
            Each question must be clear, specific, and directly relevant to the role. Use professional and formal phrasing.

            Instructions:
            1. Analyze the Role Context:
            - Role Title: {role_title}
            - Review the existing questions provided below to avoid duplication or overlap.

            2. Existing Questions:
            {json.dumps(updated_questions, indent=2)}

            3. Generate {num_questions} Interview Questions:
            - Questions should cover:
                - Technical skills: Assess expertise in the tools, technologies, and methodologies listed.
                - Job responsibilities: Test the ability to handle similar responsibilities as described in the role.
                - Soft skills: Include questions to evaluate communication, collaboration, and leadership abilities.
            - Avoid placeholders or generic text like '[insert relevant technical skill]'.

            4. Generate additional professional interview questions for the role. These questions should complement the existing ones and provide more insight into the candidate's abilities. 
            If generating additional questions, start the response with "Here are more new questions. Let me know if this helps."

            Your output must follow this exact format:
            Question 1: [Your question here]
            Question 2: [Your question here]
            ...
            """
            print("Updated questions being passed to GPT:")
            print(json.dumps(updated_questions, indent=2))

            # Call OpenAI to generate questions
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert interview question generator."},
                    {"role": "user", "content": system_prompt}
                ],
                temperature=0.7
            )
            questions_text = response.choices[0].message.content
            print("Generated Questions:", questions_text)

            # Parse the questions from the response
            new_questions = [
                line.split(": ", 1)[1] for line in questions_text.split("\n") if line.strip().startswith("Question")
            ]
            return new_questions[:num_questions]
        except Exception as e:
            print("Error during question regeneration:", e)
            return ["Could not generate new questions. Please try again."] * num_questions



# Main Function: Handles UI components


def main():
    st.title("💬 Job Interview Chat")

    # FileManager to load the questions
    file_manager = FileManager('questions.json')
    data = file_manager.load_json()
    if not data:
        return

    all_questions = data.get("all_questions", [])
    if not all_questions:
        st.warning("No questions found in the data.")
        return

    recent_question_set = max(
        all_questions, key=lambda x: datetime.fromisoformat(x["timestamp"]))

    # Display role title
    st.header(f"Role: {recent_question_set['role_title']}")

    # ConversationManager to handle questions and responses
    conversation_manager = ConversationManager(
        recent_question_set["questions"])


    # Initialize session state for questions
    if "questions" not in st.session_state:
        st.session_state.questions = recent_question_set["questions"]

    # Initialize session state for chat
    if "conversation" not in st.session_state:
        questions_text = conversation_manager.generate_question_text()
        st.session_state.conversation = [
            {"role": "assistant", "content": questions_text}
        ]
        st.session_state.responses = []

    # Display chat messages
    st.subheader("Chat")
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"])
            else:
                st.write(message["content"])

    # Always show the input box for continuous conversation
    user_input = st.chat_input("Your response...")  # Always display input box
    if user_input:
        # Handle user response
        response_text = conversation_manager.handle_response(user_input)

        if response_text:
            print("All the quesitons are --------------" , json.dumps(st.session_state.questions, indent=2) )
        
        # Update session state with user input and assistant response
        st.session_state.responses.append(user_input)
        st.session_state.conversation.append(
            {"role": "user", "content": user_input})
        st.session_state.conversation.append(
            {"role": "assistant", "content": response_text})

        # Rerun the app to display the updated conversation
        st.rerun()


if __name__ == "__main__":
    main()
