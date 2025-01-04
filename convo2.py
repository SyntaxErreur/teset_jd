import os
import openai
import streamlit as st
from openai import OpenAI
import json
import re
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize OpenAI client with API key from .env file
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

# Define the required fields structure
REQUIRED_FIELDS = {
    "Customer Details": ["Customer", "Hiring Manager"],
    "Position Details": [
        "Engagement Type", "Position Status", "Date of Qualification",
        "Position Open Date", "Number of Positions", "Account Manager/Lead",
        "Target Fulfillment Date", "Grade/Bands/Levels/Competency",
        "Education", "CTC Range", "Type of Position"
    ],
    "Technical Skills": ["Mandatory in Years", "Good to Have"],
    "Business Unit": [
        "Type of work they do", "Team Size", "Organization Structure",
        "In house/Customer project", "Composition of people in the current team"
    ],
    "Roles & Responsibilities": [
        "Cultural Expectations", "Key Roles", "Key Responsibilities",
        "Current roles", "Reports to", "Non-technical skills",
        "Screening Questions", "Consultant Note Questions"
    ],
    "Other Details": [
        "Preferred Location", "Target Date of Offer", "Date of requalification",
        "Interview Process", "Panel Availability", "Office Location",
        "Work Timings", "Notice Period", "Travel Needs", "Acceptable Job Changes",
        "Customer Value Proposition"
    ]
}


def create_analysis_prompt(jd_text):
    """Create a detailed prompt for the OpenAI API"""
    prompt = f"""You are an expert Job Description Analyzer AI. Your task is to thoroughly analyze the provided job description \
and extract all relevant information according to specific required fields. Then identify what information is missing.

Here is the job description to analyze:
{jd_text}

Please analyze this job description for the following categories and their specific fields:

1. Customer Details:
   - Identify the Customer name
   - Find the Hiring Manager information

2. Position Details:
   - Determine the Engagement Type (Full-time, Contract, etc.)
   - Position Status (New, Replacement, etc.)
   - Date of Qualification
   - Position Open Date
   - Number of Positions
   - Account Manager/Lead
   - Target Fulfillment Date
   - Grade/Bands/Levels/Competency
   - Required Education
   - CTC Range
   - Type of Position

3. Technical Skills:
   - List all Mandatory Skills with required years of experience
   - List all Good to Have skills

4. Business Unit Details:
   - Type of work they do
   - Team Size
   - Organization Structure
   - Whether it's In-house or Customer project
   - Current team composition

5. Roles & Responsibilities:
   - Cultural Expectations
   - Key Roles the person needs to perform
   - Key Responsibilities
   - Required Previous Experience/Roles
   - Reporting Structure
   - Non-technical skills required
   - Important Screening Questions
   - Required Consultant Note Questions

6. Other Details:
   - Preferred candidate location
   - Target Date of Offer
   - Date of requalification
   - Interview Process details
   - Panel Availability
   - Office Location
   - Work Timings
   - Acceptable Notice Period
   - Travel Requirements
   - Acceptable Job Changes
   - Customer Value Proposition

Please provide output in this format:
{{\\"found_fields\\": {{\\"category_name\\": {{\\"field_name\\": \\"extracted_value\\"}}}},
\\"missing_fields\\": {{\\"category_name\\": [\\"field1\\", \\"field2\\"]}}}}\n\n
For found_fields, include only the fields that are explicitly mentioned or can be clearly inferred from the job description.
For missing_fields, list all fields that are not mentioned or cannot be clearly inferred from the job description.
Be thorough and accurate in your analysis."""

    return prompt


def get_interactive_prompt(missing_fields):
    """Generate an interactive prompt based on missing fields"""
    prompt = "I noticed some important details are missing from the job description. Let me help you fill them in:\n\n"

    for category, fields in missing_fields.items():
        if fields:
            prompt += f"In the {category} category, we're missing:\n"
            for field in fields:
                prompt += f"- {field}\n"

    prompt += "\nWould you like to provide more details about these missing fields? I can guide you through each one to make your job description more comprehensive."

    return prompt


def main():
    # Set up the Streamlit page
    st.title("🤖 Job Description Analyzer Bot")

    # Initialize session states
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        # Initialize messages with a welcome message
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I'm your Job Description Analyzer. Paste your job description, and I'll help you improve it by identifying missing details."}
        ]

    # Track analysis state
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False

    if "missing_fields" not in st.session_state:
        st.session_state["missing_fields"] = {}

    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user prompt
    if prompt := st.chat_input("Enter your job description or ask a question"):
        # Append user input to session messages
        st.session_state["messages"].append(
            {"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot's response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # If analysis is not complete, treat input as job description
            if not st.session_state["analysis_complete"]:
                # Create analysis prompt
                analysis_prompt = create_analysis_prompt(prompt)

                # Call OpenAI API for analysis
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    model=st.session_state["openai_model"],
                )

                # Parse the analysis
                try:
                    analysis = json.loads(
                        chat_completion.choices[0].message.content)

                    # Show found fields
                    found_response = "Here's what I found in your job description:\n\n"
                    for category, fields in analysis["found_fields"].items():
                        if fields:
                            found_response += f"{category}:\n"
                            for field, value in fields.items():
                                found_response += f"- {field}: {value}\n"

                    # Store missing fields
                    st.session_state["missing_fields"] = analysis["missing_fields"]
                    st.session_state["analysis_complete"] = True

                    # Generate interactive prompt for missing fields
                    full_response = found_response + "\n" + \
                        get_interactive_prompt(analysis["missing_fields"])

                except Exception as e:
                    full_response = f"Sorry, I had trouble analyzing the job description. Error: {str(e)}"

            # If analysis is complete, handle conversational flow
            else:
                # Call OpenAI to generate conversational response
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant guiding a user through filling out a job description. Be friendly and conversational."},
                        *st.session_state["messages"],
                        {"role": "user", "content": prompt}
                    ],
                    model=st.session_state["openai_model"],
                )
                full_response = chat_completion.choices[0].message.content

            # Stream the response
            message_placeholder.markdown(full_response)

        # Append assistant's response to session messages
        st.session_state["messages"].append(
            {"role": "assistant", "content": full_response}
        )


if __name__ == "__main__":
    main()
