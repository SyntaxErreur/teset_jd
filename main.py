import os
import re
import json
import spacy
from openai import OpenAI
import streamlit as st
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
load_dotenv()


def main():
    st.title("🎯 Job Description Parameter Extractor")

    job_description = st.text_area("Paste Job Description", height=300)


if __name__ == "__main__":
    main()
