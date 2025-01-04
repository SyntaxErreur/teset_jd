import streamlit as st
import json

def format_category(title: str):
    """Format category title for display."""
    return f"### {title}"

def format_subcategory(title: str):
    """Format subcategory title for display."""
    return f"#### {title}"

def display_data(data: dict):
    """Display job description data in a presentable format."""
    for category, subcategories in data.items():
        st.markdown(format_category(category))
        for subcategory, details in subcategories.items():
            st.markdown(format_subcategory(subcategory.replace('_', ' ').capitalize()))
            if details:
                for item in details:
                    st.markdown(f"- {item}")
            else:
                st.markdown("*No information provided.*")
        st.markdown("---")

def main():
    st.title("📋 Job Description Viewer")

    # Load JSON data
    try:
        with open('db.json', 'r') as file:
            data = json.load(file)
        st.success("Job description data loaded successfully!")
    except FileNotFoundError:
        st.error("The file `db.json` was not found. Please make sure it exists.")
        return
    except json.JSONDecodeError:
        st.error("Error decoding `db.json`. Ensure the file contains valid JSON.")
        return

    # Display the data
    display_data(data)

if __name__ == "__main__":
    main()
