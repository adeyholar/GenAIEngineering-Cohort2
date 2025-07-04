import streamlit as st
import pandas as pd
import requests
import json
import os

# Set page configuration
st.set_page_config(
    page_title="PDF Chunk Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FastAPI service URL - CORRECTED PORT TO 9322
API_URL = "http://localhost:9322"

def search_pdf_chunks(search_string):
    """
    Sends a search query to the FastAPI backend and returns the results.
    """
    search_endpoint = f"{API_URL}/search_chunks"
    print(f"Calling API: {search_endpoint}?search_string={search_string}")
    try:
        response = requests.post(f"{search_endpoint}?search_string={search_string}")
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        json_response = response.json()
        if json_response.get("status") == "success":
            results = json_response.get("results", [])
            if results:
                return pd.DataFrame(results)
            else:
                return pd.DataFrame() # Return empty DataFrame if no results
        else:
            st.error(f"API Error: {json_response.get('message', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the FastAPI service at {API_URL}.")
        st.error("Please ensure the FastAPI application is running.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        st.error(f"Response: {response.text}")
        return None

def upload_and_chunk_pdf(pdf_file, num_chunks):
    """
    Sends a PDF file to the FastAPI backend for chunking.
    """
    chunk_endpoint = f"{API_URL}/chunk_pdf"
    
    files = {'file': (pdf_file.name, pdf_file.getvalue(), pdf_file.type)}
    data = {'num_chunks': str(num_chunks)} # num_chunks needs to be a string for Form data

    st.info(f"Uploading and chunking '{pdf_file.name}'...")
    try:
        response = requests.post(chunk_endpoint, files=files, data=data)
        response.raise_for_status()
        json_response = response.json()
        if json_response.get("status") == "success":
            st.success(json_response.get("message", "PDF chunked successfully!"))
            return True
        else:
            st.error(f"Error during chunking: {json_response.get('message', 'Unknown error')}")
            return False
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling chunk_pdf API: {e}")
        st.error("Please ensure the FastAPI service is running.")
        return False

def main():
    # App title and description
    st.title("📄 PDF Chunk Search")
    st.markdown("""
    Search through all PDF chunks for specific keywords or phrases.
    Results will show matching chunks with their source document and page number.
    """)

    # --- PDF Upload Section ---
    st.sidebar.header("Upload PDF for Chunking")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    num_chunks_input = st.sidebar.slider("Number of chunks per page", 1, 20, 5)

    if uploaded_file is not None:
        if st.sidebar.button("Process PDF"):
            if upload_and_chunk_pdf(uploaded_file, num_chunks_input):
                st.sidebar.success("PDF processed and indexed!")
                # Optionally, trigger a search or refresh after successful upload
            else:
                st.sidebar.error("Failed to process PDF.")

    st.sidebar.markdown("---")

    # --- Search Section ---
    col1, col2 = st.columns([3, 1])

    with col1:
        search_input = st.text_input("Search for keywords or phrases:",
                                       placeholder="Enter your search terms...")

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Display search results
    if search_button and search_input:
        st.subheader("Search Results")

        with st.spinner(f"Searching for '{search_input}'..."):
            results = search_pdf_chunks(search_input)

        if results is not None:
            if not results.empty:
                st.success(f"Found {len(results)} matching chunks")

                # Add a filter for filename
                if len(results['filename'].unique()) > 1:
                    file_filter = st.multiselect(
                        "Filter by document:",
                        options=sorted(results['filename'].unique()),
                        default=sorted(results['filename'].unique())
                    )

                    # Apply file filter
                    if file_filter:
                        results = results[results['filename'].isin(file_filter)]

                # Display results in an expander for each chunk
                for index, row in results.iterrows():
                    with st.expander(f"📄 {row['filename']} - Page {row['page_number']} - Chunk {row['chunk_number']}", expanded=False): # Changed to collapsed by default
                        st.markdown("""
                        <style>
                        .chunk-box {
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            padding: 10px;
                            background-color: #f9f9f9;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        st.markdown(f"<div class='chunk-box'>{row['chunk']}</div>", unsafe_allow_html=True)
                        st.caption(f"Document: {row['filename']} | Page: {row['page_number']} | Chunk: {row['chunk_number']}")
            else:
                st.warning(f"No results found for '{search_input}'")
        else:
            st.error("Could not retrieve search results.")

    if not search_button and not search_input: # Instructions visible if no search has been initiated
        st.info("Enter a search term and click 'Search' to find matching chunks from processed PDFs.")
        st.markdown("### Example searches:")
        example_searches = ["important", "data", "analysis", "conclusion"]

        for example in example_searches:
            if st.button(example):
                search_input = example
                st.rerun() # Use st.rerun() instead of experimental_rerun()

if __name__ == "__main__":
    main()