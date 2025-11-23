import streamlit as st
import pandas as pd
import io
import time
from llm_service import (
    get_dataframe_schema, 
    generate_transformation_code, 
    execute_transformation, 
    generate_formatting_code
)

# Page Config
st.set_page_config(
    page_title="Automated Excel Transformation",
    page_icon="📊",
    layout="wide"
)

def run_safe_transformation(df, user_prompt, api_key, max_retries=3):
    """
    Attempts to generate and execute code with auto-correction loop.
    """
    schema = get_dataframe_schema(df)
    last_error = None
    generated_code = None
    
    # Try initial generation
    generated_code = generate_transformation_code(schema, user_prompt, api_key)
    
    for attempt in range(max_retries + 1):
        try:
            # Test on a small subset first (head) to catch errors quickly
            # Use at least 5 rows, or all if less than 5
            test_df = df.head(5).copy()
            
            # Execute on subset
            execute_transformation(test_df, generated_code)
            
            # If successful on subset, run on full data
            final_df = execute_transformation(df, generated_code)
            return final_df, generated_code
            
        except Exception as e:
            last_error = str(e)
            st.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed. Retrying with auto-correction...")
            
            if attempt < max_retries:
                # Generate fix
                generated_code = generate_transformation_code(
                    schema, 
                    user_prompt, 
                    api_key, 
                    previous_code=generated_code, 
                    error_feedback=last_error
                )
                time.sleep(1) # Small delay
            else:
                # Out of retries
                raise RuntimeError(f"Transformation failed after {max_retries} retries.\nLast Error: {last_error}")

def run_safe_formatting(source_df, template_df, user_prompt, api_key, max_retries=3):
    """
    Attempts to generate and execute formatting code with auto-correction loop.
    """
    source_schema = get_dataframe_schema(source_df)
    template_schema = get_dataframe_schema(template_df)
    
    last_error = None
    generated_code = None
    
    # Initial generation
    generated_code = generate_formatting_code(source_schema, template_schema, user_prompt, api_key)
    
    for attempt in range(max_retries + 1):
        try:
            # Test on subset
            test_df = source_df.head(5).copy()
            
            execute_transformation(
                test_df, 
                generated_code, 
                func_name="format_data"
            )
            
            # Run on full data
            final_df = execute_transformation(
                source_df, 
                generated_code, 
                func_name="format_data"
            )
            return final_df, generated_code
            
        except Exception as e:
            last_error = str(e)
            st.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed. Retrying with auto-correction...")
            
            if attempt < max_retries:
                generated_code = generate_formatting_code(
                    source_schema, 
                    template_schema, 
                    user_prompt, 
                    api_key,
                    previous_code=generated_code,
                    error_feedback=last_error
                )
                time.sleep(1)
            else:
                raise RuntimeError(f"Formatting failed after {max_retries} retries.\nLast Error: {last_error}")

def main():
    st.title("📊 Automated Excel Transformation Tool")
    
    # Initialize Session State
    if "step1_df" not in st.session_state:
        st.session_state.step1_df = None

    # Sidebar for Configuration
    with st.sidebar:
        st.header("Configuration")
        try:
            api_key = st.secrets["google"]["api_key"]
            if api_key == "YOUR_GOOGLE_API_KEY_HERE":
                st.warning("Please configure your Google API Key in `.streamlit/secrets.toml`.")
                st.stop()
            st.success("API Key loaded.")
        except Exception:
            st.error("Secrets not found. Please create `.streamlit/secrets.toml`.")
            st.stop()

    # --- STEP 1: Initial Transformation ---
    st.header("Step 1: Data Cleaning & Logic")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Source Excel (.xlsx)", type=["xlsx"], key="source_uploader")
    with col2:
        user_prompt = st.text_area(
            "Transformation Instructions",
            height=100,
            placeholder="e.g. Rename 'Revenue' to 'Turnover', filter Region 'North'...",
            key="source_prompt"
        )

    if uploaded_file and user_prompt:
        if st.button("Run Step 1: Transform", type="primary"):
            try:
                with st.spinner("Processing Step 1..."):
                    df = pd.read_excel(uploaded_file)
                    
                    # Run safe transformation with retry loop
                    transformed_df, final_code = run_safe_transformation(df, user_prompt, api_key)
                    
                    with st.expander("View Step 1 Final Code"):
                        st.code(final_code, language="python")
                    
                    st.session_state.step1_df = transformed_df
                    st.success("Step 1 Complete!")
            except Exception as e:
                st.error(f"Error in Step 1: {e}")

    # Display Step 1 Result if available
    if st.session_state.step1_df is not None:
        st.subheader("Step 1 Result Preview")
        st.dataframe(st.session_state.step1_df.head())
        
        st.divider()
        
        # --- STEP 2: Template Formatting ---
        st.header("Step 2: Template Formatting")
        st.markdown("Upload a template Excel file and describe how to map the data from Step 1 into this structure.")
        
        col3, col4 = st.columns(2)
        with col3:
            template_file = st.file_uploader("Upload Template Excel (.xlsx)", type=["xlsx"], key="template_uploader")
        with col4:
            format_prompt = st.text_area(
                "Mapping Instructions",
                height=100,
                placeholder="e.g. Map 'Turnover' to column A, add empty rows after totals...",
                key="template_prompt"
            )
            
        if template_file and format_prompt:
            if st.button("Run Step 2: Apply Template"):
                try:
                    with st.spinner("Processing Step 2..."):
                        template_df = pd.read_excel(template_file)
                        
                        # Run safe formatting with retry loop
                        final_df, final_format_code = run_safe_formatting(
                            st.session_state.step1_df, 
                            template_df, 
                            format_prompt, 
                            api_key
                        )
                        
                        with st.expander("View Step 2 Final Code"):
                            st.code(final_format_code, language="python")
                        
                        st.success("Step 2 Complete!")
                        
                        st.subheader("Final Result Preview")
                        st.dataframe(final_df.head())
                        
                        # Download
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            final_df.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="Download Final Excel",
                            data=buffer.getvalue(),
                            file_name="final_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                except Exception as e:
                    st.error(f"Error in Step 2: {e}")

if __name__ == "__main__":
    main()
