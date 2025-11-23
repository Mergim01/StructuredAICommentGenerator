import streamlit as st
import pandas as pd
import io
import time
from llm_service import (
    get_dataframe_schema, 
    generate_transformation_code, 
    execute_transformation, 
    generate_formatting_code,
    optimize_user_prompt
)
from history_manager import load_prompt_history, save_prompt_to_history

# Page Config
st.set_page_config(
    page_title="Automated Excel Transformation",
    page_icon="📊",
    layout="wide"
)

def run_safe_transformation(df, user_prompt, api_key, preview_rows=None, max_retries=3, existing_code=None):
    """
    Attempts to generate and execute code with auto-correction loop.
    If preview_rows is set, executes on that many rows.
    If existing_code is provided, skips generation and uses it.
    """
    schema = get_dataframe_schema(df)
    last_error = None
    generated_code = existing_code
    
    # Only generate if we don't have code yet
    if not generated_code:
        generated_code = generate_transformation_code(schema, user_prompt, api_key)
    
    # Determine the working dataframe
    working_df = df.head(preview_rows) if preview_rows else df
    
    # If we are reusing code (Full Run), we typically trust it, but we can still wrap in try/except
    # If we are generating new code (Preview), we use the loop
    
    if existing_code:
        # Direct execution without retry loop (assuming it was validated in preview)
        # Or we could still retry if data issues appear on full set
        try:
             transformed_df = execute_transformation(working_df, generated_code)
             return transformed_df, generated_code
        except Exception as e:
             raise RuntimeError(f"Execution of existing code failed: {e}")

    # Generation Loop (Preview Mode)
    for attempt in range(max_retries + 1):
        try:
            # Test on subset (implicit if preview_rows is set)
            transformed_df = execute_transformation(working_df, generated_code)
            return transformed_df, generated_code
            
        except Exception as e:
            last_error = str(e)
            st.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed. Retrying with auto-correction...")
            
            if attempt < max_retries:
                generated_code = generate_transformation_code(
                    schema, 
                    user_prompt, 
                    api_key, 
                    previous_code=generated_code, 
                    error_feedback=last_error
                )
                time.sleep(1)
            else:
                raise RuntimeError(f"Transformation failed after {max_retries} retries.\nLast Error: {last_error}")

def run_safe_formatting(source_df, template_df, user_prompt, api_key, preview_rows=None, max_retries=3, existing_code=None):
    """
    Attempts to generate and execute formatting code with auto-correction loop.
    """
    source_schema = get_dataframe_schema(source_df)
    template_schema = get_dataframe_schema(template_df)
    
    last_error = None
    generated_code = existing_code
    
    if not generated_code:
        generated_code = generate_formatting_code(source_schema, template_schema, user_prompt, api_key)
    
    working_df = source_df.head(preview_rows) if preview_rows else source_df
    
    if existing_code:
        try:
            final_df = execute_transformation(working_df, generated_code, func_name="format_data")
            return final_df, generated_code
        except Exception as e:
            raise RuntimeError(f"Execution of existing code failed: {e}")

    for attempt in range(max_retries + 1):
        try:
            final_df = execute_transformation(working_df, generated_code, func_name="format_data")
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
    if "step1_code" not in st.session_state:
        st.session_state.step1_code = None
    if "preview_step1_df" not in st.session_state:
        st.session_state.preview_step1_df = None
    
    # Load Prompt History from file
    if "prompt_history" not in st.session_state:
        st.session_state.prompt_history = load_prompt_history()
        
    if "step2_code" not in st.session_state:
        st.session_state.step2_code = None
    if "preview_step2_df" not in st.session_state:
        st.session_state.preview_step2_df = None
        
    # Key for text area reset trick
    if "prompt_key" not in st.session_state:
        st.session_state.prompt_key = 0
    if "current_prompt_value" not in st.session_state:
        st.session_state.current_prompt_value = ""

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
        
        st.divider()
        st.header("Settings")
        preview_rows = st.number_input("Preview Rows (Variable X)", min_value=1, value=5, step=1)

    # --- STEP 1: Initial Transformation ---
    st.header("Step 1: Data Cleaning & Logic")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Source Excel (.xlsx)", type=["xlsx"], key="source_uploader")
    with col2:
        # Prompt Selection
        selected_prompt = st.selectbox(
            "Select from History (Optional)",
            [""] + st.session_state.prompt_history,
            key="history_select",
            index=0
        )
        
        # Update current value if history selection changes
        if selected_prompt and selected_prompt != st.session_state.current_prompt_value:
            st.session_state.current_prompt_value = selected_prompt
        
        user_prompt = st.text_area(
            "Transformation Instructions",
            height=100,
            value=st.session_state.current_prompt_value,
            placeholder="e.g. Rename 'Revenue' to 'Turnover', filter Region 'North'...",
            key=f"source_prompt_{st.session_state.prompt_key}" # Dynamic key for forcing reload
        )
        
        # Update state when user types manually
        if user_prompt != st.session_state.current_prompt_value:
             st.session_state.current_prompt_value = user_prompt

        if uploaded_file:
            if st.button("✨ Optimize Prompt", help="Uses AI to refine your instructions based on the uploaded file columns."):
                if not user_prompt:
                    st.warning("Please enter some instructions first.")
                else:
                    with st.spinner("Optimizing prompt..."):
                        try:
                            df_preview = pd.read_excel(uploaded_file, nrows=5) # Load schema only
                            schema_info = get_dataframe_schema(df_preview)
                            optimized = optimize_user_prompt(user_prompt, schema_info, api_key)
                            
                            # Update state and force rerender of text area by incrementing key
                            st.session_state.current_prompt_value = optimized
                            st.session_state.prompt_key += 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")

    if uploaded_file and user_prompt:
        # Button 1: Generate & Preview
        if st.button(f"Generate Code & Preview ({preview_rows} Rows)", type="primary"):
            # Save to history (file and session)
            if user_prompt and (not st.session_state.prompt_history or user_prompt != st.session_state.prompt_history[0]):
                save_prompt_to_history(user_prompt)
                # Refresh session state immediately to update dropdown on next rerun
                st.session_state.prompt_history = load_prompt_history()
                
            try:
                with st.spinner(f"Analyzing schema and generating code..."):
                    df = pd.read_excel(uploaded_file)
                    # Generate NEW code and preview
                    transformed_preview, generated_code = run_safe_transformation(
                        df, user_prompt, api_key, preview_rows=preview_rows
                    )
                    # Store in session state
                    st.session_state.preview_step1_df = transformed_preview
                    st.session_state.step1_code = generated_code
                    # Reset full result if we regenerated code
                    st.session_state.step1_df = None 
                    
            except Exception as e:
                st.error(f"Error in Generation: {e}")

    # Display Step 1 Preview and "Apply to All" Button
    if st.session_state.preview_step1_df is not None and st.session_state.step1_code is not None:
        st.divider()
        st.subheader("Step 1: Preview & Code")
        
        st.write("**Generated Python Code**")
        st.code(st.session_state.step1_code, language="python")
        
        st.write(f"**Preview Result ({len(st.session_state.preview_step1_df)} rows)**")
        st.dataframe(st.session_state.preview_step1_df)

        st.info("If the preview looks correct, apply the transformation to the full dataset.")
        
        if st.button("Apply to Full Dataset (Step 1)"):
            try:
                with st.spinner("Applying code to full dataset..."):
                    df = pd.read_excel(uploaded_file) # Re-read full file
                    # Execute EXISTING code on full data
                    transformed_full, _ = run_safe_transformation(
                        df, user_prompt, api_key, 
                        preview_rows=None, 
                        existing_code=st.session_state.step1_code
                    )
                    st.session_state.step1_df = transformed_full
                    st.success(f"Step 1 Full Transformation Complete! (Total rows: {len(transformed_full)})")
                    st.session_state.show_step1_final_preview = True
                    st.rerun() # Rerun to show Step 2
            except Exception as e:
                st.error(f"Error applying to full dataset: {e}")

    # Display Final Preview for Step 1 (after rerun)
    if st.session_state.get("show_step1_final_preview") and st.session_state.step1_df is not None:
        st.subheader("Data Preview (First 10 rows)")
        st.dataframe(st.session_state.step1_df.head(10))

    # --- STEP 2: Template Formatting (Only if Step 1 is fully done) ---
    if st.session_state.step1_df is not None:
        st.divider()
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
            # Button: Generate & Preview Step 2
            if st.button(f"Generate Step 2 Code & Preview ({preview_rows} Rows)", type="primary"):
                try:
                    with st.spinner("Generating Step 2 code..."):
                        template_df = pd.read_excel(template_file)
                        
                        final_preview, generated_code_2 = run_safe_formatting(
                            st.session_state.step1_df, 
                            template_df, 
                            format_prompt, 
                            api_key,
                            preview_rows=preview_rows
                        )
                        st.session_state.preview_step2_df = final_preview
                        st.session_state.step2_code = generated_code_2
                        
                except Exception as e:
                    st.error(f"Error in Step 2 Generation: {e}")

        # Display Step 2 Preview and "Apply to All" Button
        if st.session_state.preview_step2_df is not None and st.session_state.step2_code is not None:
            st.divider()
            st.subheader("Step 2: Preview & Code")
            
            st.write("**Generated Python Code**")
            st.code(st.session_state.step2_code, language="python")
            
            st.write(f"**Preview Result ({len(st.session_state.preview_step2_df)} rows)**")
            st.dataframe(st.session_state.preview_step2_df)
            
            st.info("If the preview looks correct, apply to generate the final Excel.")

            if st.button("Apply to Full Dataset (Step 2) & Download"):
                try:
                    with st.spinner("Applying Step 2 to full dataset..."):
                        template_df = pd.read_excel(template_file)
                        
                        final_full, _ = run_safe_formatting(
                            st.session_state.step1_df, 
                            template_df, 
                            format_prompt, 
                            api_key,
                            preview_rows=None,
                            existing_code=st.session_state.step2_code
                        )
                        
                        st.success(f"Process Complete! (Total rows: {len(final_full)})")
                        st.subheader("Final Result Preview (First 10 rows)")
                        st.dataframe(final_full.head(10))
                        
                        # Download
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            final_full.to_excel(writer, index=False)
                        
                        st.download_button(
                            label="Download Final Excel",
                            data=buffer.getvalue(),
                            file_name="final_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                except Exception as e:
                    st.error(f"Error in Step 2 Full Run: {e}")

if __name__ == "__main__":
    main()
