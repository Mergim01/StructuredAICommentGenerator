import streamlit as st
import pandas as pd
import io
from llm_service import get_dataframe_schema, generate_transformation_code, execute_transformation

# Page Config
st.set_page_config(
    page_title="Automated Excel Transformation",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Automated Excel Transformation Tool")
    st.markdown("""
    Upload a PnL Excel file, describe how you want it transformed, and let AI handle the rest.
    """)

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

    # File Uploader
    uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])

    # User Prompt
    user_prompt = st.text_area(
        "Transformation Instructions",
        height=150,
        placeholder="Example: Rename 'Revenue' to 'Turnover', filter out rows where 'Region' is 'North', and pivot the table..."
    )

    # Process Button
    if uploaded_file and user_prompt:
        if st.button("Transform Data", type="primary"):
            try:
                with st.spinner("Reading Excel file..."):
                    # Load Excel file
                    df = pd.read_excel(uploaded_file)
                    st.write("### Original Data Preview")
                    st.dataframe(df.head())

                with st.spinner("Analyzing schema and generating code..."):
                    # Extract Schema
                    schema = get_dataframe_schema(df)
                    
                    # Generate Code
                    generated_code = generate_transformation_code(schema, user_prompt, api_key)
                
                with st.expander("View Generated Python Code"):
                    st.code(generated_code, language="python")

                with st.spinner("Executing transformation..."):
                    # Execute Code
                    transformed_df = execute_transformation(df, generated_code)

                st.success("Transformation successful!")
                
                st.write("### Transformed Data Preview")
                st.dataframe(transformed_df.head())
                
                # Download Button
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    transformed_df.to_excel(writer, index=False)
                
                st.download_button(
                    label="Download Transformed Excel",
                    data=buffer.getvalue(),
                    file_name="transformed_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif not uploaded_file:
        st.info("Please upload an Excel file to get started.")
    elif not user_prompt:
        st.info("Please enter transformation instructions.")

if __name__ == "__main__":
    main()

