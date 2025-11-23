import google.generativeai as genai
import pandas as pd
import re
import traceback

MODEL_NAME = "gemini-2.0-flash"

def get_dataframe_schema(df: pd.DataFrame) -> str:
    """
    Extracts column names and data types from a DataFrame.
    This avoids sending actual data values for privacy reasons.
    """
    # Returning a string representation of the dtypes
    return df.dtypes.to_string()

def clean_code_block(code: str) -> str:
    """
    Removes markdown code block formatting if present.
    """
    # Matches ```python ... ``` or just ``` ... ```
    # DOTALL ensures . matches newlines
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code.strip()

def generate_transformation_code(schema_info: str, user_prompt: str, api_key: str) -> str:
    """
    Generates Python code using Google Gemini based on the schema and user prompt.
    
    Args:
        schema_info: String representation of the DataFrame schema.
        user_prompt: The user's instructions for transformation.
        api_key: The Google API key.
        
    Returns:
        String containing the generated Python code.
    """
    if not api_key:
        raise ValueError("API Key is missing. Please configure it in .streamlit/secrets.toml.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    system_instruction = f"""
    You are an expert Python Data Scientist specializing in Pandas.
    Your task is to write a Python function named `transform_data` that takes a pandas DataFrame `df` as input and returns a transformed DataFrame.
    
    STRICT RULES:
    1. You MUST define a function `def transform_data(df):`.
    2. The function MUST return the modified DataFrame.
    3. Use `import pandas as pd` or other standard libraries inside the function if needed.
    4. DO NOT include any explanation, introduction, or conclusion. RETURN ONLY THE CODE.
    5. Ensure the code is robust and handles potential type issues based on the schema.
    
    INPUT SCHEMA (Columns and Types):
    {schema_info}
    
    USER REQUEST:
    {user_prompt}
    """
    
    try:
        response = model.generate_content(system_instruction)
        if not response.text:
            raise RuntimeError("Empty response received from Gemini.")
        return clean_code_block(response.text)
    except Exception as e:
        raise RuntimeError(f"Error communicating with Gemini API: {e}")

def execute_transformation(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """
    Executes the generated code on the provided DataFrame safely.
    
    Args:
        df: The original pandas DataFrame.
        code: The Python code string containing the `transform_data` function.
        
    Returns:
        The transformed pandas DataFrame.
    """
    local_scope = {}
    
    # 1. Define the function in a local scope
    try:
        exec(code, {}, local_scope)
    except SyntaxError as e:
        raise RuntimeError(f"Generated code has syntax errors: {e}\n\nCode:\n{code}")
    except Exception as e:
        raise RuntimeError(f"Failed to define transformation function: {e}\n\nCode:\n{code}")
    
    # 2. Verify function existence
    if 'transform_data' not in local_scope:
        raise RuntimeError("The generated code did not define a function named 'transform_data'.")
    
    transform_func = local_scope['transform_data']
    
    # 3. Run the function on a copy of the DataFrame
    try:
        # Passing a deep copy to avoid mutating the original in place unexpectedly, 
        # although the pipeline should be linear.
        transformed_df = transform_func(df.copy())
    except Exception as e:
        # Capture full traceback for debugging
        error_msg = traceback.format_exc()
        raise RuntimeError(f"Error during transformation execution:\n{error_msg}")
         
    # 4. Validate output
    if not isinstance(transformed_df, pd.DataFrame):
        raise RuntimeError(f"The 'transform_data' function returned {type(transformed_df)} instead of a pandas DataFrame.")
        
    return transformed_df

