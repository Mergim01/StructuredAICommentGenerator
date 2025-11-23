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
    # Create a list of column names and types to ensure no truncation
    return "\n".join([f"{col} ({dtype})" for col, dtype in df.dtypes.items()])

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

def generate_transformation_code(schema_info: str, user_prompt: str, api_key: str, previous_code: str = None, error_feedback: str = None) -> str:
    """
    Generates Python code using Google Gemini based on the schema and user prompt.
    Can also fix code based on previous error feedback.
    
    Args:
        schema_info: String representation of the DataFrame schema.
        user_prompt: The user's instructions for transformation.
        api_key: The Google API key.
        previous_code: The code that failed (optional).
        error_feedback: The error message from the previous attempt (optional).
        
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
    3. Use `import pandas as pd` or other standard libraries INSIDE the function if needed.
    4. DO NOT include any explanation, introduction, or conclusion. RETURN ONLY THE CODE.
    5. Ensure the code is robust and handles potential type issues based on the schema.
    
    INPUT SCHEMA (Columns and Types):
    {schema_info}
    
    USER REQUEST:
    {user_prompt}
    """

    if previous_code and error_feedback:
        system_instruction += f"""
        
        PREVIOUS FAILED CODE:
        {previous_code}
        
        ERROR FEEDBACK:
        {error_feedback}
        
        INSTRUCTION:
        Fix the code based on the error feedback. Pay close attention to column names and data types.
        """
    
    try:
        response = model.generate_content(system_instruction)
        if not response.text:
            raise RuntimeError("Empty response received from Gemini.")
        return clean_code_block(response.text)
    except Exception as e:
        raise RuntimeError(f"Error communicating with Gemini API: {e}")

def generate_formatting_code(source_schema: str, template_schema: str, user_prompt: str, api_key: str, previous_code: str = None, error_feedback: str = None) -> str:
    """
    Generates Python code using Google Gemini to format the dataframe according to a template.
    Can also fix code based on previous error feedback.
    
    Args:
        source_schema: String representation of the source DataFrame schema.
        template_schema: String representation of the template DataFrame schema.
        user_prompt: The user's instructions for formatting.
        api_key: The Google API key.
        previous_code: The code that failed (optional).
        error_feedback: The error message from the previous attempt (optional).
        
    Returns:
        String containing the generated Python code with a function `format_data(df)`.
    """
    if not api_key:
        raise ValueError("API Key is missing.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    system_instruction = f"""
    You are an expert Python Data Scientist.
    Your task is to write a Python function named `format_data` that takes a pandas DataFrame `df` as input and returns a NEW DataFrame that matches the structure of the TEMPLATE.
    
    STRICT RULES:
    1. You MUST define a function `def format_data(df):`.
    2. The function MUST return a pandas DataFrame.
    3. Use `import pandas as pd` or other standard libraries INSIDE the function if needed.
    4. The returned DataFrame MUST have the columns and structure described in the TEMPLATE SCHEMA.
    5. Map the columns from the INPUT SCHEMA to the TEMPLATE SCHEMA based on the USER REQUEST.
    6. If the template has columns not present in the input, add them (empty or calculated).
    7. DO NOT include any explanation. RETURN ONLY THE CODE.
    
    INPUT SCHEMA (Current Data):
    {source_schema}
    
    TEMPLATE SCHEMA (Target Structure):
    {template_schema}
    
    USER REQUEST:
    {user_prompt}
    """

    if previous_code and error_feedback:
        system_instruction += f"""
        
        PREVIOUS FAILED CODE:
        {previous_code}
        
        ERROR FEEDBACK:
        {error_feedback}
        
        INSTRUCTION:
        Fix the code based on the error feedback. Pay close attention to column names.
        """
    
    try:
        response = model.generate_content(system_instruction)
        if not response.text:
            raise RuntimeError("Empty response received from Gemini.")
        return clean_code_block(response.text)
    except Exception as e:
        raise RuntimeError(f"Error communicating with Gemini API: {e}")

def execute_transformation(df: pd.DataFrame, code: str, func_name: str = "transform_data") -> pd.DataFrame:
    """
    Executes the generated code on the provided DataFrame safely.
    
    Args:
        df: The original pandas DataFrame.
        code: The Python code string containing the function.
        func_name: The name of the function to execute (default: "transform_data").
        
    Returns:
        The transformed pandas DataFrame.
    """
    local_scope = {}
    
    if func_name == "transform_data":
        # For step 1, ensure imports are available in local scope if the LLM forgot them inside the function
        local_scope['pd'] = pd
        
    # 1. Define the function in a local scope
    try:
        exec(code, {}, local_scope)
    except SyntaxError as e:
        raise RuntimeError(f"Generated code has syntax errors: {e}\\n\\nCode:\\n{code}")
    except Exception as e:
        raise RuntimeError(f"Failed to define transformation function: {e}\\n\\nCode:\\n{code}")
    
    # 2. Verify function existence
    if func_name not in local_scope:
        raise RuntimeError(f"The generated code did not define a function named '{func_name}'.")
    
    transform_func = local_scope[func_name]
    
    # 3. Run the function on a copy of the DataFrame
    try:
        # Passing a deep copy to avoid mutating the original in place unexpectedly
        transformed_df = transform_func(df.copy())
    except Exception as e:
        # Capture full traceback for debugging
        error_msg = traceback.format_exc()
        columns_info = df.columns.tolist()
        raise RuntimeError(f"Error during transformation execution:\\n{error_msg}\\n\\nAvailable Columns: {columns_info}")
         
    # 4. Validate output
    if not isinstance(transformed_df, pd.DataFrame):
        raise RuntimeError(f"The '{func_name}' function returned {type(transformed_df)} instead of a pandas DataFrame.")
        
    return transformed_df

