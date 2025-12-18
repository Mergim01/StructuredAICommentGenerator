import google.generativeai as genai
import pandas as pd
import re
import traceback
import os
import openai
from openai import AzureOpenAI

# Default Constants
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"
DEFAULT_AZURE_MODEL = "gpt-4.1" # Default company model
AZURE_BASE_URL = "https://api.competence-cente-cc-genai-prod.enbw-az.cloud/openai"
AZURE_API_VERSION = "2024-10-21"

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

def call_llm(prompt: str, llm_config: dict) -> str:
    """
    Unified function to call different LLM providers.
    """
    provider = llm_config.get("provider", "google")
    api_key = llm_config.get("api_key")
    model_name = llm_config.get("model_name")
    
    if provider == "azure":
        if not api_key:
             raise ValueError("API Key is missing for Azure provider.")
             
        client = openai.AzureOpenAI(
            base_url=llm_config.get("base_url", AZURE_BASE_URL),
            api_key=api_key,
            api_version=llm_config.get("api_version", AZURE_API_VERSION)
        )
        try:
            completion = client.chat.completions.create(
                model=model_name or DEFAULT_AZURE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.get("temperature", 0.0)
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error communicating with Azure OpenAI: {e}")

    elif provider == "google":
        if not api_key:
             raise ValueError("API Key is missing for Google provider.")
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name or DEFAULT_GOOGLE_MODEL)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=llm_config.get("temperature", 0.0)
                )
            )
            if not response.text:
                raise RuntimeError("Empty response received from Gemini.")
            return response.text
        except Exception as e:
             raise RuntimeError(f"Error communicating with Gemini API: {e}")
        
    else:
        raise ValueError(f"Unknown provider: {provider}")

def generate_transformation_code(schema_info: str, user_prompt: str, llm_config: dict, previous_code: str = None, error_feedback: str = None) -> str:
    """
    Generates Python code using the selected LLM based on the schema and user prompt.
    Can also fix code based on previous error feedback.
    
    Args:
        schema_info: String representation of the DataFrame schema.
        user_prompt: The user's instructions for transformation.
        llm_config: Dictionary containing LLM configuration (provider, api_key, model_name).
        previous_code: The code that failed (optional).
        error_feedback: The error message from the previous attempt (optional).
        
    Returns:
        String containing the generated Python code.
    """
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
    
    response_text = call_llm(system_instruction, llm_config)
    return clean_code_block(response_text)

def generate_formatting_code(source_schema: str, template_schema: str, user_prompt: str, llm_config: dict, previous_code: str = None, error_feedback: str = None) -> str:
    """
    Generates Python code using the selected LLM to format the dataframe according to a template.
    Can also fix code based on previous error feedback.
    
    Args:
        source_schema: String representation of the source DataFrame schema.
        template_schema: String representation of the template DataFrame schema.
        user_prompt: The user's instructions for formatting.
        llm_config: Dictionary containing LLM configuration.
        previous_code: The code that failed (optional).
        error_feedback: The error message from the previous attempt (optional).
        
    Returns:
        String containing the generated Python code with a function `format_data(df)`.
    """
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
    
    response_text = call_llm(system_instruction, llm_config)
    return clean_code_block(response_text)

def optimize_user_prompt(user_prompt: str, schema_info: str, llm_config: dict) -> str:
    """
    Optimizes the user prompt to be clearer and more precise for the LLM,
    referencing specific columns from the schema if possible.
    """
    system_instruction = f"""
    You are an expert Data Science Consultant. 
    Your goal is to refine a User's rough instructions into a precise, unambiguous prompt for a Python coding LLM.
    
    INPUT SCHEMA (Columns available):
    {schema_info}
    
    USER'S ROUGH INSTRUCTION:
    "{user_prompt}"
    
    TASK:
    Rewrite the user's instruction to be:
    1. Clear and step-by-step.
    2. Explicit about column names (match them to the schema).
    3. Concise but complete.
    4. DO NOT add conversational text. Just output the optimized prompt text.
    """
    
    try:
        response_text = call_llm(system_instruction, llm_config)
        return response_text.strip()
    except Exception as e:
        # In case of error, return original prompt
        print(f"Prompt optimization failed: {e}")
        return user_prompt

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

