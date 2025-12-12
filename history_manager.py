import json
import os

HISTORY_FILE = "prompt_history.json"

def load_prompt_history():
    """Loads prompt history from a JSON file."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_prompt_to_history(prompt):
    """Saves a new prompt to history if it doesn't exist."""
    history = load_prompt_history()
    if prompt and prompt not in history:
        history.insert(0, prompt)  # Add to top
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"Error saving history: {e}")

