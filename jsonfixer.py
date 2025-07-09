import re
import json

# Top-level variable: change this to the path of your JSON file.
JSON_FILE_PATH = "C:\Users\Oliver\Documents\Github\AI-api-building-\step-2-agent-reasoning edited personal info prompt ready for job.json [MConverter.eu] (3).json"

def fix_json(json_str):
    """
    Attempt to fix common JSON issues, such as trailing commas before a closing
    brace or bracket.
    """
    # This regex finds commas that are immediately followed by a closing curly brace or bracket
    fixed = re.sub(r",\s*([\]}])", r"\1", json_str)
    return fixed

def validate_json(json_str):
    """
    Try to load the JSON string. If it fails due to common errors, try to fix
    it using the fix_json() function and load it again.
    
    Returns:
        A tuple of (data, fixed):
          - data: the loaded JSON object if successful, or None otherwise.
          - fixed: True if fixes were applied, otherwise False.
    """
    fixed = False
    try:
        data = json.loads(json_str)
        return data, fixed
    except json.JSONDecodeError as e:
        print("Initial JSON decode error:", e)
        # Attempt to fix common issues by cleaning the JSON string
        fixed_json = fix_json(json_str)
        fixed = True
        try:
            data = json.loads(fixed_json)
            return data, fixed
        except json.JSONDecodeError as e_fixed:
            print("JSON still invalid after fix attempt:", e_fixed)
            return None, fixed

def main():
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            json_text = f.read()
    except Exception as e:
        print(f"Error reading file {JSON_FILE_PATH}: {e}")
        return
    
    data, fixed = validate_json(json_text)
    
    if data is None:
        print("Failed to validate or fix JSON.")
    else:
        if fixed:
            print("JSON was invalid and has been fixed!")
        else:
            print("JSON is valid!")
        # Output the validated (and possibly fixed) JSON in a pretty format
        print(json.dumps(data, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
