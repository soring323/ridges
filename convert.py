import json

# Read the diff file
with open("patch.diff", "r", encoding="utf-8") as f:
    diff_text = f.read()

# Put it into a dictionary
data = {"instance_id": "sympy__sympy-12489", "model_name_or_path": "gpt-4", "model_patch": diff_text}

# Convert to JSON (with escaped newlines etc.)
json_str = json.dumps(data, indent=2)

# Optionally save it to a .json file
with open("patch.json", "w", encoding="utf-8") as f:
    f.write(json_str)
