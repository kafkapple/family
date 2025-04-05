import json

def save_llm_prompts_to_txt(llm_prompts, filename="llm_prompts.txt"):
    """
    Saves the generated LLM prompts to a text file.

    Args:
        llm_prompts (str): The LLM prompts string.
        filename (str, optional): The name of the file to save the prompts to.
            Defaults to "llm_prompts.txt".
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(llm_prompts)
        print(f"LLM prompts saved to {filename}")
    except Exception as e:
        print(f"Error saving LLM prompts to file: {e}")

def parse_and_save_json_results(json_string, filename="json_results.txt"):
    """
    Parses a JSON string and saves the results to a text file.

    Args:
        json_string (str): The JSON string to parse.
        filename (str, optional): The name of the file to save the parsed results to.
            Defaults to "json_results.txt".
    """
    try:
        data = json.loads(json_string)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)  # Save with indentation for readability
        print(f"JSON results parsed and saved to {filename}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Error saving JSON results to file: {e}")

