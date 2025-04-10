import json
from typing import Dict, Any

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

def save_response_to_json(data: Dict[str, Any], output_path: str) -> None:
    """생성된 대화 데이터를 파일로 저장합니다."""
    with open(output_path+".json", 'w', encoding='utf-8') as f:
        # 메타데이터 작성
        #f.write(f"# Family Id: {family_id}\n")
        # JSON 데이터 저장
        # data = {"persona_id": persona_id, **data}
        json.dump(data, f, ensure_ascii=False, indent=2)
def save_dialogue_to_file(output_path: str, dialogue_data: Dict[str, Any], persona_id: int, i_persona, examples: str) -> None:
    """생성된 대화 데이터를 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 메타데이터 작성
        # f.write(f"# 카테고리: {category_id}\n")
        # f.write(f"# 플랜: {plan_id}\n")
        # f.write("# 사용된 예시 문장:\n")
        # dialogue_data = {"persona_id": persona_id, "persona": i_persona, "age": dialogue_data['age'],
        #             "explanation": dialogue_data['explanation'],
        #             "category": dialogue_data['category'],
        #             "plan": dialogue_data['plan'],
        #             "examples": examples,
        #             "dialogue": dialogue_data['dialogue']} 

        # # examples가 문자열일 경우 각 줄을 처리
        # if isinstance(examples, str):
        #     clean_examples = examples.strip().split('\n')
        #     for example in clean_examples:
        #         if example.strip():  # 빈 줄 제외
        #             f.write(f"{example}\n")
        # f.write("\n")
        
        # JSON 데이터 저장
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)

import json
import yaml

def call_save_results(data, filename):
    save_response_to_json(data, filename)
    # save_dialogue_to_file(data, filename)
    save_key_value_format(data, filename)
    # save_yaml_style(data, filename)

def save_key_value_format(data, filename):
    def recursive_write(d, indent=0):
        lines = []
        pad = "  " * indent
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.extend(recursive_write(v, indent + 1))
                else:
                    # 여러 줄인 경우 블록 처리
                    if isinstance(v, str) and '\n' in v:
                        lines.append(f"{pad}{k}: |")
                        lines.extend([f"{pad}  {line}" for line in v.split('\n')])
                    else:
                        lines.append(f"{pad}{k}: {v}")
        elif isinstance(d, list):
            for item in d:
                lines.append(f"{pad}-")
                lines.extend(recursive_write(item, indent + 1))
        return lines

    with open(filename+".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(recursive_write(data)))


def save_markdown_table(data, filename):
    def flatten_for_table(d):
        flat = {}
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = str(v)
        return flat

    flat_data = flatten_for_table(data)
    with open(filename+".txt", "w", encoding="utf-8") as f:
        f.write("| Key | Value |\n")
        f.write("|-----|-------|\n")
        for k, v in flat_data.items():
            f.write(f"| {k} | {v} |\n")


def save_yaml_style(data, filename):
    with open(filename+".yaml", "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)