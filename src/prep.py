import json
from typing import Dict, Any, List, Tuple
import yaml
import re
import glob
import os
from omegaconf import OmegaConf, DictConfig

def load_and_parse_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 모든 줄을 읽어옴
        lines = file.readlines()
        
        # 중괄호({)가 시작되는 줄 찾기
        start_line = 0
        for i, line in enumerate(lines):
            if '{' in line:
                start_line = i
                break
        
        # JSON 텍스트 구성
        json_text = ''.join(lines[start_line:])
        
        # JSON 문자열에서 중괄호({}) 시작과 끝 부분 찾기
        start_idx = json_text.find('{')
        end_idx = json_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_data = json_text[start_idx:end_idx]
            
            # JSON 데이터 파싱
            parsed_data = json.loads(json_data)
            return parsed_data
        else:
            return None

def load_yaml_data(file_path: str) -> Dict[str, Any]:
    """YAML 파일에서 데이터를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
def parse_json_response(response: str) -> Dict[str, Any]:
    """LLM의 응답에서 JSON 부분을 추출하고 파싱합니다. (개선됨)"""
    if not response:
        print("응답이 비어있습니다.")
        return None

    try:
        # 1. Markdown 코드 블록 제거 (```json ... ```)
        match_markdown = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response, re.DOTALL)
        if match_markdown:
            json_str = match_markdown.group(1)
            print("마크다운 코드 블록에서 JSON 추출 성공.")
        else:
            # 2. 마크다운 없으면, 첫 '{'와 마지막 '}' 사이의 내용 추출 시도
            match_brace = re.search(r"\{[\s\S]*\}", response, re.DOTALL)
            if match_brace:
                json_str = match_brace.group(0)
                print("일반 텍스트에서 JSON 추출 시도.")
            else:
                print("응답에서 JSON 형식을 찾을 수 없습니다 (마크다운/중괄호 불일치).")
                print(f"원본 응답 (앞 500자): {response[:500]}...")
                return None

        # 3. 추출된 문자열 파싱
        try:
            # 제어 문자 제거 (JSON에 유효하지 않은  -  범위, \t\n\r\f 제외)
            # 정규식: 0-8, 11, 12, 14-31 범위의 제어 문자 제거
            cleaned_json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)
            
            # 둥근 따옴표도 미리 교체
            cleaned_json_str = cleaned_json_str.replace('"', '"').replace('"', '"')
            
            # 정리된 문자열로 파싱 시도
            return json.loads(cleaned_json_str) 
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            # 오류 발생 시 원본과 정리된 문자열 일부를 비교 로깅
            print(f"파싱 시도한 원본 문자열 (앞 500자): {json_str[:500]}...") 
            if json_str != cleaned_json_str:
                print(f"정리된 문자열 (앞 500자): {cleaned_json_str[:500]}...")
            return None # 파싱 실패 시 최종 실패

    except Exception as e:
        print(f"JSON 추출/파싱 중 예상치 못한 오류 발생: {e}")
        print(f"원본 응답 (앞 500자): {response[:500]}...")
        return None
    

def sanitize_model_name(model: str) -> str:
    """Clean model name for use as file name"""
    return re.sub(r'[^\w\-\.]', '_', model.lower())
DictConfig
# Helper function to load template string from config
def load_template_str(cfg: DictConfig, key: str) -> str:
    """Load a template string from the OmegaConf configuration."""
    try:
        template_val = OmegaConf.select(cfg, key, default=None)
        if isinstance(template_val, str):
            return template_val.strip()
        print(f"템플릿 값이 문자열이 아닙니다: {type(template_val)}")
        return None
    except Exception as e:
        print(f"템플릿 로드 중 오류 발생: {e}")
        return None

# Helper function to load data from config with error handling
def load_data_from_cfg(cfg, key, data_type, default_value):
    try:
        data = OmegaConf.select(cfg, key)
        if data is not None:
            resolved_data = OmegaConf.to_container(data, resolve=True)
            if isinstance(resolved_data, data_type):
   
                return resolved_data
            else:
                
                return default_value
        else:
           
            return default_value
    except Exception as e:
    
        return default_value

def create_scoring_mappings(scoring_criteria_list: List[Dict]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """스코어링 기준 리스트에서 인덱스 <-> ID 매핑을 생성합니다."""
    scoring_index_to_id = {i: criteria['id'] for i, criteria in enumerate(scoring_criteria_list)}
    scoring_id_to_index = {criteria['id']: i for i, criteria in enumerate(scoring_criteria_list)}
    print(f"스코어링 기준 매핑 생성 완료: {len(scoring_criteria_list)}개 ({scoring_index_to_id})")
    return scoring_index_to_id, scoring_id_to_index
# --- 매핑 함수 정의 끝 --- 

# --- JSON 키 검증/수정 함수 --- 
def levenshtein_distance(s1: str, s2: str) -> int:
    """두 문자열 간의 Levenshtein 거리를 계산합니다."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def fix_json_keys(data: Dict[str, Any], expected_keys: List[str], threshold: int = 2) -> Dict[str, Any]:
    """Levenshtein 거리를 사용하여 JSON 데이터의 최상위 키를 예상 키로 수정합니다. (재귀 제거)"""
    if not isinstance(data, dict):
        return data 

    corrected_data = {}
    used_expected_keys = set()
    actual_keys = list(data.keys()) # 순회 중 변경 방지를 위해 키 목록 복사

    for actual_key in actual_keys:
        value = data[actual_key]
        best_match = None
        min_distance = float('inf')

        # 현재 키와 가장 유사한 예상 키 찾기
        for expected_key in expected_keys:
            distance = levenshtein_distance(actual_key, expected_key)
            if distance < min_distance:
                min_distance = distance
                best_match = expected_key
        
        # 거리가 임계값 이하이고 아직 사용되지 않은 예상 키면 수정된 키 사용, 아니면 원본 키 사용
        if best_match and min_distance <= threshold and best_match not in used_expected_keys:
            if actual_key != best_match:
                print(f"JSON 키 수정: '{actual_key}' -> '{best_match}' (거리: {min_distance})")
            corrected_data[best_match] = value # 재귀 호출 없이 값 그대로 할당
            used_expected_keys.add(best_match)
        else:
            # 수정 조건 미달 시 원본 키와 값 사용
            if min_distance > threshold and best_match:
                 print(f"경고: JSON 키 '{actual_key}'는 예상 키와 거리가 멀어 수정하지 않음 (최소 거리: {min_distance} to '{best_match}')")
            corrected_data[actual_key] = value # 재귀 호출 없이 값 그대로 할당
            # 만약 이 키가 예상 키 목록에 있었다면 사용된 것으로 간주 (중복 매칭 방지 위함)
            if actual_key in expected_keys:
                 used_expected_keys.add(actual_key)

    return corrected_data

def reconstruct_dialogue(dialogue_data: Dict[str, Any]) -> str:
    """
    딕셔너리에서 대화 데이터를 추출하여 텍스트로 재구성합니다.
    dialogue_data에서 'dialogue' 키(또는 유사한 키)의 값을 찾아
    각 턴의 speaker와 content를 추출하여 대화 형식의 문자열로 변환합니다.
    
    각 턴은 항상 speaker-content 쌍으로 구성되어 있다고 가정합니다.
    즉, 각 턴 딕셔너리의 첫 번째 키가 화자, 두 번째 키가 대화 내용을 나타냅니다.
    """
    # 가능한 대화 데이터 키 목록
    possible_dialogue_keys = ['dialogue', 'dialouge', 'dialgues', 'diallogue']
    dialogue_list = None
    
    # 대화 데이터 키 찾기
    for key in possible_dialogue_keys:
        if key in dialogue_data:
            dialogue_list = dialogue_data[key]
            print(f"대화 데이터 키 '{key}' 발견")
            break
    
    if not dialogue_list or not isinstance(dialogue_list, list):
        print(f"유효한 대화 데이터를 찾을 수 없습니다. 사용 가능한 키: {list(dialogue_data.keys())}")
        return "[대화 데이터 없음]"
    
    formatted_turns = []
    for i, turn in enumerate(dialogue_list):
        if not isinstance(turn, dict):
            print(f"턴 {i}가 딕셔너리가 아닙니다: {type(turn)}")
            continue
        
        # 각 턴의 키 목록 가져오기
        turn_keys = list(turn.keys())
        
        # 키가 2개가 아닌 경우 경고하고 건너뛰기
        if len(turn_keys) != 2:
            print(f"턴 {i}의 키 개수가 2개가 아닙니다 (발견된 키: {turn_keys})")
            continue
        
        # 첫 번째 키를 speaker로, 두 번째 키를 content로 간주
        speaker_key, content_key = turn_keys
        speaker = turn[speaker_key]
        content = turn[content_key]
        
        # speaker와 content가 모두 유효한지 확인
        if speaker and content and isinstance(speaker, str) and isinstance(content, str):
            # HTML 태그 제거 및 기타 정제
            cleaned_content = re.sub(r'<[^>]+>', '', content.strip())
            formatted_turns.append(f"{speaker.strip()}: {cleaned_content}")
        else:
            print(f"턴 {i} 데이터 형식 오류:")
            print(f"  - {speaker_key}: {speaker} (type: {type(speaker)})")
            print(f"  - {content_key}: {content} (type: {type(content)})")
    
    if not formatted_turns:
        return "[유효한 대화 턴 없음]"
        
    return "\n".join(formatted_turns)

def find_persona(i, folder_path="."):
    # 폴더 경로와 파일 패턴 결합
    file_pattern = os.path.join(folder_path, f"Persona_{i}_*.json")
    matching_files = glob.glob(file_pattern)
    
    # 파일이 있으면 첫 번째 파일 로드
    if matching_files:
        print(matching_files)
        return matching_files[0]
    else:
        raise FileNotFoundError(f"{file_pattern} 파일을 찾을 수 없습니다.")
        return None
    

def save_response_to_file(output_path: str, data: Dict[str, Any], family_id: int) -> None:
    """생성된 대화 데이터를 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 메타데이터 작성
        f.write(f"# Family Id: {family_id}\n")
        # JSON 데이터 저장
        json.dump(data, f, ensure_ascii=False, indent=2)
def save_dialogue_to_file(output_path: str, dialogue_data: Dict[str, Any], category_id: str, plan_id: str, examples: str) -> None:
    """생성된 대화 데이터를 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 메타데이터 작성
        f.write(f"# 카테고리: {category_id}\n")
        f.write(f"# 플랜: {plan_id}\n")
        f.write("# 사용된 예시 문장:\n")
        
        # examples가 문자열일 경우 각 줄을 처리
        if isinstance(examples, str):
            clean_examples = examples.strip().split('\n')
            for example in clean_examples:
                if example.strip():  # 빈 줄 제외
                    f.write(f"{example}\n")
        f.write("\n")
        
        # JSON 데이터 저장
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)