import os
import random
import json
from typing import List, Dict, Any, Optional, Tuple
import yaml
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass
import requests
from dotenv import load_dotenv
import re
import time
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from src.llm_client import create_llm_client
from src.llm_interface import get_llm_response
from src.logger import save_llm_prompts_to_txt, parse_and_save_json_results
from src.prep_map_category import level3_to_english
from src.prep_map_survey import FamilyPersona
from src.prompt import  generate_category_from_survey, generate_llm_prompts, create_counselor_family_prompt, get_scoring_prompt, get_category_prompt, get_plan_prompt, gen_plan_info, generate_category_info_only
# .env 파일에서 환경 변수 불러오기
load_dotenv()

def sanitize_model_name(model: str) -> str:
    """Clean model name for use as file name"""
    return re.sub(r'[^\w\-\.]', '_', model.lower())

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


# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

@dataclass
class AppConfig:
    yaml_path: str = "plans.yaml"
    llm_provider: str = "ollama" # 'openai' 또는 'ollama'
    model_name: str = "gemma3" # OpenAI: 'gpt-3.5-turbo', 'gpt-4' / Ollama: 'llama2', 'mistral' 등
    temperature: float = 0.7
    max_tokens: int = 2000
    example_count: int = 3  # 선택할 예시 문장 수
    categories: Optional[List[str]] = None  # 처리할 카테고리 목록 (None이면 전체 처리)

cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig)

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

def format_dialogue_for_scoring(dialogue_data: Dict[str, Any]) -> str:
    """대화 데이터를 스코어링용 포맷으로 변환합니다. (키 오타 및 간단한 정제 처리, 상세 로깅 추가)"""
    try:
        if not dialogue_data:
            print("대화 데이터가 비어있습니다")
            return "[대화 데이터 비어있음]"
            
        if not isinstance(dialogue_data, dict):
            print(f"대화 데이터 타입 오류: {type(dialogue_data)}")
            return "[대화 데이터 형식 오류]"
            
        # 다양한 키 변형 시도 ('dialogue', 'dialouge', 'dialgues', 'diallogue')
        dialogue = None
        possible_dialogue_keys = ['dialogue', 'dialouge', 'dialgues', 'diallogue']
        for key in possible_dialogue_keys:
            dialogue = dialogue_data.get(key)
            if dialogue is not None:
                if key != 'dialogue':
                    print(f"주의: 'dialogue' 키 대신 '{key}' 키를 사용했습니다.")
                break

        if not dialogue:
            print(f"dialogue 필드를 찾을 수 없거나 비어있습니다. 사용 가능한 키: {list(dialogue_data.keys())}")
            return "[대화 데이터 비어있음]"
            
        if not isinstance(dialogue, list):
            print(f"dialogue 필드 타입 오류: {type(dialogue)}")
            return "[대화 턴 데이터 형식 오류]"
            
        formatted_turns = []
        for i, turn in enumerate(dialogue):
            try:
                if isinstance(turn, dict):
                    # Speaker 키 처리 (다양한 변형 시도)
                    speaker = None
                    possible_speaker_keys = ['speaker', 'Speaker', 'singer'] 
                    for key in possible_speaker_keys:
                        speaker = turn.get(key)
                        if speaker:
                            break
                    
                    # Content 키 처리 (다양한 변형 시도)
                    content = None
                    possible_content_keys = ['content', 'comment']
                    for key in possible_content_keys:
                        content = turn.get(key)
                        if content:
                            break

                    # 유효성 검사 및 로깅 강화
                    speaker_valid = speaker and isinstance(speaker, str) and speaker.strip()
                    content_valid = content and isinstance(content, str) and content.strip()

                    if speaker_valid and content_valid:
                        # 간단한 HTML 태그 제거 (<a>, <b> 등) 및 앞뒤 공백 제거
                        cleaned_content = re.sub(r'<[^>]+>', '', content).strip()
                        # 깨진 문자나 제어 문자 등 추가 정제 필요시 여기에 추가 가능
                        formatted_turns.append(f"{speaker.strip()}: {cleaned_content}")
                    else:
                        # 실패 원인 구체적 로깅
                        reason = []
                        if not speaker_valid:
                            reason.append(f"speaker 누락/비거나 문자열 아님 (값: '{speaker}', 타입: {type(speaker)})")
                        if not content_valid:
                            reason.append(f"content 누락/비거나 문자열 아님 (값: '{content}', 타입: {type(content)})")
                        print(f"턴 {i} 처리 실패: { ' / '.join(reason) }")
                        print(f"   원본 턴 데이터: {turn}")
                else:
                    print(f"턴 {i} 타입 오류: {type(turn)}")
            except Exception as e:
                print(f"턴 {i} 처리 중 오류: {e}")
                continue
                
        if not formatted_turns:
            print("포맷팅된 대화 턴이 없습니다")
            return "[대화 데이터 비어있음]"
                
        return "\n".join(formatted_turns)
        
    except Exception as e:
        print(f"대화 포맷 변환 중 오류 발생: {e}")
        print(f"원본 데이터: {dialogue_data}")
        return "[대화 데이터 처리 오류]"

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

def call_llm_and_parse_json(prompt: str, llm_client: Any, max_retries: int = 7, expected_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """LLM을 호출하고 응답을 JSON으로 파싱합니다. 실패 시 재시도하고, 성공 시 키를 검증/수정합니다."""
    retries = 0
    while retries < max_retries:
        response = get_llm_response(prompt, llm_client)
        if not response:
            print(f"LLM 응답 없음. 재시도 ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(1) 
            continue

        parsed_data = parse_json_response(response)
        if parsed_data:
            # JSON 파싱 성공 후 키 검증 및 수정
            if expected_keys:
                try:
                    corrected_data = fix_json_keys(parsed_data, expected_keys)
                    return corrected_data
                except Exception as e_fix:
                    print(f"JSON 키 수정 중 오류 발생: {e_fix}")
                    print(f"원본 파싱 데이터: {parsed_data}")
                    # 키 수정 실패 시 파싱 실패로 간주하고 재시도 또는 최종 실패
                    # return parsed_data # 또는 수정 전 데이터 반환 선택
            else:
                 # expected_keys가 없으면 검증/수정 없이 반환
                 return parsed_data
        # else 블록은 파싱 실패 시 재시도 로직으로 이어짐
        print(f"JSON 파싱/키 수정 실패. 재시도 ({retries + 1}/{max_retries})...")
        if not parsed_data: # 파싱 자체가 실패한 경우만 로그 출력 (키 수정 실패는 위에서 로깅)
             print(f"실패한 응답: {response[:500]}...") 
        retries += 1
        time.sleep(1)
            
    print(f"최대 재시도 횟수({max_retries}) 도달. JSON 처리 최종 실패.")
    return None

# --- 매핑 함수 정의 --- 
def create_category_id_mappings(df_category: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    """카테고리 DataFrame에서 id_category <-> CategoryName_English 매핑을 생성합니다."""
    # CategoryName_English를 기준으로 중복 제거 후 매핑 생성
    df_unique_categories = df_category.drop_duplicates(subset=['CategoryName_English'])
    
    # id_category 컬럼이 정수형인지 확인 (필요시 타입 변환)
    if not pd.api.types.is_integer_dtype(df_unique_categories['id_category']):
        print("경고: id_category 컬럼이 정수형이 아닙니다. 변환을 시도합니다.")
        try:
            df_unique_categories['id_category'] = df_unique_categories['id_category'].astype(int)
        except ValueError as e:
            print(f"오류: id_category를 정수형으로 변환할 수 없습니다: {e}")
            raise
            
    category_id_to_name = pd.Series(
        df_unique_categories.CategoryName_English.values, 
        index=df_unique_categories.id_category
    ).to_dict()
    
    category_name_to_id = {name: id_ for id_, name in category_id_to_name.items()}
    
    print(f"카테고리 ID <-> 이름 매핑 생성 완료: {len(category_id_to_name)}개")
    # print(f"ID to Name: {category_id_to_name}") # 필요시 주석 해제
    return category_id_to_name, category_name_to_id

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
# --- JSON 키 검증/수정 함수 끝 --- 

@hydra.main(config_path="conf", config_name="config_with_categories")
def main(cfg: AppConfig) -> None:
    """메인 함수: 데이터 로드, 플랜 선택, 대화 생성을 수행합니다."""
    print(f"구성: {OmegaConf.to_yaml(cfg)}")
    
    from datetime import datetime
    timestamp = datetime.now().timestamp()
    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
    model_name = sanitize_model_name(cfg.model_name)
    name_param = f"{model_name}_{timestamp}"

    cwd = Path(hydra.utils.get_original_cwd())
    data_prep_path = cwd / Path('data/prep') 
    data_path = data_prep_path / Path('preped_category.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    data_plan_path = data_prep_path / Path('preped_plan.csv') 
    df_plan = pd.read_csv(data_plan_path, encoding='utf-8-sig')

    map_plan = df_plan[['id_plan', 'PlanName_English']]

    # --- 매핑 생성 (함수 호출, 변수명 변경) --- 
    category_id_to_name, category_name_to_id = create_category_id_mappings(df)
    
    # 스코어링 기준 로드 및 매핑 생성 (여기는 인덱스(0,1,2..) 사용 유지)
    scoring_items_dict = load_data_from_cfg(cfg, 'scoring_items', dict, {})
    scoring_criteria_list = []
    if scoring_items_dict:
        for item_id, item_data in scoring_items_dict.items():
            if isinstance(item_data, dict):
                scoring_criteria_list.append({'id': item_id, **item_data})
    scoring_index_to_id, scoring_id_to_index = create_scoring_mappings(scoring_criteria_list)
    # --- 매핑 생성 완료 ---

    prompt_category_info = generate_llm_prompts(df) # 함수명 변경 고려 (이제 프롬프트 전체가 아님)
    category_info_only = generate_category_info_only(df)
    output_path = cwd / Path("outputs") / Path(model_name) 
    
    output_persona_path = cwd / Path("outputs") / Path(model_name) / Path("1_persona")
    output_dialogue_path = cwd / Path("outputs") / Path(model_name) / Path("2_dialogue")
    output_scoring_path = cwd / Path("outputs") / Path(model_name) / Path("3_scoring")
    output_category_path = cwd / Path("outputs") / Path(model_name) / Path("4_category")
    output_final_path = cwd / Path("outputs") / Path(model_name) / Path("5_final_plan")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_dialogue_path, exist_ok=True)
    os.makedirs(output_scoring_path, exist_ok=True)
    os.makedirs(output_persona_path, exist_ok=True)
    os.makedirs(output_category_path, exist_ok=True)
    os.makedirs(output_final_path, exist_ok=True)
    prompt_path = cwd / Path("outputs") / Path(model_name) / Path("prompt")
    os.makedirs(prompt_path, exist_ok=True)
    
    # LLM API를 사용하도록 설정
    llm_client = create_llm_client(cfg)
    if llm_client is None:
        raise ValueError("LLM 클라이언트를 초기화할 수 없습니다.") 
    else:
        print("LLM 클라이언트 초기화 완료.")

    csv_path = cwd / "data" / "VirtualSurvey.csv"
    family = FamilyPersona(csv_path=csv_path)
    df_scores = pd.DataFrame()
    df_survey  =  pd.read_csv(csv_path, encoding='utf-8-sig')
    for i in tqdm(range(family.get_persona_count())):
        index_persona = f"Persona_{i}"
        print(f"\n================================{index_persona}================================")

        i_family = family.get_persona_data(i)

        prompt_family = generate_category_from_survey(i_family, prompt_category_info)
        
        prompt_category_path = os.path.join(prompt_path, f"{index_persona}_Step_1_{name_param}_prompt.txt")
        save_llm_prompts_to_txt(prompt_family, prompt_category_path)
        i_persona = i_family.get("persona")

        # LLM을 사용하여 카테고리 ID 생성 + 키 검증
        expected_category_keys = ["category_id", "explanation", "persona"] # 키 이름 변경: category_index -> category_id, "child", "parent", "age", "gender", "personality"
        family_data = call_llm_and_parse_json(prompt_family, llm_client, expected_keys=expected_category_keys)
        
        if family_data is None:
            print(f"Family {i}: 가족 데이터 생성 최종 실패. 다음 가족으로 넘어갑니다.")
            continue 
        
        selected_category_id = family_data.get("category_id") # 키 이름 변경: category_index -> category_id
        # family_persona = family_data.get("persona") # 수정 전
        # child_persona = family_persona.get("child") # 수정 전
        # parent_persona = family_persona.get("parent") # 수정 전
        # child_age = child_persona.get("age") # 수정 전
        
        # 페르소나 정보 가져오기 (NoneType 오류 방지)
        family_persona = family_data.get("persona")
        if family_persona is None or not isinstance(family_persona, dict):
            print(f"경고: Family {i}의 LLM 응답에서 유효한 'persona' 딕셔너리를 찾을 수 없습니다. 기본값을 사용합니다.")
            print(f"   LLM 응답 (family_data): {family_data}")
            child_persona = {} # 기본 빈 딕셔너리
            parent_persona = {} # 기본 빈 딕셔너리
        else:
            # persona가 존재하면 child/parent 리스트 가져오기 시도
            child_list = family_persona.get("child", [])
            parent_list = family_persona.get("parent", [])
            # 리스트가 비어있지 않으면 첫 번째 요소 사용, 아니면 빈 딕셔너리
            child_persona = child_list[0] if isinstance(child_list, list) and child_list else {}
            parent_persona = parent_list[0] if isinstance(parent_list, list) and parent_list else {}

        # 자녀 나이 가져오기 (이제 child_persona는 항상 딕셔너리)
        child_age = child_persona.get("age")
        
        print(f"Child age: {child_age if child_age is not None else '정보 없음'}")
        # 카테고리 ID 유효성 검사 (매핑에 존재하는지 확인)
        if selected_category_id is None or not isinstance(selected_category_id, int) or selected_category_id not in category_id_to_name:
            print(f"Family {i}: LLM이 유효하지 않은 카테고리 ID 반환 '{selected_category_id}' (타입: {type(selected_category_id)}). 다음 가족으로 넘어갑니다.")
            print(f"유효한 ID 목록: {list(category_id_to_name.keys())}")
            print(f"LLM 응답: {family_data}")
            # 결과 저장 (오류 분석용)
            output_file = os.path.join(output_persona_path,  f"{index_persona}_Step_1_Cat_INVALID_{selected_category_id}_{name_param}.json")
            save_response_to_file(output_file, family_data, i)
            continue 
            
        # 유효한 ID를 사용하여 실제 카테고리 이름(문자열) 가져오기
        category_name = category_id_to_name[selected_category_id] 
        print(f"====================Family Data (ID: {selected_category_id})==================={family_data}")
        print(f"\n=== 카테고리: {category_name} (ID: {selected_category_id}) ===")
        
        # 결과 저장 (정상 처리 시)
        output_file = os.path.join(output_persona_path,  f"{index_persona}_Step_1_Cat_{selected_category_id}_{name_param}.json")
        save_response_to_file(output_file, family_data, i)

        # prep_map_category 임포트 위치 변경 (필요 시점)
        
        try:
            # df_plan 필터링 시 category_id 대신 category_name 사용
            df_plans = df_plan[df_plan['CategoryName_English']==category_name]
        
            plans = df_plans['PlanName_English'].tolist() #plans_category.get(category_id, [])
            print(f"====Plans: {plans}")
            n_plan = 1
            plans = random.sample(plans, min(len(plans), n_plan))
            print(f"====Selection parameter: {n_plan} plans")
            for plan in plans:
                print(f"\n=={plan}' 플랜을 찾았습니다.")
                index_plan = int(map_plan[map_plan['PlanName_English'] == plan]['id_plan'].values[0])

                index_plan_dialogue = f"{index_persona}_Step_2_dialogue_Cat_{selected_category_id}_Plan_{index_plan}_{name_param}"
                index_plan_scoring = f"{index_persona}_Step_3_scoring_Cat_{selected_category_id}_Plan_{index_plan}_{name_param}"
                # 해당 플랜의 데이터 필터링
                i_df_plans = df_plans[df_plans['PlanName_English'] == plan]
           
                # 심리 상담사 프롬프트 생성 (category_id는 이미 문자열로 변환됨)
                prompt, examples = create_counselor_family_prompt(i_persona, category_name, i_df_plans, child_persona, parent_persona)
                prompt_dialouge_path = os.path.join(prompt_path, f"{index_plan_dialogue}_prompt.txt")
                save_llm_prompts_to_txt(prompt, prompt_dialouge_path)
                
                print(f"Prompting: {prompt}\n")
                
                # LLM을 사용하여 대화 생성 + 키 검증
                expected_dialogue_keys = ["category", "plan", "explanation", "dialogue"]
                dialogue_data = call_llm_and_parse_json(prompt, llm_client, expected_keys=expected_dialogue_keys)
                
                if dialogue_data:
                    print(f"생성된 대화 데이터:")
                    print(json.dumps(dialogue_data, ensure_ascii=False, indent=2))
                    
                    
                    output_file = os.path.join(output_dialogue_path, f"{index_plan_dialogue}.json")
                    save_dialogue_to_file(output_file, dialogue_data, category_name, plan, examples)
             
                    print(f"\n대화가 '{output_file}'에 저장되었습니다.")
                else:
                    print("대화 생성에 실패했습니다.")
                    continue # 대화 생성 실패 시 다음 plan으로 넘어감 (스코어링 건너뛰기)

                # --- 스코어링 로직 시작 --- (for plan 루프 안으로 이동)
                
                scoring_items_dict = load_data_from_cfg(cfg, 'scoring_items', dict, {})
                scoring_criteria_list = []
                if scoring_items_dict:
                    for item_id, item_data in scoring_items_dict.items():
                        if isinstance(item_data, dict):
                            scoring_criteria_list.append({'id': item_id, **item_data})

                # 스코어링 부분 수정
                # if dialogue_data and "dialogue" in dialogue_data: # dialogue_data 존재는 위에서 확인
                # 대화 데이터 포맷팅
                formatted_dialogue = format_dialogue_for_scoring(dialogue_data)
                if formatted_dialogue and formatted_dialogue != "[대화 데이터 비어있음]":
                    scoring_template_str = load_template_str(cfg, 'output_formats.scoring.template')
                    if scoring_template_str:
                        prompt = get_scoring_prompt(
                            dialogue=formatted_dialogue,
                            scoring_criteria=scoring_criteria_list,
                            output_template_str=scoring_template_str
                        )
                        
                        # 디버깅을 위한 로깅 추가
                        print("\n=== 스코어링 프롬프트 ===")
                        # print(formatted_dialogue) # 너무 길어서 주석 처리, 필요 시 해제
                        print(f"스코어링 기준 개수: {len(scoring_criteria_list)}")
                        print("========================\n")
                        
                        prompt_scoring_path = os.path.join(
                            prompt_path, 
                            f"{index_plan_scoring}_prompt.txt" # plan 변수 사용
                        )
                        save_llm_prompts_to_txt(prompt, prompt_scoring_path)
                        print(f"스코어링 프롬프트 생성됨: {prompt_scoring_path}")
                        
                        # LLM 응답 받기 + 최상위 키 검증 (scoring, explanation)
                        expected_scoring_keys = ["scoring", "explanation"]
                        scoring_data = call_llm_and_parse_json(prompt, llm_client, expected_keys=expected_scoring_keys)

                        if scoring_data:
                            # 스코어링 내부 키는 인덱스 문자열이므로 별도 처리 필요
                            # fix_json_keys는 현재 숫자 문자열 키 수정에 적합하지 않을 수 있음
                            # 우선은 LLM이 정확한 인덱스 키를 반환한다고 가정하거나,
                            # 필요시 아래 scoring_data 처리 부분에서 직접 검증/수정 로직 추가
                            output_file = os.path.join(
                                output_scoring_path,
                                f"{index_plan_scoring}_scoring.json"
                            )
                            save_response_to_file(output_file, scoring_data, i)
                            dict_score = scoring_data.get('scoring')
                            dict_explanation = scoring_data.get('explanation')
                            dict_score = {scoring_index_to_id[int(key)]: value for key, value in dict_score.items() if key.isdigit() and int(key) in scoring_index_to_id}
                            dict_explanation = {'explanation_'+scoring_index_to_id[int(key)]: value for key, value in dict_explanation.items() if key.isdigit() and int(key) in scoring_index_to_id}
                            merged_dict = {**df_survey.iloc[i,:].to_dict(), **dict_score, **dict_explanation}
                            df_i = pd.DataFrame([merged_dict])
                            df_i['persona_index'] = i
                            df_i['category_id'] = selected_category_id
                            df_i['category_name'] = category_name
                            df_i['plan_index'] = index_plan
                            df_i['plan_name'] = plan
                            df_i['child_age'] = child_age
                            
                            df_scores = pd.concat([df_scores, df_i], ignore_index=True)
            
                            print(f"스코어링 결과 저장됨: {output_file}")
                        else:
                            print("스코어링 데이터 파싱 실패")
                    else:
                        print("스코어링 템플릿 로드 실패")
                else:
                    print("대화 데이터 포맷팅 실패")
                # else: # dialogue_data가 없거나 dialogue 키가 없는 경우는 위에서 continue로 처리됨
                #     print("대화 데이터가 비어있거나 'dialogue' 필드가 없음")
                # --- 스코어링 로직 끝 ---
                print("\n" + "="*50 + "\n") # 각 plan 처리 후 구분선 출력
          
            prompt = get_category_prompt(prompt_category=category_info_only,
                    dialogue=dialogue_data,
                    scoring_results=scoring_data, # Pass the original dictionary
    
                )
            prompt_category_path = os.path.join(prompt_path, f"{index_persona}_Step_4_Category_{name_param}_prompt.txt")
            save_llm_prompts_to_txt(prompt, prompt_category_path)
            # Step 4의 예상 최상위 키 수정
            expected_category_keys = ["selected_categories"] 
            category_data = call_llm_and_parse_json(prompt, llm_client, expected_keys=expected_category_keys)
            
            # category_data가 None이거나 필요한 키가 없는 경우 처리
            if not category_data or "selected_categories" not in category_data or not isinstance(category_data["selected_categories"], list):
                 print(f"Family {i}: Step 4 카테고리 추천 데이터 처리 실패. 다음 가족으로 넘어갑니다.")
                 print(f"LLM 응답 (category_data): {category_data}")
                 continue

            output_file = os.path.join(
                                output_category_path,
                                f"{index_persona}_Step_4_Category_{name_param}.json"
                            )
            save_response_to_file(output_file, category_data, i)
            print(f"카테고리 결과 저장됨: {output_file}")

            ####### 플랜 추천 로직 시작 #######
            # category_ids 생성 시 오류 처리 강화
            category_ids = []
            for item in category_data.get("selected_categories", []):
                cat_id_str = item.get("id")
                if cat_id_str is not None and isinstance(cat_id_str, (str, int)):
                    try:
                        cat_id_int = int(cat_id_str) # 정수 변환 시도
                        # 변환된 ID가 유효한 카테고리 ID인지 추가 확인 (선택 사항)
                        if cat_id_int in category_id_to_name:
                             category_ids.append(cat_id_int)
                        else:
                             print(f"경고: 추천된 카테고리 ID '{cat_id_int}'가 유효하지 않아 무시합니다.")
                    except ValueError:
                        print(f"경고: 추천된 카테고리 ID '{cat_id_str}'를 정수로 변환할 수 없어 무시합니다.")
                else:
                     print(f"경고: 유효하지 않은 카테고리 ID 항목 발견: {item}")
            
            if not category_ids:
                print(f"Family {i}: 유효한 추천 카테고리 ID를 찾을 수 없습니다. 플랜 추천을 건너뛰니다.")
                continue
            if len(category_ids) >3:
                category_ids = category_ids[:3]
            print(f"추천된 유효 카테고리 ID 목록: {category_ids}")
            plans_info = df_plan[df_plan['id_category'].isin(category_ids)]
            relevant_plans_info = gen_plan_info(plans_info)
            prompt_plan_path = os.path.join(prompt_path, f"{index_persona}_Step_5_Plan_{name_param}_prompt.txt")
            prompt = get_plan_prompt(categories=category_data, scoring_results=scoring_data, relevant_plans_info=relevant_plans_info)
            save_llm_prompts_to_txt(prompt, prompt_plan_path)
            plan_data = call_llm_and_parse_json(prompt, llm_client, expected_keys=['conversation_analysis', 'recommended_plans'])
            output_file = os.path.join(
                                output_final_path,
                                f"{index_persona}_Step_5_Plan_{name_param}.json"
                            )
            save_response_to_file(output_file, plan_data, i)
            print(f"플랜 추천 결과 저장됨: {output_file}")

        except Exception as e:
            print(f"카테고리 '{category_name}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    df_scores.to_csv(output_path / Path("survey_with_scoring.csv"), encoding='utf-8-sig', index=True)
    
if __name__ == "__main__":
    main()
# - 이와 같은 코칭이 도움이 되었던 사례들을 천천히 떠올려 보고, 개선 되기 전의 가족 대화를 생성해 주세요.
