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
from llm_client import create_llm_client
from llm_interface import get_llm_response
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from logger import save_llm_prompts_to_txt, parse_and_save_json_results
from prompt import  create_survey_persona_prompt, generate_llm_prompts, create_counselor_family_prompt, get_scoring_prompt
import time
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
    """대화 데이터를 스코어링용 포맷으로 변환합니다. (키 오타 처리 추가)"""
    try:
        if not dialogue_data:
            print("대화 데이터가 비어있습니다")
            return "[대화 데이터 비어있음]"
            
        if not isinstance(dialogue_data, dict):
            print(f"대화 데이터 타입 오류: {type(dialogue_data)}")
            return "[대화 데이터 형식 오류]"
            
        # 'dialogue' 키를 먼저 시도하고, 없으면 오타인 'dialouge' 시도
        dialogue = dialogue_data.get('dialogue')
        if dialogue is None:
            dialogue = dialogue_data.get('dialouge') # 오타 키 시도
            if dialogue is not None:
                print("주의: 'dialogue' 키 대신 'dialouge' 키를 사용했습니다.")
            else:
                # 세 번째로 복수형 오타 'dialgues' 시도
                dialogue = dialogue_data.get('dialgues') 
                if dialogue is not None:
                    print("주의: 'dialogue' 키 대신 복수형 오타 'dialgues' 키를 사용했습니다.")

        if not dialogue:
            print("dialogue/dialouge/dialgues 필드가 없거나 비어있습니다")
            print(f"사용 가능한 키: {dialogue_data.keys()}")
            return "[대화 데이터 비어있음]"
            
        if not isinstance(dialogue, list):
            print(f"dialogue/dialouge/dialgues 필드 타입 오류: {type(dialogue)}")
            return "[대화 턴 데이터 형식 오류]"
            
        formatted_turns = []
        for i, turn in enumerate(dialogue):
            try:
                if isinstance(turn, dict):
                    speaker = turn.get('speaker', '')
                    content = turn.get('content', '')
                    if speaker and content:
                        formatted_turns.append(f"{speaker}: {content}")
                    else:
                        print(f"턴 {i} 데이터 누락: {turn}")
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

def call_llm_and_parse_json(prompt: str, llm_client: Any, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """LLM을 호출하고 응답을 JSON으로 파싱합니다. 실패 시 재시도합니다."""
    retries = 0
    while retries < max_retries:
        response = get_llm_response(prompt, llm_client)
        if not response:
            print(f"LLM 응답 없음. 재시도 ({retries + 1}/{max_retries})...")
            retries += 1
            time.sleep(1) # 간단한 지연 추가
            continue

        parsed_data = parse_json_response(response)
        if parsed_data:
            return parsed_data
        else:
            print(f"JSON 파싱 실패. 재시도 ({retries + 1}/{max_retries})...")
            print(f"실패한 응답: {response[:500]}...") # 실패한 응답 일부 로깅
            retries += 1
            time.sleep(1) # 간단한 지연 추가
            
    print(f"최대 재시도 횟수({max_retries}) 도달. JSON 파싱 최종 실패.")
    return None

# --- 매핑 함수 정의 --- 
def create_category_mappings(df_category: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int]]:
    """카테고리 DataFrame에서 인덱스 <-> 이름 매핑을 생성합니다."""
    valid_categories = df_category['CategoryName_English'].unique().tolist()
    category_index_to_name = {i: name for i, name in enumerate(valid_categories)}
    category_name_to_index = {name: i for i, name in category_index_to_name.items()}
    print(f"카테고리 매핑 생성 완료: {len(valid_categories)}개 ({category_index_to_name})")
    return category_index_to_name, category_name_to_index

def create_scoring_mappings(scoring_criteria_list: List[Dict]) -> Tuple[Dict[int, str], Dict[str, int]]:
    """스코어링 기준 리스트에서 인덱스 <-> ID 매핑을 생성합니다."""
    scoring_index_to_id = {i: criteria['id'] for i, criteria in enumerate(scoring_criteria_list)}
    scoring_id_to_index = {criteria['id']: i for i, criteria in enumerate(scoring_criteria_list)}
    print(f"스코어링 기준 매핑 생성 완료: {len(scoring_criteria_list)}개 ({scoring_index_to_id})")
    return scoring_index_to_id, scoring_id_to_index
# --- 매핑 함수 정의 끝 --- 

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

    # --- 매핑 생성 (함수 호출) --- 
    category_index_to_name, category_name_to_index = create_category_mappings(df)
    
    # 스코어링 기준 로드 및 매핑 생성
    scoring_items_dict = load_data_from_cfg(cfg, 'scoring_items', dict, {})
    scoring_criteria_list = []
    if scoring_items_dict:
        for item_id, item_data in scoring_items_dict.items():
            if isinstance(item_data, dict):
                scoring_criteria_list.append({'id': item_id, **item_data})
    scoring_index_to_id, scoring_id_to_index = create_scoring_mappings(scoring_criteria_list)
    # --- 매핑 생성 완료 ---

    prompt_category = generate_llm_prompts(df)
    output_family_path = cwd / Path("outputs") / Path(model_name) / Path("survey") 

    os.makedirs(output_family_path, exist_ok=True)
    prompt_path = cwd / Path("outputs") / Path(model_name) / Path("prompt")
    os.makedirs(prompt_path, exist_ok=True)
    
    # LLM API를 사용하도록 설정
    llm_client = create_llm_client(cfg)
    if llm_client is None:
        raise ValueError("LLM 클라이언트를 초기화할 수 없습니다.") 
    else:
        print("LLM 클라이언트 초기화 완료.")

    # plans_category = data.get("plans", {})  # plans
    
    from prep_map_survey import FamilyPersona
    csv_path = cwd / "data" / "VirtualSurvey.csv"
    family = FamilyPersona(csv_path=csv_path)
    df_scores = pd.DataFrame()
    df_survey  =  pd.read_csv(csv_path, encoding='utf-8-sig')
    for i in tqdm(range(family.get_persona_count())):
        index_persona = f"Persona_{i}"
        print(f"\n================================{index_persona}================================")

        i_family = family.get_persona_data(i)

        prompt_family = create_survey_persona_prompt(i_family, prompt_category)
        
        prompt_category_path = os.path.join(prompt_path, f"{index_persona}_Step_1_{name_param}_prompt.txt")
        save_llm_prompts_to_txt(prompt_family, prompt_category_path)
        i_persona = i_family.get("persona")

    # LLM을 사용하여 카테고리 인덱스 생성
        family_data = call_llm_and_parse_json(prompt_family, llm_client)

        
        # JSON 응답 파싱 및 유효성 검사
        if family_data is None:
            print(f"Family {i}: 가족 데이터 생성 최종 실패 (JSON 파싱/LLM 응답 없음). 다음 가족으로 넘어갑니다.")
            continue # 다음 루프 반복으로 이동
        category_index = family_data.get("category_index") # 정수 인덱스 가져오기
        output_file = os.path.join(output_family_path, f"{index_persona}_Step_1_Cat_{category_index}_{name_param}.json")
        save_response_to_file(output_file, family_data, i)
        # valid_categories = df['CategoryName_English'].unique().tolist() # 매핑으로 대체

        # 카테고리 인덱스 유효성 검사
        if category_index is None or not isinstance(category_index, int) or category_index not in category_index_to_name:
            print(f"Family {i}: LLM이 유효하지 않은 카테고리 인덱스 반환 '{category_index}' (타입: {type(category_index)}). 다음 가족으로 넘어갑니다.")
            print(f"유효한 인덱스 범위: 0 ~ {len(category_index_to_name) - 1}")
            print(f"LLM 응답: {family_data}")
            continue # 다음 루프 반복으로 이동
            
        # 유효한 인덱스를 실제 카테고리 이름(문자열)으로 변환
        category_id = category_index_to_name[category_index] # 이제 category_id는 실제 이름(문자열)
        print(f"====================Family Data (Index: {category_index})==================={family_data}")
        
        # category_id = family_data.get("category", '') # 인덱스 방식으로 변경됨
        print(f"\n=== 카테고리: {category_id} (인덱스: {category_index}) ===")
        from prep_map_category import level3_to_english

        try:
            df_plans = df_plan[df_plan['CategoryName_English']==category_id]
        
            plans = df_plans['PlanName_English'].tolist() #plans_category.get(category_id, [])
            print(f"====Plans: {plans}")
            n_plan = 1
            plans = random.sample(plans, min(len(plans), n_plan))
            print(f"====Selection parameter: {n_plan} plans")
            for plan in plans:
                print(f"\n=={plan}' 플랜을 찾았습니다.")
                index_plan = int(map_plan[map_plan['PlanName_English'] == plan]['id_plan'].values[0])

                index_plan_dialogue = f"{index_persona}_Step_2_dialogue_Cat_{category_index}_Plan_{index_plan}_{name_param}"
                index_plan_scoring = f"{index_persona}_Step_3_scoring_Cat_{category_index}_Plan_{index_plan}_{name_param}"
                # 해당 플랜의 데이터 필터링
                i_df_plans = df_plans[df_plans['PlanName_English'] == plan]
           
                # 심리 상담사 프롬프트 생성 (category_id는 이미 문자열로 변환됨)
                prompt, examples = create_counselor_family_prompt(i_persona, category_id, i_df_plans)
                prompt_dialouge_path = os.path.join(prompt_path, f"{index_plan_dialogue}_prompt.txt")
                save_llm_prompts_to_txt(prompt, prompt_dialouge_path)
                
                print(f"Prompting: {prompt}\n")
                
                # LLM을 사용하여 대화 생성
                dialogue_data = call_llm_and_parse_json(prompt, llm_client)
                
                if dialogue_data:
                    print(f"생성된 대화 데이터:")
                    print(json.dumps(dialogue_data, ensure_ascii=False, indent=2))
                    
                    # 생성된 대화를 파일에 저장
                    output_dir = os.path.join(output_family_path, category_id)
                    
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_file = os.path.join(output_dir, f"{index_plan_dialogue}.json")
                    save_dialogue_to_file(output_file, dialogue_data, category_id, plan, examples)
                    # output_file = os.path.join(output_dir, f"survey_{i}_cat_{category_id}_plan_{plan_name}_Prompt_{plan.get('id')}_{model_name}_{timestamp}.json")
                    # save_dialogue_to_file(output_file, prompt, category_id, plan.get('id', ''), examples)
                    
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
                        
                        # LLM 응답 받기
                        scoring_data = call_llm_and_parse_json(prompt, llm_client)
                        if scoring_data:
                            output_file = os.path.join(
                                output_dir, 
                                f"{index_plan_scoring}_scoring.json" # plan 변수 사용
                            )
                            save_response_to_file(output_file, scoring_data, i)
                            dict_score = scoring_data.get('scoring')
                            dict_explanation = scoring_data.get('explanation')
                            dict_score = {scoring_index_to_id[int(key)]: value for key, value in dict_score.items()}
                            dict_explanation = {'explanation_'+scoring_index_to_id[int(key)]: value for key, value in dict_explanation.items()}
                            merged_dict = {**df_survey.iloc[i,:].to_dict(), **dict_score, **dict_explanation}

                
                            df_scores = pd.concat([df_scores, pd.DataFrame([merged_dict])], ignore_index=True)

            
                            family
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
                df_scores.to_csv(os.path.join(output_family_path, f"{index_persona}_scores.csv"), encoding='utf-8-sig', index=True)

        
        except Exception as e:
            print(f"카테고리 '{category_id}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        df_scores.to_csv(os.path.join(output_family_path, f"scores.csv"), encoding='utf-8-sig', index=True)
    
if __name__ == "__main__":
    main()
# - 이와 같은 코칭이 도움이 되었던 사례들을 천천히 떠올려 보고, 개선 되기 전의 가족 대화를 생성해 주세요.
