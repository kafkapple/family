import os
import random
import json
from typing import List, Dict, Any, Optional
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
from prompt import  create_counselor_family_prompt, create_survey_persona_prompt, generate_llm_prompts
from persona_parser import PriorityInterest, FamilyPersonaInfo, FamilyPersonaParser
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
def extract_examples_from_plan(df_plan_row, cols_examples):
    """
    플랜의 예시 문장들을 깔끔하게 추출합니다.
    
    Args:
        df_plan_row (pd.Series): 플랜 데이터가 있는 DataFrame의 행
        cols_examples (list): example_ 로 시작하는 컬럼명 리스트
    
    Returns:
        list: 추출된 예시 문장들의 리스트
    """
    examples = []
    for col in cols_examples:
        example = df_plan_row[col].iloc[0]  # 첫 번째 행의 값만 가져옴
        if isinstance(example, str) and example != '-':  # 문자열이고 '-'가 아닌 경우만
            # 리스트 형태의 문자열에서 실제 문장만 추출
            clean_example = example.strip("[]'").replace("'", "")
            if clean_example:  # 빈 문자열이 아닌 경우만
                examples.append(clean_example)
    return examples

def load_yaml_data(file_path: str) -> Dict[str, Any]:
    """YAML 파일에서 데이터를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
def parse_json_response(response: str) -> Dict[str, Any]:
    """LLM의 응답에서 JSON 부분을 추출하고 파싱합니다."""
    try:
        # 응답이 None이거나 빈 문자열인 경우 처리
        if not response:
            print("응답이 비어있습니다.")
            return None
            
        # 정규 표현식을 사용하여 JSON 객체 추출
        match = re.search(r"\{[\s\S]*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                # JSON 문자열 정리 시도
                cleaned_json = re.sub(r'[\n\r\t]', '', json_str)
                return json.loads(cleaned_json)
        else:
            print("응답에서 JSON 형식을 찾을 수 없습니다.")
            print(f"원본 응답: {response}")
            return None
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        print(f"원본 응답: {response}")
        return None

def format_dialogue_for_scoring(dialogue_data: Dict[str, Any]) -> str:
    """대화 데이터를 스코어링용 포맷으로 변환합니다."""
    try:
        if not dialogue_data:
            print("대화 데이터가 비어있습니다")
            return "[대화 데이터 비어있음]"
            
        if not isinstance(dialogue_data, dict):
            print(f"대화 데이터 타입 오류: {type(dialogue_data)}")
            return "[대화 데이터 형식 오류]"
            
        dialogue = dialogue_data.get('dialogue')
        if not dialogue:
            print("dialogue 필드가 없거나 비어있습니다")
            print(f"사용 가능한 키: {dialogue_data.keys()}")
            return "[대화 데이터 비어있음]"
            
        if not isinstance(dialogue, list):
            print(f"dialogue 필드 타입 오류: {type(dialogue)}")
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

@hydra.main(config_path="conf", config_name="config_with_categories")
def main(cfg: AppConfig) -> None:
    """메인 함수: 데이터 로드, 플랜 선택, 대화 생성을 수행합니다."""
    print(f"구성: {OmegaConf.to_yaml(cfg)}")
    

    from datetime import datetime
    timestamp = datetime.now().timestamp()
    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
    model_name = sanitize_model_name(cfg.model_name)

    cwd = Path(hydra.utils.get_original_cwd())
    data_prep_path = cwd / Path('data/prep') 
    data_path = data_prep_path / Path('preped_category.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    data_plan_path = data_prep_path / Path('preped_plan.csv') 
    df_plan = pd.read_csv(data_plan_path, encoding='utf-8-sig')

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
    cols_examples = [col for col in df_plan.columns if 'example_' in col]
    from map_survey import FamilyPersona
    csv_path = cwd / "data" / "VirtualSurvey.csv"
    family = FamilyPersona(csv_path=csv_path)
    for i in tqdm(range(family.get_persona_count())):
        print(f"\n================================Family {i}================================")

        i_family = family.get_persona_data(i)
         # 가족 정보 파싱
        # parser = FamilyPersonaParser()
        # family_info = parser.parse_family_info(i_family)
        
        # # 컨텍스트가 포함된 프롬프트 생성
        # context = parser.create_context_prompt(family_info)
        
        prompt_family = create_survey_persona_prompt(i_family, prompt_category)

        prompt_category_path = os.path.join(prompt_path, f"1_prompt_category_{i}.txt")
        save_llm_prompts_to_txt(prompt_family, prompt_category_path)


    # LLM을 사용하여 대화 생성
        family_data = call_llm_and_parse_json(prompt_family, llm_client)
        
        # JSON 응답 파싱 및 유효성 검사
        if family_data is None:
            print(f"Family {i}: 가족 데이터 생성 최종 실패 (JSON 파싱/LLM 응답 없음). 다음 가족으로 넘어갑니다.")
            continue # 다음 루프 반복으로 이동

        category_id = family_data.get("category")
        valid_categories = df['CategoryName'].unique().tolist()

        if not category_id or category_id not in valid_categories:
            print(f"Family {i}: LLM이 유효하지 않은 카테고리 반환 '{category_id}'. 다음 가족으로 넘어갑니다.")
            print(f"유효한 카테고리 목록: {valid_categories}")
            print(f"LLM 응답: {family_data}")
            continue # 다음 루프 반복으로 이동
            
        print(f"====================Family Data==================={family_data}")
        
        # category_id = family_data.get("category", '') # 위에서 이미 처리
        print(f"\n=== 카테고리: {category_id} ===")
        
        output_file = os.path.join(output_family_path, f"1_family_data_survey_{i}_cat_{category_id}_{model_name}_{timestamp}.json")
        save_response_to_file(output_file, family_data, i)
        try:
            i_df_plans = df_plan[df_plan['CategoryName']==category_id]
        
            plans = i_df_plans['PlanName'].tolist() #plans_category.get(category_id, [])
            print(f"====Plans: {plans}")
            for plan in plans:
                print(f"\n=={plan}' 플랜을 찾았습니다.")
 
                # 해당 플랜의 데이터 필터링
                ii_df_plans = i_df_plans[i_df_plans['PlanName'] == plan]
           
                # examples 생성 부분
                example_list = extract_examples_from_plan(ii_df_plans, cols_examples)
                examples = [f"- {example}" for example in example_list]
                examples = "# Examples\n" + "\n".join(examples)
  
                print(f"Examples:\n{examples}")

                # 심리 상담사 프롬프트 생성
                prompt = create_counselor_family_prompt(family_data, plan, examples)
                prompt_dialouge_path = os.path.join(prompt_path, f"2_prompt_dialogue_{i}_cat_{category_id}_{model_name}_{timestamp}.txt")
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
                    
                    output_file = os.path.join(output_dir, f"2_survey_{i}_plan_{plan}_{model_name}_{timestamp}.json")
                    save_dialogue_to_file(output_file, dialogue_data, category_id, plan, examples)
                    # output_file = os.path.join(output_dir, f"survey_{i}_cat_{category_id}_plan_{plan_name}_Prompt_{plan.get('id')}_{model_name}_{timestamp}.json")
                    # save_dialogue_to_file(output_file, prompt, category_id, plan.get('id', ''), examples)
                    
                    print(f"\n대화가 '{output_file}'에 저장되었습니다.")
                else:
                    print("대화 생성에 실패했습니다.")
                    continue # 대화 생성 실패 시 다음 plan으로 넘어감 (스코어링 건너뛰기)

                # --- 스코어링 로직 시작 --- (for plan 루프 안으로 이동)
                from prompt import get_scoring_prompt
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
                            f"3_prompt_scoring_{i}_cat_{category_id}_plan_{plan}_{model_name}_{timestamp}.txt" # plan 변수 사용
                        )
                        save_llm_prompts_to_txt(prompt, prompt_scoring_path)
                        print(f"스코어링 프롬프트 생성됨: {prompt_scoring_path}")
                        
                        # LLM 응답 받기
                        scoring_data = call_llm_and_parse_json(prompt, llm_client)
                        if scoring_data:
                            output_file = os.path.join(
                                output_dir, 
                                f"3_scoring_{i}_plan_{plan}_{model_name}_{timestamp}.json" # plan 변수 사용
                            )
                            save_response_to_file(output_file, scoring_data, i)
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
        
        except Exception as e:
            print(f"카테고리 '{category_id}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
# - 이와 같은 코칭이 도움이 되었던 사례들을 천천히 떠올려 보고, 개선 되기 전의 가족 대화를 생성해 주세요.
