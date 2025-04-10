import os
import random
from typing import List, Dict, Any, Optional, Tuple
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass
from dotenv import load_dotenv
import time
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from src.llm_client import create_llm_client
from src.llm_interface import call_llm_and_parse_json
from src.logger import save_llm_prompts_to_txt, call_save_results
from src.prep_map_survey import FamilyPersona
from src.prompt import gen_prompt_category_info,  generate_category_from_survey, gen_dialogue_prompt, prep_survey_info
from src.prep_map_category import category_name_english, category_id, plan_name_english, plan_id, create_category_id_mappings
from src.prep import  load_data_from_cfg, sanitize_model_name, create_scoring_mappings

import json
import shutil
#load_and_parse_json_data, levenshtein_distance,save_dialogue_to_file
# .env 파일에서 환경 변수 불러오기
load_dotenv()
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


@hydra.main(config_path="conf", config_name="config_with_categories")
def main(cfg: AppConfig) -> None:
    """메인 함수: 데이터 로드, 플랜 선택, 대화 생성을 수행합니다."""
    print(f"구성: {OmegaConf.to_yaml(cfg)}")
    
    from datetime import datetime
    timestamp = datetime.now().timestamp()
    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')
    model_name = sanitize_model_name(cfg.model_name)
    name_param = f"Data_gen_{model_name}_{timestamp}"

    cwd = Path(hydra.utils.get_original_cwd())
    data_prep_path = cwd / Path('data/prep') 
    data_path = data_prep_path / Path('preped_category.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    data_plan_path = data_prep_path / Path('preped_plan.csv') 
    df_plan = pd.read_csv(data_plan_path, encoding='utf-8-sig')
    map_plan = df_plan[[plan_id, plan_name_english]]
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

    # prompt_category_info = generate_llm_prompts(df) # 함수명 변경 고려 (이제 프롬프트 전체가 아님)
    
    output_path = cwd / Path("outputs") / Path(name_param) 
    
    output_persona_path = output_path / Path("1_persona")
    output_dialogue_path = output_path / Path("2_dialogue")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_persona_path, exist_ok=True)
    os.makedirs(output_dialogue_path, exist_ok=True)

    prompt_path = output_path / Path("prompt")
    os.makedirs(prompt_path, exist_ok=True)
    
    shutil.copy(cwd / Path('conf') / Path('config_with_categories.yaml'), output_path)
    
    # LLM API를 사용하도록 설정
    llm_client = create_llm_client(cfg)
    if llm_client is None:
        raise ValueError("LLM 클라이언트를 초기화할 수 없습니다.") 
    else:
        print("LLM 클라이언트 초기화 완료.")

    csv_path = cwd / "data" / "VirtualSurvey.csv"
    family = FamilyPersona(csv_path=csv_path)
    df_survey  =  pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # for i, json_file in tqdm(enumerate(json_files)):
    prompt_category_info = gen_prompt_category_info(df) # 함수명 변경 고려 (이제 프롬프트 전체가 아님)
    for i in tqdm(range(family.get_persona_count())):
        index_persona = f"Persona_{i}"
        print(f"\n================================{index_persona}================================")
        i_df= df_survey.iloc[i,:]
        i_family = family.get_persona_data(i)
        child_age = i_df['아이 연령']

        df_category = df[df['min_month'] <= child_age]
        df_category = df_category[df_category['max_month'] >= child_age]
        print(f"연령에 적합한 Category id 목록: {df_category[category_id]}")
        
    #for i in tqdm(range(family.get_persona_count())):
        prompt_family = generate_category_from_survey(i_family, prompt_category_info, child_age, top_k=cfg.prompt.dialogue.top_k)
        processed_survey  = prep_survey_info(i_family, top_k=cfg.prompt.dialogue.top_k)

        prompt_category_path = os.path.join(prompt_path, f"{index_persona}_Step_1_{name_param}_prompt.txt")
        save_llm_prompts_to_txt(prompt_family, prompt_category_path)
        i_persona = i_family.get("persona")
        try:
            i_persona = i_persona[0]
        except:
            i_persona = i_persona
        
        # LLM을 사용하여 카테고리 ID 생성 + 키 검증
        expected_category_keys = [category_id, "explanation", "persona"] # 키 이름 변경: category_index -> category_id, "child", "parent", "age", "gender", "personality"
        family_data = call_llm_and_parse_json(prompt_family, llm_client, expected_keys=expected_category_keys)
        
        if family_data is None:
            print(f"Family {i}: 가족 데이터 생성 최종 실패. 다음 가족으로 넘어갑니다.")
            continue 
        
        selected_category_id = family_data.get("category_id") # 키 이름 변경: category_index -> category_id
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
        
        print(f"Child age: {child_age if child_age is not None else '정보 없음'}")
        child_age = int(child_age)  # NumPy int64를 Python int로 변환
        # 카테고리 ID 유효성 검사 (매핑에 존재하는지 확인)
        if selected_category_id is None or not isinstance(selected_category_id, int) or selected_category_id not in category_id_to_name:
            print(f"Family {i}: LLM이 유효하지 않은 카테고리 ID 반환 '{selected_category_id}' (타입: {type(selected_category_id)}). 다음 가족으로 넘어갑니다.")
            print(f"유효한 ID 목록: {list(category_id_to_name.keys())}")
            print(f"LLM 응답: {family_data}")
            # 결과 저장 (오류 분석용)
            output_file = os.path.join(output_persona_path,  f"{index_persona}_Step_1_Cat_INVALID_{selected_category_id}_{name_param}")
            #save_response_to_file(output_file, family_data, i)
            family_data ={"persona_given": i_persona, "age": child_age, "category_id": selected_category_id, "explanation": family_data.get("explanation", ""), "persona": family_data.get("persona", {})}
            call_save_results(family_data, output_file)
            continue 
            
        # 유효한 ID를 사용하여 실제 카테고리 이름(문자열) 가져오기
        i_category_name = category_id_to_name[selected_category_id] 
        print(f"====================Family Data (ID: {selected_category_id})==================={family_data}")
        print(f"\n=== 카테고리: {i_category_name} (ID: {selected_category_id}) ===")
        
        # 결과 저장 (정상 처리 시)
        output_file = os.path.join(output_persona_path,  f"{index_persona}_Step_1_Cat_{selected_category_id}_{name_param}")
        #save_response_to_file(output_file, family_data, i)

        family_data = {"persona_id": i, "persona": i_persona, "age": child_age, **family_data}
        call_save_results(family_data, output_file)
        # prep_map_category 임포트 위치 변경 (필요 시점)
        try:
            # df_plan 필터링 시 category_id 대신 category_name 사용
            df_plans = df_plan[df_plan[category_name_english]==i_category_name]
        
            df_plans = df_plans[df_plans['min_month'] <= child_age]
            df_plans = df_plans[df_plans['max_month'] >= child_age]
            plans = df_plans[plan_name_english].tolist() #plans_category.get(category_id, [])
            print(f"====Plans: {plans}")
            n_plan = cfg.prompt.dialogue.n_plan
            if n_plan == -1:
                n_plan = len(plans)
            plans = random.sample(plans, min(len(plans), n_plan))
            print(f"====Selection parameter: {n_plan} plans")
            for plan in plans:
                print(f"\n=={plan}' 플랜을 찾았습니다.")
                index_plan = int(map_plan[map_plan[plan_name_english] == plan][plan_id].values[0])

                index_plan_dialogue = f"{index_persona}_Step_2_dialogue_Cat_{selected_category_id}_Plan_{index_plan}_{name_param}"
            
                # 해당 플랜의 데이터 필터링
                i_df_plans = df_plans[df_plans[plan_name_english] == plan]
           
                # 심리 상담사 프롬프트 생성 (category_id는 이미 문자열로 변환됨)
                prompt, examples = gen_dialogue_prompt(i_persona, processed_survey, i_category_name, i_df_plans, child_persona, parent_persona, is_keywords=cfg.prompt.dialogue.is_keywords, is_description=cfg.prompt.dialogue.is_description, is_example=cfg.prompt.dialogue.is_example)
                prompt_dialouge_path = os.path.join(prompt_path, f"{index_plan_dialogue}_prompt.txt")
                save_llm_prompts_to_txt(prompt, prompt_dialouge_path)
                
                print(f"Prompting: {prompt}\n")
                
                # LLM을 사용하여 대화 생성 + 키 검증
                expected_dialogue_keys = ["category", "plan", "explanation", "dialogue"]
                dialogue_data = call_llm_and_parse_json(prompt, llm_client, expected_keys=expected_dialogue_keys)
                
                if dialogue_data:
                    # print(f"생성된 대화 데이터:")
                    # print(json.dumps(dialogue_data, ensure_ascii=False, indent=2))
                    
                    output_file = os.path.join(output_dialogue_path, f"{index_plan_dialogue}")
                    #save_dialogue_to_file(output_file, dialogue_data, i, i_persona, examples)
                
                    dialogue_data = {"persona_id": i, "persona": i_persona, "age": dialogue_data['age'],
                    "explanation": dialogue_data['explanation'],
                    "category": dialogue_data['category'],
                    "plan": dialogue_data['plan'],
                    "examples": examples,
                    "dialogue": dialogue_data['dialogue']} 
                    call_save_results(dialogue_data, output_file)
                    print(f"\n대화가 '{output_file}'에 저장되었습니다.")
                else:
                    print("대화 생성에 실패했습니다.")
                    continue # 대화 생성 실패 시 다음 plan으로 넘어감 (스코어링 건너뛰기)
        except Exception as e:
            print(f"Error: {e}")
            continue
if __name__ == "__main__":
    main()
# - 이와 같은 코칭이 도움이 되었던 사례들을 천천히 떠올려 보고, 개선 되기 전의 가족 대화를 생성해 주세요.
