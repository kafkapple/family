import os
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
from src.logger import save_llm_prompts_to_txt, call_save_results #save_response_to_file
from src.prep_map_survey import FamilyPersona
from src.prompt import get_scoring_prompt, get_category_prompt, get_plan_prompt, gen_plan_info, generate_category_info_only, prep_survey_info
from src.prep_map_category import plan_id, plan_name, category_id, category_name_english, create_category_id_mappings
from src.prep import load_data_from_cfg, load_template_str, sanitize_model_name, create_scoring_mappings, reconstruct_dialogue, find_persona
from src.evaluation import call_metric
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
    name_param = f"{model_name}_{timestamp}"

    cwd = Path(hydra.utils.get_original_cwd())
    data_prep_path = cwd / Path('data/prep') 
    data_path = data_prep_path / Path('preped_category.csv')
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df_category = df
    data_plan_path = data_prep_path / Path('preped_plan.csv') 
    df_plan = pd.read_csv(data_plan_path, encoding='utf-8-sig')

    if cfg.data.exclude_plans == 'intro':
        print(f"\n===================intro 플랜 제외===================")
        df_plan = df_plan[df_plan[category_name_english] != 'intro']
        df_category = df_category[df_category[category_name_english] != 'intro']
    elif cfg.data.exclude_plans == 'basic':
        print(f"\n===================basic 플랜 제외===================")
        df_plan = df_plan[df_plan[plan_id] >2]
        df_category = df_category[df_category[category_id] >2]
    #map_plan = df_plan[[plan_id, plan_name_english]]
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
    
    output_scoring_path = output_path / Path("3_scoring")
    output_category_path = output_path / Path("4_category")
    output_final_path = output_path / Path("5_final_plan")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_scoring_path, exist_ok=True)
    os.makedirs(output_category_path, exist_ok=True)
    os.makedirs(output_final_path, exist_ok=True)
    prompt_path = output_path / Path("prompt")
    os.makedirs(prompt_path, exist_ok=True)
    
    shutil.copy(cwd / Path('conf') / Path('config_with_categories.yaml'), output_path)
    
    # LLM API를 사용하도록 설정
    llm_client = create_llm_client(cfg)
    if llm_client is None:
        raise ValueError("LLM 클라이언트를 초기화할 수 없습니다.") 
    else:
        print("LLM 클라이언트 초기화 완료.")

    csv_path = cwd / "data" / "VirtualSurvey - VirtualSurvey.csv"
    family = FamilyPersona(csv_path=csv_path)
    # family = FamilyPersona(csv_path=csv_path)
    df_scores = pd.DataFrame()
    df_survey  =  pd.read_csv(csv_path, encoding='utf-8-sig')
    
    from src.prep import load_and_parse_json_data
    data_target_path = cwd / Path("data") / Path("target")
    # for i, json_file in tqdm(enumerate(json_files)):
    
    for i in tqdm(df_survey['일련번호']):
        i_df= df_survey.iloc[i,:]
        i_family = family.get_persona_data(i)
        i_survey_info = prep_survey_info(i_family, top_k=cfg.prompt.top_k)
        child_age = int(i_df['아이 연령'])

        df_category = df[df['min_month'] <= child_age]
        df_category = df_category[df_category['max_month'] >= child_age]

        print(f"연령에 적합한 Category id 목록: {df_category[category_id]}")
        category_info_only = generate_category_info_only(df_category)
    #for i in tqdm(range(family.get_persona_count())):
        index_persona = f"Persona_{i}"
        json_file = find_persona(i, data_target_path)

        print(json_file)
        if len(json_file) == 0:
            print(f"Persona_{i}의 데이터가 없습니다.")
            continue
        dialogue_data = load_and_parse_json_data(json_file)
        print(dialogue_data)

        index_persona = f"Persona_{i}"
        print(f"\n================================{index_persona}================================")

        # i_family = family.get_persona_data(i)
        i_persona = i_family.get("persona")
        try:
            i_persona = i_persona[0]
        except:
            i_persona = i_persona

        # --- 스코어링 로직 시작 --- (for plan 루프 안으로 이동)
        ######################
        scoring_items_dict = load_data_from_cfg(cfg, 'scoring_items', dict, {})
        scoring_criteria_list = []
        if scoring_items_dict:
            for item_id, item_data in scoring_items_dict.items():
                if isinstance(item_data, dict):
                    scoring_criteria_list.append({'id': item_id, **item_data})

        # 대화 데이터 포맷팅 부분 수정
        formatted_dialogue = reconstruct_dialogue(dialogue_data)
        if formatted_dialogue and formatted_dialogue not in ["[대화 데이터 없음]", "[유효한 대화 턴 없음]"]:
            scoring_template_str = load_template_str(cfg, 'output_formats.scoring.template')
            if scoring_template_str:
                prompt = get_scoring_prompt(
                    dialogue=formatted_dialogue,  # 재구성된 대화 문자열 사용
                    scoring_criteria=scoring_criteria_list,
                    output_template_str=scoring_template_str, 
                    detailed_criteria= cfg.prompt.scoring.detailed_criteria,
                    child_age=child_age
                )
                
                # 디버깅을 위한 로깅 추가
                print("\n=== 스코어링 프롬프트 ===")
                # print(formatted_dialogue) # 너무 길어서 주석 처리, 필요 시 해제
                print(f"스코어링 기준 개수: {len(scoring_criteria_list)}")
                print("========================\n")
                
                prompt_scoring_path = os.path.join(
                    prompt_path, 
                    f"{index_persona}_Step_3_Scoring_{name_param}.txt" # plan 변수 사용
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
                        f"{index_persona}_Step_3_Scoring_{name_param}.json"
                    )

                    scoring_data = {"persona_id": i, "persona": i_persona, "age": child_age, **scoring_data}
                    call_save_results(scoring_data, output_file)

    
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
        
        if cfg.prompt.category.enabled:
            prompt = get_category_prompt(prompt_category=category_info_only,
                    scoring_results=scoring_data, # Pass the original dictionary
                    child_age=child_age,
                    survey_info=i_survey_info
                )
            prompt_category_path = os.path.join(prompt_path, f"{index_persona}_Step_4_Category_{name_param}_prompt.txt")
            save_llm_prompts_to_txt(prompt, prompt_category_path)
            # Step 4의 예상 최상위 키 수정: overall_reason 추가
            expected_category_keys = ["selected_categories", "overall_reason"] 
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
            category_data = {"persona_id": i, "persona": i_persona, "age": child_age, **category_data}
            call_save_results(category_data, output_file)
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
            # if len(category_ids) >3:
            #     category_ids = category_ids[:3]

            print(f"추천된 유효 카테고리 ID 목록: {category_ids}")
            print("="*50)
            plans_info = df_plan[df_plan[category_id].isin(category_ids)]
            category_name = [item.get('name') for item in category_data.get('selected_categories', [])]
            category_reason = [item.get('reason') for item in category_data.get('selected_categories', [])]
            category_overall_reason = category_data.get('overall_reason')
            
        else:
            plans_info = df_plan
            category_ids = []
            category_name = []
            category_reason = []
            category_overall_reason = []

        print(f"추천된 plan id 목록: {plans_info[plan_id]}")
        plans_info = plans_info[plans_info['min_month'] <= child_age]
        plans_info = plans_info[plans_info['max_month'] >= child_age]
        print(f"연령에 적합한 plan id 목록: {plans_info[plan_id]}")
        relevant_plans_info = gen_plan_info(plans_info, is_keywords=cfg.prompt.plan.is_keywords, is_example=cfg.prompt.plan.is_example, is_description=cfg.prompt.plan.is_description)
        prompt_plan_path = os.path.join(prompt_path, f"{index_persona}_Step_5_Plan_{name_param}_prompt.txt")
        prompt = get_plan_prompt(scoring_results=scoring_data, relevant_plans_info=relevant_plans_info, child_age=child_age, dialogue=formatted_dialogue, survey_info=i_survey_info, is_dialogue=cfg.prompt.plan.is_dialogue)
        save_llm_prompts_to_txt(prompt, prompt_plan_path)
        plan_data = call_llm_and_parse_json(prompt, llm_client, expected_keys=['conversation_analysis', 'recommended_plans'])
        output_file = os.path.join(
                            output_final_path,
                            f"{index_persona}_Step_5_Plan_{name_param}.json"
                        )
        plan_data = {"persona_id": i, "persona": i_persona, "age": child_age, **plan_data}
        call_save_results(plan_data, output_file)
        print(f"플랜 추천 결과 저장됨: {output_file}")
        ###
        # 결과 취합 로직 수정: 리스트들을 먼저 추출하고 merged_dict에 포함
        dict_score = scoring_data.get('scoring')
        dict_explanation = scoring_data.get('explanation')
        dict_score = {scoring_index_to_id[int(key)]: value for key, value in dict_score.items() if key.isdigit() and int(key) in scoring_index_to_id}
        dict_explanation = {'explanation_'+scoring_index_to_id[int(key)]: value for key, value in dict_explanation.items() if key.isdigit() and int(key) in scoring_index_to_id}
        
        # 카테고리 관련 정보 추출
        #category_id = [item.get('id') for item in category_data.get('selected_categories', [])]
        
        # 플랜 관련 정보 추출
        recommended_plans = plan_data.get("recommended_plans", []) if plan_data else [] 
        plans_id = [plan.get(plan_id) for plan in recommended_plans]
        plans_name = [plan.get(plan_name) for plan in recommended_plans]
        plans_reason = [plan.get('reason') for plan in recommended_plans]
        plans_effect = [plan.get('expected_effect') for plan in recommended_plans]
        
        # 기본 데이터와 추출된 리스트/문자열들을 합쳐서 merged_dict 생성
        merged_dict = {
            **df_survey.iloc[i,:].to_dict(), 
            **dict_score, 
            **dict_explanation,
            'category_id_list': category_ids,
            'category_name_list': category_name,
            'category_reason_list': category_reason,
            'category_overall_reason': category_overall_reason,
            'plan_id_list': plans_id,
            'plan_name_list': plans_name,
            'plan_reason_list': plans_reason,
            'plan_effect_list': plans_effect,
            'child_age': child_age,
            # 처음 3개의 추천 플랜 ID를 개별 컬럼으로 추가 (안전하게 처리)
            'llm_plan_rec_1': plans_id[0] if plans_id else None,  # 리스트가 비어있지 않으면 첫 번째 값
            'llm_plan_rec_2': plans_id[1] if len(plans_id) > 1 else None,  # 두 번째 값이 있으면 사용
            'llm_plan_rec_3': plans_id[2] if len(plans_id) > 2 else None   # 세 번째 값이 있으면 사용
        }
        # 단일 행 DataFrame 생성 (이제 리스트가 값으로 들어감)
        df_i = pd.DataFrame([merged_dict])
        df_i['persona_index'] = i

        df_scores = pd.concat([df_scores, df_i], ignore_index=True)
        
    prediction_path = output_path / Path(f"survey_with_scoring_{name_param}.csv")
    evaluation_path = output_path / Path(f"evaluation_metric_{name_param}.csv")
    df_scores.to_csv(prediction_path, encoding='utf-8-sig', index=True)
    call_metric(prediction_path, evaluation_path, name_param)
    
if __name__ == "__main__":
    main()
# - 이와 같은 코칭이 도움이 되었던 사례들을 천천히 떠올려 보고, 개선 되기 전의 가족 대화를 생성해 주세요.
