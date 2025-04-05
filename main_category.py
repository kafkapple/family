import os
import random
import json
from typing import List, Dict, Any, Optional
import yaml
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import dataclass
import requests
from dotenv import load_dotenv
import re
from llm_client import create_llm_client
from llm_interface import get_llm_response
from prompt import  create_counselor_prompt

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

def load_yaml_data(file_path: str) -> Dict[str, Any]:
    """YAML 파일에서 데이터를 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def select_plan_from_category(data: Dict[str, Any], category_id: str, plan_id: str = '') -> Dict[str, Any]:
    """지정된 카테고리에서 무작위로 플랜을 선택합니다."""
    plans = data.get("plan_index", {}).get(category_id, [])
    if not plans:
        raise ValueError(f"카테고리 '{category_id}'에 플랜이 없습니다.")
    if plan_id not in plans:
        raise ValueError(f"플랜 ID '{plan_id}'는 카테고리 '{category_id}'에 존재하지 않습니다.")
    if plan_id == '':
        plan_id = random.choice(plans)
    
    # 플랜 ID가 이미 딕셔너리인 경우 그대로 반환
    if isinstance(plan_id, dict):
        return plan_id
    
    # 선택된 플랜의 상세 정보 찾기
    if category_id in data:
        for plan in data.get(category_id, []):
            if plan.get("id") == plan_id:
                return plan
    
    raise ValueError(f"플랜 ID '{plan_id}'에 대한 상세 정보를 찾을 수 없습니다.")

def select_random_examples(examples: List[str], count: int = 3) -> List[str]:
    """예시 목록에서 무작위로 지정된 수만큼 선택합니다."""
    if not examples:
        return []
    
    # 예시 수가 요청된 수보다 적으면 모든 예시 반환
    if len(examples) <= count:
        return examples
    
    return random.sample(examples, count)

def parse_json_response(response: str) -> Dict[str, Any]:
    """LLM의 응답에서 JSON 부분을 추출하고 파싱합니다."""
    try:
        # 정규 표현식을 사용하여 JSON 객체 추출
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print("응답에서 JSON 형식을 찾을 수 없습니다.")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"원본 응답: {response}")
        return None

def save_dialogue_to_file(output_path: str, dialogue_data: Dict[str, Any], category_id: str, plan_id: str, examples: List[str]) -> None:
    """생성된 대화 데이터를 파일로 저장합니다."""
    with open(output_path, 'w', encoding='utf-8') as f:
        # 메타데이터 작성
        f.write(f"# 카테고리: {category_id}\n")
        f.write(f"# 플랜: {plan_id}\n")
        f.write(f"# 사용된 예시 문장:\n")
        for example in examples:
            f.write(f"# - {example}\n")
        f.write("\n")
        
        # JSON 데이터 저장
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)

@hydra.main(config_path="conf", config_name="config_with_categories")
def main(cfg: AppConfig) -> None:
    """메인 함수: 데이터 로드, 플랜 선택, 대화 생성을 수행합니다."""
    print(f"구성: {OmegaConf.to_yaml(cfg)}")
    
    # YAML 데이터 로드
    data = load_yaml_data(cfg.yaml_path)
    
    from datetime import datetime
    timestamp = datetime.now().timestamp()
    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S')

    # 카테고리 목록 가져오기
    if cfg.categories:
        categories = cfg.categories
    else:
        categories = data.get("category_list", [])
    print(f"\n=====Categorys=====\n{categories}")

    # LLM API를 사용하도록 설정
    llm_client = create_llm_client(cfg)
    if llm_client is None:
        raise ValueError("LLM 클라이언트를 초기화할 수 없습니다.") 
    else:
        print("LLM 클라이언트 초기화 완료.")

    plans_category = data.get("plans", {})  # plans
    
    # 각 카테고리에서 플랜 선택 및 대화 생성
    for category_id in categories:
        print(f"\n=== 카테고리: {category_id} ===")
        try:
            plans = plans_category.get(category_id, [])

            for plan in plans:
               
                print(f"'{plan['id']}' 플랜을 찾았습니다.")
                print(f"Plan name: {plan.get('name')} ({plan.get('english_name')})")
                print(f"Examples: {plan.get('examples', [])}")
                examples = plan.get("examples", [])
        
            if not examples:
                print(f"카테고리 '{category_id}'의 선택된 플랜에 예시가 없습니다. 다음 카테고리로 넘어갑니다.")
                continue
            
            # 심리 상담사 프롬프트 생성
            prompt = create_counselor_prompt(plan, examples)
            print(f"Prompting: {prompt}\n")
            
            # LLM을 사용하여 대화 생성
            if cfg.llm_provider.lower() == "openai":
                #response = generate_dialogue_with_openai(prompt, cfg)
                response= get_llm_response(prompt, llm_client)
            elif cfg.llm_provider.lower() == "ollama":
                response= get_llm_response(prompt, llm_client)
                #response = generate_dialogue_with_ollama(prompt, cfg)
            else:
                raise ValueError(f"지원되지 않는 LLM 제공자: {cfg.llm_provider}")
            
            # JSON 응답 파싱
            dialogue_data = parse_json_response(response)
            
            if dialogue_data:
                print(f"생성된 대화 데이터:")
                print(json.dumps(dialogue_data, ensure_ascii=False, indent=2))
                
                # 생성된 대화를 파일에 저장
                output_dir = os.path.join(hydra.utils.get_original_cwd(), "outputs", category_id)
                
                os.makedirs(output_dir, exist_ok=True)
                
                output_file = os.path.join(output_dir, f"Dialogue_{plan.get('id')}_{cfg.model_name}_{timestamp}.json")
                save_dialogue_to_file(output_file, dialogue_data, category_id, plan.get('id', ''), examples)
                output_file = os.path.join(output_dir, f"Prompt_{plan.get('id')}_{cfg.model_name}_{timestamp}.json")
                save_dialogue_to_file(output_file, prompt, category_id, plan.get('id', ''), examples)
                
                print(f"\n대화가 '{output_file}'에 저장되었습니다.")
            else:
                print("대화 생성에 실패했습니다.")
            
            print("\n" + "="*50 + "\n")
        
        except Exception as e:
            print(f"카테고리 '{category_id}' 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
# - 이와 같은 코칭이 도움이 되었던 사례들을 천천히 떠올려 보고, 개선 되기 전의 가족 대화를 생성해 주세요.
