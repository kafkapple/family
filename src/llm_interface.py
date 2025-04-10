# integrations/llm_interface.py
import logging
from omegaconf import DictConfig
from src.prep import parse_json_response, fix_json_keys
from typing import Any, Optional, List, Dict
import time
# Assuming llm_client object has a __call__ method or similar for prediction

logging.basicConfig(level=logging.INFO) # Basic config
logger = logging.getLogger(__name__)

def get_llm_response(prompt: str,  llm_client) -> str | None:
    """Sends prompt to the LLM client and returns the response."""
    response = None # Default to None
    try:
        # Adjust this call based on the actual LLM client library's API.
        # For langchain_ollama.OllamaLLM, it's likely callable directly.
        prompt_snippet = prompt[:100].replace('\n', ' ')
        if len(prompt) > 100:
            prompt_snippet += '...'
        logger.debug(f"Sending prompt: {prompt_snippet}")

        # Call the llm_client (assumed to be callable)
        response_data = llm_client(prompt) # Assuming this returns the string directly

        # Ensure response is a string
        if isinstance(response_data, str):
            response = response_data
            response_snippet = response[:100].replace('\n', ' ') + ('...' if len(response) > 100 else '') if response else "None"
            logger.debug(f"Received response: {response_snippet}")
        else:
            logger.warning(f"LLM client returned unexpected type: {type(response_data)}. Expected str.")
            response = None # Treat non-string response as error/None

    except Exception as e:
        logger.error(f"Error getting LLM response: {e}", exc_info=True)
        response = None # Ensure response is None on error

    return response 


def call_llm_and_parse_json(prompt: str, llm_client: Any, max_retries: int = 10, expected_keys: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
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