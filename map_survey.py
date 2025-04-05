"""
가족 페르소나 프로세서 모듈 - 페르소나 데이터를 처리하고 다양한 형식으로 변환합니다.
"""
import pandas as pd
import json
import os
import glob

# 카테고리 매핑 정보
CATEGORY_ID_MAP = {
    "발달 및 정서": 1,
    "양육태도": 2,
    "소통기술": 3,
    "양육자 본인": 4,
    "상황": 5,
    "대화 대상": 6
}

CATEGORY_ENGLISH_MAP = {
    "발달 및 정서": "Development & Emotion",
    "양육태도": "Parenting Style",
    "소통기술": "Communication Skills",
    "양육자 본인": "Parent Self-care",
    "상황": "Situation",
    "대화 대상": "Communication Partner"
}

CATEGORY_NORMALIZER = {
    "소통기술": "소통 기술",
    "양육태도": "양육 태도"
}

# 유틸리티 함수
def parse_list_string(list_str):
    """문자열 형태의 리스트를 실제 리스트로 변환합니다."""
    try:
        clean_str = list_str.replace("'", "").replace("[", "").replace("]", "")
        return [item.strip() for item in clean_str.split(',')]
    except:
        return []

def normalize_category(category):
    """카테고리 이름을 정규화합니다."""
    return CATEGORY_NORMALIZER.get(category, category)

# CSV 데이터 처리 모듈
class CSVProcessor:
    def __init__(self, csv_path):
        """CSV 파일에서 데이터를 로드하고 변환합니다."""
        self.df = self._load_csv(csv_path)
    
    def _load_csv(self, csv_path):
        """CSV 파일을 로드하고 필요한 변환을 수행합니다."""
        df = pd.read_csv(csv_path)
        df_transformed = df.copy()
        
        # 카테고리 ID와 영어 이름 매핑 적용
        for i in range(1, 7):
            col_name = f"카테고리{i}"
            df_transformed[f"{col_name}_ID"] = df_transformed[col_name].map(CATEGORY_ID_MAP)
            df_transformed[f"{col_name}_English"] = df_transformed[col_name].map(CATEGORY_ENGLISH_MAP)
        
        # 관심 키워드 정리
        interest_cols = ["발달 및 정서", "양육 태도", "소통 기술", "양육자 본인", "상황", "대화 대상"]
        for col in interest_cols:
            df_transformed[col] = df_transformed[col].apply(parse_list_string)
        
        return df_transformed
    
    def get_persona_count(self):
        """페르소나 데이터의 총 개수를 반환합니다."""
        return len(self.df)
    
    def get_persona_data(self, index):
        """특정 인덱스의 페르소나 데이터를 반환합니다."""
        if index >= len(self.df):
            raise ValueError(f"인덱스 {index}가 범위를 벗어났습니다. 최대 인덱스는 {len(self.df)-1}입니다.")
        
        row = self.df.iloc[index]
        
        # 기본 페르소나 정보
        persona_data = {"persona": row["페르소나"], "priority_interests": []}
        
        # 우선순위 관심사 처리
        for i in range(1, 7):
            category_key = f"카테고리{i}"
            category_name = row[category_key]
            normalized_category = normalize_category(category_name)
            category_english = row[f"{category_key}_English"]
            category_id = row[f"{category_key}_ID"]
            
            # 해당 관심 컬럼에서 키워드 가져오기
            keywords = row[normalized_category]
            
            persona_data["priority_interests"].append({
                "priority": i,
                "category_id": int(category_id),
                "category_korean": category_name,
                "category_english": category_english,
                "keywords": keywords
            })
        
        return persona_data
    
    def get_all_persona_data(self):
        """모든 페르소나 데이터를 반환합니다."""
        return [self.get_persona_data(i) for i in range(len(self.df))]

# JSON 데이터 처리 모듈
class JSONProcessor:
    def __init__(self, json_path=None, json_data=None):
        """JSON 파일 또는 직접 데이터로 초기화합니다."""
        if json_path:
            self.data = self._load_json(json_path)
        elif json_data:
            self.data = json_data if isinstance(json_data, list) else [json_data]
        else:
            self.data = []
    
    def _load_json(self, json_path):
        """JSON 파일 또는 디렉토리에서 데이터를 로드합니다."""
        loaded_data = []
        
        try:
            if os.path.isdir(json_path):
                # 디렉토리에서 모든 JSON 파일 로드
                json_files = glob.glob(os.path.join(json_path, "*.json"))
                for file_path in json_files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        persona_data = json.load(f)
                        loaded_data.append(persona_data)
            else:
                # 단일 JSON 파일 로드
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 데이터가 리스트인지 확인
                    if isinstance(data, list):
                        loaded_data = data
                    else:
                        loaded_data = [data]
            
            return loaded_data
        except Exception as e:
            print(f"JSON 파일 로드 오류: {e}")
            return []
    
    def save_json(self, output_path, individual_files=False):
        """JSON 데이터를 파일로 저장합니다."""
        try:
            if individual_files:
                # 각 페르소나를 개별 파일로 저장
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                for idx, persona_data in enumerate(self.data):
                    file_path = os.path.join(output_path, f"persona_{idx}.json")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(persona_data, f, ensure_ascii=False, indent=2)
            else:
                # 모든 페르소나를 단일 파일로 저장
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"JSON 저장 오류: {e}")
            return False
    
    def get_persona_count(self):
        """페르소나 데이터의 총 개수를 반환합니다."""
        return len(self.data)
    
    def get_persona_data(self, index):
        """특정 인덱스의 페르소나 데이터를 반환합니다."""
        if index >= len(self.data):
            raise ValueError(f"인덱스 {index}가 범위를 벗어났습니다. 최대 인덱스는 {len(self.data)-1}입니다.")
        return self.data[index]

# 프롬프트 포맷 모듈
class PromptFormatter:
    @staticmethod
    def to_text(persona_data):
        """페르소나 데이터를 텍스트 형식으로 변환합니다."""
        persona = persona_data.get("persona", "")
        priority_interests = persona_data.get("priority_interests", [])
        
        prompt = f"가족 페르소나: {persona}\n\n"
        prompt += "우선순위 관심사 (높은 순에서 낮은 순):\n"
        
        for interest in priority_interests:
            priority = interest.get("priority", "")
            category_id = interest.get("category_id", "")
            category_korean = interest.get("category_korean", "")
            category_english = interest.get("category_english", "")
            keywords = interest.get("keywords", [])
            
            prompt += f"{priority}. {category_english} (ID: {category_id}, 한글: {category_korean})\n"
            prompt += f"   키워드: {', '.join(keywords)}\n"
        
        return prompt
    
    @staticmethod
    def to_markdown(persona_data):
        """페르소나 데이터를 마크다운 형식으로 변환합니다."""
        persona = persona_data.get("persona", "")
        priority_interests = persona_data.get("priority_interests", [])
        
        md = f"# 가족 페르소나\n\n"
        md += f"## 페르소나 정보\n\n"
        md += f"{persona}\n\n"
        md += f"## 우선순위 관심사\n\n"
        
        # 우선순위 카테고리 테이블
        md += "| 우선순위 | 카테고리 ID | 카테고리 (한글) | 카테고리 (영어) | 키워드 |\n"
        md += "|----------|-------------|----------------|----------------|--------|\n"
        
        for interest in priority_interests:
            priority = interest.get("priority", "")
            category_id = interest.get("category_id", "")
            category_korean = interest.get("category_korean", "")
            category_english = interest.get("category_english", "")
            keywords = interest.get("keywords", [])
            
            md += f"| {priority} | {category_id} | {category_korean} | {category_english} | {', '.join(keywords)} |\n"
        
        return md
    
    @staticmethod
    def save_to_markdown(persona_data_list, output_path, individual_files=False):
        """페르소나 데이터를 마크다운 파일로 저장합니다."""
        try:
            if individual_files:
                # 각 페르소나를 개별 파일로 저장
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                for idx, persona_data in enumerate(persona_data_list):
                    file_path = os.path.join(output_path, f"persona_{idx}.md")
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(PromptFormatter.to_markdown(persona_data))
            else:
                # 모든 페르소나를 단일 파일로 저장
                combined_md = "# 가족 페르소나 프롬프트 모음\n\n"
                
                for idx, persona_data in enumerate(persona_data_list):
                    combined_md += f"---\n\n## 페르소나 {idx}\n\n"
                    # 헤더 없이 내용만 가져오기 위해 분할 후 필요한 부분만 결합
                    md_content = PromptFormatter.to_markdown(persona_data)
                    lines = md_content.split('\n')[1:]  # 첫 번째 줄(# 가족 페르소나) 제외
                    combined_md += '\n'.join(lines) + "\n\n"
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(combined_md)
            
            return True
        except Exception as e:
            print(f"마크다운 저장 오류: {e}")
            return False

# 메인 인터페이스
class FamilyPersona:
    def __init__(self, csv_path=None, json_path=None):
        """CSV 또는 JSON 파일에서 페르소나 데이터를 초기화합니다."""
        if csv_path:
            self.csv_processor = CSVProcessor(csv_path)
            self.json_processor = JSONProcessor(json_data=self.csv_processor.get_all_persona_data())
        elif json_path:
            self.csv_processor = None
            self.json_processor = JSONProcessor(json_path=json_path)
        else:
            raise ValueError("CSV 또는 JSON 파일 경로를 제공해야 합니다.")
    
    def get_persona_count(self):
        """페르소나 데이터의 총 개수를 반환합니다."""
        return self.json_processor.get_persona_count()
    
    def get_persona_data(self, index):
        """특정 인덱스의 페르소나 데이터를 반환합니다."""
        return self.json_processor.get_persona_data(index)
    
    def get_all_persona_data(self):
        """모든 페르소나 데이터를 반환합니다."""
        return self.json_processor.data
    
    def get_text_prompt(self, index):
        """특정 인덱스의 페르소나를 텍스트 형식으로 반환합니다."""
        persona_data = self.get_persona_data(index)
        return PromptFormatter.to_text(persona_data)
    
    def get_markdown_prompt(self, index):
        """특정 인덱스의 페르소나를 마크다운 형식으로 반환합니다."""
        persona_data = self.get_persona_data(index)
        return PromptFormatter.to_markdown(persona_data)
    
    def save_to_json(self, output_path, individual_files=False):
        """페르소나 데이터를 JSON 파일로 저장합니다."""
        return self.json_processor.save_json(output_path, individual_files)
    
    def save_to_markdown(self, output_path, individual_files=False):
        """페르소나 데이터를 마크다운 파일로 저장합니다."""
        return PromptFormatter.save_to_markdown(self.json_processor.data, output_path, individual_files)
    
    def add_to_context(self, index, format="text"):
        """특정 페르소나를 지정한 형식으로 컨텍스트에 추가할 수 있는 형태로 반환합니다."""
        persona_data = self.get_persona_data(index)
        
        if format == "text":
            return PromptFormatter.to_text(persona_data)
        elif format == "markdown":
            return PromptFormatter.to_markdown(persona_data)
        elif format == "json":
            return json.dumps(persona_data, ensure_ascii=False, indent=2)
        else:
            raise ValueError("잘못된 형식입니다. 'text', 'markdown', 또는 'json'만 지원합니다.")


# 사용 예시
if __name__ == "__main__":
    # CSV에서 데이터 로드
    family = FamilyPersona(csv_path="data/VirtualSurvey.csv")
    
    # 텍스트 프롬프트 출력
    print("==== 텍스트 프롬프트 ====")
    print(family.get_text_prompt(0))
    
    # 마크다운 프롬프트 출력
    print("\n==== 마크다운 프롬프트 ====")
    print(family.get_markdown_prompt(0))
    
    # JSON으로 저장
    print("\n==== JSON 저장 ====")
    result = family.save_to_json("family_personas.json")
    print(f"저장 결과: {'성공' if result else '실패'}")
    
    # 마크다운으로 저장
    print("\n==== 마크다운 저장 ====")
    result = family.save_to_markdown("family_personas.md")
    print(f"저장 결과: {'성공' if result else '실패'}")
    
    # 컨텍스트에 추가 예시
    print("\n==== 컨텍스트에 추가 ====")
    context_prompt = family.add_to_context(0, format="markdown")
    print(f"컨텍스트 프롬프트:\n{context_prompt}")
    
    # JSON에서 데이터 로드
    print("\n==== JSON에서 데이터 로드 ====")
    family_from_json = FamilyPersona(json_path="family_personas.json")
    print(f"로드된 페르소나 수: {family_from_json.get_persona_count()}")