from typing import Dict, Any, List


def generate_llm_prompts(df_category):
    """
    Generates LLM prompts for each id_category in df_category.

    Args:
        df_category (pd.DataFrame): DataFrame containing category information with columns:
            - id_category
            - CategoryName
            - keyword
            - concatenated_descriptions

    Returns:
        dict: A dictionary mapping id_category to formatted LLM prompts
    """
    llm_prompts = f"""
    ## Category list    
    {df_category['CategoryName'].unique()}

    ## Category Information
    """

    for i, row in df_category.iterrows():
        prompt = f"""
        {i+1}.
        Category Name: {row['CategoryName']}
        Description: {row['concatenated_descriptions']}\n
        Related Keywords: {row['keyword']}
        
        """
# llm_prompts[row['id_category']] = prompt
        llm_prompts += prompt

    return llm_prompts


def create_survey_persona_prompt(context, category_info, top_k=6) -> str:
    persona = context.get('persona')
    priority_interests = context.get('priority_interests')
    priority_interests = priority_interests[:top_k]
    processed = ""
    for i in priority_interests:
        processed += f"- 우선순위 {i.get('priority')}. 관심 주제: {i.get('category_korean')}\\n- 키워드: {', '.join(i.get('keywords'))}\\n"
        # i_keywords= ', '.join(i.keywords) if isinstance(i.keywords, list) else i.keywords
    
    # prompt = f"""# Role  -> f-string 제거
    prompt_template = """# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰해온 심리 상담 전문가 입니다.

# Instruction
- 다음은 설문조사에 응답한 어느 가족의 정보입니다. 
- 가족 페르소나, 관심사 우선순위 카테고리 정보가 있습니다.
- 제시된 가족 페르소나와 **종합적인 관심사 우선순위**를 모두 고려하여 가장 관련성이 높은 카테고리 하나를 주어진 리스트에서 선택하세요. 
- 단순히 첫 번째 관심사에만 집중하지 말고, 전반적인 맥락과 카테고리 설명을 함께 파악하는 것이 중요합니다.

# Context
- 다음은 가족 페르소나 및 관심사 정보입니다.
- 관심사 카테고리 및 키워드는 우선 순위가 높은 것부터 나열되어 있습니다.
### Family Persona
{persona}
### Priority Interests
{processed}
## 카테고리 정보
- 다음은 카테고리 정보입니다.
{category_info}

# Output Format
- 반드시 다음 JSON 형식에 맞춰 결과를 출력해 주세요.
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
```json
{{
  "summary": "string", // 가족 정보 및 관심사를 요약한 문자열
  "persona": "string", // 입력된 페르소나 정보를 그대로 포함하는 문자열
  "category": "string", // 주어진 카테고리 리스트 중 가장 적합한 카테고리 이름 문자열
  "explanation": "string" // 위 카테고리를 선택한 이유를 설명하는 문자열
}}
```
"""
    # prompt = prompt.replace("\\\\n", "\\n") -> .format() 사용 시 불필요
    # 변수 내용에 포함될 수 있는 중괄호를 이스케이프하기 위해 .format() 사용
    prompt = prompt_template.format(persona=persona, processed=processed, category_info=category_info) 
    return prompt

def create_counselor_family_prompt(family_data, plan: Dict[str, Any], examples: List[str]) -> str:
    """심리 상담사 프롬프트를 생성합니다."""
    persona = family_data.get("persona", {})
    category_name = family_data.get("category", "")
  
    # # 키워드 문장들을 문자열로 변환
    # keywords_str = "\n".join([f"- {example}" for example in examples])
    
    # prompt = f""" -> f-string 제거
    prompt_template = """
# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰해온 심리 상담 전문가 입니다.

# Instruction
- 반드시 갈등 또는 부정적인 감정을 다양한 방식으로 포함하여 부모와 아이, 또는 양육자 간의 대화를 만들어 주세요.
- 등장 인물은 양육자인 엄마 아빠, 아이들, 주변 인물 중 맥락에 적절한 1-3인을 선택해 주세요.
- 특히 양육자 부모의 부정적인 태도가 잘 드러날 수록 좋습니다.
- 아이의 나이는 아래 주어진 정보를 바탕으로, 각 나이에 맞는 언어를 구사해야 합니다.
- 모르는 경우 3세 근처로 가정합니다. (어휘가 부족하고 유창하지 않아야 합니다.)
- 대화는 총 5 ~ 20 턴 내외로 생성해 주세요.
- Context의 예시 키워드 문장에 담긴 **핵심 의미나 의도**를 반영하는 부모의 발언을 최소 1회 이상 자연스럽게 포함하세요. (반드시 원문 그대로 사용할 필요는 없습니다.)
- 반드시 모두 한국어로 작성해 주세요.

# Context
- 다음은 도움이 필요한 대화 상황과 관련 있는 코칭 플랜 정보 입니다. 
- 예시 키워드 문장은 모두 부모가 아이에게 하는 말입니다.
## 가족 페르소나 정보
{persona}

## 카테고리 이름
{category_name}

## 코칭 플랜 이름
{plan}

## 예시 키워드 문장들
{examples}

# Output Format
- 반드시 다음 JSON 형식에 맞춰 대화 내용을 생성해 주세요.
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
```json
{{
  "category": "string", // 입력된 카테고리 이름
  "plan": "string", // 입력된 코칭 플랜 이름
  "meta_reasoning": "string", // 대화 생성 전략 및 근거 요약 (예: 갈등 상황 설정 이유, 키워드 의미 반영 방식 등)
  "strategy": "string", // 대화에 적용된 구체적인 상담 전략 설명 (예: 부모의 부정적 태도 표현 방식)
  "dialogue": [
    {{
      "speaker": "string", // 발화자 (예: "부모", "아이", "엄마", "아들")
      "content": "string" // 발화 내용
    }},
    // ... 추가 대화 턴 ...
  ]
}}
```
"""
    # .format() 사용하여 변수 삽입 및 중괄호 문제 방지
    prompt = prompt_template.format(
        persona=persona, 
        category_name=category_name, 
        plan=plan, 
        examples=examples
    )
    return prompt

from typing import List, Dict, Any
# Remove json import if no longer needed
# import json

def get_scoring_prompt(dialogue: str, scoring_criteria: List[Dict], output_template_str: str) -> str:
    """스코어링을 위한 프롬프트를 생성합니다."""
    prompt_parts = [
        "[시스템 지시사항]",
        "당신은 부모-자녀 대화 분석 전문가입니다.",
        "아래 대화 내용과 평가 기준을 바탕으로 각 항목별 점수(1~5점)와 그 이유를 상세히 설명해주세요.",
        "",
        "[대화 내용]",
        dialogue,  # 포맷팅된 대화 내용
        "",
        "[평가 기준]"
    ]
    
    # 평가 기준 추가
    for criteria in scoring_criteria:
        prompt_parts.append(f"- {criteria['id']}({criteria.get('name', '')}): {criteria.get('description', '')}")
    
    # 출력 형식 안내 추가
    prompt_parts.extend([
        "",
        "**[출력 형식 (매우 중요!)]**",
        "평가 결과는 반드시 아래와 같은 JSON 형식으로 작성해주세요.",
        "```json",
        # output_template_str 내에 설명을 직접 추가하는 대신, 외부 지침 강화
        output_template_str, 
        "```",
        "반드시 위 JSON 형식을 정확히 준수하고, 모든 키와 문자열 값은 큰따옴표(\"\")로 감싸주세요.",
        "'scoring' 객체 안에는 각 평가 기준 항목의 ID를 키로, 점수(1-5 정수)를 값으로 넣으세요.",
        "'explanation' 객체 안에는 각 평가 기준 항목의 ID를 키로, 점수 부여 이유를 상세히 작성하세요.",
        "-> **설명 작성 시에는 반드시 [대화 내용]과 [평가 기준]을 직접적으로 언급하며 근거를 제시해야 합니다.**",
        "",
        "[주의사항]",
        "- 반드시 각 평가 기준 항목에 대해 1점에서 5점 사이의 점수를 부여하세요.",
        "- 각 점수에 대한 구체적인 이유를 대화 내용을 인용하여 상세히 설명해야 합니다.",
        "- 점수는 정수만 사용하세요."
    ])
    
    return "\n".join(prompt_parts)



# - 다음의 Meta-plan 단계를 따라 생각한 뒤 대화를 생성하세요.

# ## [Meta-Plan 단계]
# 1. 이 문제 상황의 심리적 맥락은 무엇인가요?
# 2. 문제가 되는 키워드 문장이 어떻게 자연스럽게 대화에 포함될 수 있을까요?
# 3. 각 인물의 대화는 일관성을 유지하고 있나요?
