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
        """
        # Related Keywords: {row['keyword']}
        #llm_prompts[row['id_category']] = prompt
        llm_prompts += prompt

    return llm_prompts


def create_survey_persona_prompt(context, category_info, top_k=6) -> str:
    persona = context.get('persona')
    priority_interests = context.get('priority_interests')
    priority_interests = priority_interests[:top_k]
    processed = ""
    for i in priority_interests:
        processed += f"- 우선순위 {i.get('priority')}. {i.get('category_korean')}\n- 키워드: {', '.join(i.get('keywords'))}\n"
        # i_keywords= ', '.join(i.keywords) if isinstance(i.keywords, list) else i.keywords
    
    prompt = f"""# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰해온 심리 상담 전문가 입니다.

# Instruction
- 다음은 설문조사에 응답한 어느 가족의 정보입니다. 
- 가족 페르소나, 관심사 우선순위 카테고리 정보가 있습니다.
- 가장 적절한 카테고리를 주어진 리스트에서 선택해 주세요.

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
- 다음과 같은 json format 으로 결과를 출력해 주세요.
{{
"summary": 가족 정보 및 관심사 요약,
"persona": 페르소나 정보,
"category": 가장 적절한 카테고리 이름 (주어진 리스트에서),
"explanation": 카테고리 선택 이유

}}
"""
    prompt = prompt.replace("\\n", "\n")
    return prompt

def create_counselor_family_prompt(family_data, plan: Dict[str, Any], examples: List[str]) -> str:
    """심리 상담사 프롬프트를 생성합니다."""
    persona = family_data.get("persona", {})
    category_name = family_data.get("category", "")
  
    # # 키워드 문장들을 문자열로 변환
    # keywords_str = "\n".join([f"- {example}" for example in examples])
    
    prompt = f"""
# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰해온 심리 상담 전문가 입니다.

# Instruction
- 반드시 갈등 또는 부정적인 감정을 다양한 방식으로 포함하여 부모와 아이, 또는 양육자 간의 대화를 만들어 주세요.
- 특히 양육자 부모의 부정적인 태도가 잘 드러날 수록 좋습니다.
- 아이의 나이는 아래 주어진 정보를 바탕으로, 각 나이에 맞는 언어를 구사해야 합니다.
- 모르는 경우 3세 근처로 가정합니다. (어휘가 부족하고 유창하지 않아야 합니다.)
- 대화는 총 3 ~ 10 턴 내외로 생성해 주세요.
- 반드시 Context 의 예시 키워드 문장들 중 1개 이상을 대화에 포함하되, 앞 뒤 맥락에 어울리도록 자연스럽게 포함시켜 주세요.
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
- 다음과 같은 json format 으로 대화 내용을 생성해 주세요.
{{
  "category": "{category_name}",
  "plan": "{plan}",
  "meta_reasoning": "...",
  "strategy": "...",
  "dialogue": [
    {{"speaker": "부모", "content": "..."}},
    {{"speaker": "아이", "content": "..."}},
    ...
  ]
}}
"""
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
        output_template_str,
        "```",
        "반드시 위 JSON 형식을 정확히 준수하고, 모든 키와 문자열 값은 큰따옴표(\"\")로 감싸주세요.",
        "'scoring' 객체 안에는 각 평가 기준 항목의 ID를 키로, 점수(1-5)를 값으로 넣으세요.",
        "'explanation' 객체 안에는 각 평가 기준 항목의 ID를 키로, 점수 부여 이유를 상세히 작성하세요.",
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
