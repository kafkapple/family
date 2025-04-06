from typing import Dict, Any, List


role = """
# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰해온 심리 상담 전문가 입니다.
"""

def gen_plan_info(df_plan):
    """
    Generates LLM prompts listing categories with their IDs and descriptions.
    (카테고리 목록을 ID, 이름, 설명과 함께 명확하게 생성하도록 개선)

    Args:
        df_category (pd.DataFrame): DataFrame containing category information.

    Returns:
        str: A formatted string listing categories for the LLM prompt.
    """
    df_unique_categories = df_plan.drop_duplicates(subset=['id_plan'])
    prompt_lines = ["## Available Plans (ID, Name, Description)"]
    
    for _, row in df_unique_categories.iterrows():
        keyword = row['keyword']
        prompt_lines.append(f"{row['id_plan']}. {row['PlanName']}")
        prompt_lines.append(f"   Description: {row['Course_description']}")
        prompt_lines.append(f"   Keywords: {keyword}")
        prompt_lines.append("")

    return "\n".join(prompt_lines)

def generate_llm_prompts(df_category):
    """
    Generates LLM prompts listing categories with their IDs and descriptions.
    (카테고리 목록을 ID, 이름, 설명과 함께 명확하게 생성하도록 개선)

    Args:
        df_category (pd.DataFrame): DataFrame containing category information.

    Returns:
        str: A formatted string listing categories for the LLM prompt.
    """
    df_unique_categories = df_category.drop_duplicates(subset=['CategoryName_English'])
    prompt_lines = ["## Available Categories (ID, Name, Description)"]
    
    for _, row in df_unique_categories.iterrows():
        category_id_val = row['id_category']
        eng_name = row['CategoryName_English']
        korean_name = row['CategoryName']
        description = row['concatenated_descriptions']
        keyword = row['keyword']
        
        prompt_lines.append(f"{category_id_val}. {eng_name}")
        prompt_lines.append(f"   Korean Name: {korean_name}")
        prompt_lines.append(f"   Description: {description}")
        prompt_lines.append(f"   Keywords: {keyword}")
        prompt_lines.append("")

    return "\n".join(prompt_lines)


def create_survey_persona_prompt(context, category_info, top_k=3) -> str:
    persona = context.get('persona')
    priority_interests = context.get('priority_interests')
    priority_interests = priority_interests[:top_k]
    processed = ""
    for i in priority_interests:
        processed += f"- 우선순위 {i.get('priority')}. 관심 주제: {i.get('category_korean')}\\n- 키워드: {', '.join(i.get('keywords'))}\\n"
    
    prompt_template = """
{role}

# Instruction
- 제공된 가족 정보([Family Persona], [Priority Interests])와 [카테고리 정보]를 종합적으로 고려하세요.
- 단순히 첫 번째 관심사에만 집중하지 말고, 2, 3번째 관심사, 키워드, 맥락, 카테고리 설명을 함께 파악하세요.
- 가장 관련성이 높은 카테고리 하나를 선택하고, 그 카테고리의 **ID(정수)** 를 응답 JSON에 포함시키세요.
- 카테고리 목록([Available Categories])에 제공된 ID 중에서만 선택해야 합니다.
- **`explanation` 필드는 반드시 한국어로만 작성해야 합니다. 영어 단어나 문장을 사용하지 마세요.**

# Context
### Family Persona
- {persona}
### Priority Interests
- 관심사 카테고리 및 키워드는 우선 순위가 높은 것부터 나열되어 있습니다.
{processed}

## Available Categories
{category_info}

# Output Format
- 반드시 다음 JSON 형식에 맞춰 결과를 출력해 주세요.
- **`category_id` 필드에는 선택한 카테고리의 ID(정수)를 넣어야 합니다.**
- **`explanation` 필드는 아래 예시와 같이 반드시 한국어로, 선택 이유를 상세히 기술해야 합니다.**
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
```json
{{
  "category_id": int, // 예: 101 (카테고리 ID)
  "explanation": "string" // 예: "가족 페르소나의 OOO 측면과 우선순위 관심사 중 XX 키워드가 카테고리 ID 101(life_event_focused)의 설명 및 키워드와 가장 관련성이 높아 선택했습니다."
}}
```
"""
    prompt = prompt_template.format(role=role, persona=persona, processed=processed, category_info=category_info)
    return prompt

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

def create_counselor_family_prompt(i_persona, category_name: str, ii_df_plans) -> str:
    """심리 상담사 프롬프트를 생성합니다. (카테고리 이름을 직접 받도록 수정)"""
    global role
    cols_examples = [col for col in ii_df_plans.columns if 'example_' in col]
    persona = i_persona
    plan_name = ii_df_plans['PlanName'].iloc[0]
    keyword = ii_df_plans['keyword'].iloc[0]
    description = ii_df_plans['Course_description'].iloc[0]
    example_list = extract_examples_from_plan(ii_df_plans, cols_examples)
    examples = [f"- {example}" for example in example_list]
    examples = "\n".join(examples)

    print(f"Examples:\n{examples}")
  
    prompt_template = """
{role}
# Instruction
- 자주 경험했던 부모와 아이, 또는 양육자 간의 부정적인 대화를 떠올려 자연스럽게 재구성해 주세요.
- 대화는 총 5 ~ 15 턴 내외로 생성해 주세요.
- 모두 한국어로 작성해 주세요.

# Context
- 전문가 경험, 가족 페르소나, 도움이 될 수 있는 코칭 플랜 정보를 종합해 고려합니다.
- 등장 인물은 양육자인 엄마, 아빠, 아이들, 주변 인물 중 맥락에 적절한 1-3인을 선택해 주세요.
- 특히 양육자 부모의 부정적인 태도가 잘 드러날 수록 좋습니다.
- 아이의 나이는 주어진 정보를 바탕으로, 각 나이에 맞는 언어를 구사해야 합니다.
- 모르는 경우 3세 근처로 가정합니다. (어휘가 부족하고 유창하지 않아야 합니다.)
- 모든 인물의 성격 및 대화는 일관성을 유지해야 합니다.
- examples 문장에 담긴 **핵심 의미나 의도**를 반영하는 부모의 발언을 최소 1회 이상 자연스럽게 포함하세요. (반드시 원문 그대로 사용할 필요는 없습니다.)
- examples 문장은 모두 부모가 아이에게 하는 말입니다.

## 가족 페르소나 정보
{persona}

## 코칭 플랜
- 카테고리: {category_name} 
- 플랜: {plan_name}
- 키워드: {keyword}
- 설명: {description}

## examples
{examples}

# Output Format
- 반드시 다음 JSON 형식에 맞춰 대화 내용을 생성해 주세요.
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
- category, plan, explanation, dialogue, speaker, content 필드는 반드시 포함해야 하며, 정확히 같은 이름으로 포함해야 합니다.
```json
{{
  "category": "string", // 카테고리 이름
  "plan": "string", // 플랜 이름
  "explanation": "string", // 대화 생성 전략 및 근거 요약
  "dialogue": [
    {{
      "speaker": "string",
      "content": "string"
    }},
    // ...
  ]
}}
```
"""
    prompt = prompt_template.format(
        role=role,
        persona=persona, 
        category_name=category_name, # 전달받은 실제 카테고리 이름 사용
        plan_name=plan_name, 
        keyword=keyword, 
        description=description,
        examples=examples
    )
    return prompt, examples

from typing import List, Dict, Any
# Remove json import if no longer needed
# import json

def get_scoring_prompt(dialogue: str, scoring_criteria: List[Dict], output_template_str: str) -> str:
    """스코어링을 위한 프롬프트를 생성합니다. (인덱스 키 사용)"""
    # scoring_criteria 리스트에서 인덱스와 ID, 설명을 포함한 문자열 생성
    criteria_descriptions = "\n".join([
        f"- Index {idx}: `{c['id']}` ({c.get('name', '')})\n  Description: {c.get('description', '')}" 
        for idx, c in enumerate(scoring_criteria)
    ])
    # LLM에게 사용할 정확한 인덱스 키 목록 (문자열 형태)
    valid_index_keys = [str(i) for i in range(len(scoring_criteria))] 

    prompt_parts = [
        role,
        "# Instruction",
        "제공된 [Dialogue]를 [Scoring Criteria]에 따라 평가하고, 각 항목별 점수(1-5점)와 상세 이유를 JSON 형식으로 출력하세요.",
        "**매우 중요: JSON 응답의 `scoring` 및 `explanation` 객체에서는 반드시 각 평가 기준 항목의 인덱스(0, 1, 2, ...)를 문자열 키('0', '1', '2', ...)로 사용해야 합니다.**",
        "# Context",
        "## Scoring Criteria (Index: ID(Name): Description)",
        criteria_descriptions, # 인덱스와 함께 기준 표시
        "## Dialogue",
        dialogue,
        "",
        "## Output Format **[매우 중요!]**",
        "- 평가는 반드시 아래 명시된 JSON 형식을 정확히 따라야 합니다.",
        f"- 'scoring' 객체의 키는 반드시 다음 인덱스 문자열 목록에 있는 값만 **정확히** 사용해야 합니다: `{valid_index_keys}`", # 인덱스 키 목록 강조
        f"- 'explanation' 객체의 키도 반드시 동일한 인덱스 문자열 목록을 **정확히** 사용해야 합니다: `{valid_index_keys}`", # 인덱스 키 목록 강조
        "- 모든 문자열 값은 큰따옴표(\"\")로 감싸야 합니다.",
        "- 점수는 1~5 사이의 정수만 사용하세요.",
        "- **JSON 객체 외에 다른 설명이나 텍스트를 절대 포함하지 마세요.**",
        "```json",
        "{\n  \"scoring\": {\n    \"0\": 0, // Index 0 점수\n    \"1\": 0, // Index 1 점수\n    \"2\": 0, // Index 2 점수\n    \"3\": 0  // Index 3 점수\n  },\n  \"explanation\": {\n    \"0\": \"Index 0 점수 이유\",\n    \"1\": \"Index 1 점수 이유\",\n    \"2\": \"Index 2 점수 이유\",\n    \"3\": \"Index 3 점수 이유\"\n  }\n}", # 프롬프트 내에 직접 예시 제공
        "```",
        f"**다시 한번 강조합니다: `scoring` 및 `explanation` 객체의 키는 반드시 `{valid_index_keys}` 목록의 인덱스 문자열과 정확히 일치해야 합니다.**" # 키 목록 재강조
    ]

    return "\n".join(prompt_parts)

# 주석 처리된 Meta-Plan 부분은 그대로 유지하거나 필요시 제거 가능
# - 다음의 Meta-plan 단계를 따라 생각한 뒤 대화를 생성하세요.
# ... (이하 Meta-Plan 주석) ...
from typing import List, Dict, Any

def get_category_prompt(prompt_category: str,
                        conversation_summary: str, 
                        scoring_results: Dict[str, Any]):
    """Generate prompt for coaching category recommendation"""
    # Format evaluation results correctly from the expected structure
    formatted_evaluation = "평가 결과 분석 실패"
    if isinstance(scoring_results, dict): 
        scoring_dict = scoring_results.get('scoring') # Get inner dict
        explanation_dict = scoring_results.get('explanation') # Get inner dict
        
        # Check if both inner dicts exist and are dicts
        if isinstance(scoring_dict, dict) and isinstance(explanation_dict, dict):
            evaluation_lines = []
            # Iterate through the keys (criteria IDs) of the scoring dictionary
            for criterion_id in scoring_dict.keys():
                score = scoring_dict.get(criterion_id, 'N/A') # Get score using key
                explanation = explanation_dict.get(criterion_id, '설명 없음') # Get explanation using key
                # Use criterion_id (key) in the formatted string
                evaluation_lines.append(f"- {criterion_id} (점수: {score}/5): {explanation}") 
            
            if evaluation_lines:
                 formatted_evaluation = "\n".join(evaluation_lines)
            else:
                 formatted_evaluation = "유효한 평가 항목 없음"
        else:
             formatted_evaluation = "평가 결과의 scoring 또는 explanation 키/구조 오류"
    else:
        formatted_evaluation = f"평가 결과 타입 오류 (예상: dict, 실제: {type(scoring_results)})"

    # Use a regular triple-quoted string, not an f-string
    prompt = """
{role}
# Instruction
아래의 대화 분석 결과를 바탕으로, 가장 시급한 개선이 필요한 카테고리를 **1개 이상, 최대 3개까지** 선정해주세요.
# Context
## Evaluation Results
{evaluation_results_placeholder} 

# Category Information
- 반드시 카테고리 정보를 참고하여 카테고리를 선정해야 합니다.
{prompt_category}

# Output Format
**[출력 형식 (매우 중요!)]**
- 반드시 아래 명시된 **JSON 형식**으로만 응답해야 합니다.
- 최상위 키는 `"selected_categories"` (JSON 배열)와 `"overall_reason"` (문자열) 이어야 합니다.
- `"selected_categories"` 배열 안에는 선택된 각 카테고리에 대한 **JSON 객체**가 포함됩니다.
- 각 카테고리 객체는 반드시 `"id"`, `"name"`, `"priority"` (숫자 1-3), `"reason"` (문자열) 키를 가져야 합니다.
- **객체와 객체 사이에는 반드시 쉼표(,)**를 사용하세요 (배열 마지막 객체 제외).
- 모든 키와 문자열 값은 **큰따옴표("")**로 감싸야 합니다.

**[정확한 JSON 구조 예시]**
```json
{{
  "selected_categories": [
    {{"id": "선택된_카테고리_ID_1", "name": "선택된_카테고리_이름_1", "priority": 1, "reason": "선정 이유 1 (대화 내용 인용)"}},
    {{"id": "선택된_카테고리_ID_2", "name": "선택된_카테고리_이름_2", "priority": 2, "reason": "선정 이유 2 (대화 내용 인용)"}}
  ],
  "overall_reason": "종합적인 선정 이유 요약"
}}
```

[주의사항]
- 반드시 실제 대화 내용과 평가 결과를 기반으로 카테고리를 선정하세요.
- 우선순위는 1(가장 시급)~3(덜 시급) 사이의 숫자로 표시하세요.
- 선정 이유는 구체적인 대화 내용을 인용하여 설명하세요.
"""
    
    # Use .format() to insert all required values into the standard string
    return prompt.format(
        role=role,
        conversation_summary=conversation_summary,
        evaluation_results_placeholder=formatted_evaluation, # Insert the correctly formatted string
        prompt_category=prompt_category
    )

def get_plan_prompt(categories, scoring_results, relevant_plans_info: str):
    """Generate prompt for specific coaching plan recommendation"""
    prompt = f"""
{role}
- 아래 정보들을 종합적으로 고려하여, 아이에게 가장 적합한 구체적인 코칭 플랜들을 추천하고 그 이유와 기대 효과를 설명해주세요.

# Context
## Conversation Scoring
{scoring_results}

## Selected Categories
{categories}

## Relevant Plans Information
{relevant_plans_info}

# Output Format
**[출력 형식 (매우 중요!)]**
```json
{{
  "conversation_analysis": {{
    "overall_analysis": "전반적 분석",
    "strengths": "강점",
    "improvement_points": "개선점"
  }},
  "recommended_plans": [
    {{
      "category": "카테고리",
      "plan_name": "플랜명",
      "reason": "추천 이유",
      "expected_effect": "기대 효과"
    }}
  ]
}}
```
반드시 위의 JSON 형식을 정확히 준수하여 응답하세요. 특히 `recommended_plans` 배열 내 각 객체의 키(`plan_id`, `plan_name`, `category_id`, `recommend_reason`, `expected_effect`)를 정확히 사용해야 합니다.

[추천 가이드라인]
- 위 '[참고 플랜 설명]'에 제시된 플랜들을 우선적으로 고려하되, 필요시 연령 및 대화 맥락에 더 적합한 다른 플랜을 추천할 수도 있습니다.
- 추천 이유(`recommend_reason`)는 반드시 대화 내용이나 평가 결과를 근거로 구체적으로 작성해야 합니다.
- 기대 효과(`expected_effect`)는 해당 플랜을 통해 아이의 어떤 점이 개선될 수 있는지 명확하게 기술해야 합니다.
"""
    # Return the fully formatted string (using f-string for simplicity here)
    prompt.format(role=role)
    return prompt