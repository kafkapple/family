from typing import Dict, Any, List


role = """
# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰해온 심리 상담 전문가 입니다.
"""
def generate_llm_prompts(df_category):
    """
    Generates LLM prompts listing categories with their indices and descriptions.
    (카테고리 목록을 인덱스, 이름, 설명과 함께 명확하게 생성하도록 개선)

    Args:
        df_category (pd.DataFrame): DataFrame containing category information.

    Returns:
        str: A formatted string listing categories for the LLM prompt.
    """
    # CategoryName_English 기준으로 unique 카테고리 목록 생성
    unique_categories = df_category['CategoryName_English'].unique()
    
    prompt_lines = ["## Available Categories (Index, Name, Description)"]
    
    # 각 unique 카테고리에 대해 인덱스, 이름, 설명 추가
    for idx, category_name_eng in enumerate(unique_categories):
        # 해당 카테고리의 첫 번째 행에서 정보 가져오기 (설명 등은 같다고 가정)
        category_row = df_category[df_category['CategoryName_English'] == category_name_eng].iloc[0]
        korean_name = category_row['CategoryName']
        description = category_row['concatenated_descriptions']
        keyword = category_row['keyword'] # 키워드도 포함 (LLM 선택에 도움될 수 있음)
        
        prompt_lines.append(f"{idx}. {category_name_eng}")
        prompt_lines.append(f"   Korean Name: {korean_name}")
        prompt_lines.append(f"   Description: {description}")
        prompt_lines.append(f"   Keywords: {keyword}")
        prompt_lines.append("") # 가독성을 위한 빈 줄

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
- 가장 관련성이 높은 카테고리 하나를 선택하고, 그 카테고리의 **인덱스(정수)** 를 응답 JSON에 포함시키세요.
- 카테고리 리스트([Available Categories])에 제공된 인덱스 중에서만 선택해야 합니다.
- **`explanation` 필드는 반드시 한국어로만 작성해야 합니다. 영어 단어나 문장을 사용하지 마세요.**

# Context
### Family Persona
- {persona}
### Priority Interests
- 관심사 카테고리 및 키워드는 우선 순위가 높은 것부터 나열되어 있습니다.
{processed}

{category_info} 

# Output Format
- 반드시 다음 JSON 형식에 맞춰 결과를 출력해 주세요.
- **`category_index` 필드에는 선택한 카테고리의 인덱스(정수)를 넣어야 합니다.**
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
```json
{{
  "category_index": int, // 선택된 카테고리의 인덱스 (0부터 시작하는 정수)
  "explanation": "string" // 위 카테고리 인덱스를 선택한 이유를 설명하는 문자열
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
- 반드시 갈등 또는 부정적인 감정을 다양한 방식으로 포함하여 부모와 아이, 또는 양육자 간의 대화를 만들어 주세요.
- 대화는 총 5 ~ 15 턴 내외로 생성해 주세요.
- 반드시 모두 한국어로 작성해 주세요.

# Context
- 가족 페르소나, 코칭 플랜 정보를 종합해서 대화를 생성해 주세요.
- 등장 인물은 양육자인 엄마 아빠, 아이들, 주변 인물 중 맥락에 적절한 1-3인을 선택해 주세요.
- 특히 양육자 부모의 부정적인 태도가 잘 드러날 수록 좋습니다.
- 아이의 나이는 아래 주어진 정보를 바탕으로, 각 나이에 맞는 언어를 구사해야 합니다.
- 모르는 경우 3세 근처로 가정합니다. (어휘가 부족하고 유창하지 않아야 합니다.)
- 모든 인물의 성격 및 대화는 일관성을 유지해야 합니다.

- examples 문장에 담긴 **핵심 의미나 의도**를 반영하는 부모의 발언을 최소 1회 이상 자연스럽게 포함하세요. (반드시 원문 그대로 사용할 필요는 없습니다.)
- examples 문장은 모두 부모가 아이에게 하는 말입니다.

## 가족 페르소나 정보
{persona}

## 코칭 플랜
다음은 도움이 필요한 대화 상황과 관련 있는 코칭 카테고리 및 플랜 정보 입니다. 
- 카테고리: {category_name} # 전달받은 실제 카테고리 이름 사용
- 플랜: {plan_name}
- 키워드: {keyword}
- 설명: {description}

## examples
{examples}

# Output Format
- 반드시 다음 JSON 형식에 맞춰 대화 내용을 생성해 주세요.
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
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
        # output_template_str # 설정 파일의 템플릿을 직접 사용하는 대신, 아래 지침으로 대체하거나, 템플릿 자체를 수정해야 함
        "{\n  \"scoring\": {\n    \"0\": 0, // Index 0 점수\n    \"1\": 0, // Index 1 점수\n    \"2\": 0, // Index 2 점수\n    \"3\": 0  // Index 3 점수\n  },\n  \"explanation\": {\n    \"0\": \"Index 0 점수 이유\",\n    \"1\": \"Index 1 점수 이유\",\n    \"2\": \"Index 2 점수 이유\",\n    \"3\": \"Index 3 점수 이유\"\n  }\n}", # 프롬프트 내에 직접 예시 제공 (템플릿 대신)
        "```",
        f"**다시 한번 강조합니다: `scoring` 및 `explanation` 객체의 키는 반드시 `{valid_index_keys}` 목록의 인덱스 문자열과 정확히 일치해야 합니다.**" # 키 목록 재강조
    ]

    return "\n".join(prompt_parts)



# - 다음의 Meta-plan 단계를 따라 생각한 뒤 대화를 생성하세요.

# ## [Meta-Plan 단계]
# 1. 이 문제 상황의 심리적 맥락은 무엇인가요?
# 2. 문제가 되는 키워드 문장이 어떻게 자연스럽게 대화에 포함될 수 있을까요?
# 3. 각 인물의 대화는 일관성을 유지하고 있나요?
