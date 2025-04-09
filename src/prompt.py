from typing import Dict, Any, List
import json
from src.prep_map_category import category_name_english, category_name, category_id,  plan_name, plan_id, category_keywords, category_description, plan_keywords, plan_description
#theme_name_english, theme_name, theme_id, plan_name_english,

# - 전문가 경험, 가족 페르소나, 도움이 될 수 있는 코칭 플랜 정보를 종합해 고려합니다.
role = """
# Role
- 당신은 수십 년간 부모와 아이, 가족 간의 갈등 및 부정적인 대화를 관찰하고 분석해온 심리 상담 전문가 입니다.
- 당신은 따뜻하면서도 세심한 시선으로 주의 깊게 가족의 대화를 들여다 보고, 각 가족에게 상황에 적합한 코칭 플랜을 추천하고 그 이유를 논리적이면서도 쉽게 이해할 수 있도록 설명해주는 전문가 입니다.
- 당신 덕분에 많은 가족들은 육아 과정에서 더 나은 방법을 찾아가고, 더 나은 관계를 유지할 수 있습니다.
"""

def gen_plan_info(df_plan, is_keywords: bool = False, is_example: bool = False, is_description: bool = False):
    """
    Generates LLM prompts listing categories with their IDs and descriptions.
    (카테고리 목록을 ID, 이름, 설명과 함께 명확하게 생성하도록 개선)

    Args:
        df_category (pd.DataFrame): DataFrame containing category information.

    Returns:
        str: A formatted string listing categories for the LLM prompt.
    """
    df_unique_categories = df_plan.drop_duplicates(subset=[plan_id])
    prompt_lines = ["## Available Plans"]
    
    for _, row in df_unique_categories.iterrows():
        keyword = row[plan_keywords]
        prompt_lines.append(f"  ID: {row[plan_id]}")
        prompt_lines.append(f"   Plan Name: {row[plan_name]}")
        if is_description:
            prompt_lines.append(f"   Description: {row[plan_description]}")
        if is_example:
            cols_examples = ['example_'+str(i+1) for i in range(5)]
            examples = [row[col] for col in cols_examples]
            prompt_lines.append(f"   Examples: {examples}")
        if is_keywords:
            prompt_lines.append(f"   Keywords: {keyword}")
        prompt_lines.append("")

    return "\n".join(prompt_lines)

def gen_prompt_category_info(df_category):
    """
    Generates LLM prompts listing categories with their IDs and descriptions.
    (카테고리 목록을 ID, 이름, 설명과 함께 명확하게 생성하도록 개선)

    Args:
        df_category (pd.DataFrame): DataFrame containing category information.

    Returns:
        str: A formatted string listing categories for the LLM prompt.
    """
    df_unique_categories = df_category.drop_duplicates(subset=[category_name_english])
    prompt_lines = ["## Available Categories (ID, Name, Description)"]
    
    for _, row in df_unique_categories.iterrows():
        category_id_val = row[category_id]
        eng_name = row[category_name_english]
        korean_name = row[category_name]
        # description = row[category_description]
        keyword = row[category_keywords]
        
        prompt_lines.append(f"   ID: {category_id_val}")
        prompt_lines.append(f"   Category Name: {eng_name} / {korean_name}")
        # prompt_lines.append(f"   Description: {description}")
        prompt_lines.append(f"   Keywords: {keyword}")
        prompt_lines.append("")

    return "\n".join(prompt_lines)
def generate_category_info_only(df_category):
    """
    Generates LLM prompts listing categories with their IDs and descriptions.
    (카테고리 목록을 ID, 이름, 설명과 함께 명확하게 생성하도록 개선)

    Args:
        df_category (pd.DataFrame): DataFrame containing category information.

    Returns:
        str: A formatted string listing categories for the LLM prompt.
    """
    df_unique_categories = df_category.drop_duplicates(subset=[category_name_english])
    prompt_lines = ["## Available Categories"]
    
    for _, row in df_unique_categories.iterrows():
        category_id_val = row[category_id]
        eng_name = row[category_name_english]
        korean_name = row[category_name]
        # description = row[category_description]
        keyword = row[category_keywords]
        
        prompt_lines.append(f"   ID: {category_id_val}")
        prompt_lines.append(f"   Category Name: {eng_name} / {korean_name}")
        prompt_lines.append(f"   Keywords: {keyword}")
        prompt_lines.append("")

    return "\n".join(prompt_lines)

def prep_survey_info(context,  top_k=3) -> str:
    priority_interests = context.get('priority_interests')
    priority_interests = priority_interests[:top_k]
    processed = ""
    for i in priority_interests:
        processed += f"- 우선순위 {i.get('priority')}. 관심 주제: {i.get('category_korean')}\\n- 키워드: {', '.join(i.get('keywords'))}\\n"
    return processed

def generate_category_from_survey(context, category_info, child_age, top_k=3) -> str:
    persona = context.get('persona')
    processed = prep_survey_info(context, top_k)
    
    prompt_template = """
{role}

# Instruction
- 주어진 Family persona, Priority Interests 정보를 바탕으로 가족 페르소나를 생성하고, 카테고리를 선택해야 합니다.
- 가족 페르소나는 자녀와 양육자의 페르소나 정보를 각 1명 이상 포함해야 합니다.
- 카테고리 목록([Available Categories])에 제공된 ID 중에서만 선택해야 합니다.
- 제공된 가족 정보([Family Persona], [Priority Interests])와 [카테고리 정보]를 종합적으로 고려하세요.
- **`explanation` 필드는 반드시 한국어로만 작성해야 합니다. 영어 단어나 문장을 사용하지 마세요.**
- 아이 나이는 주어진 정보를 바탕으로, 각 나이에 맞는 언어를 구사해야 합니다.
# Context
### Family Persona
- 아이 나이: {child_age} 개월
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
- **`persona` 필드는 자녀와 양육자의 페르소나 정보를 포함해야 합니다.**
- 모든 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트는 포함하지 마세요.
```json
{{
  "category_id": int, // 예: 101 (카테고리 ID)
  "explanation": "string" // 예: "가족 페르소나의 OOO 측면과 우선순위 관심사 중 XX 키워드가 카테고리 ID 101(life_event_focused)의 설명 및 키워드와 가장 관련성이 높아 선택했습니다."
  "persona": {{
    "child": [{{
        "name": "string", // 자녀 이름
        "age": "string", // 자녀 나이
        "role": "string", // 자녀 역할. 예: 첫째 아들, 둘째 딸, 셋째 아들 등
        "gender": "string", // 자녀 성별
        "personality": "string", // 자녀 성격
        }},
        // ...
        ],
    "parent": [{{
        "name": "string", // 양육자 이름
        "age": "string", // 양육자 나이
        "role": "string", // 양육자 역할. 예: 엄마, 아빠,  양육자 등
        "gender": "string", // 양육자 성별
        "personality": "string", // 양육자 성격
        }},
        // ...
        ]
  }}
}}
```
"""
    prompt = prompt_template.format(role=role, persona=persona, processed=processed, 
                                    category_info=category_info, child_age=child_age)
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

def gen_dialogue_prompt(i_persona, i_category_name: str, i_df_plans, child_persona, parent_persona, is_keywords: bool = False, is_description: bool = False, is_example: bool = False) -> str:
    """심리 상담사 프롬프트를 생성합니다. (카테고리 이름을 직접 받도록 수정)"""
    global role
    cols_examples = [col for col in i_df_plans.columns if 'example_' in col]
    persona = i_persona
    i_plan_name = i_df_plans[plan_name].iloc[0]
    if is_keywords:
        i_keyword = i_df_plans[plan_keywords].iloc[0]
    else:
        i_keyword = ""
    if is_description:
        i_description = i_df_plans[plan_description].iloc[0]
    else:
        i_description = ""
    if is_example:
        example_list = extract_examples_from_plan(i_df_plans, cols_examples)
        examples = [f"- {example}" for example in example_list]
        examples = "\n".join(examples)
    else:
        examples = ""

    print(f"Examples:\n{examples}")
  
    prompt_template = """
{role}
# Instruction
- 자주 경험했던 부모와 아이, 또는 양육자 간의 다양한 갈등, 부정적인 대화를 떠올려 자연스럽게 재구성해 주세요.
- 대화는 총 5 ~ 15 턴 내외로 생성해 주세요.
- 이모지, 특수 문자 등은 사용하지 않습니다.
- 모두 한국어로 작성해 주세요.

# Context
- 등장 인물은 양육자인 엄마, 아빠, 아이들, 주변 인물 중, 주어진 맥락에 적절한 1-3인을 선택해 주세요.
- 특히 양육자 부모의 부정적인 태도가 잘 드러날 수록 좋습니다.
- 아이의 나이는 주어진 정보를 바탕으로, 각 나이에 맞는 언어를 구사해야 합니다.
- 모르는 경우 3세 근처로 가정합니다. (어휘가 부족하고 유창하지 않아야 합니다.)
- 모든 인물의 성격 및 대화는 일관성을 유지해야 합니다.
- examples 문장에 담긴 **핵심 의미나 의도**를 반영하는 부모의 발언을 최소 1회 이상 자연스럽게 포함하세요. (반드시 원문 그대로 사용할 필요는 없습니다.)
- examples 문장은 모두 부모가 아이에게 하는 말입니다.

## 가족 페르소나 정보
{persona}

## 자녀 페르소나 정보
{child_persona}

## 양육자 페르소나 정보
{parent_persona}

## 코칭 플랜
- 카테고리: {i_category_name} 
- 플랜: {i_plan_name}
- 키워드: {i_keyword}
- 설명: {i_description}

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
  "age": "string" // 아이 연령
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
        i_category_name=i_category_name, # 전달받은 실제 카테고리 이름 사용
        i_plan_name=i_plan_name, 
        i_keyword=i_keyword, 
        i_description=i_description,
        examples=examples,
        child_persona=child_persona,
        parent_persona=parent_persona
    )
    return prompt, examples

from typing import List, Dict, Any

def get_scoring_prompt(dialogue: str, scoring_criteria: List[Dict], output_template_str: str, detailed_criteria: bool = False, child_age: str = '') -> str:
    """스코어링을 위한 프롬프트를 생성합니다. (인덱스 키 사용, 상세 기준 옵션 추가)"""
    
    # scoring_criteria 리스트에서 정보를 포함한 문자열 생성
    description_lines = []
    if detailed_criteria:
        description_lines.append("[Scoring Criteria]")
        for idx, c in enumerate(scoring_criteria):
            description_lines.append(f"- Index {idx}: `{c['id']}` ({c.get('name', '')})")
            description_lines.append(f"  Description: {c.get('description', '')}")
            # Aspects 추가
            if 'aspects' in c and isinstance(c['aspects'], list):
                description_lines.append("  Aspects:")
                description_lines.extend([f"    - {aspect}" for aspect in c['aspects']])
            # Scoring Guide 추가
            if 'scoring_guide' in c and isinstance(c['scoring_guide'], dict):
                description_lines.append("  Scoring Guide:")
                for score, guide_text in sorted(c['scoring_guide'].items()):
                    description_lines.append(f"    {score}: {guide_text}")
            description_lines.append("") # 기준 간 간격 추가
    else: # 기본 (간단) 버전
        description_lines.append("[Scoring Criteria]")
        for idx, c in enumerate(scoring_criteria):
            description_lines.append(f"- Index {idx}: `{c['id']}` ({c.get('name', '')})")
            
    criteria_descriptions = "\n".join(description_lines)
    
    # LLM에게 사용할 정확한 인덱스 키 목록 (문자열 형태)
    valid_index_keys = [str(i) for i in range(len(scoring_criteria))] 

    prompt_parts = [
        role,
        "# Instruction",
        "- 제공된 [Dialogue]를 [Scoring Criteria]에 따라 평가하고, 각 항목별 점수(1-5점)와 상세 이유를 JSON 형식으로 출력하세요.",
        "- **매우 중요: JSON 응답의 `scoring` 및 `explanation` 객체에서는 반드시 각 평가 기준 항목의 인덱스(0, 1, 2, ...)를 문자열 키('0', '1', '2', ...)로 사용해야 합니다.**",
        "- 다음 [Child Age] 의 아이 나이 정보를 참고해 주세요. [Dialogue] 에 언급되는 나이는 무시해 주세요.",
        "- 부모의 양육 태도에 초점을 두고 평가해 주세요.",
        "",
        "# Context",
        
        f"[Child Age]: {child_age} 개월",
        # criteria_descriptions, # 수정 전: 위치 변경
        "[Dialogue]",
        dialogue,
        "",
        criteria_descriptions, # Dialogue 뒤로 이동하여 기준을 더 잘 볼 수 있게 함
        "",
        "## Output Format **[매우 중요!]**",
        "- 평가는 반드시 아래 명시된 JSON 형식을 정확히 따라야 합니다.",
        f"- 'scoring' 객체의 키는 반드시 다음 인덱스 문자열 목록에 있는 값만 **정확히** 사용해야 합니다: `{valid_index_keys}`",
        f"- 'explanation' 객체의 키도 반드시 동일한 인덱스 문자열 목록을 **정확히** 사용해야 합니다: `{valid_index_keys}`",
        "- 모든 문자열 값은 큰따옴표(\"\")로 감싸야 합니다.",
        "- 점수는 1~5 사이의 정수만 사용하세요.",
        "- **JSON 객체 외에 다른 설명이나 텍스트를 절대 포함하지 마세요.**",
        "```json",
        output_template_str.strip(), # YAML에서 로드한 템플릿 사용
        "```",
        f"**다시 한번 강조합니다: `scoring` 및 `explanation` 객체의 키는 반드시 `{valid_index_keys}` 목록의 인덱스 문자열과 정확히 일치해야 합니다.**"
    ]

    return "\n".join(prompt_parts)

# 주석 처리된 Meta-Plan 부분은 그대로 유지하거나 필요시 제거 가능
# - 다음의 Meta-plan 단계를 따라 생각한 뒤 대화를 생성하세요.
# ... (이하 Meta-Plan 주석) ...
from typing import List, Dict, Any

def format_scoring_results(scoring_results: Dict[str, Any]) -> str:
    formatted_evaluation = "평가 결과 분석 실패"
    if isinstance(scoring_results, dict):
        scoring_dict = scoring_results.get('scoring') # Get inner dict
        explanation_dict = scoring_results.get('explanation') # Get inner dict

        # Check if both inner dicts exist and are dicts
        if isinstance(scoring_dict, dict) and isinstance(explanation_dict, dict):
            evaluation_lines = []
            # Iterate through the keys (assumed to be index strings like '0', '1', ... based on scoring prompt)
            for index_key in explanation_dict.keys(): # explanation_dict의 키를 기준으로 반복
                score = scoring_dict.get(index_key, 'N/A') # 스코어 가져오기
                explanation_value = explanation_dict.get(index_key, '설명 없음') # 설명 가져오기
                
                # explanation_value가 문자열인지 확인 (TypeError 방지)
                if not isinstance(explanation_value, str):
                    print(f"경고: 스코어링 결과의 explanation 항목 '{index_key}' 값이 문자열이 아닙니다 (타입: {type(explanation_value)}). 값: {explanation_value}")
                    explanation = "[설명 형식 오류]" # 기본값 사용
                else:
                    explanation = explanation_value

                evaluation_lines.append(f"- 항목 {index_key} (점수: {score}/5): {explanation}") 
            
            if evaluation_lines:
                 formatted_evaluation = "\n".join(evaluation_lines)
            else:
                 formatted_evaluation = "유효한 평가 항목 없음"
        else:
             formatted_evaluation = "평가 결과의 scoring 또는 explanation 키/구조 오류"
    else:
        formatted_evaluation = f"평가 결과 타입 오류 (예상: dict, 실제: {type(scoring_results)})"

    return formatted_evaluation
def get_category_prompt(prompt_category: str,
                        child_age: str,
                        survey_info: str,
                        scoring_results: Dict[str, Any]):
    """Generate prompt for coaching category recommendation (지시 명확화, explanation 타입 검사 추가)"""
    # Format evaluation results correctly from the expected structure
    
    # 대화 내용을 JSON 문자열로 변환 (프롬프트에 포함시키기 위함)
    # dialogue 변수가 실제 대화 데이터 딕셔너리라고 가정
    #dialogue_str = json.dumps(dialogue, ensure_ascii=False, indent=2) if isinstance(dialogue, dict) else str(dialogue)
    formatted_evaluation = format_scoring_results(scoring_results)
    # 프롬프트 템플릿 수정 (지시 명확화, Dialogue 컨텍스트 추가)
    prompt_template = """
{role}
# Instruction
- **매우 중요:** 당신의 임무는 아래 제공된 [Child Age], [Survey Information], [Evaluation Results], [Category Information] **정보를 종합적으로 분석**하여, 육아 과정에 개선 및 보완이 필요한 카테고리를 3개 이상 선정하고, 그 결과를 **반드시 지정된 [Output Format]에 따라 JSON 형식으로 출력**하는 것입니다.
- 관련이 매우 낮은 카테고리는 제외하되, 가능한 많은 카테고리를 선택해 높은 관련성 및 우선순위 순서대로 나열해 주세요.

# Context
[Child Age]
- 아이 연령: {child_age} 개월

[Survey Information]
{survey_info}

[Evaluation Results]
- 다음은 육아 과정 대화에 대한 평가 결과입니다. 각 항목의 점수(낮을수록 문제)와 설명을 참고해 주세요.
{evaluation_results_placeholder}

[Category Information]
- 반드시 아래 목록에서 카테고리를 선택해야 합니다. 각 카테고리의 ID, 이름, 관련 키워드를 주의 깊게 살펴보고, 높은 관련성, 우선순위부터 차례대로 선택해 주세요.
- 만약 모든 항목의 점수가 높거나 특별한 추천 항목이 생각나지 않는 경우에는, category id: 1 "발달_단계_연령_중심" 을 우선 추천하고, 적절한 이유를 설명해 주세요.
{prompt_category}

# Output Format
**[출력 형식 (매우 중요!)]**
- **반드시** 아래 명시된 **JSON 형식**으로만 응답해야 합니다. 다른 텍스트는 절대 포함하지 마세요.
- 최상위 키는 `"selected_categories"` (JSON 배열)와 `"overall_reason"` (문자열) 이어야 합니다.
- `"selected_categories"` 배열 안에는 선택된 각 카테고리에 대한 **JSON 객체**가 포함됩니다.
- 각 카테고리 객체는 반드시 `"id"` (카테고리 정보에 명시된 ID), `"name"` (카테고리 이름), `"reason"` (선정 이유, **반드시 대화 내용과 평가 결과 내용 모두 인용**) 키를 가져야 합니다.
- 배열 내 객체와 객체 사이에는 반드시 쉼표(,)를 사용하세요 (마지막 객체 제외).
- 모든 키와 문자열 값은 **큰따옴표("")**로 감싸야 합니다.

**[정확한 JSON 구조 예시]**
```json
{{
  "selected_categories": [
    {{"id": "선택된_카테고리_ID_1", "name": "선택된_카테고리_이름_1",  "reason": "선정 이유 1 (대화 중 '...' 발언과 평가 결과 항목 X의 내용 '...'을 볼 때 시급히 개선 필요)"}},
    {{"id": "선택된_카테고리_ID_2", "name": "선택된_카테고리_이름_2",  "reason": "선정 이유 2 (대화 중 '...' 상황과 평가 결과 항목 Y의 내용 '...' 관련하여 개선 필요)"}}
  ],
  "overall_reason": "종합적인 선정 이유 요약 (예: 평가 결과를 종합할 때 ID 1과 ID 2 카테고리 개선이 가장 중요하다고 판단됨)"
}}
```

[주의사항]
- 선정 이유는 반드시 **[Dialogue] 및 [Evaluation Results] 섹션에 제시된 구체적인 내용**을 근거로 작성하세요.
- **다시 한번 강조합니다: 추가 질문이나 JSON 외의 응답은 절대 하지 마세요. 주어진 정보로 JSON만 생성하세요.**
"""

    # Use .format() to insert all required values into the standard string
    return prompt_template.format(
        role=role,
        child_age=child_age,
        evaluation_results_placeholder=formatted_evaluation,
        prompt_category=prompt_category,
        survey_info=survey_info
    )

def get_plan_prompt(scoring_results, relevant_plans_info: str, child_age: str, dialogue: str, survey_info: str, is_dialogue: bool = False):
    """Generate prompt for specific coaching plan recommendation (f-string 제거 및 포맷팅 개선)"""
    # 딕셔너리를 읽기 좋은 JSON 문자열로 포맷팅
    formatted_scoring_results = format_scoring_results(scoring_results)
    if is_dialogue:
        context = f"[Dialogue]\n{dialogue}"
    else:
        context = ""
    # 표준 문자열 사용 (f-string 아님)
    prompt_template = """
{role}
# Instruction
- 아래 정보들을 종합적으로 고려하여, 부모에게 권하고 싶은 플랜들을 3개 이상, 가능한 많이 우선 순위대로 추천하고, 그 이유와 기대 효과를 설명해주세요.
- 반드시'[Relevant Plans Information]'에 제시된 플랜들 중에서만 추천해 주세요.
- 추천 이유(`reason`)는 반드시 대화 내용이나 평가 결과를 근거로 구체적으로 작성해야 합니다.
- 기대 효과(`expected_effect`)는 해당 플랜을 통해 부모의 어떤 점이 개선될 수 있는지 명확하게 기술해야 합니다.
- 대화에 직접 드러나는 인물 및 내용을 참고하여 추천 플랜을 선택해야 합니다.
- 대화에 드러나지 않은 다른 가족에 대한 부분은 함부로 짐작하지 마세요.

# Context
[Child Age]
- 아이 연령: {child_age} 개월
- [Dialogue] 에 언급되는 나이는 무시해 주세요.

[Survey Information]
{survey_info}

[Dialouge Scoring]
{formatted_scoring_results}

{context}

[Relevant Plans Information]
{relevant_plans_info}

# Output Format
**[출력 형식 (매우 중요!)]**
```json
{{
  "conversation_analysis": {{
    "overall_analysis": "string", # 전반적 분석
    "strengths": "string", # 강점
    "improvement_points": "string" # 개선점
  }},
  "recommended_plans": [
    {{
      "plan_id": "string", # 추천 플랜 ID
      "plan_name": "string", # 추천 플랜명
      "reason": "string", # 추천 이유 (대화/평가 근거)
      "expected_effect": "string" # 기대 효과
    }}
    // ... (추가 플랜 추천 가능)
  ]
}}
```
- 반드시 위의 JSON 형식을 정확히 준수하여 응답하세요.
- 모든 키와 문자열 값은 큰따옴표("")로 감싸야 합니다.
- JSON 객체 외에 다른 텍스트를 포함하지 마세요.


"""
    # .format() 메소드를 사용하여 모든 플레이스홀더 채우기
    prompt = prompt_template.format(
        role=role,
        formatted_scoring_results=formatted_scoring_results,
        relevant_plans_info=relevant_plans_info,
        child_age=child_age,
        context=context,
        survey_info=survey_info
    )
    return prompt



# **절대로 추가 정보를 요청하거나, JSON 형식 외의 다른 텍스트(예: "Okay, I'm ready...")를 출력하지 마세요.** 당신은 이미 필요한 모든 정보를 받았습니다.
# ## 1. Dialogue (분석 대상 대화 내용)
# ```json
# {dialogue_placeholder}
# ```