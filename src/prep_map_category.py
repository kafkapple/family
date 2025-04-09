import pandas as pd
import os
from pathlib import Path
# Define mappings for ClassName and PlanName to their English equivalents
# level1_to_english = {
#     '입문': 'Intro',
#     '발달': 'Development',
#     '생활': 'Life',
#     '공감': 'Empathy'
# }
category_to_english = {
    '입문': 'intro',
    '발달_단계_연령_중심': 'developmental_age_focused',
    '시기_생활_이벤트_중심': 'life_event_focused',
    '부모_양육자_대화_및_자기관리': 'parent_communication_selfcare',
    '감정_욕구_이해': 'emotion_needs_understanding',
    '훈육_규범_설정_관련': 'discipline_norm_setting',
    '자조능력_독립성_관련': 'self_help_independence',
    '놀이_상호작용': 'play_interaction',
    '관계_소통_전반': 'relationship_communication'
}
theme_to_english = {
'입문': 'intro',
'발달': 'development',
'등하원 기관 적응': 'adaptation',
'놀이터': 'playground',
'양육자 자기연결': 'parent_self_connection',
'육아 중 부부대화': 'couple_talk',
'육아번아웃 다루기': 'parent_burnout',
'기관-동료양육자들과의 소통법': 'communication_with_peers',
'감정과 욕구': 'emotion_needs',
'생활지도': 'life_guidance',
'형제남매친구 중재': 'sibling_mediation',
'자조능력': 'self_help',
'놀이 상호작용': 'play_interaction',
'미디어 사용': 'media_use',
'거절하기와 거절듣기': 'rejection_skills',
'칭찬하기': 'giving_praise',
}
plan_to_english = {
    '체험판': 'trial',
    '첫걸음': 'first_step',
    '3T': '3t',
    'NVC': 'nvc',
    '생애초기 (intro)': 'early_life_intro',
    '두돌': 'two_years_old',
    '세돌': 'three_years_old',
    '네돌 (outro)': 'four_years_old_outro',
    '등하원 기관 적응 (만1세반)': 'adaptation_1y',
    '등하원 기관 적응 (만2세반)': 'adaptation_2y',
    '등하원 기관 적응 (유치원 만3세반)': 'adaptation_3y',
    '놀이터 (영아)': 'playground_infant',
    '놀이터 (유아)': 'playground_toddler',
    '양육자 자기연결': 'parent_self_connection',
    '육아 중 부부대화 (영아)': 'couple_talk_infant',
    '육아 중 부부대화 (유아)': 'couple_talk_toddler',
    '육아번아웃 다루기': 'parent_burnout',
    '기관-동료양육자들과의 소통법': 'communication_with_peers',
    '감정과 욕구 (태담)': 'emotion_needs_prenatal',
    '감정과 욕구 (발화 전)': 'emotion_needs_pre_speech',
    '감정과 욕구 (발화 후)': 'emotion_needs_post_speech',
    '생활지도 (두돌 전)': 'life_guidance_pre_two',
    '생활지도 (두돌에서 세돌 사이)': 'life_guidance_two_to_three',
    '생활지도 (세돌 이후)': 'life_guidance_post_three',
    '형제남매친구 중재': 'sibling_mediation',
    '자조능력 (이유식)': 'self_help_weaning',
    '자조능력 (배변)': 'self_help_toilet',
    '자조능력 (생활습관)': 'self_help_habits',
    '놀이 상호작용 (발화 전)': 'play_interaction_pre_speech',
    '놀이 상호작용 (발화 후)': 'play_interaction_post_speech',
    '미디어 입문': 'media_intro',
    '미디어 갈등': 'media_conflict',
    '거절하기와 거절듣기': 'rejection_skills',
    '칭찬하기': 'giving_praise'
}

category_id = 'plan_group_id'
category_name = 'plan_group_name'
category_name_english = 'plan_group_name_english'

theme_id ='sub_plan_group_id'
theme_name = 'sub_plan_group_name'
theme_name_english = 'sub_plan_group_name_english'

plan_id = 'plan_id'
plan_name = 'plan_name'
plan_name_english = 'plan_name_english'

plan_keywords = 'parenting_worries'
plan_description = 'course_description'

cols_text = [plan_keywords, plan_description] #, 'parenting_env', 'tag']

category_keywords = 'concatenated_keywords'
category_description = 'concatenated_descriptions'

cols_examples = ['example_'+str(i+1) for i in range(5)]
cols_month = ['min_month', 'max_month']
cols_seasonality = ['seasonality_text']
cols_numeric = cols_month + cols_seasonality
cols_text_category = [category_keywords, category_description]

cols_name = [category_name, category_name_english, plan_name, plan_name_english, theme_name, theme_name_english]
cols_id = [plan_id, category_id, theme_id]
cols = cols_id + cols_name + cols_numeric + cols_examples + cols_text
cols_category = [category_id, category_name, category_name_english] + cols_text_category + cols_month

import pandas as pd
from collections import OrderedDict

def extract_unique_index_name_pairs(df, column_pairs, sort_keys=True):
    """
    데이터프레임에서 여러 인덱스-이름 쌍을 고유하게 추출하는 함수
    
    Parameters:
    df (pandas.DataFrame): 대상 데이터프레임
    column_pairs (list): 인덱스-이름 컬럼 쌍의 리스트. 예: [('a', 'a1'), ('b', 'b1')]
    sort_keys (bool): 딕셔너리 키 정렬 여부 (기본값: True)
    
    Returns:
    dict: {
        'pairs': 각 인덱스-이름 쌍에 대한 딕셔너리
               {컬럼쌍이름: {인덱스: 이름}},
        'counts': 각 인덱스-이름 쌍의 고유 항목 수
                  {컬럼쌍이름: 항목 수},
        'total_count': 모든 고유 항목의 총 개수,
        'sorted_keys': 정렬된 키 리스트 (각 인덱스-이름 쌍별),
        'unique_values': 각 키에 해당하는 level-1의 고유 값
    }
    """
    result = {
        'pairs': {}, 
        'counts': {}, 
        'total_count': 0,
        'sorted_keys': {},
        'unique_values': {}
    }
    total_count = 0
    
    for index_col, name_col in column_pairs:
        # 해당 컬럼만 선택하고 중복 제거
        unique_pairs_df = df[[index_col, name_col]].drop_duplicates()
        
        # 딕셔너리로 변환
        pair_dict = dict(zip(unique_pairs_df[index_col], unique_pairs_df[name_col]))
        
        # 키 정렬 및 저장
        pair_key = f"{index_col}_{name_col}"
        
        if sort_keys and pair_dict:
            sorted_keys = sorted(pair_dict.keys())
            sorted_pair_dict = OrderedDict((k, pair_dict[k]) for k in sorted_keys)
            result['pairs'][pair_key] = sorted_pair_dict
            result['sorted_keys'][pair_key] = sorted_keys
        else:
            result['pairs'][pair_key] = pair_dict
            result['sorted_keys'][pair_key] = list(pair_dict.keys())
        
        # level-1 고유 값 찾기 (name_col의 고유 값)
        result['unique_values'][pair_key] = sorted(set(pair_dict.values()))
        
        # 항목 수 저장
        pair_count = len(pair_dict)
        result['counts'][pair_key] = pair_count
        total_count += pair_count
    
    result['total_count'] = total_count
    return result

def print_keys_and_values(result):
    """
    정렬된 키와 해당하는 level-1 고유 값을 출력하는 함수
    
    Parameters:
    result (dict): extract_unique_index_name_pairs 함수의 결과
    """
    print("\n정렬된 키와 고유 값:")
    for pair_key in result['pairs'].keys():
        print(f"\n{pair_key} 쌍:")
        print(f"  정렬된 키: {result['sorted_keys'][pair_key]}")
        print(f"  고유 값: {result['unique_values'][pair_key]}")
        
        # 각 키에 해당하는 값 출력
        print(f"  키별 매핑:")
        for key in result['sorted_keys'][pair_key]:
            print(f"    {key} → {result['pairs'][pair_key][key]}")

def prep_metadata(df, output_path, is_print=False):
    # Load CSV file with specified encoding and columns

    df[category_name_english] = df[category_name].map(category_to_english)
    df[plan_name_english] = df[plan_name].map(plan_to_english)
    df[theme_name_english] = df[theme_name].map(theme_to_english)
    df=df[cols]

    if is_print:
        for i in df.groupby([category_name]):
            id = i[0]
            i_df = i[1]
            print(f"================category: {id}")#id)
            for plan in i_df[plan_name]:
                print(f"===plan: {plan}")
                j_df = i_df[i_df[plan_name]==plan]
                print(f"Examples:\n {j_df[cols_examples]}")
    df.to_csv(output_path, index=True, encoding='utf-8-sig')
    return df

def prep_category(df, columns_to_keep=cols_category):
    """
    Processes the DataFrame to extract unique keywords from 'tag' column,
    replaces '-' values in 'example_' columns, concatenates unique course descriptions,
    and aggregates min_month and max_month for each category.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_keep (list): List of columns to keep in the final output.

    Returns:
        pd.DataFrame: Processed DataFrame with aggregated months and other info.
    """

    # --- Keyword processing ---
    keywords_map = {}
    keyword_source_col = plan_keywords # 정의된 변수 사용
    for group_name, group_df in df.groupby(category_id):
        keywords_map[group_name] = set()
        for tags in group_df[keyword_source_col].dropna():
            keywords_map[group_name].update(tags.split(', ')) 
    df[category_keywords] = df[category_id].apply(
        lambda x: ', '.join(sorted(keywords_map.get(x, set())))
    )

    # --- Example processing ---
    example_cols = [col for col in df.columns if 'example_' in col]
    for col in example_cols:
        first_valid_value = df[col][df[col] != '-'].iloc[0] if not df[col][df[col] == '-'].empty else None
        if first_valid_value:
            df[col] = df[col].replace('-', first_valid_value)

    # --- Description concatenation ---
    concatenated_descriptions = {}
    for i_key in df[category_id].unique():
        i_df = df[df[category_id] == i_key]
        unique_descriptions = i_df[plan_description].unique()
        concatenated_descriptions[i_key] = ' '.join(unique_descriptions)
    df[category_description] = df[category_id].map(concatenated_descriptions)

    # --- Month aggregation ---
    min_month_col = 'min_month'
    max_month_col = 'max_month'
    if min_month_col not in df.columns or max_month_col not in df.columns:
        raise ValueError(f"'{min_month_col}' or '{max_month_col}' columns not found in DataFrame.")

    aggregated_months = df.groupby(category_id).agg(
        min_month_agg=(min_month_col, 'min'),
        max_month_agg=(max_month_col, 'max')
    ).reset_index()

    # --- Create final category DataFrame ---
    # Define base columns (excluding potentially varying ones like months for unique check)
    base_columns = [col for col in columns_to_keep if col not in [min_month_col, max_month_col]]
    # Ensure category_id is included for merging
    if category_id not in base_columns:
        base_columns.insert(0, category_id)
    # Ensure the columns actually exist in df before selecting
    existing_base_columns = [col for col in base_columns if col in df.columns]
    if not existing_base_columns:
        raise ValueError("No valid base columns found to create the category DataFrame.")

    # Use set to avoid duplicate columns if category_id was already in base_columns
    df_category = df[list(set(existing_base_columns))].drop_duplicates(subset=[category_id]).sort_values(by=category_id)

    # Merge the aggregated month data
    df_category = pd.merge(df_category, aggregated_months, on=category_id, how='left')

    # Rename aggregated columns back to original names
    df_category.rename(columns={'min_month_agg': min_month_col, 'max_month_agg': max_month_col}, inplace=True)

    # Reorder columns to match the original intended order in columns_to_keep
    final_columns = [col for col in columns_to_keep if col in df_category.columns]
    df_category = df_category[final_columns]

    df_category.reset_index(drop=True, inplace=True)

    return df_category

def main():
        
    data_path='data/플랜 메타데이터 포맷 - plan_table.csv' #labeling.csv'
    df = pd.read_csv(data_path, 
                    encoding='utf-8-sig'
                    )
    # Display first few rows to verify the data
    print(df.head())
    #df['id_plan']=df.index
    # df['id_category'] = df['labeling']
    data_prep_path = Path('data/prep')
    os.makedirs(data_prep_path, exist_ok=True)
    output_path = data_prep_path / Path('metadata_plan_all.csv')
    df = prep_metadata(df=df, output_path = output_path, is_print=True)
# For recommendation
    #df = df[df[category_id]!=0] # remove 입문 플랜
    plan_path = data_prep_path / Path('preped_plan.csv')
    df.to_csv(plan_path, index=False, encoding='utf-8-sig')
    # dict_category = result['pairs']['id_category_CategoryName_English']
    # print(dict_category)
    df_category = prep_category(df, columns_to_keep = cols_category)
    print(df_category.head())
    category_path = data_prep_path / Path('preped_category.csv')
    df_category.to_csv(category_path, index=False, encoding='utf-8-sig')

        # Apply the processing function
    column_pairs = [ (category_id, category_name_english), (plan_id, plan_name_english)]#, (theme_id, theme_name_english)]
    result = extract_unique_index_name_pairs(df, column_pairs=column_pairs)
    print_keys_and_values(result)
    

if __name__ == '__main__':
    main()
