
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
level2_to_english = {
    '발달_단계_연령_중심': 'developmental_age_focused',
    '시기_생활_이벤트_중심': 'life_event_focused',
    '부모_양육자_대화_및_자기관리': 'parent_communication_selfcare',
    '감정_욕구_이해': 'emotion_needs_understanding',
    '훈육_규범_설정_관련': 'discipline_norm_setting',
    '자조능력_독립성_관련': 'self_help_independence',
    '놀이_상호작용': 'play_interaction',
    '관계_소통_전반': 'relationship_communication'
}

level3_to_english = {
    '체험판': 'Trial',
    '첫걸음': 'FirstStep',
    '3T': '3T',
    'NVC': 'NVC',
    '생애초기 (intro)': 'EarlyLifeIntro',
    '두돌': 'TwoYearsOld',
    '세돌': 'ThreeYearsOld',
    '네돌 (outro)': 'FourYearsOldOutro',
    '등하원 기관 적응 (만1세반)': 'Adaptation1Y',
    '등하원 기관 적응 (만2세반)': 'Adaptation2Y',
    '등하원 기관 적응 (유치원 만3세반)': 'Adaptation3Y',
    '놀이터 (영아)': 'PlaygroundInfant',
    '놀이터 (유아)': 'PlaygroundToddler',
    '양육자 자기연결': 'ParentSelfConnection',
    '육아 중 부부대화 (영아)': 'CoupleTalkInfant',
    '육아 중 부부대화 (유아)': 'CoupleTalkToddler',
    '육아번아웃 다루기': 'ParentBurnout',
    '기관-동료양육자들과의 소통법': 'CommunicationWithPeers',
    '감정과 욕구 (태담)': 'EmotionNeedsPrenatal',
    '감정과 욕구 (발화 전)': 'EmotionNeedsPreSpeech',
    '감정과 욕구 (발화 후)': 'EmotionNeedsPostSpeech',
    '생활지도 (두돌 전)': 'LifeGuidancePreTwo',
    '생활지도 (두돌에서 세돌 사이)': 'LifeGuidanceTwoToThree',
    '생활지도 (세돌 이후)': 'LifeGuidancePostThree',
    '형제남매친구 중재': 'SiblingMediation',
    '자조능력 (이유식)': 'SelfHelpWeaning',
    '자조능력 (배변)': 'SelfHelpToilet',
    '자조능력 (생활습관)': 'SelfHelpHabits',
    '놀이 상호작용 (발화 전)': 'PlayInteractionPreSpeech',
    '놀이 상호작용 (발화 후)': 'PlayInteractionPostSpeech',
    '미디어 입문': 'MediaIntro',
    '미디어 갈등': 'MediaConflict',
    '거절하기와 거절듣기': 'RejectionSkills',
    '칭찬하기': 'GivingPraise'
}

cols_tags = ['Course_description', 'parenting_worries', 'parenting_env', 'tag']

cols_name = ['CategoryName', 'CategoryName_English', 'PlanName', 'PlanName_English']
cols_id = ['id_category', 'id_plan']
cols_examples = ['example_'+str(i+1) for i in range(5)]

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

def mappping_category(df, output_path, is_print=False):
    # Load CSV file with specified encoding and columns

    df['CategoryName_English'] = df['CategoryName'].map(level2_to_english)
    df['PlanName_English'] = df['PlanName'].map(level3_to_english)

    df=df[cols_id+cols_name+cols_examples+cols_tags]

    if is_print:
        for i in df.groupby(['CategoryName']):
            id = i[0]
            i_df = i[1]
            print(f"================category: {id}")#id)
            for plan in i_df['PlanName']:
                print(f"===plan: {plan}")
                j_df = i_df[i_df['PlanName']==plan]
                print(f"Examples:\n {j_df[cols_examples]}")
    df.to_csv(output_path, index=True, encoding='utf-8-sig')
    return df
def process_dataframe(df, columns_to_keep = ['id_category', 'CategoryName','CategoryName_English', 'keyword', 'concatenated_descriptions']):
    """
    Processes the DataFrame to extract unique keywords from 'tag' column,
    replaces '-' values in 'example_' columns, and concatenates unique course descriptions
    for each category.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame with 'keyword' column and concatenated descriptions.
    """

    # Extract unique keywords
    keywords = set()
    for group_name, group_df in df.groupby('id_category'):
        for tags in group_df['tag'].dropna():  # Handle NaN values
            keywords.update(tags.split(', '))  # Split and update the set

    # Assign unique keywords to a new 'keyword' column (comma-separated string)
    df['keyword'] = ', '.join(sorted(keywords))  # Sort for consistency

    # Replace '-' values in 'example_' columns
    example_cols = [col for col in df.columns if 'example_' in col]
    for col in example_cols:
        # Find the first non '-' value in the column
        first_valid_value = df[col][df[col] != '-'].iloc[0] if not df[col][df[col] == '-'].empty else None
        if first_valid_value:
            df[col] = df[col].replace('-', first_valid_value)

    # Concatenate unique descriptions for each category
    concatenated_descriptions = {}
    for i_key in df['id_category'].unique():
        i_df = df[df['id_category'] == i_key]
        unique_descriptions = i_df['Course_description'].unique()
        concatenated_descriptions[i_key] = ' '.join(unique_descriptions)

    # Add concatenated descriptions to the DataFrame
    df['concatenated_descriptions'] = df['id_category'].map(concatenated_descriptions)
        # Define the columns to keep
    
    # Create a new DataFrame with only the desired columns
    df_category =  df[columns_to_keep].drop_duplicates(subset=['id_category']).sort_values(by='id_category')
    df_category.reset_index(drop=True, inplace=True)

    return df_category


def main():
        
    data_path='data/플랜 메타데이터 포맷 - labeling.csv'
    df = pd.read_csv(data_path, 
                    encoding='utf-8-sig'
                    )
    # Display first few rows to verify the data
    print(df.head())

    df = df[df['Class']!='입문']
    df['id_plan']=df.index
    df['id_category'] = df['labeling']

    data_prep_path = Path('data/prep')
    os.makedirs(data_prep_path, exist_ok=True)

    output_path = data_prep_path / Path('plan_metadata.csv')
 
    df = mappping_category(df=df, output_path = output_path, is_print=True)
    # Apply the processing function
    column_pairs = [ ('id_category', 'CategoryName_English'), ('id_plan', 'PlanName_English')]
    result = extract_unique_index_name_pairs(df, column_pairs=column_pairs)
    print_keys_and_values(result)
    dict_category = result['pairs']['id_category_CategoryName_English']
    print(dict_category)
    
    df_category = process_dataframe(df, columns_to_keep = ['id_category', 'CategoryName','CategoryName_English', 'keyword', 'concatenated_descriptions'])
    print(df_category.head())
    category_path = 'preped_category.csv'
    plan_path = 'preped_plan.csv'
    category_path = data_prep_path / Path(category_path)
    plan_path = data_prep_path / Path(plan_path)
    df_category.to_csv(category_path, index=False, encoding='utf-8-sig')
    df.to_csv(plan_path, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    main()
