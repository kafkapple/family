
import pandas as pd
import os
from pathlib import Path
# Define mappings for ClassName and PlanName to their English equivalents
level1_to_english = {
    '입문': 'Intro',
    '발달': 'Development',
    '생활': 'Life',
    '공감': 'Empathy'
}
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
columns_to_read = ['Subject', 'ClassName', 'level-1', 'CategoryName','level-2', 
                    'PlanName', 'level-3', 'example_1', 'example_2', 'example_3', 
                    'example_4', 'example_5', 'Course_description', 
                    'parenting_worries', 'parenting_env', 'tag']

cols_tags = ['Course_description', 'parenting_worries', 'parenting_env', 'tag']

cols_name = ['ClassName', 'ClassName_English', 'CategoryName', 'CategoryName_English', 'PlanName', 'PlanName_English']
cols_id = ['level-1', 'level-2', 'level-3']
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

def mappping_category(data_path, output_path, is_print=False):
    # Load CSV file with specified encoding and columns
    df = pd.read_csv(data_path, 
                    encoding='utf-8-sig',
                    usecols=columns_to_read)

    # Display first few rows to verify the data
    print(df.head())
    # Map the values in the dataframe
    df['ClassName_English'] = df['ClassName'].map(level1_to_english)
    df['CategoryName_English'] = df['CategoryName'].map(level2_to_english)
    df['PlanName_English'] = df['PlanName'].map(level3_to_english)
    # Map the values in the dataframe
    #print(cols_id, cols_name,  cols_examples)
    # Display the updated dataframe
    df=df[cols_id+cols_name+cols_examples+cols_tags]

    df=df[df['level-1']!=0]

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
    
if __name__ == '__main__':
    data_path = 'data/플랜 메타데이터 포맷 - id.csv'
    output_path = 'data/prep'
    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path) / Path('mapped_category.csv')
    mappping_category(data_path, output_path, is_print=True)