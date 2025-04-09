import pandas as pd
import os

def calc_metric(df, gt, pred, output_txt_path, n):
    df['common_count'] = df.apply(count_common_elements, axis=1, gt=gt, pred=pred, n=n)
    df['precision'] = round(df['common_count'] / n, 4)
    # 각 행마다 hit_rate 계산: common_count가 1 이상이면 1, 아니면 0
    df['hit_rate'] = df['common_count'].apply(lambda x: 1 if x >= 1 else 0)
    # 전체 행에 대한 평균 hit_rate 계산
    average_precision = round(df['precision'].mean() * 100, 2)
    average_hit_rate = round(df['hit_rate'].mean() * 100, 2)
    result = f"average_precision: {average_precision}\naverage_hit_rate: {average_hit_rate}"
    print(result)
    with open(output_txt_path, 'w') as f:
        f.write(result)
    return df

def count_common_elements(row, gt, pred, n):
    set_a = {row[gt[i-1]] for i in range(1, n+1) if pd.notna(row[gt[i-1]])}
    set_b = {row[pred[i-1]] for i in range(1, n+1) if pd.notna(row[pred[i-1]])}
    return len(set_a.intersection(set_b))

def call_metric(data_path, output_path, id, n=3):
    gt_survey = [f'설문기준 {i}순위' for i in range(1, n+1)]
    gt_dialogue = [f'대화기준 {i}순위' for i in range(1, n+1)]

    pred_llm = [f'llm_plan_rec_{i}' for i in range(1, n+1)]
    df = pd.read_csv(data_path)

    output_dir = os.path.dirname(output_path)
    output_txt_path = os.path.join(output_dir, f"evaluation_metric_{id}.txt")

    df_metric = calc_metric(df, gt_dialogue, pred_llm, output_txt_path, n)
    df_metric.to_csv(output_path, encoding='utf-8-sig', index=False)
    cols = ['persona_index','child_age']
    cols_metric =['common_count', 'precision', 'hit_rate']
    cols +=  gt_dialogue + pred_llm + cols_metric
    #cols += ['plan_id_list', 'plan_name_list']
    df_metric = df_metric[cols]
    df_metric.to_csv(output_path, encoding='utf-8-sig', index=False)

def main():
    id ='gemma3_20250409_235213'
    output_dir_for_main = f'outputs/{id}'
    os.makedirs(output_dir_for_main, exist_ok=True)
    data_path = os.path.join(output_dir_for_main, f'survey_with_scoring_{id}.csv')
    output_path = os.path.join(output_dir_for_main, f'evaluation_metric_{id}.csv')
    n = 3
    try:
        placeholder_df = pd.DataFrame(columns=[f'대화기준 {i}순위' for i in range(1, n+1)] + [f'llm_plan_rec_{i}' for i in range(1, n+1)])
        placeholder_df.to_csv(data_path, index=False)
        print(f"임시 데이터 파일 생성: {data_path}")

        call_metric(data_path, output_path, id, n)
        print(f"메인 함수 실행 완료. 결과 확인: {output_path} 및 관련 .txt 파일")
    except FileNotFoundError:
        print(f"오류: 입력 데이터 파일을 찾을 수 없습니다 - {data_path}")
        print("main_scoring.py를 먼저 실행하여 필요한 파일을 생성하세요.")
    except Exception as e:
        print(f"main 함수 실행 중 오류 발생: {e}")

if __name__ == '__main__':
    main()


