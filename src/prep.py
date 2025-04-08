import json

def parse_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 모든 줄을 읽어옴
        lines = file.readlines()
        
        # 중괄호({)가 시작되는 줄 찾기
        start_line = 0
        for i, line in enumerate(lines):
            if '{' in line:
                start_line = i
                break
        
        # JSON 텍스트 구성
        json_text = ''.join(lines[start_line:])
        
        # JSON 문자열에서 중괄호({}) 시작과 끝 부분 찾기
        start_idx = json_text.find('{')
        end_idx = json_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_data = json_text[start_idx:end_idx]
            
            # JSON 데이터 파싱
            parsed_data = json.loads(json_data)
            return parsed_data
        else:
            return None
