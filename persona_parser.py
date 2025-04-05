from dataclasses import dataclass
from typing import List, Dict, Any
import ast

@dataclass
class PriorityInterest:
    category: str
    level: int
    keywords: List[str]

@dataclass
class FamilyPersonaInfo:
    persona_description: str
    priorities: List[PriorityInterest]
    main_concerns: List[str]

class FamilyPersonaParser:
    def parse_priority_interests(self, data: Dict) -> List[PriorityInterest]:
        """CSV 데이터에서 우선순위별 관심사를 파싱합니다."""
        priorities = []
        
        # 카테고리 우선순위 (카테고리1~6)
        category_order = []
        for i in range(1, 7):
            category = data.get(f'카테고리{i}')
            if category:
                category_order.append(category)

        # 각 카테고리별 키워드 매핑
        category_keywords = {
            '발달 및 정서': self._parse_list_str(data.get('발달 및 정서', '[]')),
            '양육 태도': self._parse_list_str(data.get('양육 태도', '[]')),
            '소통 기술': self._parse_list_str(data.get('소통 기술', '[]')),
            '양육자 본인': self._parse_list_str(data.get('양육자 본인', '[]')),
            '상황': self._parse_list_str(data.get('상황', '[]')),
            '대화 대상': self._parse_list_str(data.get('대화 대상', '[]'))
        }

        # 우선순위별로 PriorityInterest 객체 생성
        for level, category in enumerate(category_order, start=1):
            # 우선순위를 역순으로 설정 (가장 높은 것이 6)
            priority_level = 7 - level
            keywords = category_keywords.get(category, [])
            
            priorities.append(PriorityInterest(
                category=category,
                level=priority_level,
                keywords=keywords
            ))

        return priorities

    def _parse_list_str(self, list_str: str) -> List[str]:
        """문자열로 된 리스트를 실제 리스트로 변환합니다."""
        try:
            return ast.literal_eval(list_str)
        except:
            return []

    def extract_main_concerns(self, persona: str) -> List[str]:
        """페르소나 설명에서 주요 고민사항을 추출합니다."""
        concerns = []
        if "고민" in persona:
            concerns.append(persona)
        return concerns

    def parse_family_info(self, i_family: Dict) -> FamilyPersonaInfo:
        """가족 정보를 파싱하여 구조화된 형태로 반환합니다."""
        persona = i_family.get('페르소나', '')
        priorities = self.parse_priority_interests(i_family)
        main_concerns = self.extract_main_concerns(persona)
        
        return FamilyPersonaInfo(
            persona_description=persona,
            priorities=priorities,
            main_concerns=main_concerns
        )

    def create_context_prompt(self, family_info: FamilyPersonaInfo) -> str:
        """파싱된 가족 정보를 바탕으로 컨텍스트 프롬프트를 생성합니다."""
        prompt_parts = [
            "# 가족 상황",
            f"페르소나: {family_info.persona_description}",
            "\n# 주요 고민사항"
        ]
        
        for concern in family_info.main_concerns:
            prompt_parts.append(f"- {concern}")
        
        prompt_parts.append("\n# 우선순위별 관심사")
        
        # 우선순위 높은 순으로 정렬
        sorted_priorities = sorted(family_info.priorities, 
                                key=lambda x: x.level, 
                                reverse=True)
        
        for priority in sorted_priorities:
            keywords_str = ", ".join(priority.keywords)
            prompt_parts.append(
                f"- 중요도 {priority.level}: {priority.category}\n  키워드: {keywords_str}"
            )
        
        return "\n".join(prompt_parts)