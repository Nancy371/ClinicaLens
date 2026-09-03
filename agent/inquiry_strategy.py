"""问诊策略 Agent：基于疾病画像补齐关键追问。"""

from typing import Any, Dict, List, Optional

from .knowledge import KnowledgeBase


class InquiryStrategyAgent:
    """轻量问诊策略角色，不调用外部服务。"""

    def __init__(self, knowledge: KnowledgeBase, max_questions: int = 5):
        self.knowledge = knowledge
        self.max_questions = max_questions

    def recommend(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """返回当前病例最值得追问的问题和红旗信号。"""
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        profiles = self.knowledge.recall_disease_profiles(
            symptoms=symptoms,
            candidate_diseases=candidate_diseases,
            top_k=4,
        )

        questions: List[str] = []
        red_flags: List[str] = []
        for profile in profiles:
            for question in profile.get("key_questions") or []:
                if question and question not in questions:
                    questions.append(str(question))
            for flag in profile.get("red_flags") or []:
                if flag and flag not in red_flags:
                    red_flags.append(str(flag))

        return {
            "questions": questions[: self.max_questions],
            "red_flags": red_flags[: self.max_questions],
            "clinical_context": self.knowledge.build_clinical_context(
                symptoms=symptoms,
                candidate_diseases=candidate_diseases,
            ),
        }
