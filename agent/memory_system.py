"""轻量结构化记忆系统。

这个模块把比赛项目里已有的三类记忆统一起来：
- WorkingCaseMemory: 单病例工作记忆，保存问诊、检查和候选诊断等临时状态。
- EpisodicMemoryAdapter: 训练病例经验，复用 DoctorMemory 的 JSON 存储。
- SemanticMemoryAdapter: 医学语义知识，复用 KnowledgeBase 的标准目录和疾病画像。
- PolicyMemoryAdapter: 策略补丁记忆，复用 PolicyStore。

设计目标是比赛友好：纯本地、无外部数据库、无向量服务依赖。
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .rag_retriever import HybridRAGRetriever


def _positive_int(value: Any, default: int) -> int:
    """把配置值安全解析为正整数。"""
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


@dataclass
class MemoryConfig:
    """记忆系统配置。"""

    working_ttl_seconds: int = 7200
    max_working_cases: int = 32
    episodic_top_k: int = 3

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MemoryConfig":
        raw = config.get("memory_system", {}) if isinstance(config, dict) else {}
        return cls(
            working_ttl_seconds=_positive_int(raw.get("working_ttl_seconds"), 7200),
            max_working_cases=_positive_int(raw.get("max_working_cases"), 32),
            episodic_top_k=_positive_int(raw.get("episodic_top_k"), 3),
        )


@dataclass
class MemoryItem:
    """标准化记忆项。"""

    memory_type: str
    content: Dict[str, Any]
    item_id: str = field(default_factory=lambda: "mem_" + uuid.uuid4().hex[:10])
    patient_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    score: float = 0.0
    tags: List[str] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = time.time()


class BaseMemory(ABC):
    """记忆组件通用接口。"""

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """返回组件统计信息。"""


class WorkingCaseMemory(BaseMemory):
    """单个病例的工作记忆。"""

    def __init__(self, patient_id: str, ttl_seconds: int = 7200):
        self.patient_id = patient_id
        self.ttl_seconds = ttl_seconds
        self.item = MemoryItem(memory_type="working", patient_id=patient_id, content={})
        self.collected_info: Dict[str, Any] = {}
        self.chat_history: List[Dict[str, str]] = []
        self.exam_results: Dict[str, Any] = {}
        self.candidate_diseases: List[Any] = []
        self.status = "active"

    def is_expired(self) -> bool:
        return (time.time() - self.item.updated_at) > self.ttl_seconds

    def update_collected_info(self, info: Dict[str, Any]) -> None:
        if info:
            self.collected_info = dict(info)
            self.item.touch()

    def update_exam_results(self, results: Dict[str, Any]) -> None:
        if results:
            self.exam_results.update(results)
            self.item.touch()

    def set_candidate_diseases(self, candidates: Optional[List[Any]]) -> None:
        self.candidate_diseases = list(candidates or [])
        self.item.touch()

    def add_dialogue(self, doctor_text: str, patient_text: str) -> None:
        if doctor_text:
            self.chat_history.append({"from": "doctor", "text": doctor_text})
        if patient_text:
            self.chat_history.append({"from": "patient", "text": patient_text})
        self.item.touch()

    def finish(self) -> None:
        self.status = "completed"
        self.item.touch()

    def to_context(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "status": self.status,
            "collected_info": self.collected_info,
            "chat_history": self.chat_history,
            "exam_results": self.exam_results,
            "candidate_diseases": self.candidate_diseases,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "status": self.status,
            "symptoms": len(self.collected_info.get("symptoms", []) or []),
            "dialogue_turns": len(self.chat_history),
            "exam_count": len(self.exam_results),
            "candidate_count": len(self.candidate_diseases),
            "expired": self.is_expired(),
        }


class EpisodicMemoryAdapter(BaseMemory):
    """病例经验记忆适配器，复用 DoctorMemory。"""

    def __init__(self, memory: Any):
        self.memory = memory

    def search(self, collected_info: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        if hasattr(self.memory, "search_relevant_experience_multi"):
            return self.memory.search_relevant_experience_multi(collected_info, top_k=top_k) or []
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        if hasattr(self.memory, "search_relevant_experience"):
            return self.memory.search_relevant_experience(symptoms, top_k=top_k) or []
        return []

    def stats(self) -> Dict[str, Any]:
        if hasattr(self.memory, "get_statistics"):
            return self.memory.get_statistics()
        return {"total": 0}


class SemanticMemoryAdapter(BaseMemory):
    """医学语义记忆适配器，复用 KnowledgeBase。"""

    def __init__(self, knowledge: Any):
        self.knowledge = knowledge

    def build_context(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> str:
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        if hasattr(self.knowledge, "build_rag_context"):
            return self.knowledge.build_rag_context(
                symptoms=symptoms,
                candidate_diseases=candidate_diseases,
            )
        if hasattr(self.knowledge, "build_clinical_context"):
            return self.knowledge.build_clinical_context(symptoms, candidate_diseases)
        return ""

    def suggest_diagnoses(self, collected_info: Dict[str, Any], top_k: int = 3) -> List[str]:
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        if hasattr(self.knowledge, "suggest_diagnoses"):
            return self.knowledge.suggest_diagnoses(symptoms=symptoms, top_k=top_k)
        return []

    def stats(self) -> Dict[str, Any]:
        return {
            "diseases": len(getattr(self.knowledge, "diseases", []) or []),
            "examinations": len(getattr(self.knowledge, "examinations", []) or []),
            "profiles": len(getattr(self.knowledge, "disease_profiles", []) or []),
        }


class PolicyMemoryAdapter(BaseMemory):
    """策略补丁记忆适配器。"""

    def __init__(self, policy_store: Any = None):
        self.policy_store = policy_store

    def match(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        include_shadow: bool = False,
    ) -> List[Dict[str, Any]]:
        if not self.policy_store:
            return []
        candidates = [str(item) for item in candidate_diseases or [] if item]
        return self.policy_store.match(collected_info, candidates, include_shadow=include_shadow)

    def render(self, patches: List[Dict[str, Any]]) -> str:
        if self.policy_store and hasattr(self.policy_store, "render_for_prompt"):
            return self.policy_store.render_for_prompt(patches)
        return ""

    def stats(self) -> Dict[str, Any]:
        patches = getattr(self.policy_store, "patches", []) if self.policy_store else []
        by_status: Dict[str, int] = {}
        for patch in patches:
            status = (patch.get("stats") or {}).get("status", "shadow")
            by_status[status] = by_status.get(status, 0) + 1
        return {"total": len(patches), "by_status": by_status}


class DoctorAgentMemory:
    """统一记忆管理器。"""

    def __init__(
        self,
        config: Dict[str, Any],
        episodic_memory: Any,
        semantic_memory: Any,
        policy_store: Any = None,
    ):
        self.config = MemoryConfig.from_config(config)
        self.episodic = EpisodicMemoryAdapter(episodic_memory)
        self.semantic = SemanticMemoryAdapter(semantic_memory)
        self.policy = PolicyMemoryAdapter(policy_store)
        self.hybrid_rag = HybridRAGRetriever(
            config=config,
            knowledge=semantic_memory,
            memory=episodic_memory,
            policy_store=policy_store,
        )
        self.working_cases: Dict[str, WorkingCaseMemory] = {}

    def start_case(self, patient_id: str) -> WorkingCaseMemory:
        self._prune_expired()
        if len(self.working_cases) >= self.config.max_working_cases:
            oldest = sorted(
                self.working_cases.values(),
                key=lambda item: item.item.updated_at,
            )[0]
            self.working_cases.pop(oldest.patient_id, None)
        memory = WorkingCaseMemory(
            patient_id=patient_id,
            ttl_seconds=self.config.working_ttl_seconds,
        )
        self.working_cases[patient_id] = memory
        return memory

    def get_case(self, patient_id: str) -> Optional[WorkingCaseMemory]:
        memory = self.working_cases.get(patient_id)
        if memory and memory.is_expired():
            self.working_cases.pop(patient_id, None)
            return None
        return memory

    def finish_case(self, patient_id: str) -> Optional[WorkingCaseMemory]:
        memory = self.get_case(patient_id)
        if memory:
            memory.finish()
        return memory

    def update_collected_info(self, patient_id: str, info: Dict[str, Any]) -> None:
        memory = self.get_case(patient_id)
        if memory:
            memory.update_collected_info(info)

    def update_exam_results(self, patient_id: str, results: Dict[str, Any]) -> None:
        memory = self.get_case(patient_id)
        if memory:
            memory.update_exam_results(results)

    def update_candidates(self, patient_id: str, candidates: Optional[List[Any]]) -> None:
        memory = self.get_case(patient_id)
        if memory:
            memory.set_candidate_diseases(candidates)

    def search_episodic(self, collected_info: Dict[str, Any], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.episodic.search(
            collected_info,
            top_k=top_k or self.config.episodic_top_k,
        )

    def build_semantic_context(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        retrieval_views: Optional[List[Any]] = None,
    ) -> str:
        context = self.hybrid_rag.build_context(
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            retrieval_views=retrieval_views,
        )
        if context:
            return context
        return self.semantic.build_context(collected_info, candidate_diseases)

    def search_rag(
        self,
        collected_info: Dict[str, Any],
        query: str = "",
        candidate_diseases: Optional[List[Any]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        retrieval_views: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self.hybrid_rag.search(
            collected_info=collected_info,
            query=query,
            candidate_diseases=candidate_diseases,
            top_k=top_k,
            score_threshold=score_threshold,
            retrieval_views=retrieval_views,
        )

    def render_rag_chunks(self, chunks: Optional[List[Dict[str, Any]]]) -> str:
        return self.hybrid_rag.render_chunks(chunks)

    def match_policy(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        include_shadow: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.policy.match(
            collected_info,
            candidate_diseases=candidate_diseases,
            include_shadow=include_shadow,
        )

    def render_policy_context(self, patches: List[Dict[str, Any]]) -> str:
        return self.policy.render(patches)

    def stats(self) -> Dict[str, Any]:
        return {
            "working": {
                "active_cases": len(self.working_cases),
                "cases": [case.stats() for case in self.working_cases.values()],
            },
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
            "policy": self.policy.stats(),
            "hybrid_rag": {
                "top_k": self.hybrid_rag.config.top_k,
                "score_threshold": self.hybrid_rag.config.score_threshold,
                "enable_mqe": self.hybrid_rag.config.enable_mqe,
                "candidate_pool_multiplier": self.hybrid_rag.config.candidate_pool_multiplier,
            },
        }

    def _prune_expired(self) -> None:
        expired = [
            patient_id
            for patient_id, memory in self.working_cases.items()
            if memory.is_expired()
        ]
        for patient_id in expired:
            self.working_cases.pop(patient_id, None)
