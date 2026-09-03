"""轻量混合 RAG 检索器。

本模块把本项目已有的知识库、病例经验和策略补丁统一成一种 chunk
结果，不依赖外部向量库，适合比赛容器直接部署。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .medical_retrieval import ExternalMedicalKnowledgeRetriever


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class HybridRAGConfig:
    """Hybrid RAG 配置。"""

    top_k: int = 8
    score_threshold: Optional[float] = 0.15
    enable_mqe: bool = True
    mqe_expansions: int = 2
    candidate_pool_multiplier: int = 4
    include_policy_shadow: bool = False
    max_chunk_chars: int = 700

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HybridRAGConfig":
        raw = config.get("rag", {}) if isinstance(config, dict) else {}
        return cls(
            top_k=_positive_int(raw.get("top_k"), 8),
            score_threshold=_optional_float(raw.get("score_threshold", 0.15)),
            enable_mqe=bool(raw.get("enable_mqe", True)),
            mqe_expansions=_positive_int(raw.get("mqe_expansions"), 2),
            candidate_pool_multiplier=_positive_int(
                raw.get("candidate_pool_multiplier"), 4
            ),
            include_policy_shadow=bool(raw.get("include_policy_shadow", False)),
            max_chunk_chars=_positive_int(raw.get("max_chunk_chars"), 700),
        )


@dataclass
class RagChunk:
    """统一 RAG chunk。"""

    chunk_id: str
    chunk_type: str
    title: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "type": self.chunk_type,
            "title": self.title,
            "text": self.text,
            "score": round(float(self.score), 4),
            "metadata": self.metadata,
        }


_MQE_RULES: List[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (
        ("胸口闷", "胸闷", "胸部发闷", "胸口堵", "胸口压榨", "胸口不舒服"),
        ("胸闷", "胸痛", "气短", "心悸", "冠心病", "心肌梗死", "肺炎"),
    ),
    (
        ("喘不上气", "喘不过气", "气短", "憋气", "呼吸困难", "气促"),
        ("呼吸困难", "气促", "胸闷", "心力衰竭", "支气管哮喘", "肺炎", "慢性阻塞性肺疾病"),
    ),
    (
        ("胸痛", "心前区痛", "胸口痛", "胸部疼痛"),
        ("胸痛", "胸闷", "大汗", "心悸", "心肌梗死", "冠心病", "肺炎", "胃溃疡"),
    ),
    (
        ("发烧", "发热", "高热", "低热"),
        ("发热", "感染", "血常规", "C反应蛋白", "肺炎", "上呼吸道感染", "急性胃肠炎"),
    ),
    (
        ("咳嗽", "咳痰", "痰多"),
        ("咳嗽", "咳痰", "发热", "胸痛", "肺炎", "支气管炎", "肺结核"),
    ),
    (
        ("腹痛", "肚子痛", "肚子疼", "上腹痛", "右下腹痛", "右上腹痛"),
        ("腹痛", "发热", "恶心", "呕吐", "阑尾炎", "胆囊炎", "胰腺炎", "急性胃肠炎"),
    ),
    (
        ("头晕", "头痛", "肢体无力", "说话不清", "口角歪"),
        ("头晕", "头痛", "肢体无力", "言语不清", "脑梗死", "脑出血", "高血压"),
    ),
    (
        ("尿频", "尿急", "尿痛", "腰痛", "血尿"),
        ("尿频", "尿急", "尿痛", "血尿", "泌尿系感染", "肾结石"),
    ),
]


class HybridRAGRetriever:
    """统一检索入口：疾病画像、标准检查、病例经验、策略补丁。"""

    def __init__(
        self,
        config: Dict[str, Any],
        knowledge: Any,
        memory: Any = None,
        policy_store: Any = None,
    ):
        self.config = HybridRAGConfig.from_config(config)
        self.knowledge = knowledge
        self.memory = memory
        self.policy_store = policy_store
        self.allowed_diagnoses = self._load_allowed_diagnoses(config)
        self.external_medical = ExternalMedicalKnowledgeRetriever(
            config=config,
            ref_dir=str((config or {}).get("ref_data_dir") or "data/ref_data"),
        )

    def search(
        self,
        collected_info: Optional[Dict[str, Any]] = None,
        query: str = "",
        candidate_diseases: Optional[List[Any]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        enable_mqe: Optional[bool] = None,
        mqe_expansions: Optional[int] = None,
        candidate_pool_multiplier: Optional[int] = None,
        include_policy_shadow: Optional[bool] = None,
        retrieval_views: Optional[Sequence[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """统一检索，返回按 score 降序排序的 chunk 字典。"""
        collected_info = collected_info or {}
        query = query or self._query_from_info(collected_info)
        if not query and not collected_info and not candidate_diseases:
            return []

        top = _positive_int(top_k, self.config.top_k)
        threshold = (
            self.config.score_threshold
            if score_threshold is None
            else score_threshold
        )
        use_mqe = self.config.enable_mqe if enable_mqe is None else bool(enable_mqe)
        expansions = _positive_int(mqe_expansions, self.config.mqe_expansions)
        multiplier = _positive_int(
            candidate_pool_multiplier,
            self.config.candidate_pool_multiplier,
        )
        include_shadow = (
            self.config.include_policy_shadow
            if include_policy_shadow is None
            else bool(include_policy_shadow)
        )

        pool = max(top * multiplier, 20)
        expanded_terms = self.expand_query(
            query=query,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            enable_mqe=use_mqe,
            mqe_expansions=expansions,
        )
        view_terms = self._retrieval_view_terms(retrieval_views or [])
        expanded_terms = _dedupe(expanded_terms + view_terms)
        candidate_names = self._candidate_names(candidate_diseases, expanded_terms, pool)
        augmented_info = self._augment_collected_info(collected_info, expanded_terms)

        chunks: List[RagChunk] = []
        chunks.extend(self._disease_profile_chunks(expanded_terms, candidate_names, pool))
        chunks.extend(self._basic_disease_chunks(expanded_terms, candidate_names, pool))
        chunks.extend(self._standard_exam_chunks(expanded_terms, candidate_names, chunks, pool))
        chunks.extend(self._experience_chunks(augmented_info, pool))
        chunks.extend(self._policy_chunks(augmented_info, candidate_names, include_shadow))
        chunks.extend(self._external_medical_chunks(retrieval_views or [], expanded_terms, pool))

        merged = self._merge_chunks(chunks)
        if threshold is not None:
            merged = [chunk for chunk in merged if chunk.score >= float(threshold)]
        merged.sort(key=lambda item: (item.score, self._type_priority(item.chunk_type)), reverse=True)
        selected = self._select_diverse(merged, top)
        return [chunk.to_dict() for chunk in selected]

    def build_context(
        self,
        collected_info: Optional[Dict[str, Any]] = None,
        query: str = "",
        candidate_diseases: Optional[List[Any]] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        retrieval_views: Optional[Sequence[Any]] = None,
    ) -> str:
        """构建可注入 prompt 的统一 RAG 上下文。"""
        chunks = self.search(
            collected_info=collected_info,
            query=query,
            candidate_diseases=candidate_diseases,
            top_k=top_k,
            score_threshold=score_threshold,
            retrieval_views=retrieval_views,
        )
        return self.render_chunks(chunks)

    def render_chunks(self, chunks: Optional[Sequence[Dict[str, Any]]]) -> str:
        """Render already-retrieved chunks without performing a second search."""
        if not chunks:
            return ""

        groups = [
            ("disease_profile", "疾病画像"),
            ("external_medical_knowledge", "外部医学检索"),
            ("standard_exam", "标准检查"),
            ("case_experience", "历史病例经验"),
            ("policy_patch", "策略补丁"),
        ]
        lines = ["【Hybrid RAG 统一检索】以下内容来自疾病画像、外部待审核检索、标准检查、历史病例和策略补丁，请结合当前病例判断："]
        for chunk_type, label in groups:
            selected = [chunk for chunk in chunks if chunk.get("type") == chunk_type]
            if not selected:
                continue
            lines.append(f"\n【{label}】")
            for idx, chunk in enumerate(selected, 1):
                score = chunk.get("score", 0.0)
                title = chunk.get("title", "")
                text = self._clip(chunk.get("text", ""), self.config.max_chunk_chars)
                lines.append(f"{idx}. {title}（score={score}）")
                if text:
                    lines.append(f"   {text}")
        return "\n".join(lines)

    def expand_query(
        self,
        query: str,
        collected_info: Optional[Dict[str, Any]] = None,
        candidate_diseases: Optional[List[Any]] = None,
        enable_mqe: bool = True,
        mqe_expansions: int = 2,
    ) -> List[str]:
        """规则版 MQE：把口语症状扩展为标准症状、疾病方向和检查线索。"""
        collected_info = collected_info or {}
        base: List[str] = []
        base.extend(self._split_medical_terms(query))
        base.extend(_as_list(collected_info.get("symptoms")))
        base.extend(self._split_medical_terms(collected_info.get("chief_complaint", "")))
        base.extend(self._split_medical_terms(collected_info.get("present_illness", "")))
        base.extend(self._candidate_names(candidate_diseases, [], 8))

        expanded = _dedupe(base)
        if enable_mqe:
            text = " ".join(str(item) for item in expanded)
            max_rule_terms = max(8, mqe_expansions * 6)
            rule_terms: List[str] = []
            for triggers, terms in _MQE_RULES:
                if any(trigger in text for trigger in triggers):
                    rule_terms.extend(terms)
            expanded.extend(_dedupe(rule_terms)[:max_rule_terms])

            try:
                suggestions = self.knowledge.suggest_diagnoses(
                    symptoms=expanded,
                    candidate_diseases=candidate_diseases or [],
                    top_k=max(3, mqe_expansions * 3),
                )
                expanded.extend(suggestions or [])
            except Exception:
                pass

        return _dedupe([str(item).strip() for item in expanded if str(item).strip()])

    def _disease_profile_chunks(
        self,
        expanded_terms: List[str],
        candidate_names: List[str],
        pool: int,
    ) -> List[RagChunk]:
        try:
            profiles = self.knowledge.recall_disease_profiles(
                symptoms=expanded_terms,
                candidate_diseases=candidate_names,
                top_k=pool,
            )
        except Exception:
            profiles = []

        chunks: List[RagChunk] = []
        for profile in profiles:
            name = profile.get("name", "")
            if not name:
                continue
            if self.allowed_diagnoses and name not in self.allowed_diagnoses:
                continue
            hit_score = float(profile.get("hit_score", 0.0) or 0.0)
            candidate_bonus = 0.12 if name in candidate_names else 0.0
            score = min(0.98, 0.45 + hit_score * 0.10 + candidate_bonus)
            chunks.append(
                RagChunk(
                    chunk_id=f"disease:{name}",
                    chunk_type="disease_profile",
                    title=name,
                    text=self._render_profile(profile),
                    score=score,
                    metadata={
                        "department": profile.get("department", ""),
                        "matched_symptoms": profile.get("matched_symptoms", []),
                        "required_exams": profile.get("required_exams", []),
                    },
                )
            )
        return chunks

    def _basic_disease_chunks(
        self,
        expanded_terms: List[str],
        candidate_names: List[str],
        pool: int,
    ) -> List[RagChunk]:
        try:
            recalls = self.knowledge.recall_diseases_by_symptoms(expanded_terms, top_k=pool)
        except Exception:
            recalls = []

        chunks: List[RagChunk] = []
        for item in recalls:
            name = item.get("name", "")
            if not name:
                continue
            if self.allowed_diagnoses and name not in self.allowed_diagnoses:
                continue
            hit_count = float(item.get("hit_count", 0.0) or 0.0)
            score = min(0.82, 0.34 + hit_count * 0.08 + (0.08 if name in candidate_names else 0.0))
            exams = []
            try:
                exams = self.knowledge.get_required_exams(candidate_diseases=[name])
            except Exception:
                exams = []
            text_parts = [
                f"科室: {item.get('department', '')}",
                f"匹配症状: {', '.join(item.get('matched_symptoms', [])[:6])}",
            ]
            if exams:
                text_parts.append(f"推荐检查: {', '.join(exams[:6])}")
            chunks.append(
                RagChunk(
                    chunk_id=f"disease:{name}",
                    chunk_type="disease_profile",
                    title=name,
                    text="；".join(part for part in text_parts if part),
                    score=score,
                    metadata={
                        "department": item.get("department", ""),
                        "matched_symptoms": item.get("matched_symptoms", []),
                        "required_exams": exams,
                    },
                )
            )
        return chunks

    def _standard_exam_chunks(
        self,
        expanded_terms: List[str],
        candidate_names: List[str],
        disease_chunks: Sequence[RagChunk],
        pool: int,
    ) -> List[RagChunk]:
        diseases = list(candidate_names)
        for chunk in disease_chunks:
            if chunk.chunk_type == "disease_profile" and chunk.title not in diseases:
                diseases.append(chunk.title)

        try:
            exams = self.knowledge.get_required_exams(
                candidate_diseases=diseases[:pool],
                symptoms=expanded_terms,
                include_optional=False,
            )
        except Exception:
            exams = []

        chunks: List[RagChunk] = []
        for exam in exams[:pool]:
            related = [
                disease
                for disease in diseases
                if exam in (self._profile_required_exams(disease) or [])
            ][:6]
            score = 0.70 if related else 0.55
            reason = f"用于支持或排除: {', '.join(related)}" if related else "由当前症状和候选疾病召回"
            chunks.append(
                RagChunk(
                    chunk_id=f"exam:{exam}",
                    chunk_type="standard_exam",
                    title=exam,
                    text=reason,
                    score=score,
                    metadata={"related_diseases": related},
                )
            )
        return chunks

    def _experience_chunks(
        self,
        collected_info: Dict[str, Any],
        pool: int,
    ) -> List[RagChunk]:
        if not self.memory:
            return []
        try:
            if hasattr(self.memory, "search_relevant_experience_multi"):
                notes = self.memory.search_relevant_experience_multi(collected_info, top_k=pool)
            elif hasattr(self.memory, "search_relevant_experience"):
                notes = self.memory.search_relevant_experience(
                    collected_info.get("symptoms", []),
                    top_k=pool,
                )
            else:
                notes = []
        except Exception:
            notes = []

        chunks: List[RagChunk] = []
        for idx, note in enumerate(notes or [], 1):
            patient_id = note.get("patient_id") or f"case_{idx}"
            score = self._score_experience(collected_info, note)
            metrics = note.get("metrics") or {}
            text = self._render_experience(note)
            chunks.append(
                RagChunk(
                    chunk_id=f"case:{patient_id}",
                    chunk_type="case_experience",
                    title=str(note.get("title") or patient_id),
                    text=text,
                    score=score,
                    metadata={
                        "patient_id": patient_id,
                        "metrics": metrics,
                        "expected_diagnosis": note.get("expected_diagnosis", []),
                        "submitted_diagnosis": note.get("submitted_diagnosis", []),
                    },
                )
            )
        return chunks

    def _policy_chunks(
        self,
        collected_info: Dict[str, Any],
        candidate_names: List[str],
        include_shadow: bool,
    ) -> List[RagChunk]:
        if not self.policy_store:
            return []
        try:
            patches = self.policy_store.match(
                collected_info=collected_info,
                candidate_diseases=candidate_names,
                include_shadow=include_shadow,
            )
        except Exception:
            patches = []

        chunks: List[RagChunk] = []
        for patch in patches or []:
            patch_id = patch.get("id") or patch.get("type") or "policy"
            stats = patch.get("stats") or {}
            status = stats.get("status", "shadow")
            score = 0.92 if status == "active" else 0.58
            action = patch.get("action") or ""
            items = patch.get("items") or []
            if items:
                action = f"{action}（关键项: {items}）"
            chunks.append(
                RagChunk(
                    chunk_id=f"policy:{patch_id}",
                    chunk_type="policy_patch",
                    title=f"{patch.get('type', 'policy')} / {status}",
                    text=action,
                    score=score,
                    metadata={
                        "patch_id": patch_id,
                        "status": status,
                        "trigger": patch.get("trigger", {}),
                    },
                )
            )
        return chunks

    def _external_medical_chunks(
        self,
        retrieval_views: Sequence[Any],
        expanded_terms: List[str],
        pool: int,
    ) -> List[RagChunk]:
        try:
            results = self.external_medical.search(
                retrieval_views=retrieval_views,
                query_terms=expanded_terms,
                top_k=min(pool, 8),
            )
        except Exception:
            results = []
        chunks: List[RagChunk] = []
        for result in results or []:
            title = str(result.title or "").strip()
            if not title:
                continue
            candidates = ", ".join(result.candidate_diseases[:4])
            exams = ", ".join(result.recommended_exams[:6])
            text_parts = [result.summary]
            if candidates:
                text_parts.append(f"候选: {candidates}")
            if exams:
                text_parts.append(f"建议检查: {exams}")
            chunks.append(
                RagChunk(
                    chunk_id=f"external:{title}",
                    chunk_type="external_medical_knowledge",
                    title=title,
                    text="; ".join(part for part in text_parts if part),
                    score=max(0.0, min(0.95, float(result.score or 0.0))),
                    metadata={
                        **dict(result.metadata or {}),
                        "candidate_diseases": list(result.candidate_diseases),
                        "recommended_exams": list(result.recommended_exams),
                        "source_label": result.source_label,
                        "submittable": False,
                    },
                )
            )
        return chunks

    @staticmethod
    def _retrieval_view_terms(retrieval_views: Sequence[Any]) -> List[str]:
        terms: List[str] = []
        for view in retrieval_views or []:
            if isinstance(view, dict):
                terms.extend(str(item) for item in view.get("terms", []) or [] if str(item))
                if view.get("query"):
                    terms.append(str(view.get("query")))
                continue
            terms.extend(str(item) for item in getattr(view, "terms", []) or [] if str(item))
            query = getattr(view, "query", "")
            if query:
                terms.append(str(query))
        return _dedupe(terms)

    def _candidate_names(
        self,
        candidate_diseases: Optional[List[Any]],
        expanded_terms: List[str],
        top_k: int,
    ) -> List[str]:
        names: List[str] = []
        for item in candidate_diseases or []:
            name = self._candidate_name(item)
            if name:
                names.append(self._normalize_diagnosis(name) or name)
        for term in expanded_terms:
            normalized = self._normalize_diagnosis(term)
            if normalized:
                names.append(normalized)
        if not names and expanded_terms:
            try:
                for rec in self.knowledge.recall_diseases_by_symptoms(expanded_terms, top_k=top_k):
                    if rec.get("name"):
                        names.append(rec["name"])
            except Exception:
                pass
        names = _dedupe(names)
        if self.allowed_diagnoses:
            names = [name for name in names if name in self.allowed_diagnoses]
        return names[:top_k]

    @staticmethod
    def _load_allowed_diagnoses(config: Dict[str, Any]) -> set:
        ref_dir = str((config or {}).get("ref_data_dir") or "data/ref_data")
        names = set()
        for filename, key in (
            ("diseases_catalog.json", "diseases"),
            ("submission_diagnosis_extensions.json", "extensions"),
        ):
            try:
                with open(os.path.join(ref_dir, filename), "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                for item in data.get(key, []) or []:
                    name = str(item.get("name") or "").strip()
                    if name:
                        names.add(name)
            except (OSError, TypeError, ValueError):
                continue
        return names

    def _candidate_name(self, item: Any) -> Optional[str]:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("disease", "diagnosis", "name"):
                if item.get(key):
                    return str(item[key])
        return None

    def _normalize_diagnosis(self, name: str) -> Optional[str]:
        try:
            return self.knowledge.normalize_diagnosis(name)
        except Exception:
            return None

    def _profile_required_exams(self, disease: str) -> List[str]:
        try:
            profile = self.knowledge.get_disease_profile(disease)
            return list((profile or {}).get("required_exams") or [])
        except Exception:
            return []

    def _render_profile(self, profile: Dict[str, Any]) -> str:
        parts = []
        if profile.get("department"):
            parts.append(f"科室: {profile['department']}")
        if profile.get("matched_symptoms"):
            parts.append(f"命中症状: {', '.join(profile['matched_symptoms'][:6])}")
        if profile.get("red_flags"):
            parts.append(f"红旗信号: {', '.join(profile['red_flags'][:6])}")
        if profile.get("key_questions"):
            parts.append(f"关键问诊: {', '.join(profile['key_questions'][:5])}")
        if profile.get("required_exams"):
            parts.append(f"必查检查: {', '.join(profile['required_exams'])}")
        if profile.get("differential_diagnoses"):
            parts.append(f"重点鉴别: {', '.join(profile['differential_diagnoses'][:6])}")
        if profile.get("treatment_principles"):
            parts.append(f"治疗原则: {', '.join(profile['treatment_principles'][:5])}")
        if profile.get("avoid_mistakes"):
            parts.append(f"易错提醒: {', '.join(profile['avoid_mistakes'][:3])}")
        return "；".join(parts)

    def _render_experience(self, note: Dict[str, Any]) -> str:
        metrics = note.get("metrics") or {}
        bits = []
        if metrics:
            bits.append(
                "评分: "
                f"诊断={metrics.get('diagnosis_accuracy', '?')}, "
                f"检查={metrics.get('exam_precision', '?')}, "
                f"治疗={metrics.get('treatment_score', '?')}"
            )
        expected = note.get("expected_diagnosis") or []
        if expected:
            bits.append(f"参考诊断: {expected}")
        exams = note.get("ordered_exams") or []
        if exams:
            bits.append(f"已做检查: {exams[:6]}")
        is_failure = note.get("memory_kind") == "failure_lesson" or metrics.get("diagnosis_accuracy", 1) < 0.8
        content = note.get("lesson") if is_failure else note.get("content")
        content = content or ""
        if content:
            bits.append(f"{'失败教训' if is_failure else '经验'}: {self._clip(str(content), 420)}")
        return "；".join(bits)

    def _score_experience(self, collected_info: Dict[str, Any], note: Dict[str, Any]) -> float:
        cur_symptoms = set(_as_list(collected_info.get("symptoms")))
        summary = note.get("collected_info_summary") or {}
        note_symptoms = set(_as_list(note.get("symptoms")) or _as_list(summary.get("symptoms")))
        score = 0.42
        if cur_symptoms and note_symptoms:
            inter = len(cur_symptoms & note_symptoms)
            union = len(cur_symptoms | note_symptoms)
            score += 0.26 * (inter / union if union else 0.0)
        quality = (note.get("metrics") or {}).get("quality_score", 0.5)
        try:
            q = float(quality)
        except (TypeError, ValueError):
            q = 0.5
        if q < 0.5:
            score += 0.01
        elif q >= 0.8:
            score += 0.10
        return min(0.86, score)

    def _augment_collected_info(
        self,
        collected_info: Dict[str, Any],
        expanded_terms: List[str],
    ) -> Dict[str, Any]:
        data = dict(collected_info or {})
        symptoms = _dedupe(_as_list(data.get("symptoms")) + expanded_terms)
        data["symptoms"] = symptoms
        return data

    def _query_from_info(self, collected_info: Dict[str, Any]) -> str:
        parts = []
        for key in ("chief_complaint", "present_illness", "past_history"):
            if collected_info.get(key):
                parts.append(str(collected_info[key]))
        parts.extend(str(item) for item in _as_list(collected_info.get("symptoms")))
        return " ".join(parts)

    def _split_medical_terms(self, text: Any) -> List[str]:
        if not text:
            return []
        raw = str(text)
        for sep in "，。！？；：、,.!?;:|/\\()（）【】[]":
            raw = raw.replace(sep, " ")
        terms = [item.strip() for item in raw.split() if item.strip()]
        if not terms and str(text).strip():
            terms = [str(text).strip()]
        return terms

    def _merge_chunks(self, chunks: Iterable[RagChunk]) -> List[RagChunk]:
        merged: Dict[str, RagChunk] = {}
        for chunk in chunks:
            if not chunk.chunk_id:
                continue
            existing = merged.get(chunk.chunk_id)
            if existing is None or chunk.score > existing.score:
                merged[chunk.chunk_id] = chunk
        return list(merged.values())

    def _select_diverse(self, chunks: List[RagChunk], top_k: int) -> List[RagChunk]:
        """保留高分排序，同时避免疾病画像把检查/经验/补丁完全挤出。"""
        if len(chunks) <= top_k:
            return chunks

        quotas = {
            "policy_patch": 1,
            "external_medical_knowledge": max(1, top_k // 5),
            "disease_profile": max(3, top_k // 2),
            "standard_exam": max(2, top_k // 4),
            "case_experience": max(1, top_k // 5),
        }
        selected: List[RagChunk] = []
        selected_ids = set()

        for chunk_type, quota in quotas.items():
            picked = [
                chunk for chunk in chunks
                if chunk.chunk_type == chunk_type and chunk.chunk_id not in selected_ids
            ][:quota]
            for chunk in picked:
                selected.append(chunk)
                selected_ids.add(chunk.chunk_id)
                if len(selected) >= top_k:
                    return sorted(
                        selected,
                        key=lambda item: (item.score, self._type_priority(item.chunk_type)),
                        reverse=True,
                    )

        for chunk in chunks:
            if chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
            if len(selected) >= top_k:
                break

        selected.sort(
            key=lambda item: (item.score, self._type_priority(item.chunk_type)),
            reverse=True,
        )
        return selected[:top_k]

    @staticmethod
    def _type_priority(chunk_type: str) -> int:
        return {
            "policy_patch": 4,
            "external_medical_knowledge": 4,
            "disease_profile": 3,
            "standard_exam": 2,
            "case_experience": 1,
        }.get(chunk_type, 0)

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe(items: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result
