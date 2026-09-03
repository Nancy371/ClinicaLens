"""Pre-submission diagnosis criticism with deterministic and bounded LLM review."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .clinical_evidence import EvidenceBundle
from .diagnosis_engine import DiagnosisDecision, DiagnosticKnowledgeBase
from .diagnosis_resolver import OpenWorldDiagnosisResolver


@dataclass
class CriticDecision:
    issues: List[str] = field(default_factory=list)
    selected_diagnoses: List[str] = field(default_factory=list)
    recommended_exams: List[str] = field(default_factory=list)
    llm_used: bool = False
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DiagnosisCritic:
    """Reject unsupported or contradicted final diagnoses before submission."""

    def __init__(
        self,
        config: Dict[str, Any],
        knowledge: DiagnosticKnowledgeBase,
        llm_chat_json: Optional[
            Callable[[List[Dict[str, str]], float], Awaitable[Optional[Dict[str, Any]]]]
        ] = None,
        resolver: Optional[OpenWorldDiagnosisResolver] = None,
    ):
        section = config.get("final_critic", {}) or {}
        self.llm_on_low_confidence = bool(section.get("llm_on_low_confidence", True))
        self.max_llm_calls = int(section.get("max_llm_calls", 1) or 1)
        self.min_remaining_seconds = float(section.get("min_remaining_seconds", 35) or 35)
        self.corrective_exam_min_seconds = float(
            section.get("corrective_exam_min_seconds", 45) or 45
        )
        self.max_corrective_exam_items = int(
            section.get("max_corrective_exams", section.get("max_corrective_exam_items", 2)) or 2
        )
        self.mode = str(section.get("mode") or "reviewer")
        self.reject_unexplained_major_findings = bool(
            section.get("reject_unexplained_major_findings", False)
        )
        diagnosis_config = config.get("diagnosis", {}) or {}
        self.trusted_threshold = float(
            diagnosis_config.get("trusted_threshold", 0.65) or 0.65
        )
        self.margin_threshold = float(
            diagnosis_config.get("margin_threshold", 0.12) or 0.12
        )
        self.knowledge = knowledge
        self.resolver = resolver or OpenWorldDiagnosisResolver(knowledge)
        self.llm_chat_json = llm_chat_json

    async def review(
        self,
        decision: DiagnosisDecision,
        evidence: EvidenceBundle,
        remaining_seconds: float,
        allow_llm: bool = True,
    ) -> CriticDecision:
        issues = self._deterministic_issues(decision)
        result = CriticDecision(
            issues=issues,
            selected_diagnoses=list(decision.final_diagnoses),
            confidence=decision.confidence,
            reason="确定性终诊审查完成。",
        )
        deterministic_selection, deterministic_reason = self._judge_selection(decision)
        if deterministic_selection:
            result.selected_diagnoses = deterministic_selection
            result.reason = deterministic_reason
        should_call = bool(issues) and (
            decision.confidence < self.trusted_threshold
            or decision.margin < self.margin_threshold
            or any(item.startswith(("hard_contradiction", "unexplained")) for item in issues)
        )
        has_evidence_gap = bool(issues) or decision.low_confidence
        if not (
            allow_llm
            and should_call
            and self.llm_on_low_confidence
            and self.max_llm_calls > 0
            and self.llm_chat_json is not None
            and remaining_seconds >= self.min_remaining_seconds
        ):
            if has_evidence_gap and remaining_seconds >= self.corrective_exam_min_seconds:
                result.recommended_exams = self._default_discriminating_exams(decision)
            return result

        prompt = self._build_prompt(decision, evidence, issues)
        try:
            raw = await self.llm_chat_json(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": "请审查候选诊断，只输出约定的 JSON 对象。",
                    },
                ],
                temperature=0.1,
            )
        except Exception as exc:
            result.reason = f"LLM Critic 调用失败，保留确定性结果：{exc}"
            return result
        if not isinstance(raw, dict):
            result.reason = "LLM Critic 未返回有效对象，保留确定性结果。"
            return result

        eligible = {
            item.diagnosis: item
            for item in decision.candidates
            if not item.hard_contradiction
            and item.matched_evidence
            and item.trusted
            and item.score >= 0.35
        }
        selected = raw.get("selected_diagnoses") or []
        if isinstance(selected, str):
            selected = [selected]
        validated: List[str] = []
        for item in selected:
            normalized = self.resolver.resolve(item).canonical_name
            if normalized in eligible and normalized not in validated:
                validated.append(normalized)
        if validated:
            result.selected_diagnoses = validated[:3]

        exams = raw.get("recommended_exams") or []
        if isinstance(exams, str):
            exams = [exams]
        allowed_exams = set(self._default_discriminating_exams(decision, limit=12))
        result.recommended_exams = [
            str(item) for item in exams
            if str(item) in allowed_exams
        ][: self.max_corrective_exam_items]
        if not result.recommended_exams and remaining_seconds >= self.corrective_exam_min_seconds:
            result.recommended_exams = self._default_discriminating_exams(decision)
        try:
            result.confidence = float(raw.get("confidence", result.confidence))
        except (TypeError, ValueError):
            pass
        result.reason = str(raw.get("reason") or "LLM Critic 完成候选内重排。")
        result.llm_used = True
        return result

    def _deterministic_issues(self, decision: DiagnosisDecision) -> List[str]:
        issues: List[str] = []
        if not decision.final_diagnoses:
            issues.append("empty_final_diagnosis")
        if decision.confidence < self.trusted_threshold:
            issues.append(f"low_confidence:{decision.confidence:.3f}")
        if decision.margin < self.margin_threshold:
            issues.append(f"narrow_margin:{decision.margin:.3f}")
        if decision.unexplained_evidence:
            issues.append("unexplained:" + ",".join(decision.unexplained_evidence[:8]))
        score_by_name = {item.diagnosis: item for item in decision.candidates}
        for name in decision.final_diagnoses:
            if not self.knowledge.is_allowed(name):
                issues.append(f"namespace_violation:{name}")
            item = score_by_name.get(name)
            if item and item.hard_contradiction:
                issues.append(f"hard_contradiction:{name}")
            if item and item.required_gaps:
                issues.append(f"evidence_gap:{name}:{'|'.join(item.required_gaps[:3])}")
            if item and self.reject_unexplained_major_findings and decision.unexplained_evidence:
                issues.append(f"judge_review_needed:{name}")
        for item in decision.candidates[:8]:
            if item.source_prior >= 0.45 and item.required_gaps and item.matched_evidence:
                issues.append(f"llm_or_rag_candidate_gap:{item.diagnosis}")
        if decision.candidates and decision.final_diagnoses:
            top = decision.candidates[0]
            selected_top = score_by_name.get(decision.final_diagnoses[0])
            if selected_top and top.score - selected_top.score >= self.margin_threshold:
                issues.append(f"higher_scored_omitted:{top.diagnosis}")
        return list(dict.fromkeys(issues))

    def _judge_selection(self, decision: DiagnosisDecision) -> tuple[List[str], str]:
        if self.mode != "judge" or not decision.candidates:
            return [], ""
        selected_name = decision.final_diagnoses[0] if decision.final_diagnoses else ""
        selected = next(
            (item for item in decision.candidates if item.diagnosis == selected_name),
            None,
        )
        eligible = [
            item for item in decision.candidates
            if item.matched_evidence
            and item.trusted
            and not item.hard_contradiction
            and item.score >= 0.35
        ]
        if not eligible:
            return [], ""
        if not selected:
            return [eligible[0].diagnosis], "Judge selected the strongest evidence-backed candidate."
        better = [
            item for item in eligible
            if item.diagnosis != selected.diagnosis
            and (
                item.score >= selected.score + self.margin_threshold
                or (
                    decision.unexplained_evidence
                    and item.explanation_score >= selected.explanation_score + 0.20
                    and item.score >= selected.score - self.margin_threshold
                )
                or (
                    item.source_prior >= 0.65
                    and item.required_gaps
                    and item.score >= selected.score - self.margin_threshold
                    and item.explanation_score >= selected.explanation_score
                )
            )
        ]
        if not better:
            return [], ""
        better.sort(
            key=lambda item: (
                item.explanation_score,
                item.source_prior,
                item.specificity,
                item.score,
            ),
            reverse=True,
        )
        replacement = better[0]
        keep = [
            item.diagnosis for item in eligible
            if item.diagnosis != replacement.diagnosis
            and item.score >= self.trusted_threshold
        ][:2]
        return [replacement.diagnosis] + keep, (
            f"Judge rejected {selected.diagnosis} because unresolved major evidence or stronger "
            f"candidate evidence favored {replacement.diagnosis}."
        )

    def _default_discriminating_exams(
        self,
        decision: DiagnosisDecision,
        limit: Optional[int] = None,
    ) -> List[str]:
        maximum = limit or self.max_corrective_exam_items
        top = self._evidence_gap_targets(decision)
        support: Dict[str, set] = {}
        relevance: Dict[str, float] = {}
        for rank, candidate in enumerate(top):
            for exam in self.knowledge.get(candidate.diagnosis).get("discriminating_exams", []) or []:
                name = str(exam).strip()
                if not name:
                    continue
                support.setdefault(name, set()).add(candidate.diagnosis)
                relevance[name] = relevance.get(name, 0.0) + candidate.score / (rank + 1)
        count = max(1, len(top))
        scored = []
        for exam, diagnoses in support.items():
            coverage = len(diagnoses)
            split = 1.0 if count > 1 and 0 < coverage < count else 0.35
            scored.append((0.65 * split + 0.35 * relevance.get(exam, 0.0), exam))
        scored.sort(reverse=True)
        return [exam for _, exam in scored[:maximum]]

    def _evidence_gap_targets(self, decision: DiagnosisDecision) -> List[Any]:
        by_name = {item.diagnosis: item for item in decision.candidates}
        targets: List[Any] = []

        for name in decision.final_diagnoses[:1]:
            candidate = by_name.get(name)
            if candidate:
                targets.append(candidate)

        priority_candidates = [
            item
            for item in decision.candidates
            if item.matched_evidence
            and not item.hard_contradiction
            and (
                item.required_gaps
                or item.source_prior >= 0.45
                or
                item.diagnosis_type.lower() in {"etiology", "metabolic", "structural"}
            )
        ]
        if priority_candidates:
            targets.append(priority_candidates[0])

        unexplained = set(decision.unexplained_evidence or [])
        if unexplained:
            for item in decision.candidates:
                if item.hard_contradiction:
                    continue
                if unexplained & set(item.matched_evidence or []):
                    targets.append(item)
                    break

        if not targets:
            targets = [
                item
                for item in decision.candidates
                if item.score >= 0.35 or item.matched_evidence
            ][:3]
        if not targets:
            targets = decision.candidates[:3]

        unique: List[Any] = []
        seen = set()
        for item in targets:
            if item.diagnosis not in seen:
                unique.append(item)
                seen.add(item.diagnosis)
        return unique[:3]

    @staticmethod
    def _build_prompt(
        decision: DiagnosisDecision,
        evidence: EvidenceBundle,
        issues: List[str],
    ) -> str:
        candidates = [
            {
                "diagnosis": item.diagnosis,
                "score": item.score,
                "required_met": item.required_met,
                "hard_contradiction": item.hard_contradiction,
                "matched_evidence": item.matched_evidence,
                "contradicted_evidence": item.contradicted_evidence,
            }
            for item in decision.candidates[:5]
        ]
        evidence_rows = [item.to_dict() for item in evidence.major()[:20]]
        return (
            "你是提交前诊断审查器。只能在给定候选中重排，禁止创造新疾病名称，"
            "required_met=false 代表证据缺口而不是排除；可以选择有充分证据但存在缺口的候选，"
            "但禁止覆盖 hard_contradiction=true 的确定性约束。"
            "检查最终诊断是否解释主要异常，并选择最多3个诊断。\n"
            f"当前问题: {issues}\n候选: {candidates}\n主要证据: {evidence_rows}\n"
            "输出 JSON: {\"selected_diagnoses\":[],\"confidence\":0.0,"
            "\"unexplained_findings\":[],\"recommended_exams\":[],\"reason\":\"\"}。"
        )
