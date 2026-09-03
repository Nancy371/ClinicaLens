"""检查策略 Agent：结合疾病画像补齐必查检查并过滤无效检查。"""

from typing import Any, Dict, List, Optional

from .exam_resolver import ALIAS, EQUIVALENT, EXACT, PARTIAL_SUBSTITUTE, ExamResolver
from .knowledge import KnowledgeBase

_GENERIC_INFLAMMATION_EXAM_MARKERS = (
    "CBC",
    "CRP",
    "ESR",
    "PCT",
    "全血细胞计数",
    "C反应蛋白",
    "红细胞沉降率",
    "降钙素原",
)
_SPECIAL_DISCRIMINATOR_EXAM_MARKERS = (
    "AFB",
    "NAAT",
    "Xpert",
    "ANCA",
    "MPO",
    "p-ANCA",
    "CT",
    "MRI",
    "血清学",
    "涂片",
    "病理",
    "活检",
    "支气管镜",
    "尿液分析",
    "肾功能",
    "屈光",
    "眼压",
    "裂隙灯",
    "痰培养",
)

_FULL_EXAM_RESOLUTION_TYPES = {EXACT, ALIAS, EQUIVALENT}
_USABLE_EXAM_RESOLUTION_TYPES = {
    EXACT,
    ALIAS,
    EQUIVALENT,
    PARTIAL_SUBSTITUTE,
}


_PEDIATRIC_CARDIAC_SCREEN_EXAMS = [
    "体格检查",
    "超声心动图",
    "心电图",
    "胸部X线",
    "心导管检查",
]

_RIGHT_HEART_VALVE_EXAMS = [
    "体格检查",
    "心电图（ECG）",
    "超声心动图",
    "三维超声心动图（3D Echo）",
    "经食管超声心动图（TEE）",
    "心脏MRI（CMR）",
]

_URINARY_SYNDROME_EXAMS = [
    "体格检查",
    "尿液分析（UA）",
    "尿培养",
    "综合代谢面板（CMP）",
    "泌尿道超声",
    "尿动力学检查（UDS）",
]

_RIB_TRAUMA_EXAMS = [
    "体格检查",
    "胸部X线检查（CXR）",
    "胸部CT扫描（Chest CT）",
]

_RETINOBLASTOMA_EXAMS = [
    "眼底摄影",
    "眼部超声",
    "磁共振成像（MRI）",
    "CT扫描（CT）",
]

_HIB_RESPIRATORY_EXAMS = [
    "全血细胞计数（CBC）",
    "血培养",
    "痰培养",
    "血清学抗体检测",
    "胸部X线检查（CXR）",
]

_IMMUNE_PNEUMONIA_EXAMS = [
    "体格检查",
    "脉搏血氧饱和度监测（SpO2）",
    "动脉血气（ABG）",
    "全血细胞计数（CBC）",
    "C反应蛋白（CRP）",
    "降钙素原（PCT）",
    "胸部X线检查（CXR）",
    "痰培养",
    "抗菌药物敏感性试验（AST）",
    "超声",
    "类风湿因子（RF）",
]

_CONGENITAL_SHUNT_EXAMS = [
    "超声心动图",
    "心电图（ECG）",
    "胸部X线检查（CXR）",
    "心导管检查",
]

_ELECTROLYTE_CRISIS_EXAMS = [
    "电解质",
    "心电图（ECG）",
    "肾功能",
    "血气分析",
]

_PULMONARY_RENAL_EXAMS = [
    "尿常规",
    "肾功能",
    "胸部CT",
    "抗核抗体",
    "血常规",
    "C反应蛋白",
]

_METABOLIC_BONE_EXAMS = [
    "电解质",
    "骨密度",
    "肝功能",
    "肾功能",
]

_ASPIRATION_PNEUMONIA_EXAMS = [
    "体格检查",
    "脉搏血氧饱和度监测（SpO2）",
    "血气分析",
    "胸部X线",
    "血常规",
    "C反应蛋白",
    "降钙素原",
    "痰培养",
    "支气管镜",
]

_ADVANCED_CARDIAC_EXAMS = {
    "三维超声心动图（3D Echo）",
    "经食管超声心动图（TEE）",
    "心脏MRI（CMR）",
    "心导管检查",
}

_STRICT_COMPANION_EXAM_PATHS = {
    frozenset({"肺不张", "支气管肺炎"}),
    frozenset({"右位心", "室间隔缺损（VSD）"}),
}

_STRONG_VERIFICATION_EXAMS = {
    "低镁血症": [
        "综合代谢面板（CMP）",
        "24小时尿电解质检测",
        "镁负荷试验",
        "心电图（ECG）",
    ],
    "维生素D缺乏性佝偻病": [
        "维生素D检测",
        "血清电解质",
        "甲状旁腺激素检测（PTH）",
        "肝功能检查（LFTs）",
        "骨转换标志物（BTMs）",
        "X线检查",
    ],
    "显微镜下多血管炎": [
        "尿液分析（UA）",
        "肾功能",
        "胸部CT扫描（Chest CT）",
        "抗核抗体",
        "全血细胞计数（CBC）",
        "C反应蛋白（CRP）",
    ],
    "肺不张": [
        "胸部X线检查（CXR）",
        "胸部CT扫描（Chest CT）",
        "支气管镜检查",
        "动脉血气（ABG）",
    ],
    "支气管肺炎": [
        "体格检查",
        "脉搏血氧饱和度监测（SpO2）",
        "胸部X线检查（CXR）",
        "全血细胞计数（CBC）",
        "C反应蛋白（CRP）",
        "降钙素原（PCT）",
        "痰培养",
        "抗菌药物敏感性试验（AST）",
    ],
    "二度房室传导阻滞": [
        "体格检查",
        "心电图（ECG）",
        "动态心电图（Holter）",
    ],
    "克里格勒-纳贾尔综合征": [
        "肝功能检查（LFTs）",
        "凝血功能全套",
        "腹部超声",
        "基因检测",
    ],
    "慢性鼻咽炎": [
        "鼻咽镜检查",
        "脱落细胞学检查",
    ],
    "小耳畸形": [
        "听性脑干反应（ABR）",
        "颞骨CT扫描（颞骨CT）",
        "基因检测",
    ],
    "急性细菌性前列腺炎": [
        "直肠指检（DRE）",
        "尿液分析（UA）",
        "尿培养",
        "全血细胞计数（CBC）",
        "抗菌药物敏感性试验（AST）",
        "前列腺超声",
    ],
}


class ExamStrategyAgent:
    """轻量检查策略角色，不调用外部服务，只做本地规则增强。"""

    def __init__(
        self,
        knowledge: KnowledgeBase,
        max_new_items: int = 10,
        discriminating_exam_max_items: int = 6,
    ):
        self.knowledge = knowledge
        self.max_new_items = max_new_items
        self.discriminating_exam_max_items = discriminating_exam_max_items
        self.exam_resolver = ExamResolver(knowledge)

    def recommend(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
        existing_results: Optional[Dict[str, Any]] = None,
        judge_decision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """返回本轮建议检查项。

        优先保留疾病画像中的必查检查，其次使用 LLM 提议的有效检查项。
        """
        symptoms = collected_info.get("symptoms", []) if collected_info else []
        existing_results = existing_results or {}
        existing_valid, _ = self.knowledge.normalize_examinations(list(existing_results.keys()))
        existing_set = set(existing_results.keys()) | set(existing_valid)

        judge_payload = self._judge_payload(judge_decision)
        proposed_valid, invalid_items = self.knowledge.normalize_examinations(
            proposed_items or []
        )
        differential_plan = self._differential_driven_plan(
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            proposed_items=proposed_valid,
            existing_results=existing_results,
            judge_decision=judge_decision,
        )
        if differential_plan:
            items = differential_plan["items"]
            considered = list(
                dict.fromkeys(proposed_valid + differential_plan["candidate_exam_pool"])
            )
            blocked_items = self._blocked_exam_items(
                considered,
                allowed=items,
                existing_results=existing_results,
            )
            return {
                "items": items,
                "strong_verification_items": [],
                "required_items": [],
                "red_flag_items": [],
                "evidence_driven_items": items,
                "information_gain": differential_plan["information_gain"],
                "exam_authorization_details": differential_plan.get(
                    "exam_authorization_details",
                    [],
                ),
                "added_required": [],
                "invalid_items": invalid_items,
                "strict_diagnosis_driven": False,
                "differential_driven": True,
                "primary_diagnosis": differential_plan["primary_diagnosis"],
                "differential_candidates": differential_plan["differential_candidates"],
                "discriminating_items": items,
                "reserved_gap_items": list(
                    differential_plan.get("reserved_gap_items") or []
                ),
                "reserved_pairwise_items": list(
                    differential_plan.get("reserved_pairwise_items") or []
                ),
                "source_decision_version": differential_plan.get(
                    "source_decision_version",
                    0,
                ),
                "source_evidence_version": differential_plan.get(
                    "source_evidence_version",
                    0,
                ),
                "blocked_items": blocked_items,
                "generic_exam_suppression_count": differential_plan.get(
                    "generic_exam_suppression_count",
                    0,
                ),
                "clinical_context": self.knowledge.build_clinical_context(
                    symptoms=symptoms,
                    candidate_diseases=differential_plan["differential_candidates"],
                ),
            }
        if judge_payload and judge_payload.get("needs_discriminating_exams"):
            blocked_items = self._blocked_exam_items(
                proposed_valid,
                allowed=[],
                existing_results=existing_results,
            )
            return {
                "items": [],
                "strong_verification_items": [],
                "required_items": [],
                "red_flag_items": [],
                "evidence_driven_items": [],
                "information_gain": {},
                "exam_authorization_details": [],
                "added_required": [],
                "invalid_items": invalid_items,
                "strict_diagnosis_driven": False,
                "differential_driven": False,
                "primary_diagnosis": str(
                    judge_payload.get("provisional_primary")
                    or judge_payload.get("primary")
                    or judge_payload.get("judge_primary")
                    or ""
                ),
                "differential_candidates": list(
                    judge_payload.get("differential_candidates") or []
                ),
                "discriminating_items": [],
                "blocked_items": blocked_items,
                "clinical_context": self.knowledge.build_clinical_context(
                    symptoms=symptoms,
                    candidate_diseases=judge_payload.get("differential_candidates") or [],
                ),
            }
        if judge_payload and str(judge_payload.get("primary_status") or "") != "locked":
            blocked_items = self._blocked_exam_items(
                proposed_valid,
                allowed=[],
                existing_results=existing_results,
            )
            return {
                "items": [],
                "strong_verification_items": [],
                "required_items": [],
                "red_flag_items": [],
                "evidence_driven_items": [],
                "information_gain": {},
                "exam_authorization_details": [],
                "added_required": [],
                "invalid_items": invalid_items,
                "strict_diagnosis_driven": False,
                "differential_driven": False,
                "primary_diagnosis": str(
                    judge_payload.get("provisional_primary")
                    or judge_payload.get("primary")
                    or judge_payload.get("judge_primary")
                    or ""
                ),
                "differential_candidates": list(
                    judge_payload.get("differential_candidates") or []
                ),
                "discriminating_items": [],
                "blocked_items": blocked_items,
                "clinical_context": self.knowledge.build_clinical_context(
                    symptoms=symptoms,
                    candidate_diseases=judge_payload.get("differential_candidates") or [],
                ),
            }

        strong_verification_items = self._strong_verification_items(
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            proposed_items=proposed_items,
        )
        ranked_items, information_gain = self._rank_by_information_gain(
            candidate_diseases=candidate_diseases or [],
            symptoms=symptoms,
            proposed_items=proposed_valid,
        )
        required_items = self.knowledge.get_required_exams(
            candidate_diseases=(candidate_diseases or [])[:3],
            symptoms=symptoms,
            include_optional=False,
        )
        red_flag_items = self._scenario_items(collected_info, candidate_diseases)
        evidence_driven_items = list(dict.fromkeys(red_flag_items + ranked_items))
        priority_proposed = self._priority_proposed_items(
            proposed_valid,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
        )
        remaining_proposed = [
            item for item in proposed_valid if item not in set(priority_proposed)
        ]
        strict_plan = self._strict_authorized_exam_plan(
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            proposed_items=proposed_items,
        )
        if strict_plan:
            merged = self.prepare_order_items(
                strict_plan["items"],
                collected_info=collected_info,
                candidate_diseases=[strict_plan["primary_diagnosis"]],
                existing_results=existing_results,
                max_items=strict_plan["max_items"],
                add_strong_verification=False,
            )
            considered = list(
                dict.fromkeys(
                    proposed_valid
                    + priority_proposed
                    + evidence_driven_items
                    + required_items
                    + remaining_proposed
                )
            )
            blocked_items = self._blocked_exam_items(
                considered,
                allowed=merged,
                existing_results=existing_results,
            )
            return {
                "items": merged,
                "strong_verification_items": [
                    item for item in strict_plan["strong_items"] if item not in existing_set
                ],
                "required_items": required_items,
                "red_flag_items": red_flag_items,
                "evidence_driven_items": evidence_driven_items,
                "information_gain": information_gain,
                "added_required": [item for item in required_items if item not in proposed_valid],
                "invalid_items": invalid_items,
                "strict_diagnosis_driven": True,
                "differential_driven": False,
                "primary_diagnosis": strict_plan["primary_diagnosis"],
                "differential_candidates": [],
                "discriminating_items": [],
                "blocked_items": blocked_items,
                "clinical_context": self.knowledge.build_clinical_context(
                    symptoms=symptoms,
                    candidate_diseases=[strict_plan["primary_diagnosis"]],
                ),
            }
        if judge_payload:
            blocked_items = self._blocked_exam_items(
                list(
                    dict.fromkeys(
                        proposed_valid
                        + priority_proposed
                        + evidence_driven_items
                        + required_items
                        + remaining_proposed
                    )
                ),
                allowed=[],
                existing_results=existing_results,
            )
            return {
                "items": [],
                "strong_verification_items": [],
                "required_items": [],
                "red_flag_items": [],
                "evidence_driven_items": [],
                "information_gain": information_gain,
                "exam_authorization_details": [],
                "added_required": [],
                "invalid_items": invalid_items,
                "strict_diagnosis_driven": False,
                "differential_driven": False,
                "primary_diagnosis": str(
                    judge_payload.get("primary") or judge_payload.get("judge_primary") or ""
                ),
                "differential_candidates": list(
                    judge_payload.get("differential_candidates") or []
                ),
                "discriminating_items": [],
                "blocked_items": blocked_items,
                "clinical_context": self.knowledge.build_clinical_context(
                    symptoms=symptoms,
                    candidate_diseases=judge_payload.get("differential_candidates") or [],
                ),
            }

        merged: List[str] = []
        for item in (
            strong_verification_items
            + priority_proposed
            + evidence_driven_items
            + required_items
            + remaining_proposed
        ):
            if item and item not in merged:
                merged.append(item)
        merged = self.prepare_order_items(
            merged,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
            existing_results=existing_results,
            max_items=self.max_new_items,
            add_strong_verification=False,
        )

        return {
            "items": merged,
            "strong_verification_items": [
                item for item in strong_verification_items if item not in existing_set
            ],
            "required_items": required_items,
            "red_flag_items": red_flag_items,
            "evidence_driven_items": evidence_driven_items,
            "information_gain": information_gain,
            "added_required": [item for item in required_items if item not in proposed_valid],
            "invalid_items": invalid_items,
            "strict_diagnosis_driven": False,
            "differential_driven": False,
            "primary_diagnosis": "",
            "differential_candidates": [],
            "discriminating_items": [],
            "blocked_items": [],
            "clinical_context": self.knowledge.build_clinical_context(
                symptoms=symptoms,
                candidate_diseases=candidate_diseases,
            ),
        }

    def _differential_driven_plan(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]],
        proposed_items: List[str],
        existing_results: Dict[str, Any],
        judge_decision: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload = self._judge_payload(judge_decision)
        if not payload:
            return {}
        exam_tasks = self._judge_exam_tasks(payload)
        raw_discriminating = [
            str(item).strip()
            for item in (
                [task.get("exam") for task in exam_tasks]
                or payload.get("discriminating_exams", [])
                or []
            )
            if str(item).strip()
        ]
        if not raw_discriminating:
            return {}
        differential_candidates = [
            self.knowledge.normalize_diagnosis(str(item)) or str(item)
            for item in (
                payload.get("differential_candidates")
                or payload.get("evidence_gap_targets")
                or candidate_diseases
                or []
            )
            if str(item).strip()
        ]
        differential_candidates = list(dict.fromkeys(differential_candidates))[:6]
        task_by_exam = self._normalized_exam_task_map(exam_tasks)
        task_exam_items = list(task_by_exam)
        exam_resolutions = self.exam_resolver.resolve_many(raw_discriminating)
        resolver_items = [
            item.resolved_exam
            for item in exam_resolutions
            if item.resolved_exam
            and item.resolution_type
            in {"exact", "alias", "equivalent", "partial_substitute"}
        ]
        normalized, _ = self.knowledge.normalize_examinations(
            resolver_items or raw_discriminating
        )
        normalized = list(dict.fromkeys(task_exam_items + normalized))
        entity_fallback_pool = self._entity_exam_fallback_pool(
            differential_candidates or candidate_diseases or []
        )
        entity_exam_resolutions = self.exam_resolver.resolve_many(entity_fallback_pool)
        entity_fallback_normalized, _ = self.knowledge.normalize_examinations(
            [
                item.resolved_exam
                for item in entity_exam_resolutions
                if item.resolved_exam
            ]
            or entity_fallback_pool
        )
        use_entity_fallback = (
            not normalized
            or self._lossy_special_exam_normalization(
                raw_discriminating,
                normalized,
                entity_fallback_normalized,
            )
        )
        entity_fallback_set = set(entity_fallback_normalized) if use_entity_fallback else set()
        if use_entity_fallback:
            normalized = list(dict.fromkeys(task_exam_items + entity_fallback_normalized))
        if not normalized:
            return {}
        resolution_by_exam = {
            exam: dict(task.get("exam_resolution") or {})
            for exam, task in task_by_exam.items()
            if isinstance(task.get("exam_resolution"), dict)
        }
        for item in list(exam_resolutions) + list(entity_exam_resolutions):
            if item.resolved_exam and item.resolved_exam not in resolution_by_exam:
                resolution_by_exam[item.resolved_exam] = item.to_dict()
        ranked, information_gain = self._rank_by_information_gain(
            candidate_diseases=differential_candidates or candidate_diseases or [],
            symptoms=(collected_info or {}).get("symptoms", []),
            proposed_items=list(dict.fromkeys(normalized + proposed_items)),
            exam_tasks=exam_tasks,
        )
        task_items_by_priority = sorted(
            task_by_exam.items(),
            key=lambda item: self._exam_task_priority_key(item[1]),
            reverse=True,
        )
        urgent_task_items = [
            exam
            for exam, task in task_items_by_priority
            if bool(task.get("urgent_safety"))
        ]
        deferred_gap_task_items = [
            exam
            for exam, task in task_items_by_priority
            if str(task.get("exam_source") or "") == "deferred_gap_closure_exam"
            and bool(task.get("priority_override"))
        ]
        pairwise_gap_task_items = [
            exam
            for exam, task in task_items_by_priority
            if str(task.get("exam_source") or "") == "pairwise_discrimination_exam"
        ]
        followup_task_items = [
            exam
            for exam, task in task_items_by_priority
            if str(task.get("exam_source") or "")
            in {"evidence_claim_followup_exam", "pattern_anchor_workup_exam"}
        ]
        conflict_task_items = [
            exam
            for exam, task in task_items_by_priority
            if str(task.get("exam_source") or "") == "conflict_adjudication_exam"
            and not bool(task.get("urgent_safety"))
        ]
        priority_task_items = list(
            dict.fromkeys(
                urgent_task_items
                + pairwise_gap_task_items
                + deferred_gap_task_items
                + followup_task_items
                + conflict_task_items
            )
        )
        high_value_proposed = [
            item
            for item in proposed_items
            if information_gain.get(item, 0.0) >= 0.35
        ]
        if payload.get("needs_discriminating_exams"):
            high_value_proposed = []
            ordered = list(
                dict.fromkeys(
                    priority_task_items
                    + [item for item in ranked if item in set(normalized)]
                )
            )
        else:
            ordered = list(
                dict.fromkeys(
                    priority_task_items
                    + high_value_proposed
                    + [item for item in ranked if item in set(normalized)]
                )
            )
        items = self._prepare_differential_order_items(
            list(dict.fromkeys(ordered)),
            collected_info=collected_info,
            candidate_diseases=differential_candidates or candidate_diseases,
            existing_results=existing_results,
            max_items=self.discriminating_exam_max_items,
            task_by_exam=task_by_exam,
        )
        if not items:
            return {}
        target_findings = [
            str(item).strip()
            for item in payload.get("discriminating_findings", []) or []
            if str(item).strip()
        ]
        authorization_details = [
            {
                "exam": item,
                "exam_source": (
                    str(task_by_exam.get(item, {}).get("exam_source") or "")
                    or (
                        "entity_exam_bundle_fallback"
                        if item in entity_fallback_set and item not in task_by_exam
                        else "judge_discriminating_exam"
                    )
                ),
                "target_candidates": list(
                    task_by_exam.get(item, {}).get("target_candidates")
                    or differential_candidates
                ),
                "target_findings": list(
                    task_by_exam.get(item, {}).get("target_findings")
                    or target_findings
                ),
                "exam_type": str(
                    task_by_exam.get(item, {}).get("exam_type")
                    or self._exam_type_for_name(item)
                ),
                "expected_effect": str(
                    task_by_exam.get(item, {}).get("expected_effect")
                    or "shift_probabilities_across_differential_pool"
                ),
                "information_gain": information_gain.get(item, 0.0),
                "allowed_reason": (
                    "exam_priority_override"
                    if bool(task_by_exam.get(item, {}).get("priority_override"))
                    else "needs_discriminating_exams"
                    if payload.get("needs_discriminating_exams")
                    else "judge_discriminating_exam"
                ),
                "blocked_reason": "",
                "exam_resolution": dict(resolution_by_exam.get(item, {})),
                "target_gap": str(task_by_exam.get(item, {}).get("target_gap") or ""),
                "target_gaps": list(task_by_exam.get(item, {}).get("target_gaps") or []),
                "priority_override": bool(task_by_exam.get(item, {}).get("priority_override")),
                "override_reason": str(task_by_exam.get(item, {}).get("override_reason") or ""),
                "priority_bucket": str(
                    task_by_exam.get(item, {}).get("priority_bucket")
                    or self._exam_task_priority_bucket(task_by_exam.get(item, {}))
                ),
                "closure_rank": task_by_exam.get(item, {}).get("closure_rank"),
                "closure_priority": task_by_exam.get(item, {}).get("closure_priority"),
                "requested_exam": str(
                    task_by_exam.get(item, {}).get("requested_exam")
                    or resolution_by_exam.get(item, {}).get("requested_exam")
                    or item
                ),
                "resolved_exam": str(
                    task_by_exam.get(item, {}).get("resolved_exam")
                    or resolution_by_exam.get(item, {}).get("resolved_exam")
                    or item
                ),
                "resolution_type": str(
                    task_by_exam.get(item, {}).get("resolution_type")
                    or resolution_by_exam.get(item, {}).get("resolution_type")
                    or ""
                ),
                "diagnostic_coverage": float(
                    task_by_exam.get(item, {}).get("diagnostic_coverage")
                    or resolution_by_exam.get(item, {}).get("diagnostic_coverage")
                    or 0.0
                ),
                "gap_diagnostic_coverage": float(
                    task_by_exam.get(item, {}).get("gap_diagnostic_coverage")
                    or 0.0
                ),
                "source_gap_value": float(
                    task_by_exam.get(item, {}).get("source_gap_value") or 0.0
                ),
                "source_gap_id": str(
                    task_by_exam.get(item, {}).get("source_gap_id") or ""
                ),
                "target_pair": list(task_by_exam.get(item, {}).get("target_pair") or []),
                "target_question": str(
                    task_by_exam.get(item, {}).get("target_question") or ""
                ),
                "target_claim": str(
                    task_by_exam.get(item, {}).get("target_claim") or ""
                ),
                "target_claims": list(
                    task_by_exam.get(item, {}).get("target_claims") or []
                ),
                "route_target_claims": list(
                    task_by_exam.get(item, {}).get("route_target_claims") or []
                ),
                "expected_evidence_concepts": list(
                    task_by_exam.get(item, {}).get("expected_evidence_concepts") or []
                ),
                "claim_requirements": list(
                    task_by_exam.get(item, {}).get("claim_requirements") or []
                ),
                "closure_routes": list(
                    task_by_exam.get(item, {}).get("closure_routes") or []
                ),
                "claim_closure_plan_version": str(
                    task_by_exam.get(item, {}).get("claim_closure_plan_version") or ""
                ),
                "clinical_question": str(
                    task_by_exam.get(item, {}).get("clinical_question")
                    or task_by_exam.get(item, {}).get("target_question")
                    or ""
                ),
                "positive_resolution_rules": list(
                    task_by_exam.get(item, {}).get("positive_resolution_rules") or []
                ),
                "negative_resolution_rules": list(
                    task_by_exam.get(item, {}).get("negative_resolution_rules") or []
                ),
                "exam_role": str(task_by_exam.get(item, {}).get("exam_role") or ""),
                "expected_arbitration_effect": dict(
                    task_by_exam.get(item, {}).get("expected_arbitration_effect") or {}
                ),
                "gap_value_rank": task_by_exam.get(item, {}).get("gap_value_rank"),
                "gap_value_components": dict(
                    task_by_exam.get(item, {}).get("gap_value_components") or {}
                ),
                "exam_gap_closure_value": float(
                    task_by_exam.get(item, {}).get("exam_gap_closure_value") or 0.0
                ),
                "candidate_score_at_decision": float(
                    task_by_exam.get(item, {}).get("candidate_score_at_decision")
                    or 0.0
                ),
                "score_gap_decoupled": bool(
                    task_by_exam.get(item, {}).get("score_gap_decoupled")
                ),
                "source_decision_version": payload.get("decision_version")
                or payload.get("case_version")
                or 0,
                "source_evidence_version": payload.get("evidence_version")
                or payload.get("case_version")
                or 0,
                "source_evidence_snapshot_hash": str(
                    payload.get("evidence_snapshot_hash") or ""
                ),
            }
            for item in items
        ]
        reserved_gap_items = [
            detail["exam"]
            for detail in authorization_details
            if detail.get("exam_source") == "deferred_gap_closure_exam"
            and detail.get("priority_override")
        ]
        reserved_pairwise_items = [
            detail["exam"]
            for detail in authorization_details
            if detail.get("exam_source") == "pairwise_discrimination_exam"
        ]
        return {
            "items": items,
            "information_gain": information_gain,
            "differential_candidates": differential_candidates,
            "primary_diagnosis": str(payload.get("primary") or payload.get("judge_primary") or ""),
            "candidate_exam_pool": normalized,
            "exam_authorization_details": authorization_details,
            "reserved_gap_items": reserved_gap_items,
            "reserved_pairwise_items": reserved_pairwise_items,
            "source_decision_version": payload.get("decision_version")
            or payload.get("case_version")
            or 0,
            "source_evidence_version": payload.get("evidence_version")
            or payload.get("case_version")
            or 0,
            "generic_exam_suppression_count": sum(
                1
                for item in normalized + proposed_items
                if self._generic_inflammation_exam(item) and item not in set(items)
            ),
        }

    def _judge_exam_tasks(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for key in ("deferred_gap_closure_tasks", "discriminating_exam_tasks"):
            for item in payload.get(key, []) or []:
                if isinstance(item, dict) and str(item.get("exam") or "").strip():
                    tasks.append(dict(item))
        tasks.extend(self._exam_tasks_from_active_gaps(payload))
        tasks.extend(self._exam_tasks_from_priority_overrides(payload))

        result: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for task in tasks:
            exam = str(task.get("exam") or "").strip()
            if not exam:
                continue
            gap = str(task.get("target_gap") or "")
            source = str(task.get("exam_source") or "")
            key = (exam, gap, source)
            if key in seen:
                continue
            seen.add(key)
            result.append(task)
        return result

    def _exam_tasks_from_active_gaps(
        self,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        active_gaps = [
            gap
            for gap in payload.get("active_evidence_gaps", []) or []
            if isinstance(gap, dict)
            and float(gap.get("gap_value") or 0.0) > 0.0
            and gap.get("closure_exams")
            and self._gap_has_exam_closure_route(gap)
        ]
        active_gaps.sort(
            key=lambda gap: float(gap.get("gap_value") or 0.0),
            reverse=True,
        )
        for fallback_gap_rank, gap in enumerate(active_gaps, start=1):
            gap_rank = int(gap.get("gap_value_rank") or fallback_gap_rank)
            candidate = str(gap.get("candidate") or "").strip()
            entity_id = str(gap.get("entity_id") or "").strip()
            target_candidates = [item for item in (candidate, entity_id) if item]
            gap_id = str(gap.get("gap_id") or "").strip()
            target = str(gap.get("target_evidence") or "").strip()
            gap_value = float(gap.get("gap_value") or 0.0)
            if gap_value < 0.58:
                continue
            for exam_rank, exam in enumerate(gap.get("closure_exams", []) or [], start=1):
                exam_text = str(exam or "").strip()
                if not exam_text:
                    continue
                if not self._exam_addresses_remaining_claims(gap, exam_text):
                    continue
                resolution = self.exam_resolver.resolve(
                    exam_text,
                    candidate=candidate or entity_id or None,
                )
                coverage = float(resolution.diagnostic_coverage or 0.0)
                closure_priority = max(0, 101 - exam_rank)
                task = {
                        "exam": exam_text,
                        "target_candidates": target_candidates,
                        "target_findings": [target] if target else [],
                        "target_gap": gap_id,
                        "target_gaps": [gap_id] if gap_id else [],
                        "target_claims": [target] if target else [],
                        "evidence_gap": dict(gap),
                        "exam_type": "deferred_gap_closure",
                        "exam_source": "deferred_gap_closure_exam",
                        "expected_effect": "close_highest_value_evidence_gap",
                        "expected_transition": dict(gap.get("expected_transition") or {}),
                        "source": ["active_evidence_gap"],
                        "priority_override": True,
                        "priority_bucket": "high_value_deferred_gap_closure",
                        "closure_rank": exam_rank,
                        "closure_priority": closure_priority,
                        "information_gain_hint": gap_value,
                        "source_gap_value": gap_value,
                        "gap_value_rank": gap_rank,
                        "gap_value_components": dict(gap.get("gap_value_components") or {}),
                        "candidate_score_at_decision": float(
                            gap.get("candidate_score_at_decision") or 0.0
                        ),
                        "score_gap_decoupled": True,
                        "requested_exam": exam_text,
                        "resolved_exam": resolution.resolved_exam or exam_text,
                        "resolution_type": resolution.resolution_type,
                        "diagnostic_coverage": coverage,
                        "gap_diagnostic_coverage": coverage,
                        "exam_gap_closure_value": round(
                            min(
                                1.0,
                                0.62 * gap_value
                                + 0.28 * coverage
                                + 0.10 * min(1.0, closure_priority / 100.0),
                            ),
                            4,
                        ),
                        "exam_resolution": resolution.to_dict(),
                        "override_reason": "highest-value evidence gap closure",
                    }
                tasks.append(self._with_evidence_question_contract(task, gap))
        return tasks

    @staticmethod
    def _with_evidence_question_contract(
        task: Dict[str, Any],
        gap: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Attach decision-question claims for gaps whose result needs interpretation."""
        result = dict(task or {})
        claim_requirements = [
            item
            for item in gap.get("claim_requirements", []) or []
            if isinstance(item, dict) and str(item.get("claim_id") or "").strip()
        ]
        closure_routes = [
            item
            for item in gap.get("closure_routes", []) or []
            if isinstance(item, dict)
        ]
        if claim_requirements:
            all_claims = [
                str(item.get("claim_id") or "").strip()
                for item in claim_requirements
                if str(item.get("claim_id") or "").strip()
            ]
            exam_text = str(result.get("exam") or result.get("requested_exam") or "")
            exam_routes = [
                route
                for route in closure_routes
                if str(route.get("route_type") or "") == "exam_result"
                and (
                    not route.get("exam")
                    or str(route.get("exam") or "") == exam_text
                    or str(route.get("exam") or "") in exam_text
                    or exam_text in str(route.get("exam") or "")
                )
            ]
            route_claims: List[str] = []
            expected_evidence: List[str] = []
            for route in exam_routes:
                route_claims.extend(str(item or "") for item in route.get("target_claims", []) or [])
                expected_evidence.extend(
                    str(item or "")
                    for item in route.get("expected_evidence_concepts", []) or []
                )
            result["target_claims"] = list(
                dict.fromkeys(
                    list(result.get("target_claims") or []) + all_claims
                )
            )
            result["route_target_claims"] = list(
                dict.fromkeys(item for item in route_claims if item)
            )
            result["expected_evidence_concepts"] = list(
                dict.fromkeys(item for item in expected_evidence if item)
            )
            result["target_findings"] = list(
                dict.fromkeys(
                    list(result.get("target_findings") or [])
                    + result["expected_evidence_concepts"]
                )
            )
            result["claim_requirements"] = claim_requirements
            result["closure_routes"] = closure_routes
            result["claim_closure_plan_version"] = str(
                gap.get("claim_closure_plan_version") or "claim_closure_plan_v1"
            )
        candidate_text = " ".join(
            str(item or "")
            for item in list(result.get("target_candidates") or [])
            + [
                gap.get("candidate"),
                gap.get("entity_id"),
                gap.get("target_evidence"),
                gap.get("gap_id"),
            ]
        )
        compact = "".join(candidate_text.lower().split())
        is_radiation_lung_gap = any(
            marker in compact
            for marker in (
                "d100058",
                "radiation",
                "radiotherapy",
                "post_radiotherapy",
                "放射",
                "放疗",
            )
        )
        if not is_radiation_lung_gap:
            return result
        claims = list(result.get("target_claims") or [])
        findings = list(result.get("target_findings") or [])
        if not claims:
            claims.extend(
                [
                    "pulmonary_morphology",
                    "radiation_field_lung_consistency",
                    "post_radiotherapy_time_window",
                ]
            )
        if not result.get("route_target_claims"):
            result["route_target_claims"] = [
                "pulmonary_morphology",
                "radiation_field_lung_consistency",
            ]
        for claim in (
            "radiation_field_lung_consistency",
            "ground_glass_opacity",
            "pulmonary_consolidation",
        ):
            if claim not in findings:
                findings.append(claim)
        for claim in ("pulmonary_morphology", "radiation_field_lung_consistency"):
            if claim not in claims:
                claims.append(claim)
        if "post_radiotherapy_time_window" not in claims:
            claims.append("post_radiotherapy_time_window")
        result["target_claims"] = claims
        result["target_findings"] = findings
        result.setdefault(
            "expected_evidence_concepts",
            [
                "ground_glass_opacity",
                "pulmonary_consolidation",
                "patchy_pulmonary_opacity",
                "lesion_within_prior_radiation_field",
            ],
        )
        result["target_question"] = (
            "Does the pulmonary imaging abnormality spatially match the prior "
            "radiation field?"
        )
        result["clinical_question"] = result["target_question"]
        result["positive_resolution_rules"] = [
            "lesion_within_prior_radiation_field",
            "opacity_within_radiation_field",
            "radiation_field_distribution",
        ]
        result["negative_resolution_rules"] = [
            "lesion_outside_prior_radiation_field",
            "diffuse_non_field_distribution",
        ]
        result["expected_arbitration_effect"] = {
            "radiation_field_lung_consistency_supported": "favor_D100058",
            "radiation_field_lung_consistency_contradicted": "demote_D100058",
            "unresolved": "remain_deferred",
        }
        result["exam_role"] = "target_claim_resolution"
        return result

    @staticmethod
    def _remaining_claim_ids(gap: Dict[str, Any]) -> List[str]:
        remaining = [
            str(item or "").strip()
            for item in gap.get("remaining_claims", []) or []
            if str(item or "").strip()
        ]
        if remaining:
            return remaining
        resolved = {
            str(item or "").strip()
            for item in list(gap.get("resolved_claims", []) or [])
            + list(gap.get("contradicted_claims", []) or [])
            + list(gap.get("conflicted_claims", []) or [])
            if str(item or "").strip()
        }
        return [
            str(item.get("claim_id") or "").strip()
            for item in gap.get("claim_requirements", []) or []
            if isinstance(item, dict)
            and str(item.get("claim_id") or "").strip()
            and str(item.get("claim_id") or "").strip() not in resolved
        ]

    @classmethod
    def _gap_has_exam_closure_route(cls, gap: Dict[str, Any]) -> bool:
        if not gap.get("claim_requirements"):
            return True
        return any(
            str(route.get("route_type") or "") == "exam_result"
            and set(
                str(item or "").strip()
                for item in route.get("target_claims", []) or []
                if str(item or "").strip()
            )
            & set(cls._remaining_claim_ids(gap))
            for route in gap.get("closure_routes", []) or []
            if isinstance(route, dict)
        )

    @classmethod
    def _exam_addresses_remaining_claims(cls, gap: Dict[str, Any], exam_text: str) -> bool:
        if not gap.get("claim_requirements"):
            return True
        remaining = set(cls._remaining_claim_ids(gap))
        if not remaining:
            return False
        for route in gap.get("closure_routes", []) or []:
            if not isinstance(route, dict) or str(route.get("route_type") or "") != "exam_result":
                continue
            route_exam = str(route.get("exam") or "")
            if route_exam and not (
                route_exam == exam_text or route_exam in exam_text or exam_text in route_exam
            ):
                continue
            route_claims = {
                str(item or "").strip()
                for item in route.get("target_claims", []) or []
                if str(item or "").strip()
            }
            if route_claims & remaining:
                return True
        return False

    def _exam_tasks_from_priority_overrides(
        self,
        payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for override in payload.get("exam_priority_overrides", []) or []:
            if not isinstance(override, dict):
                continue
            candidate = str(override.get("candidate") or "").strip()
            entity_id = str(override.get("entity_id") or "").strip()
            target_candidates = [item for item in (candidate, entity_id) if item]
            for gap in override.get("evidence_gaps", []) or []:
                if not isinstance(gap, dict):
                    continue
                gap_id = str(gap.get("gap_id") or "").strip()
                target = str(gap.get("target_evidence") or "").strip()
                for rank, exam in enumerate(gap.get("closure_exams", []) or [], start=1):
                    exam_text = str(exam or "").strip()
                    if not exam_text:
                        continue
                    resolution = self.exam_resolver.resolve(
                        exam_text,
                        candidate=candidate or entity_id or None,
                    )
                    task = {
                            "exam": exam_text,
                            "target_candidates": target_candidates,
                            "target_findings": [target] if target else [],
                            "target_gap": gap_id,
                            "target_gaps": [gap_id] if gap_id else [],
                            "target_claims": [target] if target else [],
                            "evidence_gap": dict(gap),
                            "exam_type": "deferred_gap_closure",
                            "exam_source": "deferred_gap_closure_exam",
                            "expected_effect": "close_high_value_deferred_evidence_gap",
                            "expected_transition": dict(gap.get("expected_transition") or {}),
                            "source": ["deferred_gap_closure"],
                            "priority_override": True,
                            "priority_bucket": "high_value_deferred_gap_closure",
                            "closure_rank": rank,
                            "closure_priority": max(0, 101 - rank),
                            "information_gain_hint": float(
                                gap.get("gap_value")
                                or override.get("max_gap_value")
                                or override.get("deferred_priority")
                                or 0.9
                            ),
                            "source_gap_value": float(
                                gap.get("gap_value")
                                or override.get("max_gap_value")
                                or 0.0
                            ),
                            "gap_value_rank": int(gap.get("gap_value_rank") or rank),
                            "gap_value_components": dict(
                                gap.get("gap_value_components") or {}
                            ),
                            "candidate_score_at_decision": float(
                                gap.get("candidate_score_at_decision") or 0.0
                            ),
                            "score_gap_decoupled": True,
                            "requested_exam": exam_text,
                            "resolved_exam": resolution.resolved_exam or exam_text,
                            "resolution_type": resolution.resolution_type,
                            "diagnostic_coverage": float(
                                resolution.diagnostic_coverage or 0.0
                            ),
                            "gap_diagnostic_coverage": float(
                                resolution.diagnostic_coverage or 0.0
                            ),
                            "exam_gap_closure_value": round(
                                min(
                                    1.0,
                                    0.62
                                    * float(
                                        gap.get("gap_value")
                                        or override.get("max_gap_value")
                                        or 0.0
                                    )
                                    + 0.28 * float(resolution.diagnostic_coverage or 0.0)
                                    + 0.10 * max(0.0, (101 - rank) / 100.0),
                                ),
                                4,
                            ),
                            "exam_resolution": resolution.to_dict(),
                            "override_reason": str(
                                override.get("override_reason")
                                or "high-value deferred evidence gap"
                            ),
                        }
                    tasks.append(self._with_evidence_question_contract(task, gap))
        return tasks

    def _prepare_differential_order_items(
        self,
        items: List[str],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]],
        existing_results: Dict[str, Any],
        max_items: Optional[int],
        task_by_exam: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        existing_valid, _ = self.knowledge.normalize_examinations(
            list((existing_results or {}).keys())
        )
        existing_set = set((existing_results or {}).keys()) | set(existing_valid)
        prepared: List[str] = []

        for item in items or []:
            text = str(item or "").strip()
            if not text:
                continue
            if text in task_by_exam:
                candidates = [text]
            else:
                normalized, _ = self.knowledge.normalize_examinations([text])
                candidates = normalized or [text]
            for exam in candidates:
                if exam and exam not in existing_set and exam not in prepared:
                    prepared.append(exam)

        prepared = self._filter_contextual_items(
            prepared,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
        )
        if max_items is None:
            return prepared
        limit = max(0, int(max_items))
        if len(prepared) <= limit:
            return prepared
        urgent = [
            item
            for item in prepared
            if bool(task_by_exam.get(item, {}).get("urgent_safety"))
        ]
        reserved = [
            item
            for item in prepared
            if str(task_by_exam.get(item, {}).get("exam_source") or "")
            == "deferred_gap_closure_exam"
            and bool(task_by_exam.get(item, {}).get("priority_override"))
        ]
        pairwise_reserved = [
            item
            for item in prepared
            if str(task_by_exam.get(item, {}).get("exam_source") or "")
            == "pairwise_discrimination_exam"
        ]
        selected: List[str] = []
        for item in urgent + pairwise_reserved + reserved + prepared:
            if item not in selected:
                selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _entity_exam_fallback_pool(
        self,
        candidate_diseases: Optional[List[Any]],
    ) -> List[str]:
        pool: List[str] = []
        if not hasattr(self.knowledge, "get_discriminating_exam_bundle"):
            return pool
        for candidate in candidate_diseases or []:
            name = self.knowledge._candidate_name(candidate)
            if not name and hasattr(candidate, "diagnosis"):
                name = str(getattr(candidate, "diagnosis") or "")
            if not name:
                name = str(candidate or "")
            bundle = self.knowledge.get_discriminating_exam_bundle(name)
            for item in bundle or []:
                text = str(item or "").strip()
                if text and text not in pool:
                    pool.append(text)
        return pool

    @staticmethod
    def _lossy_special_exam_normalization(
        raw_items: List[str],
        normalized: List[str],
        entity_fallback_normalized: List[str],
    ) -> bool:
        if not raw_items or not normalized or not entity_fallback_normalized:
            return False
        raw_text = " ".join(str(item or "") for item in raw_items).lower()
        special_markers = (
            "cta",
            "\u589e\u5f3a",
            "\u9020\u5f71",
            "\u52a8\u8109",
            "\u8840\u7ba1",
        )
        if not any(marker in raw_text for marker in special_markers):
            return False
        generic_normalized = {
            "CT\u626b\u63cf\uff08CT\uff09",
            "\u78c1\u5171\u632f\u6210\u50cf\uff08MRI\uff09",
        }
        normalized_set = set(normalized or [])
        if not normalized_set or not normalized_set <= generic_normalized:
            return False
        return any(item not in generic_normalized for item in entity_fallback_normalized)

    @staticmethod
    def _judge_payload(judge_decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not judge_decision:
            return {}
        if isinstance(judge_decision, dict):
            return judge_decision
        if hasattr(judge_decision, "to_dict"):
            return judge_decision.to_dict()
        return {
            key: getattr(judge_decision, key)
            for key in (
                "primary",
                "judge_primary",
                "primary_status",
                "needs_discriminating_exams",
                "provisional_primary",
                "locked_primary",
                "differential_candidates",
                "evidence_gap_targets",
                "discriminating_exams",
                "discriminating_exam_tasks",
                "active_evidence_gaps",
                "deferred_gap_closure_tasks",
                "exam_priority_overrides",
                "discriminating_findings",
            )
            if hasattr(judge_decision, key)
        }

    def _normalized_exam_task_map(
        self,
        exam_tasks: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for task in exam_tasks or []:
            if not isinstance(task, dict):
                continue
            requested_exam = str(task.get("exam") or "").strip()
            if not requested_exam:
                continue
            target_candidates = [
                str(item or "").strip()
                for item in task.get("target_candidates", []) or []
                if str(item or "").strip()
            ]
            resolution = self.exam_resolver.resolve(
                requested_exam,
                candidate=target_candidates[0] if target_candidates else None,
            )
            task_resolution = dict(task.get("exam_resolution") or {})
            if (
                task_resolution.get("resolved_exam")
                and str(task_resolution.get("resolution_type") or "")
                in _USABLE_EXAM_RESOLUTION_TYPES
            ):
                resolution_payload = task_resolution
            else:
                resolution_payload = resolution.to_dict()
            normalized: List[str] = []
            if resolution.resolved_exam and resolution.resolution_type in _USABLE_EXAM_RESOLUTION_TYPES:
                if self._preserve_specialty_exam_name(task, resolution.resolved_exam):
                    normalized = [resolution.resolved_exam]
                else:
                    normalized, _ = self.knowledge.normalize_examinations(
                        [resolution.resolved_exam]
                    )
                    if not normalized:
                        normalized = [resolution.resolved_exam]
            if not normalized:
                normalized, _ = self.knowledge.normalize_examinations([requested_exam])
            if not normalized:
                if str(task.get("exam_source") or "") != "pairwise_discrimination_exam":
                    continue
                normalized = [requested_exam]
                resolution_payload = {
                    **resolution_payload,
                    "requested_exam": requested_exam,
                    "resolved_exam": requested_exam,
                    "resolution_type": "unresolved_catalog_gap",
                    "diagnostic_coverage": float(
                        task.get("diagnostic_coverage") or 0.0
                    ),
                    "reason": "pairwise gap exam retained for audit despite catalog miss",
                }
            current = dict(task)
            for exam in normalized:
                current_for_exam = dict(current)
                current_for_exam["exam"] = exam
                current_for_exam["requested_exam"] = str(
                    resolution_payload.get("requested_exam") or requested_exam
                )
                current_for_exam["resolved_exam"] = exam
                current_for_exam["resolution_type"] = str(
                    resolution_payload.get("resolution_type")
                    or resolution.resolution_type
                    or ""
                )
                current_for_exam["diagnostic_coverage"] = float(
                    current_for_exam.get("diagnostic_coverage")
                    or resolution_payload.get("diagnostic_coverage")
                    or 0.0
                )
                current_for_exam["exam_resolution"] = {
                    **resolution_payload,
                    "resolved_exam": exam,
                    "resolution_type": current_for_exam["resolution_type"],
                    "diagnostic_coverage": current_for_exam["diagnostic_coverage"],
                }
                existing = result.get(exam)
                if existing is None:
                    result[exam] = current_for_exam
                else:
                    result[exam] = self._merge_normalized_exam_tasks(
                        existing,
                        current_for_exam,
                    )
        return result

    @staticmethod
    def _preserve_specialty_exam_name(
        task: Dict[str, Any],
        exam: str,
    ) -> bool:
        source = str((task or {}).get("exam_source") or "")
        if source == "deferred_gap_closure_exam":
            if not bool((task or {}).get("priority_override")):
                return False
        elif source not in {
            "evidence_claim_followup_exam",
            "pattern_anchor_workup_exam",
            "pairwise_discrimination_exam",
        }:
            return False
        text = str(exam or "").lower()
        return any(
            marker in text
            for marker in (
                "cta",
                "cect",
                "\u589e\u5f3a",
                "\u9020\u5f71",
                "\u52a8\u8109",
                "\u8840\u7ba1",
                "\u9aa8\u9ad3",
                "\u6d41\u5f0f",
                "\u514d\u75ab\u5206\u578b",
                "\u878d\u5408\u57fa\u56e0",
                "\u5206\u5b50\u68c0\u6d4b",
                "\u7ec6\u80de\u9057\u4f20",
                "\u67d3\u8272\u4f53\u6838\u578b",
                "bmab",
            )
        )

    def _merge_normalized_exam_tasks(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> Dict[str, Any]:
        keep_incoming = self._exam_task_priority_key(incoming) > self._exam_task_priority_key(existing)
        result = dict(incoming if keep_incoming else existing)
        for key in ("target_candidates", "target_findings", "target_gaps", "target_claims", "source"):
            values: List[Any] = []
            for item in list(existing.get(key) or []) + list(incoming.get(key) or []):
                if item not in values:
                    values.append(item)
            if values:
                result[key] = values
        result["information_gain_hint"] = max(
            float(existing.get("information_gain_hint") or 0.0),
            float(incoming.get("information_gain_hint") or 0.0),
        )
        result["closure_priority"] = max(
            int(existing.get("closure_priority") or 0),
            int(incoming.get("closure_priority") or 0),
        )
        result["closure_rank"] = min(
            int(existing.get("closure_rank") or 9999),
            int(incoming.get("closure_rank") or 9999),
        )
        result["source_gap_value"] = max(
            float(existing.get("source_gap_value") or 0.0),
            float(incoming.get("source_gap_value") or 0.0),
        )
        result["exam_gap_closure_value"] = max(
            float(existing.get("exam_gap_closure_value") or 0.0),
            float(incoming.get("exam_gap_closure_value") or 0.0),
        )
        result["score_gap_decoupled"] = bool(
            existing.get("score_gap_decoupled") or incoming.get("score_gap_decoupled")
        )
        if incoming.get("gap_value_components") and (
            float(incoming.get("source_gap_value") or 0.0)
            >= float(existing.get("source_gap_value") or 0.0)
        ):
            result["gap_value_components"] = dict(incoming.get("gap_value_components") or {})
            result["gap_value_rank"] = incoming.get("gap_value_rank")
            result["candidate_score_at_decision"] = incoming.get(
                "candidate_score_at_decision"
            )
        if (
            keep_incoming
            and incoming.get("exam_source") == "deferred_gap_closure_exam"
            and incoming.get("priority_override")
        ):
            for key in (
                "exam_source",
                "exam_type",
                "expected_effect",
                "priority_override",
                "priority_bucket",
                "override_reason",
                "target_gap",
                "evidence_gap",
                "expected_transition",
                "requested_exam",
                "resolved_exam",
                "resolution_type",
                "diagnostic_coverage",
                "gap_diagnostic_coverage",
                "source_gap_value",
                "gap_value_rank",
                "gap_value_components",
                "exam_gap_closure_value",
                "candidate_score_at_decision",
                "score_gap_decoupled",
                "exam_resolution",
            ):
                if incoming.get(key) not in (None, "", [], {}):
                    result[key] = (
                        dict(incoming.get(key))
                        if isinstance(incoming.get(key), dict)
                        else incoming.get(key)
                    )
        if keep_incoming and incoming.get("exam_source") == "pairwise_discrimination_exam":
            for key in (
                "exam_source",
                "exam_type",
                "expected_effect",
                "priority_bucket",
                "source_gap_id",
                "target_pair",
                "target_question",
                "target_claim",
                "exam_role",
                "expected_arbitration_effect",
            ):
                if incoming.get(key) not in (None, "", [], {}):
                    result[key] = (
                        dict(incoming.get(key))
                        if isinstance(incoming.get(key), dict)
                        else incoming.get(key)
                    )
        return result

    def _strict_authorized_exam_plan(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        primary = self._strict_primary_diagnosis(
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
        )
        if not primary:
            return {}
        strong_items = self._strong_verification_items_for_disease(primary)
        if not strong_items:
            return {}
        if self._has_advanced_cardiac_signal(collected_info, candidate_diseases):
            proposed_valid, _ = self.knowledge.normalize_examinations(proposed_items or [])
            for item in proposed_valid:
                if item in _ADVANCED_CARDIAC_EXAMS and item not in strong_items:
                    strong_items.append(item)
        for companion in self._explicit_candidate_names(candidate_diseases):
            if companion == primary:
                continue
            if not self._is_strict_exam_companion(primary, companion):
                continue
            for item in self._strong_verification_items_for_disease(companion):
                if item and item not in strong_items:
                    strong_items.append(item)
        required_items = self.knowledge.get_required_exams(
            candidate_diseases=[primary],
            symptoms=(collected_info or {}).get("symptoms", []),
            include_optional=False,
        )
        items = list(dict.fromkeys(strong_items + required_items))
        max_items = min(self.max_new_items, max(1, len(strong_items)))
        return {
            "primary_diagnosis": primary,
            "items": items,
            "strong_items": strong_items,
            "max_items": max_items,
        }

    @staticmethod
    def _exam_task_priority_bucket(task: Dict[str, Any]) -> str:
        source = str((task or {}).get("exam_source") or "")
        if bool((task or {}).get("urgent_safety")):
            return "urgent_safety"
        if source == "deferred_gap_closure_exam" and bool(
            (task or {}).get("priority_override")
        ):
            return "high_value_deferred_gap_closure"
        if source in {
            "evidence_claim_followup_exam",
            "pattern_anchor_workup_exam",
        }:
            return "targeted_evidence_followup"
        if source == "pairwise_discrimination_exam":
            return "high_value_pairwise_gap_closure"
        if source == "conflict_adjudication_exam":
            return "conflict_adjudication"
        exam_type = str((task or {}).get("exam_type") or "")
        if exam_type == "generic_inflammation":
            return "generic_lab_context"
        return "general_discrimination"

    @classmethod
    def _exam_task_priority_key(cls, task: Dict[str, Any]) -> tuple:
        bucket_name = cls._exam_task_priority_bucket(task)
        bucket = {
            "urgent_safety": 7,
            "high_value_pairwise_gap_closure": 6,
            "high_value_deferred_gap_closure": 5,
            "targeted_evidence_followup": 4,
            "conflict_adjudication": 3,
            "general_discrimination": 2,
            "generic_lab_context": 1,
        }.get(bucket_name, 0)
        closure_priority = int((task or {}).get("closure_priority") or 0)
        closure_rank = int((task or {}).get("closure_rank") or 9999)
        source_gap_value = float((task or {}).get("source_gap_value") or 0.0)
        exam_gap_closure_value = float(
            (task or {}).get("exam_gap_closure_value") or 0.0
        )
        gap_coverage = float((task or {}).get("gap_diagnostic_coverage") or 0.0)
        diagnostic_coverage = float((task or {}).get("diagnostic_coverage") or 0.0)
        specialty_followup_priority = (
            cls._specialty_followup_priority(task)
            if bucket_name == "targeted_evidence_followup"
            else 0
        )
        return (
            bucket,
            source_gap_value,
            exam_gap_closure_value,
            closure_priority,
            gap_coverage,
            diagnostic_coverage,
            specialty_followup_priority,
            -closure_rank,
            float((task or {}).get("information_gain_hint") or 0.0),
        )

    @staticmethod
    def _specialty_followup_priority(task: Dict[str, Any]) -> int:
        exam = str((task or {}).get("exam") or "")
        requested = str((task or {}).get("requested_exam") or "")
        resolved = str((task or {}).get("resolved_exam") or "")
        targets = " ".join(str(item or "") for item in (task or {}).get("target_candidates", []) or [])
        text = f"{exam} {requested} {resolved} {targets}".lower()
        compact = "".join(text.split())
        if (
            "\u9aa8\u9ad3\u7a7f\u523a" in compact
            or "\u9aa8\u9ad3\u6d3b\u68c0" in compact
            or "bmab" in compact
        ):
            return 100
        if "\u9aa8\u9ad3\u6d41\u5f0f" in compact or "\u6d41\u5f0f\u7ec6\u80de" in compact or "\u514d\u75ab\u5206\u578b" in compact:
            return 96
        if "\u767d\u8840\u75c5\u878d\u5408\u57fa\u56e0" in compact or "\u878d\u5408\u57fa\u56e0" in compact:
            return 92
        if "\u7ec6\u80de\u9057\u4f20" in compact or "\u67d3\u8272\u4f53\u6838\u578b" in compact:
            return 88
        if "cta" in compact or "\u53f3\u5fc3\u58f0\u5b66\u9020\u5f71" in compact:
            return 84
        if "\u589e\u5f3a" in compact or "cect" in compact:
            return 80
        if "\u5916\u5468\u8840\u6d82\u7247" in compact:
            return 72
        if "\u5168\u8840\u7ec6\u80de\u8ba1\u6570" in compact or "\u8840\u5e38\u89c4" in compact or "cbc" in compact:
            return 38
        if "\u8840\u6c89" in compact or "esr" in compact:
            return 12
        return 0

    def _strict_primary_diagnosis(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> str:
        raw_by_name = self._candidate_items_by_name(candidate_diseases)
        explicit_names = list(raw_by_name)
        for name in explicit_names:
            if self._strong_verification_items_for_disease(name):
                if not self._candidate_supported_by_context(
                    name,
                    collected_info,
                    len(explicit_names),
                    raw_by_name.get(name),
                ):
                    continue
                return name
        profiles = self.knowledge.recall_disease_profiles(
            symptoms=(collected_info or {}).get("symptoms", []),
            candidate_diseases=explicit_names,
            top_k=3,
        )
        for profile in profiles:
            name = str(profile.get("name") or "")
            try:
                hit_score = float(profile.get("hit_score", 0) or 0)
            except (TypeError, ValueError):
                hit_score = 0.0
            if hit_score < 2:
                continue
            if self._strong_verification_items_for_disease(name):
                return name
        return ""

    def _candidate_supported_by_context(
        self,
        disease: str,
        collected_info: Dict[str, Any],
        candidate_count: int,
        candidate_item: Optional[Any] = None,
    ) -> bool:
        if self._candidate_has_objective_support(candidate_item):
            return True
        profile = self.knowledge.get_disease_profile(disease) or {}
        text = self._case_text_without_candidates(collected_info)
        if not text.strip():
            return candidate_count <= 1
        if disease and disease in text:
            return True
        for alias in profile.get("aliases", []) or []:
            alias_text = str(alias).strip()
            if alias_text and alias_text in text:
                return True
        hit_count = 0
        for term in self._profile_context_terms(profile):
            if not term:
                continue
            if term in text or (len(term) >= 3 and any(term in part or part in term for part in text.split())):
                hit_count += 1
            if hit_count >= 2:
                return True
        return candidate_count <= 1 and hit_count >= 1

    @staticmethod
    def _candidate_has_objective_support(candidate_item: Optional[Any]) -> bool:
        if not candidate_item:
            return False
        if not isinstance(candidate_item, dict):
            required = getattr(candidate_item, "required_met", False)
            matched = getattr(candidate_item, "matched_evidence", []) or []
            score = float(getattr(candidate_item, "coverage_score", 0.0) or 0.0)
        else:
            required = bool(candidate_item.get("required_met", False))
            matched = candidate_item.get("matched_evidence", []) or []
            try:
                score = float(
                    candidate_item.get("coverage_score")
                    or candidate_item.get("explanatory_coverage")
                    or 0.0
                )
            except (TypeError, ValueError):
                score = 0.0
        if required or score >= 0.45:
            return True
        for item in matched:
            finding = str(item or "")
            if finding and not finding.startswith("symptom:") and not finding.startswith("field:"):
                return True
        return False

    @classmethod
    def _case_text_without_candidates(cls, collected_info: Dict[str, Any]) -> str:
        parts: List[str] = []
        for key in (
            "chief_complaint",
            "present_illness",
            "past_history",
            "personal_history",
            "physical_signs",
            "raw_responses",
            "question_focus",
        ):
            value = (collected_info or {}).get(key)
            if value:
                parts.append(str(value))
        for symptom in (collected_info or {}).get("symptoms", []) or []:
            parts.append(str(symptom))
        return " ".join(parts)

    @classmethod
    def _profile_context_terms(cls, profile: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for field_name in (
            "common_symptoms",
            "red_flags",
            "hallmark_findings",
            "discriminating_features",
        ):
            for item in profile.get(field_name, []) or []:
                for term in cls._extract_profile_terms(item):
                    if term and term not in terms:
                        terms.append(term)
        return terms

    @classmethod
    def _extract_profile_terms(cls, item: Any) -> List[str]:
        if isinstance(item, str):
            return [item.strip()] if item.strip() else []
        if isinstance(item, dict):
            terms: List[str] = []
            for key in ("terms", "keywords", "term"):
                value = item.get(key)
                if isinstance(value, list):
                    terms.extend(str(part).strip() for part in value if str(part).strip())
                elif value:
                    terms.append(str(value).strip())
            return [term for term in terms if term]
        if isinstance(item, list):
            terms: List[str] = []
            for child in item:
                terms.extend(cls._extract_profile_terms(child))
            return terms
        return []

    def _candidate_items_by_name(
        self,
        candidate_diseases: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for item in candidate_diseases or []:
            name = self.knowledge._candidate_name(item)
            if not name and hasattr(item, "diagnosis"):
                name = str(getattr(item, "diagnosis") or "")
            if not name:
                continue
            standard = self.knowledge.normalize_diagnosis(name) or name
            result.setdefault(standard, item)
        return result

    def _explicit_candidate_names(
        self,
        candidate_diseases: Optional[List[Any]] = None,
    ) -> List[str]:
        names: List[str] = []
        for item in candidate_diseases or []:
            name = self.knowledge._candidate_name(item)
            if not name and hasattr(item, "diagnosis"):
                name = str(getattr(item, "diagnosis") or "")
            if not name:
                continue
            standard = self.knowledge.normalize_diagnosis(name) or name
            if standard not in names:
                names.append(standard)
        return names

    def _strong_verification_items_for_disease(self, disease: str) -> List[str]:
        standard = self.knowledge.normalize_diagnosis(disease) or disease
        profile = self.knowledge.get_disease_profile(standard) or {}
        raw_items = (
            list(profile.get("strong_verification_exams") or [])
            or list(_STRONG_VERIFICATION_EXAMS.get(standard, []) or [])
        )
        normalized, _ = self.knowledge.normalize_examinations(raw_items)
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _is_strict_exam_companion(primary: str, companion: str) -> bool:
        return frozenset({primary, companion}) in _STRICT_COMPANION_EXAM_PATHS

    def _blocked_exam_items(
        self,
        items: List[str],
        allowed: List[str],
        existing_results: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        existing_valid, _ = self.knowledge.normalize_examinations(
            list((existing_results or {}).keys())
        )
        existing_set = set((existing_results or {}).keys()) | set(existing_valid)
        allowed_set = set(allowed or [])
        blocked: List[str] = []
        for item in items or []:
            if not item or item in existing_set or item in allowed_set:
                continue
            if item not in blocked:
                blocked.append(item)
        return blocked

    def prepare_order_items(
        self,
        items: List[str],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        existing_results: Optional[Dict[str, Any]] = None,
        max_items: Optional[int] = None,
        add_strong_verification: bool = True,
    ) -> List[str]:
        """Normalize, strengthen and gate examination names before submission."""
        existing_results = existing_results or {}
        normalized, _ = self.knowledge.normalize_examinations(items or [])
        merged: List[str] = []
        if add_strong_verification:
            for item in self._strong_verification_items(
                collected_info=collected_info,
                candidate_diseases=candidate_diseases,
                proposed_items=items,
            ):
                if item and item not in merged:
                    merged.append(item)
        for item in normalized:
            if item and item not in merged:
                merged.append(item)

        if add_strong_verification:
            strict_plan = self._strict_authorized_exam_plan(
                collected_info=collected_info,
                candidate_diseases=candidate_diseases,
                proposed_items=items,
            )
            if strict_plan:
                allowed = set(strict_plan["items"])
                merged = [item for item in merged if item in allowed]
                if not merged:
                    merged = list(strict_plan["items"])
                if max_items is None:
                    max_items = strict_plan["max_items"]

        merged = self._filter_contextual_items(
            merged,
            collected_info=collected_info,
            candidate_diseases=candidate_diseases,
        )
        existing_valid, _ = self.knowledge.normalize_examinations(list(existing_results.keys()))
        existing_set = set(existing_results.keys()) | set(existing_valid)
        result = [item for item in merged if item not in existing_set]
        if max_items is not None:
            return result[: max(0, int(max_items))]
        return result

    def _strong_verification_items(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
    ) -> List[str]:
        diseases: List[str] = []
        raw_by_name = self._candidate_items_by_name(candidate_diseases)
        explicit_names = list(raw_by_name)
        for standard in explicit_names:
            if not self._candidate_supported_by_context(
                standard,
                collected_info,
                len(explicit_names),
                raw_by_name.get(standard),
            ):
                continue
            if standard not in diseases:
                diseases.append(standard)

        symptoms = collected_info.get("symptoms", []) if collected_info else []
        for profile in self.knowledge.recall_disease_profiles(
            symptoms=symptoms,
            candidate_diseases=diseases,
            top_k=5,
        ):
            name = str(profile.get("name") or "")
            if name and name not in diseases:
                diseases.append(name)

        if self._needs_low_magnesium_verification(
            collected_info, candidate_diseases, proposed_items
        ):
            diseases.append("低镁血症")
        if self._needs_metabolic_bone_workup(collected_info, candidate_diseases):
            diseases.append("维生素D缺乏性佝偻病")
        if self._needs_pulmonary_renal_workup(collected_info, candidate_diseases):
            diseases.append("显微镜下多血管炎")
        if self._needs_aspiration_pneumonia_workup(collected_info, candidate_diseases):
            diseases.extend(["肺不张", "支气管肺炎"])

        items: List[str] = []
        for disease in list(dict.fromkeys(diseases)):
            profile = self.knowledge.get_disease_profile(disease) or {}
            raw_items = (
                list(profile.get("strong_verification_exams") or [])
                or list(_STRONG_VERIFICATION_EXAMS.get(disease, []))
            )
            normalized, _ = self.knowledge.normalize_examinations(raw_items)
            for exam in normalized:
                if exam and exam not in items:
                    items.append(exam)
        return items

    def _rank_by_information_gain(
        self,
        candidate_diseases: List[Any],
        symptoms: List[Any],
        proposed_items: List[str],
        exam_tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[str], Dict[str, float]]:
        """Rank exams by relevance and ability to separate the top candidates."""
        candidates: List[str] = []
        for item in candidate_diseases:
            if isinstance(item, dict):
                value = item.get("disease") or item.get("diagnosis") or item.get("name")
            else:
                value = item
            if not value:
                continue
            normalized = self.knowledge.normalize_diagnosis(str(value)) or str(value)
            if normalized not in candidates:
                candidates.append(normalized)
            if len(candidates) >= 6:
                break

        exam_support: Dict[str, set] = {}
        relevance: Dict[str, float] = {}
        task_type: Dict[str, str] = {}
        task_findings: Dict[str, set] = {}
        task_priority: Dict[str, tuple] = {}
        task_by_exam = self._normalized_exam_task_map(exam_tasks or [])
        for exam, task in task_by_exam.items():
            targets = [
                self.knowledge.normalize_diagnosis(str(item)) or str(item)
                for item in task.get("target_candidates", []) or []
                if str(item).strip()
            ]
            targets = [item for item in targets if item in set(candidates)] or list(candidates)
            exam_support.setdefault(exam, set()).update(targets)
            task_type[exam] = str(task.get("exam_type") or self._exam_type_for_name(exam))
            task_findings.setdefault(exam, set()).update(
                str(item).strip()
                for item in task.get("target_findings", []) or []
                if str(item).strip()
            )
            relevance[exam] = relevance.get(exam, 0.0) + float(
                task.get("information_gain_hint", 0.8) or 0.8
            )
            if task.get("priority_override"):
                relevance[exam] = relevance.get(exam, 0.0) + 1.25
            if str(task.get("exam_source") or "") == "deferred_gap_closure_exam":
                relevance[exam] = relevance.get(exam, 0.0) + 1.0
            if str(task.get("exam_source") or "") == "pairwise_discrimination_exam":
                relevance[exam] = relevance.get(exam, 0.0) + 1.15
            task_priority[exam] = max(
                task_priority.get(exam, (0, 0, 0.0, 0.0, -9999, 0.0)),
                self._exam_task_priority_key(task),
            )
        for rank, disease in enumerate(candidates):
            profile = self.knowledge.get_disease_profile(disease) or {}
            raw_items: List[str] = []
            for field_name in (
                "discriminating_exams",
                "strong_verification_exams",
                "required_exams",
            ):
                raw_items.extend(profile.get(field_name) or [])
            normalized_items, _ = self.knowledge.normalize_examinations(raw_items)
            for exam in normalized_items:
                exam_support.setdefault(exam, set()).add(disease)
                relevance[exam] = relevance.get(exam, 0.0) + 1.0 / (rank + 1)

        if not exam_support:
            fallback = self.knowledge.get_required_exams(
                candidate_diseases=candidates,
                symptoms=symptoms,
                include_optional=False,
            )
            for exam in fallback:
                exam_support.setdefault(exam, set()).add("symptom_recall")
                relevance[exam] = max(relevance.get(exam, 0.0), 0.7)

        for exam in proposed_items:
            exam_support.setdefault(exam, set())
            relevance[exam] = relevance.get(exam, 0.0) + 0.05

        candidate_count = max(1, len(candidates))
        max_relevance = max(relevance.values(), default=1.0)
        scores: Dict[str, float] = {}
        for exam, supported in exam_support.items():
            coverage = len(supported)
            if candidate_count <= 1:
                discrimination = 0.7 if coverage else 0.0
            elif 0 < coverage < candidate_count:
                discrimination = 1.0
            elif coverage == candidate_count:
                discrimination = 0.90
            else:
                discrimination = 0.0
            multi_candidate = 1.0 if coverage >= 2 else 0.0
            relevance_score = relevance.get(exam, 0.0) / max_relevance
            exam_type = task_type.get(exam) or self._exam_type_for_name(exam)
            type_score = self._exam_type_score(exam_type)
            finding_score = min(1.0, len(task_findings.get(exam, set())) / 4.0)
            score = (
                0.32 * discrimination
                + 0.24 * multi_candidate
                + 0.22 * type_score
                + 0.12 * finding_score
                + 0.10 * relevance_score
            )
            if task_by_exam.get(exam, {}).get("priority_override"):
                score += 0.60
            if str(task_by_exam.get(exam, {}).get("exam_source") or "") == "deferred_gap_closure_exam":
                score += 0.75
            if str(task_by_exam.get(exam, {}).get("exam_source") or "") == "pairwise_discrimination_exam":
                score += 0.70
            if exam_type == "generic_inflammation" and coverage < 2:
                score *= 0.45
            scores[exam] = round(score, 4)

        ranked = sorted(
            scores,
            key=lambda item: (
                task_priority.get(item, (0, 0, 0.0, 0.0, -9999, 0.0)),
                scores[item],
                self._exam_type_score(task_type.get(item) or self._exam_type_for_name(item)),
                len(exam_support.get(item, set())),
                relevance.get(item, 0.0),
            ),
            reverse=True,
        )
        return ranked, scores

    @staticmethod
    def _generic_inflammation_exam(exam: str) -> bool:
        text = str(exam or "")
        return any(marker in text for marker in _GENERIC_INFLAMMATION_EXAM_MARKERS)

    @staticmethod
    def _special_discriminator_exam(exam: str) -> bool:
        text = str(exam or "")
        return any(marker in text for marker in _SPECIAL_DISCRIMINATOR_EXAM_MARKERS)

    def _exam_type_for_name(self, exam: str) -> str:
        if self._generic_inflammation_exam(exam):
            return "generic_inflammation"
        if self._special_discriminator_exam(exam):
            return "special_discriminator"
        return "confirmatory"

    @staticmethod
    def _exam_type_score(exam_type: str) -> float:
        return {
            "deferred_gap_closure": 1.25,
            "conflict_adjudication": 1.08,
            "evidence_claim_verification": 1.05,
            "pattern_anchor_workup": 1.02,
            "special_discriminator": 1.0,
            "shared_discriminator": 0.78,
            "confirmatory": 0.50,
            "generic_inflammation": 0.12,
        }.get(str(exam_type or ""), 0.35)

    def _scenario_items(
        self,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> List[str]:
        items: List[str] = []
        if self._needs_electrolyte_crisis_workup(collected_info, candidate_diseases):
            items.extend(_ELECTROLYTE_CRISIS_EXAMS)
        if self._needs_pulmonary_renal_workup(collected_info, candidate_diseases):
            items.extend(_PULMONARY_RENAL_EXAMS)
        if self._needs_metabolic_bone_workup(collected_info, candidate_diseases):
            items.extend(_METABOLIC_BONE_EXAMS)
        if self._needs_aspiration_pneumonia_workup(collected_info, candidate_diseases):
            items.extend(_ASPIRATION_PNEUMONIA_EXAMS)
        if self._needs_pediatric_cardiac_screen(collected_info):
            items.extend(_PEDIATRIC_CARDIAC_SCREEN_EXAMS)
        if self._needs_right_heart_valve_workup(collected_info, candidate_diseases):
            items.extend(_RIGHT_HEART_VALVE_EXAMS)
        if self._needs_urinary_syndrome_workup(collected_info, candidate_diseases):
            items.extend(_URINARY_SYNDROME_EXAMS)
        if self._needs_rib_trauma_workup(collected_info, candidate_diseases):
            items.extend(_RIB_TRAUMA_EXAMS)
        if self._needs_retinoblastoma_workup(collected_info, candidate_diseases):
            items.extend(_RETINOBLASTOMA_EXAMS)
        if self._needs_hib_respiratory_workup(collected_info, candidate_diseases):
            items.extend(_HIB_RESPIRATORY_EXAMS)
        if self._needs_immune_pneumonia_workup(collected_info, candidate_diseases):
            items.extend(_IMMUNE_PNEUMONIA_EXAMS)
        if self._needs_congenital_shunt_workup(collected_info, candidate_diseases):
            items.extend(_CONGENITAL_SHUNT_EXAMS)
        normalized, _ = self.knowledge.normalize_examinations(items)
        return normalized

    def _priority_proposed_items(
        self,
        proposed_items: List[str],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> List[str]:
        if not self._has_advanced_cardiac_signal(collected_info, candidate_diseases):
            return []
        return [item for item in proposed_items if item in _ADVANCED_CARDIAC_EXAMS]

    def _filter_contextual_items(
        self,
        items: List[str],
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> List[str]:
        advanced_cardiac = self._has_advanced_cardiac_signal(
            collected_info, candidate_diseases
        )
        if advanced_cardiac:
            return items
        return [item for item in items if item not in _ADVANCED_CARDIAC_EXAMS]

    @staticmethod
    def _needs_pediatric_cardiac_screen(collected_info: Dict[str, Any]) -> bool:
        if not collected_info:
            return False

        parts: List[str] = []
        for key in (
            "chief_complaint",
            "present_illness",
            "past_history",
            "personal_history",
            "physical_signs",
            "raw_responses",
            "question_focus",
        ):
            value = collected_info.get(key)
            if value:
                parts.append(str(value))
        for symptom in collected_info.get("symptoms", []) or []:
            parts.append(str(symptom))
        text = " ".join(parts)

        pediatric = any(token in text for token in ("宝宝", "婴儿", "患儿", "吃奶", "喂奶"))
        respiratory = any(token in text for token in ("呼吸急促", "呼吸明显变快", "气促", "呼吸困难", "喘"))
        perfusion = any(token in text for token in ("发绀", "青紫", "口周发绀", "嘴巴周围发青"))
        feeding = any(token in text for token in ("吃奶减少", "喂养困难", "喂奶", "吃奶"))
        sweating = any(token in text for token in ("多汗", "出汗"))

        return pediatric and respiratory and (perfusion or feeding or sweating)

    @staticmethod
    def _case_text(
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> str:
        parts: List[str] = []
        for key in (
            "chief_complaint",
            "present_illness",
            "past_history",
            "personal_history",
            "physical_signs",
        ):
            value = (collected_info or {}).get(key)
            if value:
                parts.append(str(value))
        for symptom in (collected_info or {}).get("symptoms", []) or []:
            parts.append(str(symptom))
        for candidate in candidate_diseases or []:
            if isinstance(candidate, dict):
                parts.extend(str(v) for v in candidate.values() if v)
            else:
                parts.append(str(candidate))
        return " ".join(parts)

    @classmethod
    def _needs_electrolyte_crisis_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        gi_loss = any(token in text for token in ("腹泻", "拉肚子", "水样便", "呕吐"))
        neuromuscular = any(token in text for token in ("抽筋", "手足", "痉挛", "乏力", "无力"))
        cardiac = any(token in text for token in ("心悸", "心慌", "QT", "心律失常"))
        neuro = any(token in text for token in ("意识模糊", "头晕", "黑朦"))
        candidate = any(token in text for token in ("低镁血症", "低钾", "低钙", "电解质"))
        return candidate or (gi_loss and (neuromuscular or cardiac or neuro))

    @classmethod
    def _needs_pulmonary_renal_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        pulmonary = any(token in text for token in ("咯血", "咳血", "血痰", "肺泡出血"))
        renal = any(
            token in text
            for token in ("尿色", "深色尿", "血尿", "蛋白尿", "脚踝水肿", "水肿", "肾功能", "少尿")
        )
        candidate = any(token in text for token in ("显微镜下多血管炎", "血管炎", "ANCA", "肾肺"))
        return candidate or (pulmonary and renal)

    @classmethod
    def _needs_metabolic_bone_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        bone = any(token in text for token in ("腿痛", "骨痛", "跛行", "O型腿", "X型腿", "步态"))
        function = any(token in text for token in ("运动耐力下降", "运动耐量下降", "活动后", "下肢功能障碍", "上下楼"))
        pediatric = any(token in text for token in ("儿童", "患儿", "孩子", "青少年", "岁"))
        candidate = any(token in text for token in ("佝偻病", "维生素D", "代谢性骨病", "骨软化"))
        return candidate or (bone and (function or pediatric))

    @classmethod
    def _needs_low_magnesium_verification(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
        proposed_items: Optional[List[str]] = None,
    ) -> bool:
        text = " ".join(
            [
                cls._case_text(collected_info, candidate_diseases),
                " ".join(str(item) for item in proposed_items or []),
            ]
        )
        gi_loss = any(token in text for token in ("腹泻", "拉肚子", "水样便", "呕吐"))
        neuromuscular = any(token in text for token in ("抽筋", "手足", "痉挛", "乏力", "无力"))
        cardiac = any(token in text for token in ("心悸", "心慌", "QT", "QTc", "心律失常"))
        explicit = any(token in text for token in ("低镁血症", "低镁", "血镁", "镁负荷", "hypomagnesemia"))
        magnesium_hint = "镁" in text and (gi_loss or neuromuscular or cardiac)
        return explicit or magnesium_hint or (gi_loss and neuromuscular and cardiac)

    @classmethod
    def _needs_aspiration_pneumonia_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        aspiration = any(token in text for token in ("呛咳", "误吸", "吞咽困难", "进食时"))
        respiratory = any(token in text for token in ("咳嗽", "发热", "呼吸困难", "气促", "低氧", "痰"))
        candidate = any(token in text for token in ("肺不张", "支气管肺炎", "吸入性肺炎", "黏液栓"))
        return candidate or (aspiration and respiratory)

    @classmethod
    def _has_strong_noncardiac_path(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        return (
            cls._needs_electrolyte_crisis_workup(collected_info, candidate_diseases)
            or cls._needs_pulmonary_renal_workup(collected_info, candidate_diseases)
            or cls._needs_metabolic_bone_workup(collected_info, candidate_diseases)
            or cls._needs_aspiration_pneumonia_workup(collected_info, candidate_diseases)
        )

    @classmethod
    def _has_advanced_cardiac_signal(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        return any(
            token in text
            for token in (
                "二尖瓣",
                "三尖瓣",
                "肺动脉瓣",
                "瓣膜",
                "心脏杂音",
                "杂音",
                "先心",
                "房间隔缺损",
                "ASD",
                "VSD",
                "右心",
                "肺动脉高压",
                "心导管",
                "重度反流",
            )
        )

    @classmethod
    def _needs_right_heart_valve_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        direct = any(
            token in text
            for token in ("三尖瓣", "肺动脉瓣", "右心", "肺动脉高压", "肺动脉瓣狭窄")
        )
        syndrome = any(token in text for token in ("心力衰竭", "心衰", "气短", "下肢水肿", "腿肿"))
        cardiopulmonary = any(token in text for token in ("心悸", "心慌", "呼吸困难", "肺动脉高压"))
        return direct or (
            syndrome
            and cardiopulmonary
            and not cls._needs_congenital_shunt_workup(collected_info, candidate_diseases)
        )

    @classmethod
    def _needs_urinary_syndrome_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        urinary_symptoms = any(token in text for token in ("尿急", "尿频", "尿痛", "排尿烧灼"))
        urinary_candidates = any(token in text for token in ("尿道综合征", "泌尿系感染", "膀胱过度活动", "逼尿肌"))
        return urinary_symptoms or urinary_candidates

    @classmethod
    def _needs_rib_trauma_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        chest_wall = any(token in text for token in ("胸壁", "肋骨", "肋部", "左侧胸"))
        trauma = any(token in text for token in ("外伤", "车门", "夹伤", "撞伤", "摔伤"))
        pleuritic = any(token in text for token in ("深呼吸", "咳嗽", "转身", "呼吸受限"))
        return (chest_wall and trauma) or (chest_wall and pleuritic)

    @classmethod
    def _needs_retinoblastoma_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        leukocoria = any(token in text for token in ("瞳孔发白", "白瞳", "猫眼反光", "白色反光"))
        strabismus = any(token in text for token in ("内斜视", "斜视"))
        tumor_candidate = any(token in text for token in ("视网膜母细胞瘤", "眼内肿瘤"))
        return leukocoria or (strabismus and tumor_candidate)

    @classmethod
    def _needs_hib_respiratory_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        prolonged = any(token in text for token in ("6周", "六周", "迁延", "反复发热", "反复咳嗽"))
        respiratory = any(token in text for token in ("咳嗽", "发热", "脓痰", "浓痰", "喘息", "呼吸困难"))
        ear = any(token in text for token in ("耳痛", "中耳炎", "耳流脓"))
        candidate = any(token in text for token in ("流感嗜血杆菌", "Hib"))
        return candidate or (prolonged and respiratory and ear)

    @classmethod
    def _needs_immune_pneumonia_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        pneumonia = any(token in text for token in ("肺炎", "发热", "咳嗽", "呼吸困难"))
        immune = any(token in text for token in ("类风湿", "糖皮质激素", "免疫抑制", "脾肿大", "粒细胞减少", "费尔蒂"))
        severe = any(token in text for token in ("低氧", "SpO", "呼吸困难", "重症"))
        return pneumonia and (immune or severe and "搬入新公寓" in text)

    @classmethod
    def _needs_congenital_shunt_workup(
        cls,
        collected_info: Dict[str, Any],
        candidate_diseases: Optional[List[Any]] = None,
    ) -> bool:
        text = cls._case_text(collected_info, candidate_diseases)
        direct = any(token in text for token in ("房间隔缺损", "ASD", "先心", "左向右分流", "继发孔"))
        recurrent_lung = any(token in text for token in ("反复肺炎", "两次肺炎", "多次肺炎", "反复呼吸道感染"))
        exertional = any(token in text for token in ("活动后", "玩耍", "运动后", "跑几步", "乏力", "心悸", "喘息"))
        pediatric = any(token in text for token in ("孩子", "患儿", "儿童", "宝宝", "家长"))
        return direct or (recurrent_lung and exertional and pediatric)
