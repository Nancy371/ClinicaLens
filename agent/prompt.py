"""
Prompt 模板和输出格式约束模块。

参赛者需要修改此模块来调整模型输入和输出格式。
包含：
- 系统角色 prompt
- 初始问诊 prompt（含历史经验注入）
- 追问 prompt（含信息缺口分析）
- 检查申请 prompt（含鉴别诊断引导）
- 诊断 prompt（含经验参考）
- 反思 prompt（含结构化分析框架）
"""

import json
from typing import Any, Dict, List, Optional


class DoctorPrompt:
    """医生 Agent 的 Prompt 模板管理类。

    参赛者可以修改各类 prompt 模板来优化模型输入，
    也可以调整输出格式约束来引导模型生成更规范的结果。
    """

    # ============ 系统角色 Prompt ============

    SYSTEM_ROLE = """你是一位经验丰富的临床医生，正在对一位患者进行诊疗。
你的目标是：
1. 通过问诊收集全面的患者信息（主诉、现病史、既往史、用药史、过敏史、家族史等）
2. 根据问诊结果合理选择检查项目
3. 综合问诊和检查结果，做出准确诊断
4. 制定安全、有效、个性化的治疗方案

请遵循以下原则：
- 问诊要有针对性，避免无关问题
- 检查选择要有依据，避免过度检查
- 诊断要有鉴别依据
- 治疗方案要考虑安全性、有效性和个性化
- 诊断和检查名称必须尽量使用项目 data/ref_data/ 中的标准名称
- 输出必须严格遵循指定的 JSON 格式"""

    # ============ 初始问诊 Prompt ============

    INITIAL_INQUIRY_TEMPLATE = """你是一位临床医生，正在接诊一位新患者。
请通过问诊了解患者的主要症状和基本情况。

需要了解的信息包括：
1. 主诉：最主要的不适及其持续时间
2. 现病史：发病过程、症状特点、伴随症状
3. 既往史：过去的疾病和手术史
4. 用药史：目前使用的药物
5. 过敏史：药物和食物过敏
6. 家族史：家族中的遗传疾病和常见疾病

{experience_section}

请生成一个初始问诊问题，引导患者描述主要症状。
只输出问题本身，不要输出其他内容。"""

    # ============ 追问 Prompt ============

    FOLLOW_UP_TEMPLATE = """你是一位临床医生，正在对患者进行追问。

已收集的患者信息：
{collected_info}

对话历史：
{chat_history}

{thinking_section}

{experience_section}

根据以上分析，生成一个最有针对性的追问问题。

重点关注：
1. 鉴别诊断中 differentiating_info 提到的关键区分信息
2. key_unknowns 中列出的关键未知项
3. 尚未了解的病史方面（既往史、过敏史、家族史等）
4. 需要进一步明确的症状细节（诱因、加重/缓解因素、时间规律等）

如果所有关键信息已经收集完毕，请返回空字符串。
只输出追问问题本身，不要输出其他内容。"""

    # ============ 检查申请 Prompt ============

    EXAMINATION_TEMPLATE = """你是一位临床医生，需要根据问诊信息选择检查项目。

已收集的患者信息：
{collected_info}

已有的检查结果：
{exam_results}

{thinking_section}

{experience_section}

请根据鉴别诊断分析，选择最能确认或排除关键诊断的检查项目。

选择原则：
1. 检查项目应使用标准医学名称
2. 优先选择能区分鉴别诊断中最高可能性疾病的检查
3. 优先选择 differentiating_info 中提到的检查
4. 避免重复和不必要的检查
5. 如果已有检查结果，基于结果决定是否需要补充检查

请输出需要申请的检查项目列表（JSON 数组格式），例如：
["血常规", "胸部CT"]

只输出 JSON 数组，不要输出其他内容。"""

    # ============ 规划 Prompt（全局诊疗策略） ============

    PLANNING_TEMPLATE = """你是一位资深临床医生，正在为患者制定整体诊疗策略。

当前已知信息：
{collected_info}

已有检查结果：
{exam_results}

当前阶段：{phase}

历史对话摘要：
{chat_history_summary}

{experience_section}

{previous_plan_section}

请制定全局诊疗策略规划，将宏观目标分解为具体的、可执行的步骤。

输出格式（JSON）：
{{
    "primary_hypothesis": "最可能的诊断假设",
    "hypothesis_confidence": 0.5,
    "differential_diagnoses": [
        {{"diagnosis": "疾病名称", "likelihood": 0.3, "key_evidence": "支持依据", "needed_info": "还需确认的信息"}}
    ],
    "strategy": {{
        "current_phase": "inquiry|examination|diagnosis|treatment",
        "phase_goal": "当前阶段的具体目标",
        "priority_actions": [
            {{"action": "ask|examine|diagnose|treat", "target": "具体目标", "reason": "为什么这个行动最重要"}}
        ],
        "info_gaps": ["关键信息缺口1", "关键信息缺口2"],
        "decision_points": ["需要做出决策的关键节点"]
    }},
    "phase_plan": {{
        "inquiry": {{"focus": "问诊重点", "key_questions": ["关键问题1", "关键问题2"], "stop_condition": "问诊停止条件"}},
        "examination": {{"focus": "检查重点", "key_exams": ["关键检查1", "关键检查2"], "stop_condition": "检查停止条件"}},
        "diagnosis": {{"approach": "诊断思路", "differentials_to_resolve": "需要鉴别的诊断"}},
        "treatment": {{"principles": "治疗原则", "considerations": "注意事项"}}
    }},
    "risk_assessment": {{
        "urgent_findings": ["需要紧急处理的情况"],
        "red_flags": ["危险信号"],
        "safety_constraints": ["安全约束"]
    }}
}}

注意：
1. primary_hypothesis 应基于当前已有信息推理，信息不足时给出最合理的初始假设
2. priority_actions 按优先级排列，第一个行动就是下一步应该执行的
3. phase_plan 为每个阶段规划具体策略，即使当前不在该阶段也要预判
4. risk_assessment 关注安全性和紧急性，避免遗漏危险情况
5. 只输出JSON，不要输出其他内容"""

    # ============ 反思批判 Prompt（自我批评与策略调整） ============

    REFLECTION_CRITICISM_TEMPLATE = """你是一位临床医生的内在批判者，需要对当前诊疗策略进行严格审查和批判。

当前诊疗计划：
{current_plan}

已收集的患者信息：
{collected_info}

已有检查结果：
{exam_results}

当前思考结果（鉴别诊断）：
{thinking_result}

已执行的行动历史：
{action_history}

请从以下维度进行批判性反思：

1. **假设偏差**：primary_hypothesis 是否可能错误？有哪些被忽视的可能性？
2. **信息盲区**：是否有关键信息尚未收集但被策略忽略了？
3. **行动优先级**：priority_actions 的排序是否合理？是否有更紧急的行动？
4. **风险遗漏**：risk_assessment 是否遗漏了重要风险？
5. **阶段转换**：当前阶段判断是否正确？是否应该提前或延后转换阶段？
6. **认知偏差**：是否存在确认偏差（只寻找支持假设的证据）？是否充分考虑了反驳证据？

输出格式（JSON）：
{{
    "criticisms": [
        {{
            "aspect": "批判维度",
            "issue": "发现的问题",
            "severity": "high|medium|low",
            "suggestion": "改进建议"
        }}
    ],
    "plan_revision": {{
        "hypothesis_revised": "修订后的主要假设（如无需修订则为null）",
        "priority_actions_revised": [
            {{"action": "ask|examine|diagnose|treat", "target": "具体目标", "reason": "修订原因"}}
        ],
        "phase_change": {{"from": "当前阶段", "to": "建议阶段", "reason": "转换原因"}},
        "new_risks": ["新发现的风险"],
        "missing_info": ["被忽略的关键信息"]
    }},
    "overall_assessment": "on_track|needs_adjustment|needs_replan",
    "confidence_in_plan": 0.7
}}

注意：
1. 批判必须具体、可操作，不要泛泛而谈
2. severity=high 的问题必须立即修正
3. overall_assessment: on_track=计划良好继续执行, needs_adjustment=需要微调, needs_replan=需要重新规划
4. 只输出JSON，不要输出其他内容"""

    # ============ 思考链 Prompt（合并充分性判断） ============

    THINKING_TEMPLATE = """你是一位临床医生，正在分析当前诊疗状态，进行临床推理。

已收集的患者信息：
{collected_info}

已有检查结果：
{exam_results}

对话历史：
{chat_history}

当前阶段：{phase}

{experience_section}

{evidence_summary}

请进行临床推理，同时判断当前信息是否足够做出诊断。

输出格式（JSON）：
{{
    "differential_diagnosis": [
        {{"diagnosis": "疾病名称", "likelihood": 0.6, "supporting_evidence": "支持依据", "differentiating_info": "区分该诊断需要的信息"}}
    ],
    "key_unknowns": ["需要了解的信息1", "需要了解的信息2"],
    "is_sufficient": false,
    "next_action": "下一步行动建议",
    "action_reasoning": "为什么这个行动最有价值",
    "clinical_pattern_proposals": [
        {{
            "pattern_type": "exposure_temporal_organ_injury",
            "pattern_name": "optional_relation_pattern_name",
            "evidence_bindings": [
                {{"evidence_id": "structured_finding_name", "role": "support", "expected_polarity": "positive", "relation_slot": "exposure"}}
            ],
            "relations": [
                {{"type": "temporal_after", "from_evidence_ref": "exposure_finding", "to_evidence_ref": "manifestation_finding"}}
            ],
            "suggested_family": "optional_family_hint",
            "suggested_diseases": ["specific disease name"],
            "missing_evidence_requests": []
        }}
    ]
}}

注意：
1. 鉴别诊断必须基于当前已有信息推理，至少列出2-3个可能性，按可能性从高到低排列
2. 每个诊断的 differentiating_info 应明确指出区分它还需要什么信息或检查
3. key_unknowns 聚焦于能区分最可能诊断的关键信息
4. is_sufficient: 如果当前信息已足够做出诊断则为 true，否则为 false
   - 问诊阶段：主诉明确、现病史完整、既往史/过敏史已了解、症状细节足以鉴别 → true
   - 检查阶段：检查结果已能支持明确诊断、无需补充检查排除鉴别 → true
5. next_action 应明确说明是为了确认还是排除哪个诊断
6. clinical_pattern_proposals 是可选字段，最多 3 个；只能引用上方结构化临床证据中出现的 finding 名称作为 evidence_id。
7. clinical_pattern_proposals 只提出可追溯的关系型候选召回假设，不代表诊断成立；不得引用 reasoning、疾病名、检查计划或不存在的证据作为 evidence。
8. 不确定时返回空数组；不要为了填字段编造 Pattern。
9. 只输出JSON，不要输出其他内容"""

    # ============ 诊断 Prompt ============

    DIAGNOSIS_TEMPLATE = """你是一位临床医生，需要根据问诊和检查结果做出诊断。

患者信息：
{collected_info}

检查结果：
{exam_results}

对话历史：
{chat_history}

{experience_section}

{rag_context}

{evidence_summary}

{candidate_table}

{critic_feedback}

{catalog_section}

Clinical pattern hypothesis instructions:
- Fill clinical_pattern_hypotheses only when the provided structured findings support a multi-evidence pattern that may recall a missed specific disease.
- Use only explicit structured finding names as evidence_id values; do not cite reasoning text, candidate names, ordered exam names, or inferred facts.
- Prefer patterns involving exposure-time-organ injury, post-infectious multi-system syndromes, or mechanism-specific objective findings.
- A pattern hypothesis is not a diagnosis and is not confirmatory evidence; it only proposes controlled candidate recall for later verification.
- Return an empty clinical_pattern_hypotheses array when no traceable pattern is present.

请综合分析以上信息，做出诊断并制定治疗方案。

输出格式（JSON）：
{{
    "diagnosis": ["最可能的具体临床疾病名称"],
    "diagnosis_candidates": [
        {{"name": "候选疾病名称", "confidence": 0.0, "supporting_evidence": ["证据"]}}
    ],
    "clinical_pattern_hypotheses": [
        {{
            "pattern_name": "optional_pattern_name",
            "pattern_type": "temporal_causal_multievidence",
            "evidence_bindings": [
                {{"evidence_id": "structured_finding_name", "role": "support", "expected_polarity": "positive", "relation_slot": "exposure"}}
            ],
            "relations": [
                {{"type": "temporal_after", "from": "exposure_finding", "to": "manifestation_finding"}}
            ],
            "suggested_diseases": [
                {{"name": "specific disease name", "canonical_id": "", "hypothesis_confidence": 0.0}}
            ],
            "missing_evidence_requests": []
        }}
    ],
    "treatment_plan": "治疗方案描述",
    "reasoning": "诊断和治疗依据"
}}

注意：
1. 候选诊断可以使用比目录更具体的规范临床病名；不要为了迁就目录把明确病因泛化成症状或综合征
2. 如果有多个可能的诊断，按可能性从高到低排列
3. 必须说明支持证据和关键反证；不得把阴性结果、正常参考范围或括号中的示例当成阳性证据
4. 治疗方案应具体、可执行，并服从年龄、过敏史和禁忌证
5. 你的输出只是开放候选；系统会进行别名、上下位和相似名称标准化，再由证据评分与提交前审查决定最终提交名
6. 只输出 JSON，不要输出其他内容"""

    # ============ 反思 Prompt ============

    REFLECTION_TEMPLATE = """你是一位临床医生，正在反思自己的诊疗过程。

评估报告：
{report}

患者信息：
{collected_info}

检查结果：
{exam_results}

请从以下维度进行结构化反思：

1. **问诊反思**：
   - 问诊是否全面？遗漏了哪些关键信息？
   - 问诊顺序是否合理？是否浪费了轮次？
   - 改进建议：下次如何更高效地收集信息

2. **检查反思**：
   - 检查选择是否精准？有无遗漏关键检查？
   - 是否存在过度检查？哪些检查可以省略？
   - 改进建议：如何根据症状更精准地选择检查

3. **诊断反思**：
   - 诊断是否准确？如果偏差，原因是什么？
   - 鉴别诊断是否充分？是否考虑了其他可能性？
   - 改进建议：如何提高诊断准确率

4. **治疗反思**：
   - 治疗方案是否安全、有效、个性化？
   - 是否有更好的治疗选择？
   - 改进建议：如何优化治疗方案

5. **关键经验总结**：
   - 从这个病例中学到的最重要的经验是什么？
   - 遇到类似症状时应该注意什么？

请输出结构化的反思总结。"""

    # ============ 信息提取 Prompt ============

    INFO_EXTRACTION_TEMPLATE = """你是一位临床医生，需要从患者的回复中提取结构化信息。

患者回复：
{patient_response}

已有信息：
{existing_info}

请提取以下结构化信息（JSON 格式）：
{{
    "chief_complaint": "主诉（主要症状+持续时间）",
    "present_illness": "现病史描述",
    "symptoms": ["症状1", "症状2"],
    "symptom_details": {{
        "症状名": {{"onset": "起病时间", "severity": "严重程度", "duration": "持续时间", "triggers": "诱因", "relieving": "缓解因素"}}
    }},
    "past_history": "既往史",
    "medication_history": "用药史",
    "allergy_history": "过敏史",
    "family_history": "家族史",
    "personal_history": "个人史（吸烟、饮酒等）"
}}

注意：
1. 只提取患者回复中明确提到的信息
2. 保留已有信息中已提取的字段
3. 症状使用标准医学术语
4. 如果某字段无信息，保留空字符串或空数组
5. 只输出 JSON，不要输出其他内容"""

    # ============ 信息充分性判断 Prompt ============

    INFO_SUFFICIENCY_TEMPLATE = """你是一位临床医生，需要判断已收集的患者信息是否足够做出诊断。

已收集的患者信息：
{collected_info}

已进行的问诊轮次：{ask_rounds} / {max_ask_rounds}

请判断信息是否足够，考虑以下方面：
1. 主诉和现病史是否明确
2. 既往史、过敏史是否已了解
3. 症状细节是否足够进行鉴别诊断
4. 是否有关键信息缺失

输出格式（JSON）：
{{
    "is_sufficient": true/false,
    "missing_aspects": ["缺失的方面1", "缺失的方面2"],
    "reasoning": "判断依据"
}}

只输出 JSON，不要输出其他内容。"""

    # ============ 检查充分性判断 Prompt ============

    EXAM_SUFFICIENCY_TEMPLATE = """你是一位临床医生，需要判断已进行的检查是否足够做出诊断。

患者信息：
{collected_info}

已有检查结果：
{exam_results}

已进行的检查轮次：{exam_rounds} / {max_exam_rounds}

请判断检查是否足够，考虑以下方面：
1. 现有检查结果是否支持明确诊断
2. 是否需要补充检查来排除鉴别诊断
3. 检查结果之间是否一致
4. 是否存在无法解释的异常指标

输出格式（JSON）：
{{
    "is_sufficient": true/false,
    "additional_exams_needed": ["需要补充的检查1"],
    "reasoning": "判断依据"
}}

只输出 JSON，不要输出其他内容。"""

    # ============ 构建方法 ============

    def _build_experience_section(self, relevant_experience: List[Dict[str, Any]]) -> str:
        """构建历史经验注入段落。

        Args:
            relevant_experience: 相关历史经验列表

        Returns:
            格式化的经验注入文本
        """
        if not relevant_experience:
            return ""

        lines = ["以下是类似病例的历史经验，请参考但不要照搬："]
        for i, exp in enumerate(relevant_experience[:3], 1):
            is_failure = exp.get("memory_kind") == "failure_lesson" or (
                (exp.get("metrics") or {}).get("diagnosis_accuracy", 1) < 0.8
            )
            content = exp.get("lesson", "") if is_failure else exp.get("content", "")
            metrics = exp.get("metrics", {})
            if metrics:
                da = metrics.get("diagnosis_accuracy", "?")
                ep = metrics.get("exam_precision", "?")
                ts = metrics.get("treatment_score", "?")
                label = "失败教训" if is_failure else "成功经验"
                lines.append(f"\n{label}{i}（诊断准确率={da}, 检查精确率={ep}, 治疗评分={ts}）：")
            else:
                lines.append(f"\n经验{i}：")
            lines.append(content)

        return "\n".join(lines)

    def _build_thinking_section(self, thinking: Dict[str, Any]) -> str:
        """将思考结果格式化为注入段落。

        Args:
            thinking: _think() 方法返回的思考结果

        Returns:
            格式化的思考注入文本
        """
        if not thinking:
            return ""

        lines = ["【当前鉴别诊断分析】"]

        dd = thinking.get("differential_diagnosis", [])
        if dd:
            for i, d in enumerate(dd[:3], 1):
                name = d.get("diagnosis", "未知")
                likelihood = d.get("likelihood", "?")
                evidence = d.get("supporting_evidence", "无")
                diff_info = d.get("differentiating_info", "无")
                lines.append(f"  {i}. {name}（可能性: {likelihood}）")
                lines.append(f"     支持依据: {evidence}")
                lines.append(f"     区分所需: {diff_info}")
        else:
            lines.append("  （暂无明确鉴别诊断）")

        unknowns = thinking.get("key_unknowns", [])
        if unknowns:
            lines.append(f"\n关键未知项: {', '.join(unknowns)}")

        action = thinking.get("next_action", "")
        if action:
            lines.append(f"\n建议下一步: {action}")
            reasoning = thinking.get("action_reasoning", "")
            if reasoning:
                lines.append(f"理由: {reasoning}")

        return "\n".join(lines)

    def _build_knowledge_section(self, knowledge_context: str) -> str:
        """将知识库 RAG 上下文包装为可注入的段落。

        Args:
            knowledge_context: KnowledgeBase.build_rag_context() 的返回文本

        Returns:
            带边界标识的知识段落；空输入返回空串
        """
        if not knowledge_context or not str(knowledge_context).strip():
            return ""
        return (
            "\n=== 医学知识库参考（静态词典召回，非病例经验，仅缩小搜索空间）===\n"
            f"{knowledge_context}\n"
            "=== 知识库参考结束 ===\n"
        )
    def build_thinking_prompt(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        phase: str,
        relevant_experience: Optional[List[Dict[str, Any]]] = None,
        knowledge_context: str = "",
        evidence_summary: str = "",
    ) -> str:
        """构建思考链 prompt。

        Args:
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            chat_history: 对话历史
            phase: 当前阶段（"inquiry" 或 "examination"）
            relevant_experience: 相关历史经验

        Returns:
            思考链 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2) if exam_results else "暂无检查结果"
        history_str = "\n".join(
            f"{'医生' if msg['from'] == 'doctor' else '患者'}: {msg['text']}"
            for msg in chat_history
        ) if chat_history else "暂无对话历史"
        experience_section = self._build_experience_section(relevant_experience or [])

        phase_desc = "问诊阶段" if phase == "inquiry" else "检查阶段"

        knowledge_section = self._build_knowledge_section(knowledge_context)

        return self.SYSTEM_ROLE + knowledge_section + "\n\n" + self.THINKING_TEMPLATE.format(
            collected_info=info_str,
            exam_results=exam_str,
            chat_history=history_str,
            phase=phase_desc,
            experience_section=experience_section,
            evidence_summary=evidence_summary or "【结构化临床证据】暂无。",
        )

    def build_initial_inquiry_prompt(
        self, relevant_experience: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """构建初始问诊 prompt。

        Args:
            relevant_experience: 相关历史经验

        Returns:
            初始问诊 prompt 文本
        """
        experience_section = self._build_experience_section(relevant_experience or [])
        return self.SYSTEM_ROLE + "\n\n" + self.INITIAL_INQUIRY_TEMPLATE.format(
            experience_section=experience_section
        )

    def build_follow_up_prompt(
        self,
        collected_info: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        relevant_experience: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建追问 prompt。

        Args:
            collected_info: 已收集的患者信息
            chat_history: 对话历史
            relevant_experience: 相关历史经验
            thinking: 思考链结果（鉴别诊断 + 关键未知项）

        Returns:
            追问 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        history_str = "\n".join(
            f"{'医生' if msg['from'] == 'doctor' else '患者'}: {msg['text']}"
            for msg in chat_history
        )
        experience_section = self._build_experience_section(relevant_experience or [])
        thinking_section = self._build_thinking_section(thinking) if thinking else ""

        return self.SYSTEM_ROLE + "\n\n" + self.FOLLOW_UP_TEMPLATE.format(
            collected_info=info_str,
            chat_history=history_str,
            thinking_section=thinking_section,
            experience_section=experience_section,
        )

    def build_examination_prompt(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        relevant_experience: Optional[List[Dict[str, Any]]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        knowledge_context: str = "",
    ) -> str:
        """构建检查申请 prompt。

        Args:
            collected_info: 收集到的患者信息
            exam_results: 已有的检查结果
            relevant_experience: 相关历史经验
            thinking: 思考链结果（鉴别诊断 + 关键未知项）
            knowledge_context: 知识库 RAG 上下文文本

        Returns:
            检查申请 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2)
        experience_section = self._build_experience_section(relevant_experience or [])
        thinking_section = self._build_thinking_section(thinking) if thinking else ""
        knowledge_section = self._build_knowledge_section(knowledge_context)

        return self.SYSTEM_ROLE + knowledge_section + "\n\n" + self.EXAMINATION_TEMPLATE.format(
            collected_info=info_str,
            exam_results=exam_str,
            thinking_section=thinking_section,
            experience_section=experience_section,
        )

    def build_diagnosis_prompt(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        relevant_experience: Optional[List[Dict[str, Any]]] = None,
        standard_diseases: Optional[List[str]] = None,
        rag_context: str = "",
        evidence_summary: str = "",
        candidate_table: str = "",
        critic_feedback: str = "",
    ) -> str:
        """构建诊断 prompt。

        Args:
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            chat_history: 对话历史
            relevant_experience: 相关历史经验
            standard_diseases: 标准疾病名称列表

        Returns:
            诊断 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2)
        history_str = "\n".join(
            f"{'医生' if msg['from'] == 'doctor' else '患者'}: {msg['text']}"
            for msg in chat_history
        )
        experience_section = self._build_experience_section(relevant_experience or [])
        catalog_section = (
            "【开放候选与标准化】\n"
            "你可以提出目录外但医学上规范、具体且有证据支持的疾病候选，"
            "不要输出缩写、待确诊或自造名称。系统不会直接提交你的原始名称；"
            "它会映射到官方 catalog 或 evaluation 已确认的受控细分名称，"
            "无法可靠映射的候选只进入审计和后续学习。"
        )
        if standard_diseases:
            catalog_section += "\n优先标准化目标名称：" + "、".join(standard_diseases)

        return self.SYSTEM_ROLE + "\n\n" + self.DIAGNOSIS_TEMPLATE.format(
            collected_info=info_str,
            exam_results=exam_str,
            chat_history=history_str,
            experience_section=experience_section,
            rag_context=rag_context or "【Hybrid RAG】未召回到可靠上下文。",
            evidence_summary=evidence_summary or "【结构化临床证据】暂无。",
            candidate_table=candidate_table or "【证据评分候选】暂无。",
            critic_feedback=critic_feedback or "",
            catalog_section=catalog_section,
        )

    def build_reflection_prompt(
        self,
        report: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        knowledge_context: str = "",
        relevant_experience: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建反思 prompt（三源 RAG：知识库 + 经验 + 当前病例）。

        Args:
            report: 评估报告
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            knowledge_context: 知识库 RAG 上下文文本
            relevant_experience: 相关历史经验（memory 层）

        Returns:
            反思 prompt 文本
        """
        report_str = json.dumps(report, ensure_ascii=False, indent=2)
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2)
        knowledge_section = self._build_knowledge_section(knowledge_context)
        experience_section = self._build_experience_section(relevant_experience or [])
        extra = ""
        if experience_section:
            extra += "\n\n" + experience_section

        return self.SYSTEM_ROLE + knowledge_section + extra + "\n\n" + self.REFLECTION_TEMPLATE.format(
            report=report_str,
            collected_info=info_str,
            exam_results=exam_str,
        )

    def build_info_extraction_prompt(
        self, patient_response: str, existing_info: Dict[str, Any]
    ) -> str:
        """构建信息提取 prompt。

        Args:
            patient_response: 患者回复文本
            existing_info: 已有的患者信息

        Returns:
            信息提取 prompt 文本
        """
        info_str = json.dumps(existing_info, ensure_ascii=False, indent=2)
        return self.SYSTEM_ROLE + "\n\n" + self.INFO_EXTRACTION_TEMPLATE.format(
            patient_response=patient_response,
            existing_info=info_str,
        )

    def build_info_sufficiency_prompt(
        self, collected_info: Dict[str, Any], ask_rounds: int, max_ask_rounds: int
    ) -> str:
        """构建信息充分性判断 prompt。

        Args:
            collected_info: 已收集的信息
            ask_rounds: 当前问诊轮次
            max_ask_rounds: 最大问诊轮次

        Returns:
            信息充分性判断 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        return self.SYSTEM_ROLE + "\n\n" + self.INFO_SUFFICIENCY_TEMPLATE.format(
            collected_info=info_str,
            ask_rounds=ask_rounds,
            max_ask_rounds=max_ask_rounds,
        )

    def build_exam_sufficiency_prompt(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        exam_rounds: int,
        max_exam_rounds: int,
    ) -> str:
        """构建检查充分性判断 prompt。

        Args:
            collected_info: 收集到的患者信息
            exam_results: 检查结果
            exam_rounds: 当前检查轮次
            max_exam_rounds: 最大检查轮次

        Returns:
            检查充分性判断 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2)
        return self.SYSTEM_ROLE + "\n\n" + self.EXAM_SUFFICIENCY_TEMPLATE.format(
            collected_info=info_str,
            exam_results=exam_str,
            exam_rounds=exam_rounds,
            max_exam_rounds=max_exam_rounds,
        )

    def build_planning_prompt(
        self,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        chat_history: List[Dict[str, str]],
        phase: str,
        relevant_experience: Optional[List[Dict[str, Any]]] = None,
        previous_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        """构建全局诊疗策略规划 prompt。

        Args:
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            chat_history: 对话历史
            phase: 当前阶段
            relevant_experience: 相关历史经验
            previous_plan: 上一轮规划结果（用于策略调整）

        Returns:
            规划 prompt 文本
        """
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2) if exam_results else "暂无检查结果"

        # 对话历史摘要（过长时截断）
        if chat_history:
            history_lines = [
                f"{'医生' if msg['from'] == 'doctor' else '患者'}: {msg['text'][:100]}"
                for msg in chat_history[-10:]  # 最近10轮
            ]
            chat_history_summary = "\n".join(history_lines)
        else:
            chat_history_summary = "暂无对话历史"

        experience_section = self._build_experience_section(relevant_experience or [])

        # 上一轮规划信息
        if previous_plan:
            previous_plan_section = "上一轮诊疗策略：\n" + json.dumps(
                previous_plan, ensure_ascii=False, indent=2
            )
        else:
            previous_plan_section = "这是首次制定诊疗策略。"

        phase_desc = {
            "initial": "初始接诊",
            "inquiry": "问诊阶段",
            "examination": "检查阶段",
            "diagnosis": "诊断阶段",
            "treatment": "治疗阶段",
        }.get(phase, phase)

        return self.SYSTEM_ROLE + "\n\n" + self.PLANNING_TEMPLATE.format(
            collected_info=info_str,
            exam_results=exam_str,
            phase=phase_desc,
            chat_history_summary=chat_history_summary,
            experience_section=experience_section,
            previous_plan_section=previous_plan_section,
        )

    def build_reflection_criticism_prompt(
        self,
        current_plan: Dict[str, Any],
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        thinking_result: Optional[Dict[str, Any]] = None,
        action_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """构建反思批判 prompt。

        Args:
            current_plan: 当前诊疗计划
            collected_info: 已收集的患者信息
            exam_results: 已有检查结果
            thinking_result: 最近一次思考结果
            action_history: 已执行的行动历史

        Returns:
            反思批判 prompt 文本
        """
        plan_str = json.dumps(current_plan, ensure_ascii=False, indent=2)
        info_str = json.dumps(collected_info, ensure_ascii=False, indent=2)
        exam_str = json.dumps(exam_results, ensure_ascii=False, indent=2) if exam_results else "暂无检查结果"

        if thinking_result:
            thinking_str = json.dumps(thinking_result, ensure_ascii=False, indent=2)
        else:
            thinking_str = "暂无思考结果"

        if action_history:
            history_lines = []
            for i, action in enumerate(action_history[-10:], 1):
                action_type = action.get("type", "?")
                target = action.get("target", "")
                result_summary = action.get("result_summary", "")
                history_lines.append(f"  {i}. {action_type}: {target} → {result_summary}")
            action_history_str = "\n".join(history_lines)
        else:
            action_history_str = "暂无行动历史"

        return self.SYSTEM_ROLE + "\n\n" + self.REFLECTION_CRITICISM_TEMPLATE.format(
            current_plan=plan_str,
            collected_info=info_str,
            exam_results=exam_str,
            thinking_result=thinking_str,
            action_history=action_history_str,
        )
