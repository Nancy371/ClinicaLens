"""
记忆逻辑模块。

参赛者需要修改或替换此模块来管理训练反思、病例经验和检索记忆。
升级版使用结构化 JSON 存储记忆，支持：
- 症状关键词检索
- 诊断模式匹配
- 低分病例优先参考（从失败中学习）
- 经验质量评分

注意：如果使用数据库或对象存储等外部服务，需要使用可直接访问的外部存储服务。
部署环境不能依赖 docker-compose 拉起额外服务。
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DoctorMemory:
    """医生 Agent 的记忆管理类。

    使用结构化 JSON 文件存储记忆，支持高效检索和质量评估。
    """

    def __init__(self, config: Dict[str, Any]):
        """初始化记忆管理器。

        Args:
            config: 配置字典，包含 memory 相关配置
        """
        self.config = config
        memory_config = config.get("memory", {})
        self.json_path = memory_config.get("json_path", "outputs/runtime_state/memory.json")
        self.replay_path = memory_config.get(
            "diagnostic_replay_path",
            "outputs/runtime_state/diagnostic_replay.jsonl",
        )
        self.max_notes = memory_config.get("max_notes", 200)
        self.max_note_chars = memory_config.get("max_note_chars", 1000)

        # 加载已有记忆
        self.notes: List[Dict[str, Any]] = []
        self._load_memory()

    def _load_memory(self) -> None:
        """从本地文件加载记忆。"""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self.notes = json.load(f)
                logger.info(f"[Memory] 已加载 {len(self.notes)} 条记忆")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[Memory] 加载记忆文件失败: {e}，将创建新文件")
                self.notes = []
        else:
            # 尝试从旧版 Markdown 格式迁移
            md_path = self.config.get("memory", {}).get("md_path", "data/memory_data/memory.md")
            if os.path.exists(md_path):
                logger.info(f"[Memory] 发现旧版 Markdown 记忆文件，尝试迁移")
                self.notes = self._migrate_from_markdown(md_path)
                if self.notes:
                    self._save_to_file()
                    logger.info(f"[Memory] 迁移了 {len(self.notes)} 条记忆到 JSON 格式")
            else:
                logger.info(f"[Memory] 记忆文件不存在，将创建新文件: {self.json_path}")

    def _migrate_from_markdown(self, md_path: str) -> List[Dict[str, Any]]:
        """从旧版 Markdown 格式迁移记忆。

        Args:
            md_path: Markdown 文件路径

        Returns:
            迁移后的记忆条目列表
        """
        notes = []
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()

            current_note = None
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("## "):
                    if current_note:
                        notes.append(current_note)
                    current_note = {
                        "title": line[3:].strip(),
                        "content": "",
                        "created_at": datetime.now().isoformat(),
                    }
                elif current_note and line:
                    current_note["content"] += line + "\n"

            if current_note:
                notes.append(current_note)
        except Exception as e:
            logger.warning(f"[Memory] Markdown 迁移失败: {e}")

        return notes

    def save_case_experience(
        self,
        patient_id: str,
        report: Dict[str, Any],
        reflection: str,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        evidence: Optional[Dict[str, Any]] = None,
        diagnosis_decision: Optional[Dict[str, Any]] = None,
        error_types: Optional[List[str]] = None,
    ) -> None:
        """保存病例经验到记忆。

        Args:
            patient_id: 患者 ID
            report: 评估报告
            reflection: 反思总结
            collected_info: 收集到的患者信息
            exam_results: 检查结果
        """
        # 提取评估指标
        diagnosis_accuracy = _score_value(
            report.get("diagnosisAccuracy", report.get("diagnosis_accuracy", 0))
        )
        exam_precision = _score_value(
            report.get("examinationPrecision", report.get("examination_precision", 0))
        )
        treatment_score = _score_value(
            report.get("treatmentOverallScore", report.get("treatment_overall_score", 0))
        )

        # 提取诊断详情
        symptoms = collected_info.get("symptoms", [])
        diagnosis_detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(diagnosis_detail, dict):
            diagnosis_detail = {}
        submitted_diagnosis = diagnosis_detail.get("submitted") or report.get("diagnosis") or []
        expected_diagnosis = diagnosis_detail.get("expected") or report.get("finalDiagnosis") or []
        if isinstance(submitted_diagnosis, str):
            submitted_diagnosis = [submitted_diagnosis]
        if isinstance(expected_diagnosis, str):
            expected_diagnosis = [expected_diagnosis]

        # 提取检查详情
        ordered_exams = list(exam_results.keys()) if exam_results else []

        # 计算综合质量评分（0-1）
        quality_score = (diagnosis_accuracy + exam_precision + treatment_score) / 3
        memory_kind = (
            "success"
            if diagnosis_accuracy >= 0.8 and quality_score >= 0.7
            else "failure_lesson"
        )
        error_types = list(dict.fromkeys(error_types or []))
        if memory_kind == "failure_lesson":
            lesson = (
                f"参考诊断：{'、'.join(str(item) for item in expected_diagnosis) or '未提供'}。"
                f"错误类型：{'、'.join(error_types) or '待归因'}。"
                "后续相似病例应围绕参考诊断的关键支持证据和反证重新裁决。"
            )
        else:
            lesson = "该病例达到成功范例阈值，可参考其正确诊断和检查路径。"

        # 构建结构化经验条目
        note = {
            "title": f"病例经验: {patient_id}",
            "content": reflection[:self.max_note_chars] if len(reflection) > self.max_note_chars else reflection,
            "memory_kind": memory_kind,
            "lesson": lesson,
            "error_types": error_types,
            "created_at": datetime.now().isoformat(),
            "patient_id": patient_id,
            "symptoms": symptoms,
            "submitted_diagnosis": submitted_diagnosis,
            "expected_diagnosis": expected_diagnosis,
            "ordered_exams": ordered_exams,
            "evidence_summary": evidence or {},
            "diagnosis_decision": diagnosis_decision or {},
            "metrics": {
                "diagnosis_accuracy": diagnosis_accuracy,
                "exam_precision": exam_precision,
                "treatment_score": treatment_score,
                "quality_score": quality_score,
            },
            "collected_info_summary": {
                "chief_complaint": collected_info.get("chief_complaint", ""),
                "symptoms": symptoms,
                "symptom_details": collected_info.get("symptom_details", {}),
                "past_history": collected_info.get("past_history", ""),
                "age": collected_info.get("age", ""),
                "gender": collected_info.get("gender", ""),
                "age_bucket": self._bucket_age(collected_info.get("age", "")),
            },
        }

        # 添加到记忆列表
        self.notes.append(note)

        # 控制记忆数量，优先保留低分病例（从失败中学习）和近期病例
        if len(self.notes) > self.max_notes:
            self._prune_notes()

        # 保存到文件
        self._save_to_file()

        logger.info(
            f"[Memory] 已保存病例经验: {patient_id}, "
            f"质量评分={quality_score:.2f}, "
            f"诊断={diagnosis_accuracy}, 检查={exam_precision}, 治疗={treatment_score}"
        )

    def save_diagnostic_replay(
        self,
        patient_id: str,
        collected_info: Dict[str, Any],
        exam_results: Dict[str, Any],
        evidence: Dict[str, Any],
        diagnosis_decision: Dict[str, Any],
        report: Dict[str, Any],
        error_types: Optional[List[str]] = None,
        llm_candidates: Optional[List[str]] = None,
        rag_chunks: Optional[List[Dict[str, Any]]] = None,
        case_audit: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a full diagnosis trace for deterministic offline replay."""
        detail = report.get("diagnosisDetail") or report.get("diagnosis_detail") or {}
        if not isinstance(detail, dict):
            detail = {}
        row = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "patient_id": patient_id,
            "collected_info": collected_info or {},
            "exam_results": exam_results or {},
            "evidence": evidence or {},
            "diagnosis_decision": diagnosis_decision or {},
            "llm_candidates": list(llm_candidates or []),
            "rag_chunks": list(rag_chunks or []),
            "submitted": detail.get("submitted") or report.get("diagnosis") or [],
            "expected": detail.get("expected") or report.get("finalDiagnosis") or [],
            "evaluation": report or {},
            "error_types": list(dict.fromkeys(error_types or [])),
            "case_audit": {
                "elapsed_seconds": (case_audit or {}).get("elapsed_seconds"),
                "timed_out": bool((case_audit or {}).get("timed_out", False)),
                "critic": (case_audit or {}).get("critic") or {},
            },
        }
        directory = os.path.dirname(self.replay_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(self.replay_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"[Memory] 保存诊断回放失败: {exc}")

    def _prune_notes(self) -> None:
        """裁剪记忆，保留最有价值的条目。

        策略：
        1. 保留所有低分病例（quality_score < 0.5），从失败中学习
        2. 保留近期高分病例（quality_score >= 0.8），作为成功参考
        3. 按时间排序，删除最旧的中等分数病例
        """
        if len(self.notes) <= self.max_notes:
            return

        # 分类
        low_score = [n for n in self.notes if n.get("metrics", {}).get("quality_score", 1) < 0.5]
        high_score = [n for n in self.notes if n.get("metrics", {}).get("quality_score", 0) >= 0.8]
        medium_score = [n for n in self.notes if 0.5 <= n.get("metrics", {}).get("quality_score", 0) < 0.8]

        # 保留所有低分和近期高分
        keep = low_score + high_score

        # 如果还不够，从中等分数中按时间倒序补充
        remaining_slots = self.max_notes - len(keep)
        if remaining_slots > 0:
            medium_score.sort(key=lambda n: n.get("created_at", ""), reverse=True)
            keep.extend(medium_score[:remaining_slots])

        # 按时间排序
        keep.sort(key=lambda n: n.get("created_at", ""))

        removed = len(self.notes) - len(keep)
        self.notes = keep[-self.max_notes:]
        if removed > 0:
            logger.info(f"[Memory] 裁剪记忆: 移除 {removed} 条，保留 {len(self.notes)} 条")

    def _save_to_file(self) -> None:
        """将记忆保存到 JSON 文件。"""
        tmp_path = ""
        try:
            target_dir = os.path.dirname(self.json_path) or "."
            os.makedirs(target_dir, exist_ok=True)
            tmp_path = os.path.join(
                target_dir,
                f".{os.path.basename(self.json_path)}.{os.getpid()}.{uuid.uuid4().hex}.tmp",
            )

            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.notes, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp_path, self.json_path)
                tmp_path = ""
            except OSError:
                # 某些受限 Windows 沙盒禁止重命名替换，退化为直接写入以保证功能可用。
                with open(self.json_path, "w", encoding="utf-8") as f:
                    json.dump(self.notes, f, ensure_ascii=False, indent=2)

        except Exception as e:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            logger.warning(f"[Memory] 保存记忆文件失败: {e}")

    def search_relevant_experience(
        self, symptoms: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """检索与当前症状相关的历史经验。

        升级版检索策略：
        1. 症状关键词匹配（权重 0.4）
        2. 诊断模式匹配（权重 0.3）
        3. 经验质量评分（权重 0.3，低分病例权重更高以从失败学习）

        Args:
            symptoms: 当前症状列表
            top_k: 返回最相关的 top_k 条经验

        Returns:
            相关经验列表
        """
        if not self.notes or not symptoms:
            return []

        scored_notes = []

        for note in self.notes:
            score = 0.0

            # 1. 症状关键词匹配（0.4）
            note_symptoms = note.get("symptoms", [])
            if note_symptoms:
                overlap = len(set(symptoms) & set(note_symptoms))
                total = len(set(symptoms) | set(note_symptoms))
                symptom_score = overlap / total if total > 0 else 0
                score += symptom_score * 0.4

            # 2. 内容关键词匹配（补充症状匹配）
            content = note.get("content", "")
            content_symptoms = note.get("collected_info_summary", {}).get("symptoms", [])
            if content_symptoms:
                overlap = len(set(symptoms) & set(content_symptoms))
                total = len(set(symptoms) | set(content_symptoms))
                content_score = overlap / total if total > 0 else 0
                score += content_score * 0.2

            # 3. 经验质量评分。失败病例只作为教训，不再获得额外召回奖励。
            quality = note.get("metrics", {}).get("quality_score", 0.5)
            if note.get("memory_kind") == "failure_lesson" or quality < 0.5:
                quality_weight = 0.08
            elif quality >= 0.8:
                quality_weight = 0.45
            else:
                quality_weight = 0.25
            score += quality_weight * 0.3

            # 4. 时间衰减（近期经验更相关）
            created_at = note.get("created_at", "")
            if created_at:
                try:
                    created = datetime.fromisoformat(created_at)
                    days_ago = (datetime.now() - created).days
                    time_decay = max(0.5, 1.0 - days_ago / 365)  # 一年内衰减到 0.5
                    score *= time_decay
                except (ValueError, TypeError):
                    pass

            if score > 0:
                scored_notes.append((score, note))

        # 按相关度排序
        scored_notes.sort(key=lambda x: x[0], reverse=True)

        return [note for _, note in scored_notes[:top_k]]

    def search_by_diagnosis(
        self, diagnosis_keywords: List[str], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """按诊断关键词检索历史经验。

        Args:
            diagnosis_keywords: 诊断关键词列表
            top_k: 返回数量

        Returns:
            相关经验列表
        """
        if not self.notes or not diagnosis_keywords:
            return []

        scored_notes = []
        for note in self.notes:
            expected = note.get("expected_diagnosis", [])
            submitted = note.get("submitted_diagnosis", [])

            # 匹配正确诊断和提交诊断
            all_diagnoses = expected + submitted
            overlap = sum(1 for kw in diagnosis_keywords if any(kw in d for d in all_diagnoses))

            if overlap > 0:
                quality = note.get("metrics", {}).get("quality_score", 0.5)
                scored_notes.append((overlap + quality, note))

        scored_notes.sort(key=lambda x: x[0], reverse=True)
        return [note for _, note in scored_notes[:top_k]]

    def get_low_score_cases(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """获取低分病例，用于重点学习。

        Args:
            threshold: 质量评分阈值

        Returns:
            低分病例列表
        """
        return [
            note for note in self.notes
            if note.get("metrics", {}).get("quality_score", 1) < threshold
        ]

    def get_all_notes(self) -> List[Dict[str, Any]]:
        """获取所有记忆条目。"""
        return self.notes

    def get_notes_by_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """获取特定患者的记忆条目。"""
        return [
            note for note in self.notes
            if note.get("patient_id") == patient_id
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息。"""
        if not self.notes:
            return {"total": 0}

        metrics = [n.get("metrics", {}) for n in self.notes]
        quality_scores = [m.get("quality_score", 0) for m in metrics]

        return {
            "total": len(self.notes),
            "avg_quality": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "low_score_count": len(self.get_low_score_cases()),
            "symptoms_covered": list(set(
                s for n in self.notes for s in n.get("symptoms", [])
            )),
        }

    def clear_notes(self) -> None:
        """清空所有记忆。"""
        self.notes = []
        self._save_to_file()
        logger.info("[Memory] 已清空所有记忆")

    # ==========================================================
    # 多维召回：人口学 + 主诉 + 症状 + 既往史
    # ==========================================================
    @staticmethod
    def _bucket_age(age: Any) -> str:
        """将年龄归一到年龄段桶，用于跨患者匹配。"""
        try:
            a = int(str(age).strip())
        except (ValueError, TypeError, AttributeError):
            return ""
        if a < 3:
            return "婴幼儿"
        if a < 14:
            return "儿童"
        if a < 18:
            return "青少年"
        if a < 40:
            return "青年"
        if a < 60:
            return "中年"
        if a < 75:
            return "老年"
        return "高龄"

    @staticmethod
    def _tokenize(text: str) -> set:
        """粗粒度中文分词：按标点/空白切分为 2-4 字子串集合。"""
        if not text:
            return set()
        s = str(text)
        # 去除标点
        for ch in "，。！？；：、,.!?;: \t\n\r()（）【】[]":
            s = s.replace(ch, " ")
        tokens = set()
        for w in s.split():
            if len(w) >= 2:
                tokens.add(w)
            # 2-gram
            for i in range(len(w) - 1):
                tokens.add(w[i:i + 2])
        return tokens

    def get_score_baseline(
        self,
        final_dx: Optional[str] = None,
        default: float = 0.7,
    ) -> float:
        """返回同疾病的历史综合分基线（不足 3 条则退化到全局均值/默认值）。

        Args:
            final_dx: 目标疾病名（用于按疾病过滤）
            default: 无历史时的兜底基线

        Returns:
            基线分数（0-1 之间）
        """
        if not self.notes:
            return default

        def _quality(n: Dict[str, Any]) -> Optional[float]:
            m = n.get("metrics") or {}
            q = m.get("quality_score")
            try:
                return float(q) if q is not None else None
            except (TypeError, ValueError):
                return None

        # 优先按疾病过滤
        pool: List[float] = []
        if final_dx:
            fd = str(final_dx)
            for n in self.notes:
                for tag in ("submitted_diagnosis", "expected_diagnosis"):
                    v = n.get(tag) or []
                    if any(fd in str(x) for x in v):
                        q = _quality(n)
                        if q is not None:
                            pool.append(q)
                        break
        if len(pool) < 3:
            pool = [q for q in (_quality(n) for n in self.notes) if q is not None]
        if not pool:
            return default
        return sum(pool) / len(pool)

    def search_relevant_experience_multi(
        self,
        collected_info: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """多维召回：症状 + 主诉 + 既往史 + 年龄段 + 性别。

        权重分配（总和 1.0）:
            - 症状 Jaccard   : 0.40
            - 主诉 token 重叠: 0.20
            - 既往史 token 重叠: 0.15
            - 年龄段命中     : 0.10
            - 性别命中       : 0.05
            - 质量权重(低分优先): 0.10
        再乘时间衰减 (365 天衰减到 0.5)。

        Args:
            collected_info: 当前病例的收集信息
            top_k: 返回条数

        Returns:
            按相关度降序的经验列表
        """
        if not self.notes or not collected_info:
            return []

        cur_symptoms = set(collected_info.get("symptoms", []) or [])
        cur_chief = self._tokenize(collected_info.get("chief_complaint", ""))
        cur_past = self._tokenize(collected_info.get("past_history", ""))
        cur_age_bucket = self._bucket_age(collected_info.get("age", ""))
        cur_gender = str(collected_info.get("gender", "")).strip()

        scored = []
        for note in self.notes:
            summary = note.get("collected_info_summary", {}) or {}
            note_symptoms = set(note.get("symptoms", []) or summary.get("symptoms", []) or [])
            note_chief = self._tokenize(summary.get("chief_complaint", ""))
            note_past = self._tokenize(summary.get("past_history", ""))
            note_age_bucket = summary.get("age_bucket", "")
            note_gender = str(summary.get("gender", "")).strip()

            score = 0.0

            # 1) 症状 Jaccard
            if cur_symptoms and note_symptoms:
                inter = len(cur_symptoms & note_symptoms)
                union = len(cur_symptoms | note_symptoms)
                score += 0.40 * (inter / union if union else 0)

            # 2) 主诉重叠
            if cur_chief and note_chief:
                inter = len(cur_chief & note_chief)
                union = len(cur_chief | note_chief)
                score += 0.20 * (inter / union if union else 0)

            # 3) 既往史重叠
            if cur_past and note_past:
                inter = len(cur_past & note_past)
                union = len(cur_past | note_past)
                score += 0.15 * (inter / union if union else 0)

            # 4) 年龄段命中
            if cur_age_bucket and note_age_bucket and cur_age_bucket == note_age_bucket:
                score += 0.10

            # 5) 性别命中
            if cur_gender and note_gender and cur_gender == note_gender:
                score += 0.05

            # 6) 质量权重：成功范例优先；失败病例由渲染层作为教训展示。
            quality = note.get("metrics", {}).get("quality_score", 0.5)
            if note.get("memory_kind") == "failure_lesson" or quality < 0.5:
                score += 0.01
            elif quality >= 0.8:
                score += 0.10

            # 7) 时间衰减
            created_at = note.get("created_at", "")
            if created_at:
                try:
                    days_ago = (datetime.now() - datetime.fromisoformat(created_at)).days
                    time_decay = max(0.5, 1.0 - days_ago / 365)
                    score *= time_decay
                except (ValueError, TypeError):
                    pass

            if score > 0:
                scored.append((score, note))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:top_k]]


def _score_value(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0
