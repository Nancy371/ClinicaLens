export const STAGES = [
  ["connect_records", "获取病历", "把异常放回完整背景"],
  ["safety_triage", "安全分流", "先排除需要立即就医的情况"],
  ["confirm_records", "核对病历", "避免错误字段进入判断"],
  ["ready_for_assessment", "辅助判断", "理解证据与候选变化"],
  ["appointment_preparation", "挂号准备", "把理解转化为行动"],
  ["awaiting_doctor", "医生回传", "让结论接受真实验证"],
  ["medication_active", "复诊与用药", "执行医生制定的方案"],
];

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function formatDate(value, includeTime = true) {
  if (!value) return "待医生确认";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return escapeHtml(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "long", day: "numeric",
    ...(includeTime ? { hour: "2-digit", minute: "2-digit", hour12: false } : {}),
  }).format(date);
}

export function stageIndex(name) {
  if (name === "emergency") return 1;
  const index = STAGES.findIndex(([key]) => key === name);
  return index < 0 ? 0 : index;
}

export function sourceLabel(source = {}) {
  const type = source.type || "system";
  const map = {
    sandbox: "沙箱来源",
    sandbox_hospital: "沙箱医院回传",
    hospital: "医院回传",
    uploaded_document: "上传文档",
    user: "用户提供",
    system: "系统规则",
  };
  return `${map[type] || "已记录来源"} · ${source.label || source.locator || "来源已保留"}`;
}

export function verificationLabel(status) {
  return {
    imported: "待你确认",
    extracted: "待核对提取结果",
    user_confirmed: "用户已确认",
    doctor_confirmed: "医生已确认",
    hospital_confirmed: "医院签名确认",
    disputed: "患者已提出争议",
    needs_correction: "需要更正",
    unresolved: "尚未确认",
  }[status] || "状态待确认";
}

export function emptyState(title, copy, action = "") {
  return `<article class="empty-state"><span aria-hidden="true">○</span><h2>${escapeHtml(title)}</h2><p>${escapeHtml(copy)}</p>${action}</article>`;
}

export function journeySteps(journey) {
  const active = stageIndex(journey.current_stage);
  return `<ol class="journey-steps">${STAGES.map(([key, title, value], index) => {
    const done = index < active || (key === "medication_active" && journey.current_stage === key);
    const current = index === active;
    return `<li class="${done ? "is-done" : ""} ${current ? "is-current" : ""}"><span>${done ? "✓" : index + 1}</span><div><b>${escapeHtml(title)}</b><small>${escapeHtml(value)}</small></div></li>`;
  }).join("")}</ol>`;
}

export function boundaryPanel() {
  return `<aside class="boundary-panel"><b>医疗边界</b><p>AI 可提供候选方向、证据解释、就医准备和无剂量的治疗路径参考。确诊、检查医嘱、个体处方和剂量由医生负责。</p></aside>`;
}
