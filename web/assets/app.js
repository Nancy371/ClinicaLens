import { CareApi, ApiError } from "./api.js";
import { escapeHtml, formatDate, boundaryPanel } from "./ui.js";

const api = new CareApi();
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const HISTORY_FIELDS = ["conditions", "surgeries", "current_medications", "allergies", "family_history", "social_history"];
const HISTORY_LABELS = { conditions: "疾病史", surgeries: "手术史", current_medications: "当前用药", allergies: "过敏史", family_history: "家族史", social_history: "个人与暴露史" };
const NAV = {
  patient: [["consultation", "问诊", "问"], ["case", "病例", "历"], ["aftercare", "治后", "护"]],
  clinician: [["clinician-case", "病例", "历"], ["clinician-exams", "检查", "检"], ["diagnosis", "诊断", "诊"], ["clinician-treatment", "治疗", "治"]],
};

const state = {
  authenticated: false, user: null, previewRole: "patient", samples: {}, journey: null,
  selectedVersion: 4, pendingQuestion: "", pendingAction: null, busy: false, metrics: null, testReport: null,
};

function role() { return state.authenticated ? state.user?.role || "patient" : state.previewRole; }
function journey() { return state.journey || state.samples[role()] || null; }
function writable() { return state.authenticated && Boolean(state.journey); }
function list(items, empty = "暂无") { return items?.length ? `<ul>${items.map((item) => `<li>${escapeHtml(typeof item === "string" ? item : item.name || item.label || JSON.stringify(item))}</li>`).join("")}</ul>` : `<p class="muted-copy">${escapeHtml(empty)}</p>`; }

function toast(message, tone = "normal") {
  const element = $("#toast"); element.textContent = message; element.dataset.tone = tone; element.classList.add("is-visible");
  window.clearTimeout(toast.timer); toast.timer = window.setTimeout(() => element.classList.remove("is-visible"), 3200);
}
function openDialog(id) { const dialog = $(id); if (dialog && !dialog.open) dialog.showModal(); }
function closeDialog(id) { const dialog = $(id); if (dialog?.open) dialog.close(); }

function requireRole(expected, action = null) {
  if (!state.authenticated) { state.pendingAction = action; openDialog("#auth-dialog"); return false; }
  if (role() !== expected) { toast(`当前是${role() === "patient" ? "患者" : "医生"}账户，不能执行此操作。`, "error"); return false; }
  if (!state.journey && expected === "clinician") { toast("请先领取患者的一次性病例授权。", "error"); openDialog("#account-dialog"); return false; }
  return true;
}

function homeView() { return role() === "patient" ? "consultation" : "clinician-case"; }
function showView(name) {
  const allowed = new Set([...NAV[role()].map((item) => item[0]), "trust"]);
  const target = allowed.has(name) ? name : homeView();
  $$(".app-view").forEach((view) => view.classList.toggle("is-active", view.dataset.view === target));
  $$("#bottom-nav [data-go]").forEach((button) => button.classList.toggle("is-active", button.dataset.go === target));
  window.history.replaceState(null, "", `#${target}`);
  if (target === "trust") renderTrust();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderChrome() {
  const currentRole = role();
  $("#bottom-nav").innerHTML = NAV[currentRole].map(([view, label, icon]) => `<button type="button" data-go="${view}"><span>${icon}</span><b>${label}</b></button>`).join("");
  $("#bottom-nav").dataset.columns = String(NAV[currentRole].length);
  $("#brand-subtitle").textContent = currentRole === "patient" ? "患者健康旅程" : "临床决策支持";
  $("#role-switch").hidden = state.authenticated;
  $$('[data-preview-role]').forEach((button) => button.classList.toggle("is-active", button.dataset.previewRole === currentRole));
  $("#connection-state").dataset.state = state.authenticated ? "connected" : "sample";
  $("#connection-state span").textContent = state.authenticated ? `${currentRole === "patient" ? "患者" : "医生"}账户` : "虚构样例";
  $("#profile-button span").textContent = state.authenticated ? (currentRole === "patient" ? "患" : "医") : "访";
  $("#account-phone").textContent = state.authenticated ? `${state.user.phone_masked} · ${currentRole === "patient" ? "患者账户" : "医生账户（服务端认证）"}` : "尚未登录；可切换查看两个角色的完整虚构样例。";
  $("#access-action-title").textContent = currentRole === "patient" ? "生成医生访问授权" : "领取患者病例";
  $("#access-action-copy").textContent = currentRole === "patient" ? "十分钟有效、单次使用，可随时撤销" : "输入患者提供的 8 位一次性授权码";
  $("#export-data").hidden = currentRole !== "patient";
}

function previewBanner() {
  if (writable()) return "";
  return `<div class="fictional-banner"><b>完整虚构病例</b><span>${state.authenticated ? "当前医生账户尚无授权病例，下面仅展示公开样例；所有写操作已关闭。" : "未登录预览，可浏览全部功能；写操作会要求对应角色登录。"}</span></div>`;
}

function patientExplanation(explanation) {
  if (!explanation) return `<section class="plain-explanation"><h2>还没有 AI 通俗解读</h2><p>医院结果进入并完成辅助判断后，这里会解释发生了什么、为什么重要和下一步做什么。</p></section>`;
  return `<section class="plain-explanation">
    <div class="panel-heading"><div><p class="eyebrow">PATIENT EXPLANATION · v${escapeHtml(explanation.assessment_version)}</p><h2>${escapeHtml(explanation.headline)}</h2></div><span>${escapeHtml(explanation.doctor_confirmation?.label)}</span></div>
    <p class="explanation-lead">${escapeHtml(explanation.summary)}</p>
    <div class="explanation-columns"><div><h3>最关键的三条依据</h3>${list(explanation.key_evidence)}</div><div><h3>哪些反证降低了其他方向</h3>${list(explanation.contradictions)}</div><div><h3>还缺什么</h3>${list(explanation.missing_information)}</div></div>
    <div class="danger-strip"><b>仍需留意的危险情况</b>${list((explanation.dangerous_conditions || []).map((item) => `${item.name}：${item.action}`))}</div>
    <p class="next-action"><b>现在最需要做：</b>${escapeHtml(explanation.next_action)}</p><small>${escapeHtml(explanation.boundary)}</small>
  </section>`;
}

function consultationMessage(message) {
  if (message.kind === "assessment_update") {
    const explanation = message.patient_explanation;
    return `<article class="chat-message"><span>AI</span><div class="consult-answer assessment-update"><div class="answer-heading"><span>检查结果更新 · v${escapeHtml(explanation?.assessment_version)}</span><b>${escapeHtml(explanation?.headline)}</b></div><p>${escapeHtml(explanation?.summary)}</p><p><b>下一步：</b>${escapeHtml(explanation?.next_action)}</p><button type="button" class="text-link" data-go="case">在病例中查看全部依据</button></div></article>`;
  }
  if (message.role === "user") return `<article class="chat-message is-user"><span>我</span><p>${escapeHtml(message.text)}</p></article>`;
  const answer = message.answer || {};
  return `<article class="chat-message"><span>AI</span><div class="consult-answer ${answer.urgency === "emergency" ? "is-emergency" : ""}"><div class="answer-heading"><span>${answer.urgency === "emergency" ? "立即行动" : "直接回答"}</span><b>${escapeHtml(answer.direct_answer)}</b></div><div class="answer-grid"><div><dt>为什么</dt><dd>${escapeHtml(answer.basis)}</dd></div><div><dt>下一步</dt><dd>${escapeHtml(answer.next_action)}</dd></div></div><details><summary>医生可能继续问什么</summary>${list(answer.follow_up_questions)}</details><p class="answer-boundary">${escapeHtml(answer.boundary)}</p></div></article>`;
}

function renderConsultation() {
  const item = journey(); if (!item) return;
  const messages = item.consultation?.messages || [];
  $("#consultation-content").innerHTML = `${previewBanner()}<div class="consult-hero"><div><p class="eyebrow">CONSULTATION</p><h1>你现在最担心什么？</h1><p>先判断是否需要急诊，再帮助你选科室、整理医生会问的问题和就医材料。</p></div><span class="source-badge">规则分流优先</span></div>
    <section class="consult-shell"><div class="chat-thread">${messages.length ? messages.slice(-6).map(consultationMessage).join("") : `<article class="chat-message"><span>AI</span><div class="consult-intro"><h2>你可以直接问：“我胸口痛，这严重吗？”</h2><p>系统会先问危险信号；没有完成安全分流前，不会进入诊断解释。</p></div></article>`}</div>
    <div class="quick-questions">${(item.consultation?.quick_questions || []).map((question) => `<button type="button" data-question="${escapeHtml(question)}">${escapeHtml(question)}</button>`).join("")}</div>
    <form id="consultation-form" class="consult-composer"><textarea id="consultation-input" placeholder="例如：需要现在去急诊，还是明天挂门诊？" maxlength="500" required></textarea><button class="button button-primary" type="submit">发送</button></form><p class="consult-disclaimer">回答用于安全分流和就医导航，不替代医生诊断。</p></section>`;
}

function rawCaseDocument(item, compact = false) {
  const document = item.raw_case_document || {};
  return `<div class="case-layout ${compact ? "is-compact" : ""}"><nav class="case-nav"><b>病历章节</b>${(document.sections || []).map((section) => `<a href="#raw-${section.id}">${escapeHtml(section.title)}</a>`).join("")}</nav><article class="raw-document"><header><div><p class="eyebrow">ORIGINAL RECORD</p><h2>${escapeHtml(document.title || "完整病历原文")}</h2></div><span>原文不随 AI 分析改写</span></header>${(document.sections || []).map((section) => `<section id="raw-${section.id}"><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.text)}</p></section>`).join("")}</article></div>`;
}

function historySummary(value) {
  if (Array.isArray(value)) return value.map((item) => typeof item === "string" ? item : Object.values(item).filter(Boolean).join(" · ")).join("；") || "未记录";
  if (value && typeof value === "object") return Object.values(value).filter(Boolean).join("；") || "未记录";
  return value || "未记录";
}
function historyPanel(item, editable = false) {
  const history = item.clinical_history || {};
  return `<section class="history-confirmation"><div class="panel-heading"><div><p class="eyebrow">CLINICAL HISTORY</p><h2>疾病史、用药史与过敏史</h2></div><span>${history.confirmation_status === "confirmed" ? "已逐项确认" : "需要逐项确认"}</span></div><div class="history-grid">${HISTORY_FIELDS.map((key) => `<article><header><b>${HISTORY_LABELS[key]}</b>${editable ? `<select data-history-status="${key}"><option value="unconfirmed" ${history.field_statuses?.[key] === "unconfirmed" ? "selected" : ""}>未确认</option><option value="confirmed" ${history.field_statuses?.[key] === "confirmed" ? "selected" : ""}>与原文一致</option><option value="unknown" ${history.field_statuses?.[key] === "unknown" ? "selected" : ""}>不了解</option></select>` : `<span>${escapeHtml(history.field_statuses?.[key] || "未确认")}</span>`}</header><p>${escapeHtml(historySummary(history[key]))}</p></article>`).join("")}</div>${editable ? `<button class="button button-primary" type="button" data-action="save-history">保存病史确认</button>` : ""}</section>`;
}

function reportStatus(status) {
  return { hospital_confirmed: "医院签名", user_confirmed: "用户已确认", disputed: "患者争议", extracted: "待对照原文" }[status] || status || "待确认";
}
function renderExamReports(reports, { clinician = false, dispute = false } = {}) {
  if (!reports?.length) return `<section class="report-empty"><h2>医院暂未返回检查报告</h2><p>请先尝试重新同步；同步失败或缺失报告时再使用上传兜底。</p></section>`;
  return `<div class="report-ledger">${reports.map((report) => `<section class="report-block" data-status="${escapeHtml(report.verification_status)}"><header><div><p>${escapeHtml(report.hospital)} · ${formatDate(report.observed_at)}</p><h3>${escapeHtml(report.title)}</h3><small>报告号 ${escapeHtml(report.report_no)} · ${escapeHtml(report.source?.label)}</small></div><div><span>${reportStatus(report.verification_status)}</span>${dispute && report.verification_status !== "disputed" ? `<button class="text-link" type="button" data-dispute-report="${escapeHtml(report.id)}">结果可能有误</button>` : ""}</div></header>
      <table class="exam-table"><thead><tr><th>指标或发现</th><th>结果</th><th>医院参考范围</th><th>状态</th><th>${clinician ? "证据角色 / 诊断影响" : "这个指标是什么意思"}</th><th>${clinician ? "原始报告定位" : "对当前诊断的影响 / 来源"}</th></tr></thead><tbody>${(report.observations || []).map((observation) => `<tr><th data-label="指标或发现">${escapeHtml(observation.name)}</th><td data-label="结果"><b>${escapeHtml(observation.value)}</b> ${escapeHtml(observation.unit)}</td><td data-label="医院参考范围">${escapeHtml(observation.reference_range_display || "原报告未提供")}</td><td data-label="状态"><span class="result-status" data-status="${escapeHtml(observation.interpretation_status)}">${escapeHtml(observation.interpretation_status)}</span></td><td data-label="${clinician ? "证据角色 / 诊断影响" : "这个指标是什么意思"}">${clinician ? `<b>${escapeHtml(observation.evidence_role)}</b><br>${escapeHtml(observation.diagnostic_impact)}` : escapeHtml(observation.patient_explanation || "暂无经过审核的通俗解释，请咨询医生。")}</td><td data-label="${clinician ? "原始报告定位" : "对当前诊断的影响 / 来源"}">${clinician ? escapeHtml(observation.source_locator) : `${escapeHtml(observation.diagnostic_impact)}<small>${escapeHtml(observation.source_locator)} · ${escapeHtml(report.source?.label)}</small>`}</td></tr>`).join("")}</tbody></table>${report.dispute ? `<p class="report-dispute">患者争议：${escapeHtml(report.dispute.reason)} · 已退出当前证据集</p>` : ""}</section>`).join("")}</div>`;
}

function renderPatientCase() {
  const item = journey(); if (!item) return;
  const reports = item.exam_reports || [];
  const latest = (item.patient_explanations || []).at(-1);
  const syncFailed = item.hospital_sync_status === "failed";
  $("#case-content").innerHTML = `${previewBanner()}<div class="fictional-banner"><b>病例原文</b><span>${escapeHtml(item.raw_case_document?.notice || "完整虚构病例，仅用于功能体验。")}</span></div>${rawCaseDocument(item)}${historyPanel(item, writable() && role() === "patient")}
    <section class="exam-report-section"><div class="sync-toolbar"><div><p class="eyebrow">HOSPITAL CONNECTOR</p><h2>检查结果</h2><p>${escapeHtml(item.hospital_connection?.display_name || "尚未连接医院")} · ${item.hospital_sync_status === "completed" ? "同步完成" : syncFailed ? "同步失败" : "等待同步"} · 最后同步 ${item.last_hospital_sync_at ? formatDate(item.last_hospital_sync_at) : "—"}</p></div><div><button class="button button-secondary" type="button" data-action="sync-records">重新同步医院</button>${syncFailed || !reports.length ? `<button class="button button-ghost" type="button" data-action="upload-fallback">上传报告兜底</button>` : ""}</div></div>${renderExamReports(reports, { dispute: writable() && role() === "patient" })}</section>
    ${patientExplanation(latest)}${boundaryPanel()}`;
}

function renderClinicianCase() {
  const item = journey(); if (!item) return;
  const corrections = (item.timeline || []).filter((event) => ["clinical_history_updated", "exam_report_disputed"].includes(event.type));
  $("#clinician-case-content").innerHTML = `${previewBanner()}${rawCaseDocument(item, true)}${historyPanel(item, false)}<section class="timeline-panel"><div class="panel-heading"><div><p class="eyebrow">CONSULTATION TIMELINE</p><h2>问诊与患者纠错</h2></div><span>${(item.consultation?.messages || []).length} 条问诊消息</span></div><div class="clinical-timeline">${(item.consultation?.messages || []).map((message) => `<div><time>${formatDate(message.created_at)}</time><b>${message.kind === "assessment_update" ? "检查结果更新" : message.role === "user" ? "患者提问" : "安全分流回答"}</b><p>${escapeHtml(message.text || message.answer?.direct_answer || message.patient_explanation?.headline || "")}</p></div>`).join("") || `<p>暂无问诊记录</p>`}</div><h3>患者纠错与争议</h3>${list(corrections.map((event) => `${event.title}：${event.detail}`), "暂无纠错记录")}</section>`;
}

function recommendationRows(recommendations) {
  return recommendations.map((recommendation) => `<section class="recommendation-row" data-status="${escapeHtml(recommendation.status)}"><div class="recommendation-index"><b>${escapeHtml(recommendation.priority)}</b><small>${escapeHtml(recommendation.timing)}</small></div><div><h3>${escapeHtml(recommendation.clinical_question)}</h3><p><b>具体项目：</b>${escapeHtml((recommendation.items || []).join("、"))}</p><details><summary>查看条件、风险与诊断影响</summary><div class="recommendation-details"><div><b>前置条件</b>${list(recommendation.prerequisites)}</div><div><b>风险</b>${list(recommendation.risks)}</div><div><b>怎样改变诊断</b><p>${escapeHtml(recommendation.expected_impact)}</p></div></div></details>${recommendation.decision ? `<p class="decision-audit">${escapeHtml(recommendation.decision.action)} · ${escapeHtml(recommendation.decision.rationale)} · ${formatDate(recommendation.decision.decided_at)}</p>` : ""}</div><button class="button button-secondary" type="button" data-decision-kind="exam" data-decision-id="${escapeHtml(recommendation.id)}">${recommendation.status === "proposed" ? "处理建议" : "更新决策"}</button></section>`).join("");
}

function renderClinicianExams() {
  const item = journey(); if (!item) return;
  $("#clinician-exams-content").innerHTML = `${previewBanner()}<section class="professional-results"><div class="panel-heading"><div><p class="eyebrow">EXISTING RESULTS</p><h2>已有结果</h2></div><span>医院报告定位保留</span></div>${renderExamReports(item.exam_reports || [], { clinician: true })}</section><section class="recommendation-ledger"><div class="panel-heading"><div><p class="eyebrow">AI RECOMMENDED EXAMS</p><h2>AI 推荐检查</h2></div><span>${(item.exam_recommendations || []).length} 组 · 决策支持</span></div><p class="panel-copy">确认或修改后只创建带医生来源的沙箱检查医嘱；真实部署需由 HospitalConnector 对接院内医嘱。</p>${recommendationRows(item.exam_recommendations || [])}</section>`;
}

function differential(item, index) {
  return `<article class="differential-card" data-trend="${escapeHtml(item.trend)}"><header><span>${index + 1}</span><div><h3>${escapeHtml(item.name)}</h3><p>${escapeHtml(item.reason)}</p></div><b>${escapeHtml(item.level)} · ${escapeHtml(item.trend)}</b></header><div class="evidence-columns"><div><small>支持</small>${list(item.supporting)}</div><div data-kind="against"><small>反证</small>${list(item.contradicting)}</div><div data-kind="gap"><small>未解决</small>${list(item.unresolved)}</div></div></article>`;
}

function renderDiagnosis() {
  const item = journey(); if (!item) return;
  const versions = item.assessment_versions || [];
  const selected = versions.find((entry) => Number(entry.version) === Number(state.selectedVersion)) || versions.at(-1);
  if (selected) state.selectedVersion = selected.version;
  $("#diagnosis-content").innerHTML = `${previewBanner()}<section class="diagnosis-sequence"><div class="panel-heading"><div><p class="eyebrow">VERSIONED REASONING</p><h2>新检查是否改变 primary diagnosis？</h2></div><button class="button button-secondary" type="button" data-action="rerun-diagnosis">重新运行当前证据</button></div><div class="sequence-line">${versions.map((version) => `<button type="button" data-version="${version.version}" class="${selected?.version === version.version ? "is-active" : ""}"><span>v${version.version}</span><b>${escapeHtml(version.primary_diagnosis?.name)}</b><small>${version.change_from_previous?.changed ? "主诊断已改变" : "方向延续"}</small></button>`).join("")}</div></section>${selected ? `<section class="assessment-panel"><div class="assessment-label"><span>AI 决策支持 · v${selected.version}</span><b>${escapeHtml(selected.primary_diagnosis?.status)}</b></div><div class="assessment-hero"><div><p>当前 primary diagnosis</p><h2>${escapeHtml(selected.primary_diagnosis?.name)}</h2><span>${escapeHtml(selected.urgency?.label)}</span></div><div class="uncertainty-meter"><small>不确定性</small><b>${escapeHtml(selected.uncertainty?.label)}</b></div></div><p class="reasoning-copy">${escapeHtml(selected.primary_diagnosis?.reasoning)}</p><article class="diagnosis-change ${selected.change_from_previous?.changed ? "has-change" : ""}"><span>${selected.change_from_previous?.changed ? "诊断发生变化" : "本版方向"}</span><p>${selected.change_from_previous?.previous ? `${escapeHtml(selected.change_from_previous.previous)} → ` : ""}<b>${escapeHtml(selected.change_from_previous?.current)}</b></p><small>触发证据：${escapeHtml(selected.change_from_previous?.why)}</small></article><div class="panel-heading"><div><p class="eyebrow">DIFFERENTIALS</p><h2>重要鉴别诊断、支持与反证</h2></div><span>不展示未经验证的精确概率</span></div><div class="differential-list">${(selected.differentials || []).map(differential).join("")}</div><div class="diagnosis-columns"><section><h2>有没有漏掉危险疾病？</h2>${(selected.dangerous_conditions || []).map((condition) => `<article class="danger-condition"><header><b>${escapeHtml(condition.name)}</b><span>${escapeHtml(condition.status)}</span></header><p>${escapeHtml(condition.evidence)}</p><small>${escapeHtml(condition.action)}</small></article>`).join("")}</section><section><h2>哪些检查还缺？</h2>${(selected.missing_exams || []).map((exam) => `<article class="missing-exam"><header><b>${escapeHtml(exam.name)}</b><span>${escapeHtml(exam.priority)}</span></header><p>${escapeHtml(exam.purpose)}</p><small>${escapeHtml(exam.status)}</small></article>`).join("")}</section></div><div class="limits-panel"><h3>不确定性与能力边界</h3>${list([...(selected.uncertainty?.gaps || []), ...(selected.limitations || [])])}</div></section>` : `<section class="report-empty"><h2>当前没有诊断版本</h2><p>医生获得患者授权并完成证据确认后才能重新运行。</p></section>`}`;
}

function guidelineLinks(items) { return `<div class="source-links">${(items || []).map((item) => `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.name)} ↗</a>`).join("")}</div>`; }
function renderClinicianTreatment() {
  const item = journey(); if (!item) return;
  const recommendation = (item.treatment_recommendations || [])[0];
  const doctor = item.doctor_plan;
  $("#clinician-treatment-content").innerHTML = `${previewBanner()}${recommendation ? `<section class="treatment-reference"><div class="panel-heading"><div><p class="eyebrow">AI TREATMENT PATH · 非处方</p><h2>${escapeHtml(recommendation.title)}</h2></div><span>${escapeHtml(recommendation.status)}</span></div><p>${escapeHtml(recommendation.boundary)}</p><div class="reference-columns"><div><h3>治疗目标</h3>${list(recommendation.goals)}</div><div><h3>采用前确认</h3>${list(recommendation.prerequisites)}</div><div><h3>主要风险</h3>${list(recommendation.risks)}</div><div><h3>监测</h3>${list(recommendation.monitoring)}</div></div>${guidelineLinks(recommendation.guidelines)}${recommendation.decision ? `<p class="decision-audit">医生决策：${escapeHtml(recommendation.decision.action)} · ${escapeHtml(recommendation.decision.rationale)}</p>` : ""}<button class="button button-primary" type="button" data-decision-kind="treatment" data-decision-id="${escapeHtml(recommendation.id)}">确认、修改或拒绝路径</button></section>` : ""}<section class="doctor-comparison"><div class="panel-heading"><div><p class="eyebrow">CURRENT DOCTOR PLAN</p><h2>当前医生方案</h2></div><span>${doctor ? "医生来源" : "尚未回传"}</span></div>${doctor ? `<h3>${escapeHtml((doctor.diagnoses || []).join("、"))}</h3><p>${escapeHtml(doctor.care_summary)}</p><h3>具体治疗记录（只读）</h3>${list((doctor.treatments || []).map((entry) => `${entry.name} · ${entry.route} · ${entry.schedule}`))}` : `<p>AI 路径决策不会创建处方。具体药名、剂量和疗程仍需医院回传或医生处方。</p>`}</section>`;
}

function educationPanel(education) {
  if (!education || education.review_status === "missing") return `<p class="education-missing">${escapeHtml(education?.notice || "暂无经过审核的药物说明，请咨询医生或药师。")}</p>`;
  return `<details class="medication-education"><summary>查看详细用药说明</summary><div class="education-grid"><div><h4>治疗目的</h4><p>${escapeHtml(education.purpose)}</p></div><div><h4>常见情况</h4>${list(education.common_effects)}</div><div class="is-warning"><h4>立即就医警示</h4>${list(education.urgent_warnings)}</div><div><h4>重要相互作用</h4>${list(education.interactions)}</div><div><h4>监测任务</h4>${list(education.monitoring)}</div><div><h4>漏服处理</h4><p>${escapeHtml(education.missed_dose)}</p></div></div><small>${escapeHtml(education.knowledge_source)} · ${escapeHtml(education.reviewed_on)}</small></details>`;
}
function renderAftercare() {
  const item = journey(); if (!item) return; const doctor = item.doctor_plan; const medications = item.medications || [];
  $("#aftercare-content").innerHTML = `${previewBanner()}${item.confirmed_treatment_direction ? `<section class="confirmed-direction"><p class="eyebrow">CLINICIAN CONFIRMED DIRECTION</p><h2>${escapeHtml(item.confirmed_treatment_direction.title)}</h2><p>${escapeHtml(item.confirmed_treatment_direction.rationale)}</p><small>${escapeHtml(item.confirmed_treatment_direction.boundary)}</small></section>` : ""}${doctor ? `<section class="doctor-comparison"><div class="panel-heading"><div><p class="eyebrow">DOCTOR RESULT</p><h2>AI 解读与医生结论并列</h2></div><span>${escapeHtml(doctor.source?.label)}</span></div><div class="comparison-grid"><article><small>此前 AI 通俗解读</small><h3>${escapeHtml((item.patient_explanations || []).at(-1)?.headline || "尚未形成")}</h3></article><article class="is-doctor"><small>医生最终结论</small><h3>${escapeHtml((doctor.diagnoses || []).join("、"))}</h3><p>${escapeHtml(doctor.care_summary)}</p></article></div></section>` : `<section class="report-empty"><h2>等待医生方案回传</h2><p>AI 治疗路径参考不会创建药物任务。具体药物、剂量、复诊和监测只能来自医生记录。</p><button class="button button-primary" type="button" data-action="load-doctor-plan">${writable() ? "同步虚构医生出院记录" : "登录患者账户后体验回传"}</button></section>`}${item.followups?.length ? `<section class="followup-panel"><div class="panel-heading"><div><p class="eyebrow">FOLLOW-UP</p><h2>复诊计划</h2></div><span>医生来源</span></div>${item.followups.map((followup) => `<article><b>${formatDate(followup.scheduled_at)}</b><div><h3>${escapeHtml(followup.title)}</h3><p>${escapeHtml(followup.source?.label)}</p></div><span>${escapeHtml(followup.status)}</span></article>`).join("")}</section>` : ""}<section class="medication-list"><div class="panel-heading"><div><p class="eyebrow">MEDICATION MANAGEMENT</p><h2>药物说明与执行记录</h2></div><span>${medications.length} 项医生处方</span></div>${medications.map((medication) => `<article class="medication-card"><header><div><span>${escapeHtml(medication.route)} · ${escapeHtml(medication.purpose)}</span><h3>${escapeHtml(medication.name)}</h3></div><b>医生处方</b></header><blockquote>${escapeHtml(medication.prescription_original)}</blockquote><dl><div><dt>剂量</dt><dd>${escapeHtml(medication.dose)}</dd></div><div><dt>时间/频次</dt><dd>${escapeHtml(medication.frequency)}</dd></div><div><dt>疗程</dt><dd>${escapeHtml(medication.course)}</dd></div></dl>${educationPanel(medication.education)}<div class="medication-actions"><button class="button button-secondary" data-medication="${escapeHtml(medication.id)}" data-event="taken">已执行</button><button class="button button-ghost" data-medication="${escapeHtml(medication.id)}" data-event="missed">漏服/错过</button><button class="button button-ghost" data-medication="${escapeHtml(medication.id)}" data-event="adverse">记录不良反应</button></div><small>${escapeHtml(medication.boundary)}</small></article>`).join("") || `<p class="muted-copy">医生处方回传后才会出现具体药物与剂量。</p>`}</section>`;
}

function metricValue(value) { return typeof value === "number" ? `${Math.round(value * 100)}%` : "未生成"; }
async function renderTrust() {
  if (!state.metrics) try { state.metrics = await api.metrics(); } catch { state.metrics = null; }
  if (!state.testReport) try { state.testReport = await fetch("/data/test-report.json", { cache: "no-store" }).then((response) => response.ok ? response.json() : null); } catch { state.testReport = null; }
  const metrics = state.metrics || {}, report = state.testReport;
  $("#trust-content").innerHTML = `<section class="trust-source"><span>数据来源</span><b>${escapeHtml(metrics.dataset || "尚无回放报告")} · n=${escapeHtml(metrics.cases || 0)}</b><p>${escapeHtml(metrics.method || "未生成")}</p><p>${escapeHtml(metrics.disclaimer || "该结果不代表临床准确率。")}</p></section><div class="metric-grid"><article><small>Recall@5</small><b>${metricValue(metrics.candidate_recall_at_5)}</b><p>固定回归样本中，正确方向是否进入前5</p></article><article><small>Top-1</small><b>${metricValue(metrics.top1_accuracy)}</b><p>固定样本首位命中</p></article><article><small>Exact Match</small><b>${metricValue(metrics.exact_match_rate)}</b><p>标准化结果完全匹配</p></article><article><small>反证误判率</small><b>${metricValue(metrics.negation_false_positive_rate)}</b><p>阴性证据被误当阳性</p></article></div><section class="test-report" data-status="${report?.status || "unverified"}"><div><p class="eyebrow">CURRENT BUILD</p><h2>${report?.status === "passed" ? "当前构建已验证" : "当前构建未验证"}</h2><p>${escapeHtml(report?.source || "没有成功测试产物")}</p></div><div><strong>${report?.status === "passed" ? `${report.passed} tests passed` : "—"}</strong><span>${escapeHtml(report?.commit || "无提交版本")} · ${escapeHtml(report?.generated_on || "无生成时间")}</span></div></section><section class="trust-boundaries"><h2>正确理解这些数字</h2>${list(["样本量固定为 n=7，来自仓库确定性回归样本。", "不调用真实患者数据，也不调用 LLM。", "指标用于发现代码回归，不代表临床准确率或真实世界泛化能力。", "测试数只读取当前成功构建产物，不在页面硬编码。"])} </section>`;
}

function renderAll() { renderChrome(); renderConsultation(); renderPatientCase(); renderAftercare(); renderClinicianCase(); renderClinicianExams(); renderDiagnosis(); renderClinicianTreatment(); }
async function reloadJourney() {
  if (!state.authenticated || !state.journey?.id) return;
  state.journey = role() === "patient" ? await api.getJourney(state.journey.id) : await api.clinicianJourney(state.journey.id);
  state.selectedVersion = state.journey.assessment_versions?.at(-1)?.version || state.selectedVersion; renderAll();
}
async function runAction(task) { if (state.busy) return; state.busy = true; try { await task(); } catch (error) { toast(error instanceof ApiError ? error.message : "操作失败，请稍后重试。", "error"); } finally { state.busy = false; } }

function beginQuestion(message) { const clean = String(message || "").trim(); if (!clean) return toast("请先写下你最担心的问题。", "error"); if (!requireRole("patient", () => beginQuestion(clean))) return; state.pendingQuestion = clean; openDialog("#triage-dialog"); }
async function saveHistory() { if (!requireRole("patient", saveHistory)) return; const history = state.journey.clinical_history || {}; const field_statuses = Object.fromEntries(HISTORY_FIELDS.map((key) => [key, $(`[data-history-status="${key}"]`)?.value || "unconfirmed"])); await api.updateHistory(state.journey.id, { ...history, field_statuses }); await reloadJourney(); toast("病史确认状态已保存。", "success"); }
async function syncHospital() { if (!requireRole("patient", syncHospital)) return; if (!state.journey.hospital_connection) await api.connectHospital(); await api.syncRecords(); await reloadJourney(); toast("已从明确标记的沙箱医院同步检查结果。", "success"); }
async function loadDoctorPlan() { if (!requireRole("patient", loadDoctorPlan)) return; const sample = state.samples.patient; const source = sample.doctor_plan; const result = await api.doctorDocument(state.journey.id, { source_type: "sandbox_hospital", diagnoses: source.diagnoses, care_summary: source.care_summary, examination_orders: source.examination_orders, treatments: source.treatments, confirmed_evidence: source.comparison?.confirmed_evidence || [], revisions: source.comparison?.revisions || [], followup_at: sample.followups?.[0]?.scheduled_at, prescriptions: sample.medications.map((item) => ({ name: item.name, dose: item.dose, frequency: item.frequency, course: item.course, route: item.route, purpose: item.purpose, prescription_original: item.prescription_original, next_at: item.next_at })) }); await reloadJourney(); toast(`已回传医生记录并创建 ${result.medications.length} 项医生来源任务。`, "success"); }

async function initialize() {
  try { [state.samples.patient, state.samples.clinician] = await Promise.all([api.sampleJourney("patient"), api.sampleJourney("clinician")]); state.selectedVersion = state.samples.clinician.assessment_versions?.at(-1)?.version || 4; }
  catch (error) { $("#connection-state").dataset.state = "error"; $("#connection-state span").textContent = "样例不可用"; toast(error.message || "无法读取虚构病例。", "error"); return; }
  try { const session = await api.session(); state.authenticated = true; state.user = session.user; state.previewRole = session.user.role; state.journey = session.journeys?.[0] || null; }
  catch (error) { if (!(error instanceof ApiError) || error.status !== 401) toast(error.message, "error"); }
  renderAll(); showView((window.location.hash || `#${homeView()}`).slice(1));
}

document.addEventListener("submit", (event) => {
  if (event.target.id === "consultation-form") { event.preventDefault(); beginQuestion($("#consultation-input").value); }
  if (event.target.id === "triage-form") { event.preventDefault(); const signs = [...new FormData(event.target).getAll("danger_signs")]; closeDialog("#triage-dialog"); runAction(async () => { await api.consultation(state.journey.id, state.pendingQuestion, signs); state.pendingQuestion = ""; event.target.reset(); await reloadJourney(); showView("consultation"); }); }
  if (event.target.id === "auth-form") { event.preventDefault(); runAction(async () => { await api.verifyOtp($("#phone-input").value, $("#otp-input").value); const session = await api.session(); state.authenticated = true; state.user = session.user; state.previewRole = session.user.role; state.journey = session.journeys?.[0] || null; closeDialog("#auth-dialog"); renderAll(); showView(homeView()); toast(`已登录${role() === "patient" ? "患者" : "医生"}账户。`, "success"); const pending = state.pendingAction; state.pendingAction = null; if (pending) pending(); }); }
  if (event.target.id === "upload-form") { event.preventDefault(); if (!requireRole("patient")) return; const form = new FormData(event.target); closeDialog("#upload-dialog"); runAction(async () => { const result = await api.upload(form); toast(`${result.notice} 当前状态：待用户确认。`, "success"); }); }
  if (event.target.id === "decision-form") { event.preventDefault(); if (!requireRole("clinician")) return; const kind = $("#decision-kind").value, id = $("#decision-id").value, action = $("#decision-action").value, rationale = $("#decision-rationale").value.trim(), lines = $("#decision-edits").value.split(/\n/).map((item) => item.trim()).filter(Boolean); const payload = { action, rationale, edits: action === "modified" && lines.length ? (kind === "exam" ? { items: lines } : { pathways: lines.map((name) => ({ name })) }) : {} }; closeDialog("#decision-dialog"); runAction(async () => { if (kind === "exam") await api.decideExamRecommendation(state.journey.id, id, payload); else await api.decideTreatmentRecommendation(state.journey.id, id, payload); await reloadJourney(); toast("医生决策已记录并保留审计。", "success"); }); }
});

document.addEventListener("click", (event) => {
  const go = event.target.closest("[data-go]"); if (go) { event.preventDefault(); closeDialog("#account-dialog"); showView(go.dataset.go); return; }
  if (event.target.closest("[data-home]")) { event.preventDefault(); showView(homeView()); return; }
  const roleButton = event.target.closest("[data-preview-role]"); if (roleButton && !state.authenticated) { state.previewRole = roleButton.dataset.previewRole; state.journey = null; renderAll(); showView(homeView()); return; }
  const question = event.target.closest("[data-question]"); if (question) { beginQuestion(question.dataset.question); return; }
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "save-history") runAction(saveHistory);
  if (action === "sync-records") runAction(syncHospital);
  if (action === "upload-fallback") { if (requireRole("patient", () => openDialog("#upload-dialog"))) openDialog("#upload-dialog"); }
  if (action === "load-doctor-plan") runAction(loadDoctorPlan);
  if (action === "rerun-diagnosis") { if (!requireRole("clinician")) return; runAction(async () => { const result = await api.rerunClinicianAssessment(state.journey.id); await reloadJourney(); state.selectedVersion = result.assessment.version; renderDiagnosis(); toast(`已生成诊断版本 v${result.assessment.version}。`, "success"); }); }
  if (action === "care-access") { if (!state.authenticated) return openDialog("#auth-dialog"); if (role() === "patient") runAction(async () => { const grant = await api.createCareAccessGrant(state.journey.id); $("#access-inline").innerHTML = `<b class="grant-code">${escapeHtml(grant.code)}</b><small>10 分钟有效 · 单次使用</small>`; toast("一次性授权码已生成。", "success"); }); else $("#access-inline").innerHTML = `<label>8 位授权码<input id="grant-code-input" maxlength="8" autocomplete="off" /></label><button class="button button-primary" type="button" data-action="redeem-access">领取病例</button>`; }
  if (action === "redeem-access") { const code = $("#grant-code-input")?.value || ""; runAction(async () => { await api.redeemCareAccessGrant(code); const data = await api.clinicianJourneys(); state.journey = data.journeys?.[0] || null; closeDialog("#account-dialog"); renderAll(); showView("clinician-case"); toast("病例授权已领取。", "success"); }); }
  const dispute = event.target.closest("[data-dispute-report]"); if (dispute) { if (!requireRole("patient")) return; const reason = window.prompt("请说明哪一项结果可能有误（提交后该报告会退出当前证据集）：", "报告内容与医院原件不一致"); if (reason) runAction(async () => { await api.disputeExamReport(state.journey.id, dispute.dataset.disputeReport, reason); await reloadJourney(); toast("报告已标记争议，当前 AI 判断已撤回。", "success"); }); }
  const decision = event.target.closest("[data-decision-kind]"); if (decision) { if (!requireRole("clinician")) return; $("#decision-kind").value = decision.dataset.decisionKind; $("#decision-id").value = decision.dataset.decisionId; $("#decision-title").textContent = decision.dataset.decisionKind === "exam" ? "处理检查建议" : "处理治疗路径"; $("#decision-rationale").value = ""; $("#decision-edits").value = ""; openDialog("#decision-dialog"); }
  const version = event.target.closest("[data-version]")?.dataset.version; if (version) { state.selectedVersion = Number(version); renderDiagnosis(); }
  const medication = event.target.closest("[data-medication]"); if (medication) { if (!requireRole("patient")) return; if (!state.journey.medications?.some((item) => item.id === medication.dataset.medication)) return toast("请先完成医生记录回传。", "error"); runAction(async () => { await api.medicationEvent(medication.dataset.medication, medication.dataset.event, "患者在治后页记录"); await reloadJourney(); toast("已记录，不会修改医生处方。", "success"); }); }
  if (event.target.closest("#profile-button")) openDialog("#account-dialog");
  const close = event.target.closest("[data-close]"); if (close) closeDialog(`#${close.dataset.close}-dialog`);
});

$("#request-otp").addEventListener("click", () => runAction(async () => { const result = await api.requestOtp($("#phone-input").value); $("#otp-help").textContent = result.development_code ? `本地测试验证码：${result.development_code}` : "验证码已发送，请在 5 分钟内输入。"; if (result.development_code) $("#otp-input").value = result.development_code; }));
$("#sign-out").addEventListener("click", () => { if (!state.authenticated) return closeDialog("#account-dialog"); runAction(async () => { await api.logout(); state.authenticated = false; state.user = null; state.journey = null; state.previewRole = "patient"; api.csrf = ""; closeDialog("#account-dialog"); renderAll(); showView("consultation"); toast("已退出，继续浏览完整虚构病例。", "success"); }); });
$("#export-data").addEventListener("click", (event) => { if (!state.authenticated || role() !== "patient") { event.preventDefault(); openDialog("#auth-dialog"); return; } event.currentTarget.href = api.exportUrl(); });
if ("serviceWorker" in navigator) window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
initialize();
