import { CareApi, ApiError } from "./api.js";
import { escapeHtml, formatDate, boundaryPanel } from "./ui.js";

const api = new CareApi();
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const HISTORY_FIELDS = ["conditions", "surgeries", "current_medications", "allergies", "family_history", "social_history"];
const HISTORY_LABELS = { conditions: "疾病史", surgeries: "手术史", current_medications: "当前用药", allergies: "过敏史", family_history: "家族史", social_history: "个人与暴露史" };
const NAV = {
  patient: [["consultation", "问诊", "问"], ["case", "我的情况", "况"], ["aftercare", "治后", "护"]],
  clinician: [["clinician-case", "病例", "历"], ["clinician-exams", "检查", "检"], ["diagnosis", "诊断", "诊"], ["clinician-treatment", "治疗", "治"]],
};

const state = {
  authenticated: false, user: null, previewRole: "patient", samples: {}, journey: null,
  selectedVersion: 4, pendingQuestion: "", pendingAction: null, busy: false, metrics: null, testReport: null, activeGrant: null,
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

function patientReasoningGraph(graph) {
  if (!graph?.nodes?.length) return `<p class="muted-copy">当前还没有形成可展示的判断路径。</p>`;
  const layers = [
    [["symptom", "history"], "我的症状和情况"],
    [["exam_result"], "关键检查结果"],
    [["evidence"], "AI 找到的共同模式"],
    [["hypothesis"], "仍需比较的方向"],
    [["diagnosis"], "当前判断"],
  ].map(([types, label]) => ({ label, nodes: graph.nodes.filter((node) => types.includes(node.type)) })).filter((layer) => layer.nodes.length);
  const labels = Object.fromEntries(graph.nodes.map((node) => [node.id, node.label]));
  return `<div class="patient-reasoning-graph" aria-label="从症状到当前判断的推理关系">${layers.map((layer, index) => `<div class="reasoning-layer"><small>${escapeHtml(layer.label)}</small><div>${layer.nodes.map((node) => `<article data-node-type="${escapeHtml(node.type)}"><b>${escapeHtml(node.label)}</b><p>${escapeHtml(node.plain_text)}</p></article>`).join("")}</div></div>${index < layers.length - 1 ? `<span class="reasoning-arrow" aria-hidden="true">↓</span>` : ""}`).join("")}<details class="reasoning-relations"><summary>查看信息之间的具体关系</summary><ul>${(graph.edges || []).map((edge) => `<li data-relation="${escapeHtml(edge.relation)}"><span>${escapeHtml(labels[edge.source] || edge.source)}</span><b>${escapeHtml(graph.legend?.[edge.relation] || edge.label)}</b><span>${escapeHtml(labels[edge.target] || edge.target)}</span><small>${escapeHtml(edge.label)}</small></li>`).join("")}</ul></details></div>`;
}

function professionalDetails(level) {
  const details = level?.terms || [];
  return `<details class="patient-professional-details"><summary>查看专业名词、原始数值和来源</summary><div class="professional-term-list">${details.map((item) => `<article><header><b>${escapeHtml(item.term)}</b><span>${escapeHtml(item.value)}</span></header><p>${escapeHtml(item.meaning)}</p><small>来源：${escapeHtml(item.source)}</small></article>`).join("") || `<p class="muted-copy">当前还没有专业检查详情。</p>`}</div><p>${escapeHtml(level?.notice || "专业信息用于和医生核对。")}</p></details>`;
}

function patientExplanation(explanation) {
  if (!explanation) return `<section class="plain-explanation"><h2>还没有 AI 通俗解读</h2><p>医院结果进入并完成辅助判断后，这里会解释发生了什么、为什么重要和下一步做什么。</p></section>`;
  const levels = explanation.language_levels || {};
  return `<section class="plain-explanation">
    <div class="panel-heading"><div><p class="eyebrow">WHY THIS ASSESSMENT · v${escapeHtml(explanation.assessment_version)}</p><h2>为什么 AI 这样判断</h2></div><span>${escapeHtml(explanation.doctor_confirmation?.label)}</span></div>
    <div class="language-level-one"><small>先看一句话</small><p>${escapeHtml(levels.level_1 || explanation.summary)}</p></div>
    <div class="language-level-two"><h3>判断是怎样一步步形成的</h3>${patientReasoningGraph(explanation.reasoning_graph)}<ol>${(levels.level_2 || []).map((step) => `<li>${escapeHtml(step)}</li>`).join("")}</ol></div>
    <div class="explanation-columns patient-evidence-groups"><div><h3>支持这个判断</h3>${list(explanation.key_evidence)}</div><div><h3>反对或降低其他判断</h3>${list(explanation.contradictions)}</div><div><h3>还没有确认</h3>${list(explanation.missing_information)}</div></div>
    ${professionalDetails(levels.level_3)}
    <div class="danger-strip"><b>仍需留意的危险情况与对应检查</b>${list((explanation.dangerous_conditions || []).map((item) => `${item.name}：${item.action}；对应检查：${(item.exams || []).slice(0, 3).join("、") || "由医生评估"}`))}</div>
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
  return `<article class="chat-message"><span>AI</span><div class="consult-answer ${answer.urgency === "emergency" ? "is-emergency" : ""}"><div class="answer-heading"><span>${answer.urgency === "emergency" ? "立即行动" : answer.intent === "intake" ? "继续补全" : "直接回答"}</span><b>${escapeHtml(answer.direct_answer)}</b></div><div class="answer-grid"><div><dt>为什么</dt><dd>${escapeHtml(answer.basis)}</dd></div><div><dt>下一步</dt><dd>${escapeHtml(answer.next_action)}</dd></div></div>${answer.intent !== "intake" && answer.follow_up_questions?.length ? `<details><summary>医生可能继续问什么</summary>${list(answer.follow_up_questions)}</details>` : ""}<p class="answer-boundary">${escapeHtml(answer.boundary)}</p></div></article>`;
}

function intakeProgress(intake) {
  const progress = intake?.progress || [];
  const completed = progress.filter((item) => item.complete);
  const remaining = progress.filter((item) => !item.complete);
  return `<section class="intake-progress"><div><small>当前已了解</small>${completed.length ? `<ul>${completed.map((item) => `<li>✓ ${escapeHtml(item.label)}</li>`).join("")}</ul>` : `<p>还在等待你的第一个问题。</p>`}</div><div><small>还需要</small>${remaining.length ? `<ul>${remaining.map((item) => `<li>○ ${escapeHtml(item.label)}</li>`).join("")}</ul>` : `<p>必要信息已补全。</p>`}</div></section>`;
}

function intakeSummary(intake) {
  const summary = intake?.summary || {};
  return `<section class="intake-summary"><div class="panel-heading"><div><p class="eyebrow">SUMMARY CONFIRMATION</p><h2>这是我目前理解的信息</h2></div><span>确认后才进入辅助判断</span></div><dl>${Object.entries(summary).map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl><div class="summary-actions"><button class="button button-primary" type="button" data-action="confirm-intake-summary">确认信息无误</button><button class="button button-ghost" type="button" data-action="correct-intake-summary">修改或补充</button></div></section>`;
}

function renderConsultation() {
  const item = journey(); if (!item) return;
  const messages = item.consultation?.messages || [];
  const intake = item.consultation_state || {};
  const pending = intake.pending_question;
  const awaitingSummary = intake.completion_status === "awaiting_summary_confirmation";
  const ready = intake.completion_status === "ready_for_assessment";
  const needsSafety = pending?.response_type === "safety_screen";
  $("#consultation-content").innerHTML = `${previewBanner()}<div class="consult-hero"><div><p class="eyebrow">CONSULTATION</p><h1>你现在最担心什么？</h1><p>先判断是否需要急诊，再帮助你选科室、整理医生会问的问题和就医材料。</p></div><span class="source-badge">规则分流优先</span></div>
    ${intakeProgress(intake)}<section class="consult-shell"><div class="chat-thread">${messages.length ? messages.slice(-8).map(consultationMessage).join("") : `<article class="chat-message"><span>AI</span><div class="consult-intro"><h2>你可以直接说：“我胸口痛，这严重吗？”</h2><p>我会记住你已经回答的信息，只追问仍缺少的部分。</p></div></article>`}</div>
    ${awaitingSummary ? intakeSummary(intake) : ready ? `<section class="intake-ready"><b>信息摘要已确认</b><p>后续新增症状仍会触发必要的安全复核；你也可以继续向我提问。</p><button class="button button-secondary" type="button" data-go="case">查看当前情况与检查</button></section>` : pending ? `<section class="pending-intake-question"><small>接下来需要了解</small><h2>${escapeHtml(pending.text)}</h2><p><b>为什么问：</b>${escapeHtml(pending.why)}</p>${needsSafety ? `<button class="button button-primary" type="button" data-action="open-safety-screen">完成危险信号确认</button>` : ""}</section>` : ""}
    ${!awaitingSummary && !needsSafety ? `<div class="quick-questions">${!intake.known_facts?.chief_complaint ? (item.consultation?.quick_questions || []).map((question) => `<button type="button" data-question="${escapeHtml(question)}">${escapeHtml(question)}</button>`).join("") : ""}</div><form id="consultation-form" class="consult-composer"><textarea id="consultation-input" placeholder="${escapeHtml(pending?.text || "继续提问或补充新的症状")}" maxlength="500" required></textarea><button class="button button-primary" type="submit">发送</button></form>` : ""}<p class="consult-disclaimer">回答用于安全分流、信息整理和辅助判断，不替代医生诊断。</p></section>`;
}

function rawCaseDocument(item, compact = false) {
  const document = item.raw_case_document || {};
  const sourceCount = (document.generated_from_message_ids || []).length;
  return `<div class="case-layout ${compact ? "is-compact" : ""}"><nav class="case-nav"><b>问诊病历章节</b>${(document.sections || []).map((section) => `<a href="#raw-${section.id}">${escapeHtml(section.title)}</a>`).join("")}</nav><article class="raw-document"><header><div><p class="eyebrow">CONSULTATION-GENERATED NOTE · v${escapeHtml(document.version || 1)}</p><h2>${escapeHtml(document.title || "问诊生成病例原文")}</h2></div><span>${document.status === "confirmed" ? "患者已确认" : "AI 整理草稿"}</span></header><p class="case-provenance">来自 ${sourceCount} 条患者问诊消息；不包含医院检查、AI 诊断或医生结论。</p>${(document.sections || []).map((section) => `<section id="raw-${section.id}"><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.text)}</p><small>问诊来源：${(section.source_message_ids || []).map((id) => escapeHtml(id)).join("、") || "已确认病史字段"}</small></section>`).join("")}${document.missing_information?.length ? `<div class="case-missing"><b>仍未补全</b>${list(document.missing_information)}</div>` : ""}</article></div>`;
}

function historySummary(value) {
  if (Array.isArray(value)) return value.map((item) => typeof item === "string" ? item : Object.values(item).filter(Boolean).join(" · ")).join("；") || "未记录";
  if (value && typeof value === "object") return Object.values(value).filter(Boolean).join("；") || "未记录";
  return value || "未记录";
}
function historyEditorValue(key, value) {
  if (key === "social_history") {
    const labels = { smoking: "吸烟", alcohol: "饮酒", occupation: "职业", exposures: "暴露" };
    return Object.entries(value || {}).map(([name, detail]) => `${labels[name] || name}｜${detail || ""}`).join("\n");
  }
  if (!Array.isArray(value)) return String(value || "");
  if (key === "current_medications") return value.map((item) => typeof item === "string" ? item : [item.name, item.dose, item.frequency].filter(Boolean).join("｜")).join("\n");
  if (key === "allergies") return value.map((item) => typeof item === "string" ? item : [item.allergen, item.reaction, item.severity, item.status].filter(Boolean).join("｜")).join("\n");
  if (["conditions", "surgeries"].includes(key)) return value.map((item) => typeof item === "string" ? item : [item.name, item.detail].filter(Boolean).join("｜")).join("\n");
  return value.map((item) => typeof item === "string" ? item : historySummary(item)).join("\n");
}
function parseHistoryEditorValue(key, text) {
  const rows = String(text || "").split(/\r?\n/).map((row) => row.trim()).filter(Boolean).slice(0, 20);
  if (key === "social_history") {
    const names = { "吸烟": "smoking", "饮酒": "alcohol", "职业": "occupation", "暴露": "exposures", smoking: "smoking", alcohol: "alcohol", occupation: "occupation", exposures: "exposures" };
    return Object.fromEntries(rows.map((row) => { const [label, ...rest] = row.split(/[｜|]/); return [names[label.trim()] || "exposures", rest.join("｜").trim() || label.trim()]; }));
  }
  if (key === "current_medications") return rows.map((row) => { const [name, dose = "待核对", frequency = "待核对"] = row.split(/[｜|]/).map((part) => part.trim()); return { name, dose, frequency, source: "患者修正" }; });
  if (key === "allergies") return rows.map((row) => { const [allergen, reaction = "待核对", severity = "unknown", status = "patient_reported"] = row.split(/[｜|]/).map((part) => part.trim()); return { allergen, reaction, severity, status }; });
  if (["conditions", "surgeries"].includes(key)) return rows.map((row) => { const [name, ...detail] = row.split(/[｜|]/).map((part) => part.trim()); return { name, detail: detail.join("｜") || "患者修正" }; });
  return rows;
}
function correctionValue(value) {
  if (typeof value === "string") return value;
  return historySummary(value);
}
function correctionHistory(items, { compact = false } = {}) {
  const corrections = [...(items || [])].sort((a, b) => String(b.timestamp || "").localeCompare(String(a.timestamp || "")));
  return `<section class="correction-history ${compact ? "is-compact" : ""}"><div class="panel-heading"><div><p class="eyebrow">INFORMATION CHANGES</p><h2>我提供和修正过的信息</h2></div><span>${corrections.length} 条可追溯修改</span></div>${corrections.length ? `<ol>${corrections.map((item) => `<li><time>${formatDate(item.timestamp)}</time><div><header><b>${escapeHtml(item.field_label || item.field)}</b><span>${escapeHtml(item.source?.label || "来源未标明")}</span></header><p><del>${escapeHtml(correctionValue(item.old_value))}</del><i>→</i><strong>${escapeHtml(correctionValue(item.new_value))}</strong></p>${item.reason ? `<small>修改原因：${escapeHtml(item.reason)}</small>` : ""}<small>${escapeHtml(item.impact || "已记录到后续评估。")}${item.affected_assessment_version ? ` · 影响 ${escapeHtml(item.affected_assessment_version)}` : ""}</small></div></li>`).join("")}</ol>` : `<p class="muted-copy">你修改病史、质疑检查结果或医生修正判断后，变化和影响会显示在这里。</p>`}</section>`;
}
function historyPanel(item, editable = false) {
  const history = item.clinical_history || {};
  return `<section class="history-confirmation"><div class="panel-heading"><div><p class="eyebrow">CLINICAL HISTORY</p><h2>疾病史、用药史与过敏史</h2></div><span>${history.confirmation_status === "confirmed" ? "已逐项确认" : "需要逐项确认"}</span></div><div class="history-grid">${HISTORY_FIELDS.map((key) => `<article><header><b>${HISTORY_LABELS[key]}</b>${editable ? `<select data-history-status="${key}"><option value="unconfirmed" ${history.field_statuses?.[key] === "unconfirmed" ? "selected" : ""}>未确认</option><option value="confirmed" ${history.field_statuses?.[key] === "confirmed" ? "selected" : ""}>与记录一致</option><option value="unknown" ${history.field_statuses?.[key] === "unknown" ? "selected" : ""}>不了解</option></select>` : `<span>${escapeHtml(history.field_statuses?.[key] || "未确认")}</span>`}</header><p>${escapeHtml(historySummary(history[key]))}</p>${editable ? `<button class="text-link" type="button" data-edit-history="${key}">修改这项信息</button>` : ""}</article>`).join("")}</div>${editable ? `<button class="button button-primary" type="button" data-action="save-history">保存病史确认</button>` : ""}</section>`;
}

function reportStatus(status) {
  return { hospital_confirmed: "医院签名", user_confirmed: "用户已确认", disputed: "患者争议", extracted: "待对照原文" }[status] || status || "待确认";
}
function renderExamReports(reports, { clinician = false, dispute = false } = {}) {
  if (!reports?.length) return `<section class="report-empty"><h2>医院暂未返回检查报告</h2><p>请先尝试重新同步；同步失败或缺失报告时再使用上传兜底。</p></section>`;
  return `<div class="report-ledger">${reports.map((report) => `<section class="report-block" data-status="${escapeHtml(report.verification_status)}"><header><div><p>${escapeHtml(report.hospital)} · ${formatDate(report.observed_at)}</p><h3>${escapeHtml(report.title)}</h3><small>报告号 ${escapeHtml(report.report_no)} · ${escapeHtml(report.source?.label)}</small></div><div><span>${reportStatus(report.verification_status)}</span>${dispute && report.verification_status !== "disputed" ? `<button class="text-link" type="button" data-dispute-report="${escapeHtml(report.id)}">结果可能有误</button>` : ""}</div></header>
      <table class="exam-table"><thead><tr><th>指标或发现</th><th>结果</th><th>医院参考范围</th><th>状态</th><th>${clinician ? "趋势 / 证据角色 / 版本" : "这个指标是什么意思"}</th><th>${clinician ? "诊断影响 / 原始报告定位" : "对当前诊断的影响 / 来源"}</th></tr></thead><tbody>${(report.observations || []).map((observation) => `<tr><th data-label="指标或发现">${escapeHtml(observation.name)}</th><td data-label="结果"><b>${escapeHtml(observation.value)}</b> ${escapeHtml(observation.unit)}</td><td data-label="医院参考范围">${escapeHtml(observation.reference_range_display || "原报告未提供")}</td><td data-label="状态"><span class="result-status" data-status="${escapeHtml(observation.interpretation_status)}">${escapeHtml(observation.interpretation_status)}</span></td><td data-label="${clinician ? "趋势 / 证据角色 / 版本" : "这个指标是什么意思"}">${clinician ? `${escapeHtml(observation.trend)}<br><b>${escapeHtml(observation.evidence_role)} · ${escapeHtml(observation.entered_assessment_version)}</b>` : escapeHtml(observation.patient_explanation || "暂无经过审核的通俗解释，请咨询医生。")}</td><td data-label="${clinician ? "诊断影响 / 原始报告定位" : "对当前诊断的影响 / 来源"}">${clinician ? `${escapeHtml(observation.diagnostic_impact)}<small>${escapeHtml(observation.source_locator)}</small>` : `${escapeHtml(observation.diagnostic_impact)}<small>${escapeHtml(observation.source_locator)} · ${escapeHtml(report.source?.label)}</small>`}</td></tr>`).join("")}</tbody></table>${report.dispute ? `<p class="report-dispute">患者争议：${escapeHtml(report.dispute.reason)} · 已退出当前证据集</p>` : ""}</section>`).join("")}</div>`;
}

function renderPatientCase() {
  const item = journey(); if (!item) return;
  const reports = item.exam_reports || [];
  const latest = (item.patient_explanations || []).at(-1);
  const syncFailed = item.hospital_sync_status === "failed";
  const needsFallback = syncFailed || reports.length < 7;
  const patientMessages = (item.consultation?.messages || []).filter((message) => message.role === "user" && message.text);
  const currentHeadline = latest?.headline || "还没有形成辅助判断";
  const currentSummary = latest?.summary || "完成问诊、病史确认和检查同步后，这里会说明目前可能是什么情况。";
  $("#case-content").innerHTML = `${previewBanner()}<section class="patient-overview"><div><p class="eyebrow">CURRENT SITUATION</p><h2>我现在可能是什么情况</h2><strong>${escapeHtml(currentHeadline)}</strong><p>${escapeHtml(currentSummary)}</p></div><span>${escapeHtml(latest?.doctor_confirmation?.label || "等待更多信息")}</span></section>
    <section class="patient-information"><div class="panel-heading"><div><p class="eyebrow">WHAT I SHARED</p><h2>我告诉过 AI 什么</h2></div><span>${patientMessages.length} 条信息</span></div>${patientMessages.length ? `<ol class="patient-fact-list">${patientMessages.map((message) => `<li><time>${formatDate(message.created_at)}</time><p>${escapeHtml(message.text)}</p></li>`).join("")}</ol>` : `<p class="muted-copy">完成问诊后，你提供的信息会显示在这里。</p>`}</section>
    ${historyPanel(item, writable() && role() === "patient")}
    <section class="exam-report-section"><div class="sync-toolbar"><div><p class="eyebrow">HOSPITAL CONNECTOR</p><h2>检查结果</h2><p>${escapeHtml(item.hospital_connection?.display_name || "尚未连接医院")} · ${item.hospital_sync_status === "completed" ? "同步完成" : syncFailed ? "同步失败" : "等待同步"} · 最后同步 ${item.last_hospital_sync_at ? formatDate(item.last_hospital_sync_at) : "—"}</p></div><div><button class="button button-secondary" type="button" data-action="sync-records">重新同步医院</button>${needsFallback ? `<button class="button button-ghost" type="button" data-action="upload-fallback">上传报告兜底</button>` : ""}</div></div>${renderExamReports(reports, { dispute: writable() && role() === "patient" })}</section>
    ${patientExplanation(latest)}<section class="patient-next-action"><p class="eyebrow">NEXT ACTION</p><h2>我现在应该做什么</h2><p>${escapeHtml(latest?.next_action || "先完成问诊和病史确认，再根据已有信息安排下一步。")}</p></section>${correctionHistory(item.information_corrections)}${boundaryPanel()}`;
}

function renderClinicianCase() {
  const item = journey(); if (!item) return;
  $("#clinician-case-content").innerHTML = `${previewBanner()}${rawCaseDocument(item, true)}${historyPanel(item, false)}<section class="timeline-panel"><div class="panel-heading"><div><p class="eyebrow">CONSULTATION TIMELINE</p><h2>问诊记录</h2></div><span>${(item.consultation?.messages || []).length} 条问诊消息</span></div><div class="clinical-timeline">${(item.consultation?.messages || []).map((message) => `<div><time>${formatDate(message.created_at)}</time><b>${message.kind === "assessment_update" ? "检查结果更新" : message.role === "user" ? "患者提问" : "安全分流回答"}</b><p>${escapeHtml(message.text || message.answer?.direct_answer || message.patient_explanation?.headline || "")}</p></div>`).join("") || `<p>暂无问诊记录</p>`}</div></section>${correctionHistory(item.information_corrections, { compact: true })}`;
}

function recommendationRows(recommendations) {
  return recommendations.map((recommendation) => `<section class="recommendation-row" data-recommendation-id="${escapeHtml(recommendation.id)}" data-status="${escapeHtml(recommendation.status)}"><div class="recommendation-index"><b>${escapeHtml(recommendation.priority)}</b><small>${escapeHtml(recommendation.timing)}</small></div><div><h3>${escapeHtml(recommendation.clinical_question)}</h3><p><b>具体项目：</b>${escapeHtml((recommendation.items || []).join("、"))}</p><details><summary>查看条件、风险与诊断影响</summary><div class="recommendation-details"><div><b>前置条件</b>${list(recommendation.prerequisites)}</div><div><b>风险</b>${list(recommendation.risks)}</div><div><b>怎样改变诊断</b><p>${escapeHtml(recommendation.expected_impact)}</p></div></div></details>${recommendation.decision ? `<p class="decision-audit">${escapeHtml(recommendation.decision.action)} · ${escapeHtml(recommendation.decision.rationale)} · ${formatDate(recommendation.decision.decided_at)}</p>` : ""}</div><button class="button button-secondary" type="button" data-decision-kind="exam" data-decision-id="${escapeHtml(recommendation.id)}">${recommendation.status === "proposed" ? "处理建议" : "更新决策"}</button></section>`).join("");
}

function renderClinicianExams() {
  const item = journey(); if (!item) return;
  $("#clinician-exams-content").innerHTML = `${previewBanner()}<section class="professional-results"><div class="panel-heading"><div><p class="eyebrow">EXISTING RESULTS</p><h2>已有结果</h2></div><span>医院报告定位保留</span></div>${renderExamReports(item.exam_reports || [], { clinician: true })}</section><section class="recommendation-ledger"><div class="panel-heading"><div><p class="eyebrow">AI RECOMMENDED EXAMS</p><h2>AI 推荐检查</h2></div><span>${(item.exam_recommendations || []).length} 组 · 决策支持</span></div><p class="panel-copy">确认或修改后只创建带医生来源的沙箱检查医嘱；真实部署需由 HospitalConnector 对接院内医嘱。</p>${recommendationRows(item.exam_recommendations || [])}</section>`;
}

function differential(item, index) {
  return `<article class="differential-card" data-trend="${escapeHtml(item.trend)}"><header><span>${index + 1}</span><div><h3>${escapeHtml(item.name)}</h3><p>${escapeHtml(item.reason)}</p></div><b>${escapeHtml(item.level)} · ${escapeHtml(item.trend)}</b></header><div class="evidence-columns"><div><small>支持</small>${list(item.supporting)}</div><div data-kind="against"><small>反证</small>${list(item.contradicting)}</div><div data-kind="gap"><small>未解决</small>${list(item.unresolved)}</div></div></article>`;
}

function safetyMatrix(items) {
  if (!items?.length) return `<p class="muted-copy">当前版本尚未生成疾病—检查对应关系。</p>`;
  return `<div class="safety-matrix">${items.map((condition) => `<article class="safety-row"><header><div><span>${escapeHtml(condition.risk_level)}</span><h3>${escapeHtml(condition.condition_name)}</h3></div><b>${escapeHtml(condition.current_status)}</b></header><p>${escapeHtml(condition.why_it_might_be_missed)}</p><div class="safety-grid"><div><small>当前证据</small>${list(condition.supporting_evidence, "尚无足够证据")}</div><div><small>用于排查的检查</small>${list(condition.exam_items)}</div><div><small>结果怎样改变判断</small><p>${escapeHtml(condition.expected_result_effect)}</p></div></div><div class="exam-link-tags">${(condition.exam_links || []).map((link) => `<button type="button" data-go="clinician-exams" data-exam-link="${escapeHtml(link.recommendation_id)}">查看对应检查 · ${escapeHtml(link.recommendation_id)}</button>`).join("")}</div></article>`).join("")}</div>`;
}

function renderDiagnosis() {
  const item = journey(); if (!item) return;
  const versions = item.assessment_versions || [];
  const selected = versions.find((entry) => Number(entry.version) === Number(state.selectedVersion)) || versions.at(-1);
  if (selected) state.selectedVersion = selected.version;
  $("#diagnosis-content").innerHTML = `${previewBanner()}<section class="diagnosis-sequence"><div class="panel-heading"><div><p class="eyebrow">VERSIONED REASONING</p><h2>新检查是否改变 primary diagnosis？</h2></div><button class="button button-secondary" type="button" data-action="rerun-diagnosis">重新运行当前证据</button></div><div class="sequence-line">${versions.map((version) => `<button type="button" data-version="${version.version}" class="${selected?.version === version.version ? "is-active" : ""}"><span>v${version.version}</span><b>${escapeHtml(version.primary_diagnosis?.name)}</b><small>${version.change_from_previous?.changed ? "主诊断已改变" : "方向延续"}</small></button>`).join("")}</div></section>${selected ? `<section class="assessment-panel"><div class="assessment-label"><span>AI 决策支持 · v${selected.version}</span><b>${escapeHtml(selected.primary_diagnosis?.status)}</b></div><div class="assessment-hero"><div><p>当前 primary diagnosis</p><h2>${escapeHtml(selected.primary_diagnosis?.name)}</h2><span>${escapeHtml(selected.urgency?.label)}</span></div><div class="uncertainty-meter"><small>不确定性</small><b>${escapeHtml(selected.uncertainty?.label)}</b></div></div><p class="reasoning-copy">${escapeHtml(selected.primary_diagnosis?.reasoning)}</p><article class="diagnosis-change ${selected.change_from_previous?.changed ? "has-change" : ""}"><span>${selected.change_from_previous?.changed ? "诊断发生变化" : "本版方向"}</span><p>${selected.change_from_previous?.previous ? `${escapeHtml(selected.change_from_previous.previous)} → ` : ""}<b>${escapeHtml(selected.change_from_previous?.current)}</b></p><small>触发证据：${escapeHtml(selected.change_from_previous?.why)}</small></article><div class="panel-heading"><div><p class="eyebrow">DIFFERENTIALS</p><h2>重要鉴别诊断、支持与反证</h2></div><span>不展示未经验证的精确概率</span></div><div class="differential-list">${(selected.differentials || []).map(differential).join("")}</div><div class="panel-heading"><div><p class="eyebrow">SAFETY MATRIX</p><h2>可能漏掉的疾病，对应要做哪些检查？</h2></div><span>疾病与检查一一关联</span></div>${safetyMatrix(selected.safety_matrix)}<div class="limits-panel"><h3>不确定性与能力边界</h3>${list([...(selected.uncertainty?.gaps || []), ...(selected.limitations || [])])}</div></section>` : `<section class="report-empty"><h2>当前没有诊断版本</h2><p>医生获得患者授权并完成证据确认后才能重新运行。</p></section>`}`;
}

function guidelineLinks(items) { return `<div class="source-links">${(items || []).map((item) => `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.name)} ↗</a>`).join("")}</div>`; }
function prescriptionLines(items) { return (items || []).map((item) => [item.medication || item.name, item.dose, item.frequency, item.duration || item.course, item.route].map((value) => String(value || "").trim()).join("｜")).join("\n"); }
function doseOptionRows(items) {
  return `<div class="dose-option-list">${(items || []).map((item) => `<article><header><div><span>${escapeHtml(item.route)}</span><h3>${escapeHtml(item.medication)}</h3></div><b>${escapeHtml(item.dose)}</b></header><p>${escapeHtml(item.frequency)}</p><small>${escapeHtml(item.dose_source)} · ${escapeHtml(item.origin)}</small><details><summary>适用前提与分阶段剂量</summary>${list(item.requires)}${item.schedule?.length ? `<ol>${item.schedule.map((step) => `<li><b>${escapeHtml(step.period)}</b> ${escapeHtml(step.dose)}</li>`).join("")}</ol>` : ""}</details></article>`).join("")}</div>`;
}
function renderClinicianTreatment() {
  const item = journey(); if (!item) return;
  const recommendation = (item.treatment_recommendations || [])[0];
  const doctor = item.doctor_plan;
  const drafts = item.prescription_drafts || [];
  const activeDraft = [...drafts].reverse().find((draft) => draft.status === "draft");
  const signed = (item.signed_prescriptions || []).at(-1);
  $("#clinician-treatment-content").innerHTML = `${previewBanner()}${recommendation ? `<section class="treatment-reference"><div class="panel-heading"><div><p class="eyebrow">AI GUIDELINE PATH · 第一步</p><h2>${escapeHtml(recommendation.title)}</h2></div><span>${escapeHtml(recommendation.status)}</span></div><p>${escapeHtml(recommendation.boundary)}</p><div class="reference-columns"><div><h3>治疗目标</h3>${list(recommendation.goals)}</div><div><h3>采用前确认</h3>${list(recommendation.prerequisites)}</div><div><h3>主要风险</h3>${list(recommendation.risks)}</div><div><h3>监测</h3>${list(recommendation.monitoring)}</div></div><h3>AI 结构化剂量建议</h3>${doseOptionRows(recommendation.dose_options)}${list(recommendation.non_dosed_support, "暂无需医生补充的支持治疗")}${guidelineLinks(recommendation.guidelines)}${recommendation.decision ? `<p class="decision-audit">医生路径决策：${escapeHtml(recommendation.decision.action)} · ${escapeHtml(recommendation.decision.rationale)}</p>` : ""}<button class="button button-primary" type="button" data-decision-kind="treatment" data-decision-id="${escapeHtml(recommendation.id)}">确认、修改或拒绝路径</button></section>` : ""}${activeDraft ? `<section class="prescription-draft"><div class="panel-heading"><div><p class="eyebrow">PRESCRIPTION DRAFT · 第二步</p><h2>剂量草稿等待医生签署</h2></div><span>患者不可见</span></div><p>${escapeHtml(activeDraft.boundary)}</p>${doseOptionRows(activeDraft.items)}<div class="draft-actions"><button class="button button-primary" type="button" data-sign-draft="${escapeHtml(activeDraft.id)}">核对并签署</button><button class="button button-ghost" type="button" data-cancel-draft="${escapeHtml(activeDraft.id)}">取消草稿</button></div></section>` : ""}${signed ? `<section class="signed-prescription"><div class="panel-heading"><div><p class="eyebrow">SIGNED</p><h2>医生已签署处方</h2></div><span>${formatDate(signed.signed_at)}</span></div><p>${escapeHtml(signed.notice)}</p>${doseOptionRows(signed.items)}</section>` : ""}<section class="doctor-comparison"><div class="panel-heading"><div><p class="eyebrow">CURRENT DOCTOR PLAN</p><h2>当前医生诊断与方案</h2></div><span>${doctor ? "医生来源" : "尚未回传"}</span></div>${doctor ? `<h3>${escapeHtml((doctor.diagnoses || []).join("、"))}</h3><p>${escapeHtml(doctor.care_summary)}</p><h3>具体治疗记录</h3>${list((doctor.treatments || []).map((entry) => `${entry.name} · ${entry.route} · ${entry.schedule}`), "等待医生签署处方")}` : `<p>可以先确认路径并生成草稿；签署前仍必须取得医生确认诊断。</p>`}</section>`;
}

function educationPanel(education) {
  if (!education || education.review_status === "missing") return `<p class="education-missing">${escapeHtml(education?.notice || "暂无经过审核的药物说明，请咨询医生或药师。")}</p>`;
  return `<details class="medication-education"><summary>查看详细用药说明</summary><div class="education-grid"><div><h4>治疗目的</h4><p>${escapeHtml(education.purpose)}</p></div><div><h4>常见情况</h4>${list(education.common_effects)}</div><div class="is-warning"><h4>立即就医警示</h4>${list(education.urgent_warnings)}</div><div><h4>重要相互作用</h4>${list(education.interactions)}</div><div><h4>监测任务</h4>${list(education.monitoring)}</div><div><h4>漏服处理</h4><p>${escapeHtml(education.missed_dose)}</p></div></div><small>${escapeHtml(education.knowledge_source)} · ${escapeHtml(education.reviewed_on)}</small></details>`;
}
function renderAftercare() {
  const item = journey(); if (!item) return; const doctor = item.doctor_plan; const medications = item.medications || [];
  const provenance = item.treatment_provenance || [];
  $("#aftercare-content").innerHTML = `${previewBanner()}${item.confirmed_treatment_direction ? `<section class="confirmed-direction"><p class="eyebrow">CLINICIAN CONFIRMED DIRECTION</p><h2>${escapeHtml(item.confirmed_treatment_direction.title)}</h2><p>${escapeHtml(item.confirmed_treatment_direction.rationale)}</p><small>${escapeHtml(item.confirmed_treatment_direction.boundary)}</small></section>` : ""}${doctor ? `<section class="doctor-comparison"><div class="panel-heading"><div><p class="eyebrow">DOCTOR RESULT</p><h2>AI 解读与医生结论并列</h2></div><span>${escapeHtml(doctor.source?.label)}</span></div><div class="comparison-grid"><article><small>此前 AI 通俗解读 · ${escapeHtml(doctor.comparison?.ai_assessment_version_id || "对应版本")}</small><h3>${escapeHtml(doctor.comparison?.ai_primary || (item.patient_explanations || []).at(-1)?.headline || "尚未形成")}</h3></article><article class="is-doctor"><small>医生最终结论</small><h3>${escapeHtml((doctor.diagnoses || []).join("、"))}</h3><p>${escapeHtml(doctor.care_summary)}</p></article></div><div class="comparison-notes"><div><b>得到确认的依据</b>${list(doctor.comparison?.confirmed_evidence)}</div><div><b>医生修正</b>${list(doctor.comparison?.revisions)}</div></div></section>` : `<section class="report-empty"><h2>等待医生方案回传</h2><p>AI 可以整理指南路径和剂量草稿，但在医生确认诊断并第二次签署前不会创建患者处方或提醒。</p><button class="button button-primary" type="button" data-action="load-doctor-plan">${writable() ? "同步虚构医生出院记录" : "登录患者账户后体验回传"}</button></section>`}${provenance.length ? `<section class="treatment-provenance"><div class="panel-heading"><div><p class="eyebrow">TREATMENT PROVENANCE</p><h2>治疗从建议到执行的来源链</h2></div><span>${provenance.length} 步</span></div><ol>${provenance.map((step) => `<li><i></i><div><b>${escapeHtml(step.label)}</b><small>${formatDate(step.at)}</small></div></li>`).join("")}</ol></section>` : ""}${item.followups?.length ? `<section class="followup-panel"><div class="panel-heading"><div><p class="eyebrow">FOLLOW-UP</p><h2>复诊计划</h2></div><span>医生来源</span></div>${item.followups.map((followup) => `<article><b>${formatDate(followup.scheduled_at)}</b><div><h3>${escapeHtml(followup.title)}</h3><p>${escapeHtml(followup.source?.label)}</p></div><span>${escapeHtml(followup.status)}</span></article>`).join("")}</section>` : ""}<section class="medication-list"><div class="panel-heading"><div><p class="eyebrow">MEDICATION MANAGEMENT</p><h2>药物说明与执行记录</h2></div><span>${medications.length} 项医生签署处方</span></div>${medications.map((medication) => `<article class="medication-card"><header><div><span>${escapeHtml(medication.route)} · ${escapeHtml(medication.purpose)}</span><h3>${escapeHtml(medication.name)}</h3></div><b>${medication.source?.type === "clinician_signed_ai_path" ? "医生签署" : "医生处方"}</b></header><blockquote>${escapeHtml(medication.prescription_original)}</blockquote><dl><div><dt>剂量</dt><dd>${escapeHtml(medication.dose)}</dd></div><div><dt>时间/频次</dt><dd>${escapeHtml(medication.frequency)}</dd></div><div><dt>疗程</dt><dd>${escapeHtml(medication.course)}</dd></div></dl>${educationPanel(medication.education)}<div class="medication-actions"><button class="button button-secondary" data-medication="${escapeHtml(medication.id)}" data-event="taken">已执行</button><button class="button button-ghost" data-medication="${escapeHtml(medication.id)}" data-event="missed">漏服/错过</button><button class="button button-ghost" data-medication="${escapeHtml(medication.id)}" data-event="adverse">记录不良反应</button></div><small>${escapeHtml(medication.boundary)}</small></article>`).join("") || `<p class="muted-copy">医生第二次签署后才会出现具体药物、剂量与提醒。</p>`}</section>`;
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

function beginQuestion(message) { const clean = String(message || "").trim(); if (!clean) return toast("请先写下你最担心的问题。", "error"); if (!requireRole("patient", () => beginQuestion(clean))) return; const intake = state.journey.consultation_state || {}; if (intake.pending_question?.response_type === "safety_screen") { state.pendingQuestion = clean; openDialog("#triage-dialog"); return; } runAction(async () => { await api.consultation(state.journey.id, clean); await reloadJourney(); showView("consultation"); }); }
async function saveHistory() { if (!requireRole("patient", saveHistory)) return; const history = state.journey.clinical_history || {}; const field_statuses = Object.fromEntries(HISTORY_FIELDS.map((key) => [key, $(`[data-history-status="${key}"]`)?.value || "unconfirmed"])); await api.updateHistory(state.journey.id, { ...history, field_statuses }); await reloadJourney(); toast("病史确认状态已保存。", "success"); }
async function syncHospital() { if (!requireRole("patient", syncHospital)) return; if (!state.journey.hospital_connection) await api.connectHospital(); await api.syncRecords(); await reloadJourney(); toast("已从明确标记的沙箱医院同步检查结果。", "success"); }
async function loadDoctorPlan() { if (!requireRole("patient", loadDoctorPlan)) return; const sample = state.samples.patient; const source = sample.doctor_plan; await api.doctorDocument(state.journey.id, { source_type: "sandbox_hospital", diagnoses: source.diagnoses, care_summary: source.care_summary, examination_orders: source.examination_orders, treatments: [], confirmed_evidence: source.comparison?.confirmed_evidence || [], revisions: source.comparison?.revisions || [], followup_at: sample.followups?.[0]?.scheduled_at, prescriptions: [] }); await reloadJourney(); toast("已回传虚构医生确诊记录；处方仍需医生确认路径并二次签署。", "success"); }

async function renderCareLinks() {
  if (!state.authenticated || role() !== "patient" || !state.journey) return;
  const payload = await api.careTeamLinks(state.journey.id);
  const active = (payload.care_team_links || []).filter((item) => item.status === "active");
  $("#access-inline").innerHTML = `${state.activeGrant ? `<b class="grant-code">${escapeHtml(state.activeGrant.code)}</b><small>10 分钟有效 · 单次使用</small>` : ""}<div class="care-link-list">${active.map((item) => `<div><span><b>已授权医生</b><small>仅可读取当前病例</small></span><button class="text-link" type="button" data-revoke-link="${escapeHtml(item.id)}">撤销</button></div>`).join("") || `<small>当前没有已生效的医生授权。</small>`}</div>`;
}

async function initialize() {
  try { [state.samples.patient, state.samples.clinician] = await Promise.all([api.sampleJourney("patient"), api.sampleJourney("clinician")]); state.selectedVersion = state.samples.clinician.assessment_versions?.at(-1)?.version || 4; }
  catch (error) { $("#connection-state").dataset.state = "error"; $("#connection-state span").textContent = "样例不可用"; toast(error.message || "无法读取虚构病例。", "error"); return; }
  try { const session = await api.session(); state.authenticated = true; state.user = session.user; state.previewRole = session.user.role; state.journey = session.journeys?.[0] || null; }
  catch (error) { if (!(error instanceof ApiError) || error.status !== 401) toast(error.message, "error"); }
  renderAll(); showView((window.location.hash || `#${homeView()}`).slice(1));
}

document.addEventListener("submit", (event) => {
  if (event.target.id === "consultation-form") { event.preventDefault(); beginQuestion($("#consultation-input").value); }
  if (event.target.id === "triage-form") { event.preventDefault(); const signs = [...new FormData(event.target).getAll("danger_signs")]; closeDialog("#triage-dialog"); runAction(async () => { await api.consultation(state.journey.id, state.pendingQuestion, { dangerSigns: signs }); state.pendingQuestion = ""; event.target.reset(); await reloadJourney(); showView("consultation"); }); }
  if (event.target.id === "auth-form") { event.preventDefault(); runAction(async () => { await api.verifyOtp($("#phone-input").value, $("#otp-input").value); const session = await api.session(); state.authenticated = true; state.user = session.user; state.previewRole = session.user.role; state.journey = session.journeys?.[0] || null; state.activeGrant = null; closeDialog("#auth-dialog"); renderAll(); showView(homeView()); toast(`已登录${role() === "patient" ? "患者" : "医生"}账户。`, "success"); const pending = state.pendingAction; state.pendingAction = null; if (pending) pending(); }); }
  if (event.target.id === "upload-form") { event.preventDefault(); if (!requireRole("patient")) return; const form = new FormData(event.target); closeDialog("#upload-dialog"); runAction(async () => { const result = await api.upload(form); toast(`${result.notice} 当前状态：待用户确认。`, "success"); }); }
  if (event.target.id === "history-edit-form") { event.preventDefault(); if (!requireRole("patient")) return; const key = $("#history-edit-field").value; const value = parseHistoryEditorValue(key, $("#history-edit-value").value); const reason = $("#history-edit-reason").value.trim(); const statuses = { ...(state.journey.clinical_history?.field_statuses || {}), [key]: "confirmed" }; closeDialog("#history-edit-dialog"); runAction(async () => { const result = await api.updateHistory(state.journey.id, { [key]: value, field_statuses: statuses, reason }); await reloadJourney(); toast(result.corrections_created ? "修改已保存，受影响的 AI 判断已撤回。" : "内容没有变化，确认状态已保存。", "success"); }); }
  if (event.target.id === "intake-correction-form") { event.preventDefault(); if (!requireRole("patient")) return; const field = $("#intake-correction-field").value; const newValue = $("#intake-correction-value").value.trim(); const reason = $("#intake-correction-reason").value.trim(); closeDialog("#intake-correction-dialog"); runAction(async () => { await api.consultation(state.journey.id, "", { correction: { field, new_value: newValue, reason } }); await reloadJourney(); showView("consultation"); toast("问诊信息已修改，请重新核对摘要。", "success"); }); }
  if (event.target.id === "decision-form") { event.preventDefault(); if (!requireRole("clinician")) return; const kind = $("#decision-kind").value, id = $("#decision-id").value, action = $("#decision-action").value, rationale = $("#decision-rationale").value.trim(), lines = $("#decision-edits").value.split(/\n/).map((item) => item.trim()).filter(Boolean); const payload = { action, rationale, edits: action === "modified" && lines.length ? (kind === "exam" ? { items: lines } : { pathways: lines.map((name) => ({ name })) }) : {} }; closeDialog("#decision-dialog"); runAction(async () => { if (kind === "exam") await api.decideExamRecommendation(state.journey.id, id, payload); else await api.decideTreatmentRecommendation(state.journey.id, id, payload); await reloadJourney(); toast("医生决策已记录并保留审计。", "success"); }); }
  if (event.target.id === "prescription-form") { event.preventDefault(); if (!requireRole("clinician")) return; const draftId = $("#prescription-draft-id").value; const items = $("#prescription-items").value.split(/\n/).map((line) => line.split(/[｜|]/).map((part) => part.trim())).filter((parts) => parts.length >= 3 && parts[0] && parts[1] && parts[2]).map(([medication, dose, frequency, duration = "", route = "口服"]) => ({ medication, dose, frequency, duration, route })); const acknowledgements = [...new FormData(event.target).getAll("prescription_check")]; const rationale = $("#prescription-rationale").value.trim(); closeDialog("#prescription-dialog"); runAction(async () => { const result = await api.signPrescriptionDraft(state.journey.id, draftId, { items, acknowledgements, rationale }); await reloadJourney(); toast(`处方已签署，创建 ${result.medications.length} 项患者任务。`, "success"); }); }
});

document.addEventListener("click", (event) => {
  const go = event.target.closest("[data-go]"); if (go) { event.preventDefault(); closeDialog("#account-dialog"); showView(go.dataset.go); const targetId = go.dataset.examLink; if (targetId) window.setTimeout(() => { const row = document.querySelector(`[data-recommendation-id="${CSS.escape(targetId)}"]`); row?.classList.add("is-highlighted"); row?.scrollIntoView({ behavior: "smooth", block: "center" }); }, 60); return; }
  if (event.target.closest("[data-home]")) { event.preventDefault(); showView(homeView()); return; }
  const roleButton = event.target.closest("[data-preview-role]"); if (roleButton && !state.authenticated) { state.previewRole = roleButton.dataset.previewRole; state.journey = null; renderAll(); showView(homeView()); return; }
  const question = event.target.closest("[data-question]"); if (question) { beginQuestion(question.dataset.question); return; }
  const action = event.target.closest("[data-action]")?.dataset.action;
  if (action === "save-history") runAction(saveHistory);
  if (action === "generate-case-document") { if (!requireRole("patient")) return; runAction(async () => { await api.generateConsultationCase(state.journey.id); await reloadJourney(); toast("已根据最新问诊生成新的病例整理稿。", "success"); }); }
  if (action === "confirm-case-document") { if (!requireRole("patient")) return; runAction(async () => { await api.confirmConsultationCase(state.journey.id, state.journey.raw_case_document.id); await reloadJourney(); toast("本版问诊病例已确认。", "success"); }); }
  if (action === "sync-records") runAction(syncHospital);
  if (action === "upload-fallback") { if (requireRole("patient", () => openDialog("#upload-dialog"))) openDialog("#upload-dialog"); }
  if (action === "open-safety-screen") { if (!requireRole("patient")) return; state.pendingQuestion = ""; openDialog("#triage-dialog"); }
  if (action === "confirm-intake-summary") { if (!requireRole("patient")) return; runAction(async () => { await api.consultation(state.journey.id, "", { summaryConfirmed: true }); await reloadJourney(); toast("信息摘要已确认，可以进入下一步。", "success"); }); }
  if (action === "correct-intake-summary") { if (!requireRole("patient")) return; const facts = state.journey.consultation_state?.known_facts || {}; const select = $("#intake-correction-field"); select.innerHTML = Object.entries({ chief_complaint: "主要不适", onset: "发病时间", progression: "变化过程", severity: "最严重程度", associated_symptoms: "伴随症状", medical_history: "既往疾病", surgery_history: "手术史", medication: "当前用药", allergy: "过敏情况", family_history: "家族史", exposure_history: "吸烟、职业与暴露" }).map(([key, label]) => `<option value="${key}">${label}</option>`).join(""); const fill = () => { $("#intake-correction-value").value = facts[select.value] || ""; }; select.onchange = fill; fill(); $("#intake-correction-reason").value = ""; openDialog("#intake-correction-dialog"); }
  if (action === "load-doctor-plan") runAction(loadDoctorPlan);
  if (action === "rerun-diagnosis") { if (!requireRole("clinician")) return; runAction(async () => { const result = await api.rerunClinicianAssessment(state.journey.id); await reloadJourney(); state.selectedVersion = result.assessment.version; renderDiagnosis(); toast(`已生成诊断版本 v${result.assessment.version}。`, "success"); }); }
  if (action === "care-access") { if (!state.authenticated) return openDialog("#auth-dialog"); if (role() === "patient") runAction(async () => { state.activeGrant = await api.createCareAccessGrant(state.journey.id); await renderCareLinks(); toast("一次性授权码已生成。", "success"); }); else $("#access-inline").innerHTML = `<label>8 位授权码<input id="grant-code-input" maxlength="8" autocomplete="off" /></label><button class="button button-primary" type="button" data-action="redeem-access">领取病例</button>`; }
  if (action === "redeem-access") { const code = $("#grant-code-input")?.value || ""; runAction(async () => { await api.redeemCareAccessGrant(code); const data = await api.clinicianJourneys(); state.journey = data.journeys?.[0] || null; closeDialog("#account-dialog"); renderAll(); showView("clinician-case"); toast("病例授权已领取。", "success"); }); }
  const dispute = event.target.closest("[data-dispute-report]"); if (dispute) { if (!requireRole("patient")) return; const reason = window.prompt("请说明哪一项结果可能有误（提交后该报告会退出当前证据集）：", "报告内容与医院原件不一致"); if (reason) runAction(async () => { await api.disputeExamReport(state.journey.id, dispute.dataset.disputeReport, reason); await reloadJourney(); toast("报告已标记争议，当前 AI 判断已撤回。", "success"); }); }
  const editHistory = event.target.closest("[data-edit-history]"); if (editHistory) { if (!requireRole("patient")) return; const key = editHistory.dataset.editHistory; $("#history-edit-field").value = key; $("#history-edit-title").textContent = `修改${HISTORY_LABELS[key] || "健康信息"}`; $("#history-edit-value").value = historyEditorValue(key, state.journey.clinical_history?.[key]); $("#history-edit-reason").value = ""; $("#history-edit-help").textContent = key === "current_medications" ? "每行一项：药名｜剂量｜频次。修改后当前 AI 判断会撤回。" : key === "allergies" ? "每行一项：过敏原｜反应｜严重程度。不了解时请关闭窗口并选择“不了解”。" : key === "social_history" ? "每行一项：吸烟/饮酒/职业/暴露｜具体情况。" : "每行填写一项；需要补充说明时可用“名称｜详情”。"; openDialog("#history-edit-dialog"); }
  const revoke = event.target.closest("[data-revoke-link]"); if (revoke) { if (!requireRole("patient")) return; runAction(async () => { await api.revokeCareTeamLink(revoke.dataset.revokeLink); await renderCareLinks(); toast("医生病例访问权已立即撤销。", "success"); }); }
  const decision = event.target.closest("[data-decision-kind]"); if (decision) { if (!requireRole("clinician")) return; $("#decision-kind").value = decision.dataset.decisionKind; $("#decision-id").value = decision.dataset.decisionId; $("#decision-title").textContent = decision.dataset.decisionKind === "exam" ? "处理检查建议" : "处理治疗路径"; $("#decision-rationale").value = ""; $("#decision-edits").value = ""; openDialog("#decision-dialog"); }
  const signDraft = event.target.closest("[data-sign-draft]"); if (signDraft) { if (!requireRole("clinician")) return; const draft = (state.journey.prescription_drafts || []).find((item) => item.id === signDraft.dataset.signDraft); if (!draft) return toast("处方草稿不存在或已失效。", "error"); $("#prescription-draft-id").value = draft.id; $("#prescription-items").value = prescriptionLines(draft.items); $("#prescription-rationale").value = ""; $("#prescription-form").querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = false; }); openDialog("#prescription-dialog"); }
  const cancelDraft = event.target.closest("[data-cancel-draft]"); if (cancelDraft) { if (!requireRole("clinician")) return; const rationale = window.prompt("请输入取消处方草稿的理由：", "当前信息不足，暂不签署"); if (rationale) runAction(async () => { await api.cancelPrescriptionDraft(state.journey.id, cancelDraft.dataset.cancelDraft, rationale); await reloadJourney(); toast("处方草稿已取消。", "success"); }); }
  const version = event.target.closest("[data-version]")?.dataset.version; if (version) { state.selectedVersion = Number(version); renderDiagnosis(); }
  const medication = event.target.closest("[data-medication]"); if (medication) { if (!requireRole("patient")) return; if (!state.journey.medications?.some((item) => item.id === medication.dataset.medication)) return toast("请先完成医生记录回传。", "error"); runAction(async () => { await api.medicationEvent(medication.dataset.medication, medication.dataset.event, "患者在治后页记录"); await reloadJourney(); toast("已记录，不会修改医生处方。", "success"); }); }
  if (event.target.closest("#profile-button")) { openDialog("#account-dialog"); if (state.authenticated && role() === "patient" && state.journey) runAction(renderCareLinks); }
  const close = event.target.closest("[data-close]"); if (close) closeDialog(`#${close.dataset.close}-dialog`);
});

$("#request-otp").addEventListener("click", () => runAction(async () => { const result = await api.requestOtp($("#phone-input").value); $("#otp-help").textContent = result.development_code ? `本地测试验证码：${result.development_code}` : "验证码已发送，请在 5 分钟内输入。"; if (result.development_code) $("#otp-input").value = result.development_code; }));
$("#sign-out").addEventListener("click", () => { if (!state.authenticated) return closeDialog("#account-dialog"); runAction(async () => { await api.logout(); state.authenticated = false; state.user = null; state.journey = null; state.previewRole = "patient"; state.activeGrant = null; api.csrf = ""; closeDialog("#account-dialog"); renderAll(); showView("consultation"); toast("已退出，继续浏览完整虚构病例。", "success"); }); });
$("#export-data").addEventListener("click", (event) => { if (!state.authenticated || role() !== "patient") { event.preventDefault(); openDialog("#auth-dialog"); return; } event.currentTarget.href = api.exportUrl(); });
if ("serviceWorker" in navigator) window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js").catch(() => {}));
initialize();
