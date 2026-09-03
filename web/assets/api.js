const config = window.CLINICALENS_CONFIG || {};

export class ApiError extends Error {
  constructor(message, status = 0, code = "request_failed", payload = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.payload = payload;
  }
}

export class CareApi {
  constructor() {
    this.base = String(config.apiBaseUrl || "").replace(/\/$/, "");
    this.csrf = "";
  }

  async request(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeout || 15000);
    const headers = new Headers(options.headers || {});
    let body = options.body;
    if (body && !(body instanceof FormData) && typeof body !== "string") {
      headers.set("Content-Type", "application/json");
      body = JSON.stringify(body);
    }
    const method = String(options.method || "GET").toUpperCase();
    if (!["GET", "HEAD", "OPTIONS"].includes(method) && this.csrf && !path.includes("/auth/otp/")) {
      headers.set("X-CSRF-Token", this.csrf);
    }
    try {
      const response = await fetch(`${this.base}${path}`, {
        method,
        headers,
        body,
        credentials: "include",
        cache: "no-store",
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new ApiError(payload.message || payload.error || `请求失败（${response.status}）`, response.status, payload.error, payload);
      }
      if (payload.csrf_token) this.csrf = payload.csrf_token;
      return payload;
    } catch (error) {
      if (error.name === "AbortError") throw new ApiError("请求超时，请稍后重试。", 0, "request_timeout");
      if (error instanceof ApiError) throw error;
      throw new ApiError("暂时无法连接健康档案服务。", 0, "network_unavailable");
    } finally {
      window.clearTimeout(timeout);
    }
  }

  session() { return this.request("/api/v1/session"); }
  requestOtp(phone) { return this.request("/api/v1/auth/otp/request", { method: "POST", body: { phone } }); }
  verifyOtp(phone, code) { return this.request("/api/v1/auth/otp/verify", { method: "POST", body: { phone, code } }); }
  logout() { return this.request("/api/v1/session", { method: "DELETE", body: {} }); }
  sampleJourney(audience = "patient") { return this.request(`/api/sample/journey?audience=${encodeURIComponent(audience)}`); }
  connectHospital() { return this.request("/api/v1/hospital-connections", { method: "POST", body: { consent: true } }); }
  syncRecords(simulate = "success") { return this.request("/api/v1/records/sync", { method: "POST", body: { simulate }, timeout: 20000 }); }
  getJourneys() { return this.request("/api/v1/journeys"); }
  getJourney(id) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}`); }
  consultation(id, message, dangerSigns = []) {
    return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/consultation/messages`, { method: "POST", body: { message, danger_signs: dangerSigns } });
  }
  updateHistory(id, payload) {
    return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/clinical-history`, { method: "PATCH", body: payload });
  }
  syncBatch(id, batchKey) {
    return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/record-batches/${encodeURIComponent(batchKey)}/sync`, { method: "POST", body: {} });
  }
  assessmentVersions(id) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/assessment-versions`); }
  examReports(id) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/exam-reports`); }
  patientExplanations(id) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/patient-explanations`); }
  disputeExamReport(journeyId, reportId, reason) { return this.request(`/api/v1/journeys/${encodeURIComponent(journeyId)}/exam-reports/${encodeURIComponent(reportId)}`, { method: "PATCH", body: { disputed: true, reason } }); }
  triage(id, dangerSigns) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/triage`, { method: "POST", body: { danger_signs: dangerSigns } }); }
  reviewRecord(journeyId, recordId, confirmed, correction = "") {
    return this.request(`/api/v1/journeys/${encodeURIComponent(journeyId)}/records/${encodeURIComponent(recordId)}`, { method: "PATCH", body: { confirmed, correction } });
  }
  startAssessment(id) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/assessments`, { method: "POST", body: { consent: true }, timeout: 20000 }); }
  assessmentRun(id) { return this.request(`/api/v1/assessment-runs/${encodeURIComponent(id)}`); }
  appointmentPlan(id) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/appointment-plan`, { method: "POST", body: {} }); }
  bookingStatus(id, booked) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/appointment-plan`, { method: "PATCH", body: { booked } }); }
  doctorDocument(id, payload) { return this.request(`/api/v1/journeys/${encodeURIComponent(id)}/doctor-documents`, { method: "POST", body: payload }); }
  medicationEvent(id, type, note = "") { return this.request(`/api/v1/medications/${encodeURIComponent(id)}/events`, { method: "POST", body: { type, note } }); }
  upload(form) { return this.request("/api/v1/record-imports", { method: "POST", body: form, timeout: 30000 }); }
  createCareAccessGrant(journeyId) { return this.request("/api/v1/care-access-grants", { method: "POST", body: { journey_id: journeyId } }); }
  redeemCareAccessGrant(code) { return this.request("/api/v1/care-access-grants/redeem", { method: "POST", body: { code } }); }
  revokeCareTeamLink(id) { return this.request(`/api/v1/care-team-links/${encodeURIComponent(id)}`, { method: "DELETE", body: {} }); }
  clinicianJourneys() { return this.request("/api/v1/clinician/journeys"); }
  clinicianJourney(id) { return this.request(`/api/v1/clinician/journeys/${encodeURIComponent(id)}`); }
  clinicianExamRecommendations(id) { return this.request(`/api/v1/clinician/journeys/${encodeURIComponent(id)}/exam-recommendations`); }
  decideExamRecommendation(journeyId, recommendationId, payload) { return this.request(`/api/v1/clinician/journeys/${encodeURIComponent(journeyId)}/exam-recommendations/${encodeURIComponent(recommendationId)}/decision`, { method: "POST", body: payload }); }
  decideTreatmentRecommendation(journeyId, recommendationId, payload) { return this.request(`/api/v1/clinician/journeys/${encodeURIComponent(journeyId)}/treatment-recommendations/${encodeURIComponent(recommendationId)}/decision`, { method: "POST", body: payload }); }
  rerunClinicianAssessment(id) { return this.request(`/api/v1/clinician/journeys/${encodeURIComponent(id)}/assessments`, { method: "POST", body: {} }); }
  subscribe(subscription) { return this.request("/api/v1/push-subscriptions", { method: "POST", body: { subscription } }); }
  metrics() { return this.request("/api/sample/metrics"); }
  exportUrl() { return `${this.base}/api/v1/account/export`; }
}
