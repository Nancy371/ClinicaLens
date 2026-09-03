# ClinicaLens

**患者能理解、医生能核对的连续健康旅程**

ClinicaLens 围绕一个持续的健康事件，帮助用户完成：

> 患者问诊与安全分流 → 医院检查同步 → 患者通俗解读 → 医生检查/诊断/治疗决策支持 → 接收医生结果 → 执行复诊与用药任务

它解决的问题不是“再生成一个诊断”，而是让用户知道依据来自哪里、候选方向为什么变化、有哪些反证、还缺什么，以及什么时候必须交给医生。

## 离线公开 Demo

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FNancy371%2FClinicaLens)

根目录 `render.yaml` 当前只创建一个免费 Static Site，发布 `web/` 中的脱敏离线体验。它不创建 API 服务、不接收任意病例输入，也不需要模型、患者服务或数据库凭证。点击上方按钮后，在 Render 复核资源并批准部署即可。

## 医疗边界

- AI 提供候选方向、证据解释、反证、紧急度、就医准备和无个体剂量的治疗路径参考。
- 医生负责确诊、检查医嘱、个体治疗决策、处方和剂量。
- AI 输出不能创建复诊或用药任务；只有医院回传或医生文档可以。
- 命中咯血、明显呼吸困难或意识异常等危险信号时，普通流程立即停止并提示急诊。
- 本项目不提供线上问诊、电子处方、院内号源或支付。

## 当前产品闭环

首屏是问诊输入框。第一条纵向旅程使用明确标记的“肺—肾多器官异常”完整虚构病例：

1. 未登录可切换患者版和医生版完整虚构样例；登录后角色由服务端决定，不能在前端切换。
2. 问诊先运行咯血、低氧、静息气促和意识异常的确定性安全分流。
3. 对照完整虚构病例原文，逐项确认疾病史、用药史、过敏史、家族史和暴露史。
4. 检查以医院接口同步为主，按医院—日期—报告连续表格展示；上传仅在同步失败或缺失时兜底。
5. 患者查看与诊断版本绑定的通俗解释；医生查看 `v1–v4` 主诊断变化、鉴别、支持/反证、缺失检查和危险疾病。
6. 医生逐项确认、修改或拒绝具体检查建议与无剂量治疗路径，所有决定保留理由和审计。
7. 同步医生诊断、复诊计划和处方，患者查看审核药物说明并记录执行、错过和不良反应。

页面包含医院接口超时、证据未确认、危险症状、低确定性和推送未配置等非成功状态，不会把每条路径伪装为成功。

## 本地运行

推荐 Python 3.10 或 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tools\build_demo_assets.py
python -m agent
```

打开 `http://127.0.0.1:7860/`。

本地默认测试账号：

- 患者手机号：任意合法中国大陆手机号，例如 `13800138000`
- 沙箱医生手机号：`13900000000`（由 `CARE_DEV_CLINICIAN_PHONES` 服务端配置）
- 验证码：`246810`

开发验证码会由登录接口明确返回；当 `CARE_PUBLIC_DEPLOYMENT=true` 且没有配置短信供应商时，登录入口返回 `503`，不会伪装短信发送成功。

## 架构

```text
移动端 Web / PWA
  ├─ 患者版：问诊 / 病例与报告 / 治后
  └─ 医生版：病例 / 检查 / 诊断 / 治疗
          │
          ▼
aiohttp /api/v1
  ├─ OTP 会话与 CSRF
  ├─ 服务端角色、一次性病例授权与撤销
  ├─ CareJourney 领域服务
  ├─ SandboxHospitalConnector
  ├─ Agent 决策支持投影
  ├─ 医生来源门禁
  └─ 复诊、用药与 PWA Push
          │
          ├─ PostgreSQL：生产结构化状态
          ├─ SQLite：本地开发与测试
          ├─ S3 兼容对象存储：生产文档
          └─ Redis / Upstash：公开样例限流
```

生产 PostgreSQL 结构位于 `db/postgres.sql`。设置 `CARE_DATABASE_URL` 或 `DATABASE_URL` 为 PostgreSQL 地址后，服务使用 `asyncpg` 并自动初始化结构；未设置时使用 `.care_data/clinicalens.db`。

上传文档在本地开发时保存到被忽略的 `.care_data/uploads/`；公开部署必须配置 S3 兼容对象存储，否则上传入口关闭。

## 患者端与医生端 API

### 登录与会话

- `POST /api/v1/auth/otp/request`
- `POST /api/v1/auth/otp/verify`
- `GET /api/v1/session`
- `DELETE /api/v1/session`

会话使用 `HttpOnly` Cookie。除 OTP 外的写请求都必须携带 `X-CSRF-Token`。

### 健康事件

- `POST /api/v1/hospital-connections`
- `POST /api/v1/records/sync`
- `POST /api/v1/record-imports`
- `GET/POST /api/v1/journeys`
- `GET /api/v1/journeys/{journey_id}`
- `GET /api/v1/journeys/{journey_id}/exam-reports`
- `PATCH /api/v1/journeys/{journey_id}/exam-reports/{report_id}`
- `GET /api/v1/journeys/{journey_id}/patient-explanations`
- `POST /api/v1/journeys/{journey_id}/consultation/messages`
- `PATCH /api/v1/journeys/{journey_id}/clinical-history`
- `POST /api/v1/journeys/{journey_id}/record-batches/{batch_key}/sync`
- `GET /api/v1/journeys/{journey_id}/assessment-versions`
- `POST /api/v1/journeys/{journey_id}/triage`
- `PATCH /api/v1/journeys/{journey_id}/records/{record_id}`
- `POST /api/v1/journeys/{journey_id}/assessments`
- `GET /api/v1/assessment-runs/{run_id}`
- `POST /api/v1/journeys/{journey_id}/appointment-plan`
- `POST /api/v1/journeys/{journey_id}/doctor-documents`

### 病例授权与医生工作区

- `POST /api/v1/care-access-grants`
- `POST /api/v1/care-access-grants/redeem`
- `DELETE /api/v1/care-team-links/{link_id}`
- `GET /api/v1/clinician/journeys`
- `GET /api/v1/clinician/journeys/{journey_id}`
- `GET /api/v1/clinician/journeys/{journey_id}/exam-recommendations`
- `POST /api/v1/clinician/journeys/{journey_id}/exam-recommendations/{recommendation_id}/decision`
- `POST /api/v1/clinician/journeys/{journey_id}/treatment-recommendations/{recommendation_id}/decision`

### 复诊、用药和通知

- `GET /api/v1/followups`
- `GET /api/v1/medications`
- `POST /api/v1/medications/{medication_id}/events`
- `POST/DELETE /api/v1/push-subscriptions`
- `GET /api/v1/account/export`
- `DELETE /api/v1/account`

原 `POST /test` 协议保持兼容。该接口的内部治疗字段不会进入 C 端 `CareJourney`。

完整虚构只读旅程使用 `GET /api/sample/journey`；其余只读样例继续使用 `/api/sample/*`。原 `/api/demo/*` 暂时作为兼容别名。虚构身份与联系方式均为人工构造，不对应真实患者。

## 可信度与测试

7 例固定回归样本位于 `tests/fixtures/diagnostic_replay_cases.jsonl`，通过 `tools/build_demo_assets.py` 生成：

- Recall@5
- Top-1
- Exact Match
- 命名空间合法率
- 反证误判率

报告始终同时展示 `n=7`、数据来源、生成日期和方法；这些指标只说明固定回归集上的工程行为，不代表临床准确率。

当前在 Python 3.12 环境执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

使用 `python tools/build_test_report.py` 运行并生成 `web/data/test-report.json`；页面只读取该构建产物，没有成功报告时显示“当前构建未验证”，不硬编码测试通过数。

## 生产配置

参考 `.env.example`。至少需要：

- `CARE_AUTH_SECRET`
- `CARE_DATABASE_URL`
- `CARE_SMS_PROVIDER_URL` 与供应商凭证
- `CARE_S3_BUCKET`、端点和访问凭证
- `CARE_ALLOWED_ORIGINS`
- `CARE_COOKIE_SECURE=true`
- `CARE_VAPID_PUBLIC_KEY`、`CARE_VAPID_PRIVATE_KEY`、`CARE_VAPID_SUBJECT`

没有生产凭证的能力必须关闭并向用户明确说明。安全边界与数据处理见 `SECURITY.md`。
