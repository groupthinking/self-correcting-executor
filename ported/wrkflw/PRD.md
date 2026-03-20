```markdown
# ConstructionOnline — MVP PRD

Version: 0.1
Author: Copilot / groupthinking
Date: 2025-09-07

## Prompt Yourself: What exactly are we building?
I want to build ConstructionOnline: an AI/agent-first, mobile-first Construction Management SaaS that replaces fragmented tools with a single Project OS that automates workflows, centralizes field→office communications, and delivers actionable, predictive intelligence to contractors, project managers, and developers. MVP focuses on daily reporting, tasks, photos, document attachments, basic RFI flow, and an AI daily-summary copilot.

---

## Objective
Deliver an MVP that demonstrates clear ROI for SMB contractors by:
- Reducing meeting time via automated daily reports
- Improving field→office visibility (photos, tasks, RFIs)
- Demonstrating AI-generated weekly summaries and risk flags

Target outcome (90 days): onboard 3–5 pilot customers, collect measurable time-savings metrics, iterate to paid beta.

---

## Scope (MVP)
Core features:
1. Authentication & RBAC (Admin, Project Manager, Foreman, Subcontractor)
2. Project creation & membership
3. Task management (create, assign, update, comments)
4. Photo & document upload (mobile + web) with S3 signed-URL uploads
5. Daily Log capture & AI daily-summary generator (RAG-based)
6. Simple RFI: draft → approval → send (internal)
7. Notifications (in-app, email)
8. Mobile app (Expo React Native) for field capture and offline queue
9. Basic Audit log & download of daily logs
10. API endpoints (GraphQL or REST) for integrations (OpenAPI/GraphQL schema)

Out-of-scope for MVP:
- Full ERP/BIM integrations, advanced CPM/Gantt editing, change order automation, enterprise SAML/SCIM, granular cost control.

---

## Personas & Goals
- Project Manager (PM)
  - Goals: get daily visibility, reduce status meetings, manage RFIs and tasks, produce weekly summaries for stakeholders.
- Foreman / Field Worker
  - Goals: report progress quickly, upload photos, create tasks/RFIs while offline, receive assignments.
- Subcontractor
  - Goals: receive assignments, submit trade-specific updates and photos, respond to RFIs.

---

## Detailed User Flows

### Flow: PM — Create Project & Invite Team
1. PM logs in.
2. PM clicks "Create Project" → enters name, location, start/end dates, baseline budget (optional).
3. PM invites users (email) and assigns roles.
4. Invitations sent (email with signup link). Invited users accept, join project.
Acceptance criteria:
- Project created with unique ID.
- Invited email receives link with token that expires (48 hours).
- Role assigned is honored on subsequent actions.

### Flow: Foreman — Capture Daily Log (mobile-first)
1. Foreman opens mobile app, selects project, taps "New Daily Log".
2. Foreman captures photos, types notes, selects weather, marks tasks completed, adds time entries (optional).
3. If offline: content is saved locally and queued for upload.
4. On connectivity, photos are uploaded to signed S3 URLs; metadata and structured log saved to DB.
5. A "daily log created" notification is sent to PM.
Acceptance criteria:
- Photo and metadata successfully persisted and visible in web UI.
- Offline queue persists across app restarts; syncs when online.
- Notifications triggered and received.

### Flow: Subcontractor — Receive & Update Task
1. Subcontractor receives notification (in-app or email) for assigned task.
2. Opens task, adds comment, attaches a photo, marks progress (Not Started → In Progress → Complete).
3. Completion triggers a log entry and optionally notifies PM.
Acceptance criteria:
- Task state changes recorded with actor and timestamp.
- Attachment links are accessible to authorized project members.

### Flow: PM — RFI Draft & Send (internal for MVP)
1. PM or Foreman creates RFI with attached photo/drawing, description, and due date.
2. PM selects approver (another PM or Admin).
3. Approver reviews draft in web UI; approves or requests changes.
4. Once approved, RFI is "Open" and tracked.
Acceptance criteria:
- RFI lifecycle (Draft → Pending Approval → Open → Closed) enforced.
- Approver notifications sent; approvals recorded.

### Flow: AI Daily Summary Agent
1. Scheduled job (e.g., 18:00 local project time) triggers RAG pipeline:
   - Retrieve daily logs, photos captions, tasks completed, open RFIs from last 24h.
   - Run summarization prompt producing: Overview, Issues, Photos of note, Tasks due, Risk flags.
2. Draft summary stored as "AI Summary (draft)" in project notes and sent to PM for review.
3. PM can accept/modify and publish summary; publishing creates an exportable PDF.
Acceptance criteria:
- Summary produced within service-level time (e.g., < 30s for small projects).
- Draft stored with provenance (what data used, embeddings retrieval IDs).
- Human-in-the-loop required to publish.

---

## Feature List + Acceptance Criteria (detailed)

1. Authentication & RBAC
   - AC1.1: User can sign up / sign in via email+password.
   - AC1.2: Admins can invite users and assign roles; roles restrict access accordingly.
   - AC1.3: JWTs issued expire and can be revoked.

2. Projects
   - AC2.1: Projects support metadata (name, location, dates).
   - AC2.2: Project members list is editable by Admin/PM.

3. Tasks
   - AC3.1: Create task with title, description, assignee, due date, priority.
   - AC3.2: Task history shows who changed status and when.
   - AC3.3: Tasks support attachments and comments.

4. Uploads (Photos & Docs)
   - AC4.1: Mobile can upload via signed S3 URLs.
   - AC4.2: Uploads link back to a daily log or task.
   - AC4.3: File type & size validated; malicious files blocked.

5. Daily Log
   - AC5.1: Foreman can create daily log with photos, text, task status.
   - AC5.2: Daily logs are time-stamped and immutable once published (edits create a new version).
   - AC5.3: Exportable PDF/CSV for a date range.

6. AI Daily Summary
   - AC6.1: Summaries include structured sections (Overview, Issues, Photos, Recommendations).
   - AC6.2: Summaries show source citations (which logs photos/tasks were used).
   - AC6.3: Publishing requires PM approval with an audit trail.

7. RFI
   - AC7.1: RFI draft/approval workflow enforced.
   - AC7.2: RFIs attached to logs and tasks.
   - AC7.3: RFIs track Replies and status.

8. Notifications
   - AC8.1: In-app and email notifications for task assignment, RFI events, daily log creation, and AI summary draft.
   - AC8.2: Users can mute project notifications.

9. Mobile offline sync
   - AC9.1: Local queue persists content and retries upload.
   - AC9.2: Conflict resolution: server wins, but an edit history is stored.

10. Audit & Security
    - AC10.1: All actions logged with actor, time, source (web/mobile).
    - AC10.2: Files encrypted at rest (S3 SSE) and in transit (TLS).

---

## Non-Functional Requirements
- Scalability: Service must handle 1k concurrent projects in MVP infra with autoscaling plans.
- Performance: UI actions under 200ms for common reads; AI summary within 30s for small projects.
- Reliability: 99.9% uptime for core APIs (SLAs later).
- Security: SOC2-aligned controls to start; encryption at rest/in transit; RBAC enforced server-side.
- Privacy: GDPR-ready data deletion workflows.

---

## Data Model (high level)
- Users {id, name, email, role, last_active}
- Projects {id, name, location, start_date, end_date, members[]}
- DailyLog {id, project_id, author_id, date, notes, attachments[], weather, synced_at}
- Task {id, project_id, title, description, assignee_id, status, history[], attachments[]}
- RFI {id, project_id, creator_id, approver_id, status, attachments[], history[]}
- Attachment {id, url, content_type, size, uploaded_by, checksum}
- AI_Summary {id, project_id, generated_at, model, prompt_version, content, sources[]}
- Audit {id, actor_id, action, resource_type, resource_id, timestamp, meta}

---

## API Surface (MVP)
- Auth
  - POST /auth/signup
  - POST /auth/login
  - POST /auth/invite
- Projects
  - GET /projects
  - POST /projects
  - GET /projects/:id
- Daily Logs
  - POST /projects/:id/daily-logs
  - GET /projects/:id/daily-logs?date=
- Tasks
  - POST /projects/:id/tasks
  - PATCH /projects/:id/tasks/:taskId
- Uploads
  - POST /uploads/signed-url (returns S3 signed PUT URL)
- AI
  - POST /projects/:id/ai/daily-summary (trigger or manual)
  - GET /projects/:id/ai/daily-summary/:id
- RFI
  - POST /projects/:id/rfis
  - PATCH /projects/:id/rfis/:rfiId/approve
- Notifications
  - GET /users/:id/notifications
  - POST /users/:id/notifications/read

Provide both GraphQL schema and OpenAPI v3 for the above in repo (seed file required).

---

## AI / Agent Design (MVP specifics)
- Pipeline:
  1. Ingest: daily logs metadata + photo captions (use CPU OCR optionally).
  2. Chunk & embed textual content → store embeddings in Vector DB.
  3. Retrieval: gather high-similarity docs for the last 24–72 hours.
  4. LLM call with prompt templates (versioned) to produce draft summary.
- Safety:
  - Agent must produce structured JSON with sections; function-calling or strict JSON schema enforced.
  - All actions that create server-side mutations (e.g., create task) are in draft mode and require user confirmation.
  - Keep data access scoping (agent uses service account with only project scope).
- Observability:
  - Log prompt, model used, tokens, latency, top retrieval sources (IDs).
- Storage:
  - Store agent transcripts and allow PM to view source citations.

References: LangChain patterns for agents, OpenAI function-calling for deterministic structured outputs, Pinecone/Redis Vector for embeddings.

---

## Milestones & Timeline (90-day sprint)
- Week 0–2: Finalize PRD, UX flows, GraphQL/OpenAPI stubs, seeded fixtures.
- Week 2–6: Backend core (auth, projects, tasks, uploads), frontend skeleton, mobile capture (offline queue).
- Week 6–10: Daily log flow complete, basic notifications, RFI minimal workflow, AI RAG prototype (daily summary).
- Week 10–12: Pilot deploy, onboard 3–5 pilot customers, collect feedback & metrics.

Deliverables per milestone listed in repo as GitHub issues.

---

## Success Metrics (KPIs)
- Time-to-first-daily-log (target < 24h after sign-up)
- % active projects with ≥1 photo/day (target 40% for pilot)
- AI Summary adoption rate: % of summaries published by PMs (target 50% within pilot)
- Activation → Trial → Paid conversion (initial target 10–15%)
- 3-month retention (pilot) >= 70%

---

## Rollout & GTM (pilot plan)
- Offer free 6-week pilot to local SMB contractors.
- Provide onboarding playbook & dedicated onboarding contact for pilot accounts.
- Collect 2 case studies in pilot regions for marketing.
- Pricing: simple per-project-per-month with free tier for single project (intro).

---

## Testing & QA
- Unit tests for all business logic (target 70% coverage MVP).
- E2E tests: mobile capture → upload → daily log visible in web.
- Security tests: auth flows, file uploads, XSS, SSRF scanning.
- AI tests: prompt regression tests and A/B variants; store sample inputs/outputs.

---

## Dev Checklist (initial)
- Add repo files:
  - PRD.md (this file)
  - openapi.yaml and graphql/schema.graphql
  - infra/terraform for S3 & minimal infra
  - seed-data.json with sample projects and users
- CI: GitHub Actions for lint, unit tests, build
- Observability: Sentry + Prometheus + Grafana dashboard skeleton
- SOC2 readiness docs & runbook (ops/ folder)
- Create GitHub issues for every milestone & feature

---

## Risks & Mitigations
- Adoption risk: field teams resist change — mitigate with mobile-first UX and offline support.
- Data privacy/regulation: sensitive contracts/drawings — implement encryption, access controls, and data deletion workflows.
- Cost risk (LLM): high token usage — mitigate with caching, aggregations, and hybrid models + cost limits per project.

---

## Appendix
- Tech stack recommendation (short)
  - Backend: TypeScript (NestJS) + PostgreSQL + Redis + S3
  - AI: Python microservice (LangChain) + Vector DB (Pinecone/Redis)
  - Frontend: React + Next.js
  - Mobile: Expo React Native
- Links to reference patterns:
  - LangChain docs, OpenAI function-calling docs, AWS S3 presigned upload patterns, SOC2 readiness checklist templates.

---
```