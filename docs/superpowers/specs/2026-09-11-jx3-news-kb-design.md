# JX3 News Knowledge Base Plugin Design

## Goal

Build an AstrBot plugin that ingests JX3API news and announcements into a local knowledge base, answers natural-language questions using the configured AstrBot providers, extracts reminder-worthy activity deadlines, sends whitelist-scoped reminders, and provides a Plugin Page for news management.

## Functional Scope

### Ingestion

- Data source: `GET {api_base_url}/news/records?limit={limit}`.
- First run: fetch 50 records.
- Scheduled run: fetch 10 records at `00:00` Asia/Shanghai by default.
- Manual run: fetch 1 record or catch up with 50 records from Plugin Page.
- Deduplicate by URL and `desc.id`; store `date`, `title`, `type`, `url`, cleaned text, raw JSON, publish/update timestamps, and content fingerprint.
- If an existing announcement's `updatetime` or content hash changes, append a revision instead of overwriting history.
- Embedding indexing is best-effort: full-text retrieval remains functional without an embedding provider.

### Access Control

- One whitelist controls queries and reminders.
- `whitelist_groups` controls group chats; `whitelist_users` controls private chats.
- `allow_group` and `allow_private` are independent switches.
- The plugin has no chat commands.
- Plugin message handling only runs after a real wake: `event.is_at_or_wake_command` must be true. (`event.is_wake` is forced to True for every message matched by a plugin event listener and cannot be used here.)
- A lightweight LLM relevance check suppresses unrelated awakened messages unless no LLM is available; in that case, retrieval may run directly.

### Retrieval and Answers

- SQLite is the local store.
- FTS5 indexes cleaned announcement chunks.
- Embedding vectors are stored in SQLite and compared with cosine similarity in Python.
- If configured, a Rerank provider reranks merged candidates.
- Query flow: relevance classification, query rewriting, full-text recall, vector recall, merge and rerank, latest-first weighting, LLM synthesis.
- Answers must cite announcement date, title, and URL.
- Latest information takes precedence. Historical information is mentioned only when the user asks about changes or explicitly requests history.

### Activity Reminders

Reminders include:

- In-game activities: sign-in, tasks, token collection, token exchange, reward claiming.
- Free coupons and free items, including free transmog, face-shape, and body-shape vouchers.
- Disappearance time of a free coupon or item, even after its parent activity ends.
- Rewards with explicit claim windows.

Reminders exclude:

- Test-server activities.
- Offline activities.
- Questionnaires.
- Discount coupons, discount vouchers, percentage-off promotions, and spend thresholds without a separate claim deadline.
- Punishments, maintenance, bug fixes, gameplay-only adjustments.
- Announcements without a definite deadline or disappearance time.
- Already-expired items.

Reminder schedule defaults:

- `reminder_enabled`: true
- `reminder_days_before`: 1
- `reminder_send_time`: `10:00`

Each reminder contains activity name, action, deadline, explanation, source title/date, and source URL. A short activity window shorter than 24 hours is reminded 30 minutes before start instead of one day before end.

### WebUI

The Plugin Page is embedded in AstrBot WebUI through `pages/jx3-news/index.html`. It has no separate port or password.

Features:

- Status overview.
- Announcement search, filtering, detail view, and extracted activity view.
- Hard deletion with a non-dialog, two-step confirmation: enable the confirmation checkbox, enter `DELETE`, then press the second enabled delete button.
- Manual fetch of 1 or 50 records.
- Full-text and embedding rebuild actions.
- Fetch/error log display.

Hard deletion removes the announcement, chunks, embeddings, activity records, pending reminders, and reminder logs. A fingerprint tombstone prevents the same record from being re-ingested.

## Configuration

```json
{
  "api_base_url": "https://www.jx3api.com",
  "news_records_path": "/news/records",
  "api_token": "",
  "initial_fetch_limit": 50,
  "daily_fetch_limit": 10,
  "daily_fetch_time": "00:00",
  "timezone": "Asia/Shanghai",
  "catchup_fetch_limit": 50,
  "whitelist_groups": [],
  "whitelist_users": [],
  "allow_group": true,
  "allow_private": true,
  "llm_provider_id": "",
  "llm_model": "",
  "embedding_provider_id": "",
  "reranker_provider_id": "",
  "reminder_enabled": true,
  "reminder_days_before": 1,
  "reminder_send_time": "10:00"
}
```

Empty provider IDs use AstrBot defaults. Rerank failure degrades gracefully.

