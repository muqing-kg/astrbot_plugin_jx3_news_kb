/* JX3 news knowledge base Plugin Page. */
const bridge = window.AstrBotPluginPage;

const state = {
  announcements: { page: 1, pageSize: 20, total: 0, q: "", type: "" },
  currentAnnouncementId: null,
};

const $ = (id) => document.getElementById(id);

function fmtDateTime(value) {
  if (!value) return "—";
  const text = String(value).replace("T", " ");
  return text.slice(0, 16);
}

function showError(outputEl, error) {
  outputEl.hidden = false;
  outputEl.classList.add("error");
  outputEl.textContent = `操作失败：${error.message || error}`;
}

function showInfo(outputEl, message) {
  outputEl.hidden = false;
  outputEl.classList.remove("error");
  outputEl.textContent = message;
}

async function callApi(fn, outputEl) {
  try {
    const result = await fn();
    if (outputEl) {
      showInfo(outputEl, typeof result === "string" ? result : JSON.stringify(result, null, 2));
    }
    return result;
  } catch (error) {
    if (outputEl) showError(outputEl, error);
    return null;
  }
}

/* ---------- tabs ---------- */
function setupTabs() {
  $("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (!tab) return;
    document.querySelectorAll(".tab").forEach((item) =>
      item.classList.toggle("active", item === tab),
    );
    document.querySelectorAll(".panel").forEach((panel) =>
      panel.classList.toggle("active", panel.id === `panel-${tab.dataset.tab}`),
    );
    if (tab.dataset.tab === "logs") loadLogs();
    if (tab.dataset.tab === "reminders") loadActivities();
  });
}

/* ---------- overview ---------- */
function renderStats(data) {
  const counts = data.counts || {};
  const providers = data.providers || {};
  const embedding = providers.embedding || {};
  const reranker = providers.reranker || {};
  const cards = [
    ["公告总数", counts.announcements],
    ["文本分块", counts.chunks],
    ["向量数", counts.embeddings],
    ["活动抽取", counts.activities],
    ["待发提醒（条）", data.pending_reminders],
    ["提醒目标", data.reminder_targets ?? 0],
    ["后台任务", data.scheduler_alive ? "运行中" : "未运行"],
    ["Embedding 维度", embedding.available ? (embedding.dim ?? "探测中") : "未启用"],
    ["Reranker", reranker.available ? "已启用" : "未启用"],
  ];
  $("stat-cards").innerHTML = cards
    .map(
      ([label, value]) =>
        `<div class="stat-card"><div class="num">${value ?? 0}</div><div class="label">${label}</div></div>`,
    )
    .join("");

  const last = data.last_fetch;
  $("last-fetch").innerHTML = last
    ? [
        `开始：${fmtDateTime(last.started_at)}`,
        `结果：${Number(last.success) === 1 ? "成功" : "失败"}`,
        `抓取条数：${last.fetch_limit}，入库 ${last.inserted_count}，修订 ${last.revised_count}，跳过 ${last.skipped_count}`,
        last.error ? `错误：${escapeHtml(last.error)}` : "",
      ]
        .filter(Boolean)
        .join("<br />")
    : "还没有抓取记录。首次安装后可点击“补抓最近 50 条”建立初始知识库。";

  const upcoming = data.upcoming_reminders || [];
  const noTargets =
    data.reminder_enabled && (data.reminder_targets ?? 0) === 0;
  if (noTargets) {
    $("upcoming-reminders").innerHTML =
      '<div class="hint danger">提醒已启用，但白名单中没有可用的完整会话地址，提醒不会发送。请在插件配置的群白名单中填写：平台ID:GroupMessage:群号（私聊为 平台ID:FriendMessage:用户号），保存后重载插件。</div>';
  } else if (!upcoming.length) {
    $("upcoming-reminders").innerHTML = '<div class="muted">暂无待发提醒。</div>';
  } else {
    const groups = new Map();
    for (const item of upcoming) {
      if (!groups.has(item.target_id)) groups.set(item.target_id, []);
      groups.get(item.target_id).push(item);
    }
    $("upcoming-reminders").innerHTML =
      '<div class="target-grid">' +
      [...groups.entries()]
        .map(
          ([target, items]) => `
          <div class="target-card">
            <div class="target-name">
              <span>${escapeHtml(target)}</span>
              <span class="target-count">${items.length}</span>
            </div>
            ${items
              .map(
                (item) => `
                <div class="target-item">
                  <div>${escapeHtml(item.name)}</div>
                  <div class="muted">${fmtDateTime(item.scheduled_at)}</div>
                </div>`,
              )
              .join("")}
          </div>`,
        )
        .join("") +
      "</div>";
  }
}

async function loadStats() {
  const data = await callApi(() => bridge.apiGet("stats"), null);
  if (data) renderStats(data);
}

function setupActions() {
  $("btn-fetch-1").addEventListener("click", () => runFetch(1));
  $("btn-fetch-10").addEventListener("click", () => runFetch(10));
  $("btn-fetch-50").addEventListener("click", () => runFetch(50));
  $("btn-rebuild-fts").addEventListener("click", async () => {
    const result = await callApi(
      () => bridge.apiPost("rebuild/fts", {}),
      $("action-result"),
    );
    if (result) {
      showInfo($("action-result"), `全文索引已重建，共 ${result.chunks} 个分块。`);
    }
    loadStats();
  });
  $("btn-rebuild-emb").addEventListener("click", async () => {
    const result = await callApi(
      () => bridge.apiPost("rebuild/embeddings", {}),
      $("action-result"),
    );
    if (result) {
      showInfo(
        $("action-result"),
        `已清理 ${result.cleared} 条旧向量，重新生成 ${result.indexed} 条。`,
      );
    }
    loadStats();
  });
}

async function runFetch(limit) {
  const result = await callApi(
    () => bridge.apiPost("fetch", { limit }),
    $("action-result"),
  );
  if (result) {
    showInfo(
      $("action-result"),
      `抓取完成：返回 ${result.returned} 条，入库 ${result.inserted} 条，修订 ${result.revised} 条，跳过 ${result.skipped} 条。`,
    );
  }
  loadStats();
}

/* ---------- announcements ---------- */
async function loadAnnouncements() {
  const { page, pageSize, q, type } = state.announcements;
  const data = await callApi(
    () => bridge.apiGet("announcements", { page, page_size: pageSize, q, type }),
    null,
  );
  if (!data) return;
  state.announcements.total = data.total;

  const typeSelect = $("ann-type");
  if (typeSelect.options.length <= 1) {
    for (const name of data.types || []) {
      typeSelect.add(new Option(name, name));
    }
  }

  $("ann-tbody").innerHTML = (data.items || [])
    .map(
      (item) => `
        <tr>
          <td>${escapeHtml(fmtDateTime(item.announcement_date))}</td>
          <td>${escapeHtml(item.type || "")}</td>
          <td class="wrap"><a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.title)}</a></td>
          <td>${item.activity_count}</td>
          <td>
            <button class="secondary" data-detail="${item.id}">详情</button>
            <button class="danger-btn" data-del="${item.id}">删除</button>
          </td>
        </tr>`,
    )
    .join("");

  const totalPages = Math.max(1, Math.ceil(data.total / pageSize));
  $("ann-page-info").textContent = `第 ${page} / ${totalPages} 页，共 ${data.total} 条`;
  $("ann-prev").disabled = page <= 1;
  $("ann-next").disabled = page >= totalPages;
}

function setupAnnouncements() {
  $("btn-ann-search").addEventListener("click", () => {
    state.announcements.q = $("ann-search").value.trim();
    state.announcements.type = $("ann-type").value;
    state.announcements.page = 1;
    loadAnnouncements();
  });
  $("ann-search").addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("btn-ann-search").click();
  });
  $("ann-prev").addEventListener("click", () => {
    state.announcements.page -= 1;
    loadAnnouncements();
  });
  $("ann-next").addEventListener("click", () => {
    state.announcements.page += 1;
    loadAnnouncements();
  });
  $("ann-tbody").addEventListener("click", (event) => {
    const detailBtn = event.target.closest("[data-detail]");
    if (detailBtn) {
      openDetail(detailBtn.dataset.detail);
      return;
    }
    const delBtn = event.target.closest("[data-del]");
    if (delBtn) openDeleteZone(Number(delBtn.dataset.del));
  });
  $("btn-detail-close").addEventListener("click", () => {
    $("ann-detail").hidden = true;
    state.currentAnnouncementId = null;
  });
}

async function openDetail(id) {
  state.currentAnnouncementId = id;
  const data = await callApi(() => bridge.apiGet(`announcements/${id}`), null);
  if (!data) return;
  const announcement = data.announcement;

  $("ann-detail").hidden = false;
  $("detail-title").textContent = announcement.title;
  $("detail-meta").innerHTML = [
    `ID：${announcement.id}`,
    `类型：${escapeHtml(announcement.type || "")}`,
    `日期：${escapeHtml(fmtDateTime(announcement.announcement_date))}`,
    `修订于源站：${escapeHtml(fmtDateTime(announcement.updated_at_source))}`,
    `分块：${data.chunk_count}`,
    `链接：<a href="${escapeHtml(announcement.url)}" target="_blank" rel="noreferrer">${escapeHtml(announcement.url)}</a>`,
  ].join("　·　");

  $("detail-content").textContent = announcement.content_text || "（无正文）";
  $("detail-raw").textContent = JSON.stringify(announcement.raw_json, null, 2);

  const chunks = data.chunks || [];
  $("detail-chunk-count").textContent = chunks.length;
  $("detail-chunks").innerHTML = chunks.length
    ? chunks
        .map(
          (chunk) => `
          <div class="chunk-item">
            <div class="muted">分块 #${chunk.chunk_index + 1}${chunk.embedding_updated_at ? ` · 向量生成于 ${fmtDateTime(chunk.embedding_updated_at)}` : " · 无向量"}</div>
            <pre class="content">${escapeHtml(chunk.content)}</pre>
          </div>`,
        )
        .join("")
    : '<div class="muted">没有分块。</div>';

  const revisions = data.revisions || [];
  $("detail-revisions").innerHTML = revisions.length
    ? `<table class="data-table"><thead><tr><th>#</th><th>标题</th><th>源站更新时间</th><th>内容指纹</th></tr></thead><tbody>${
        revisions
          .map(
            (item) =>
              `<tr><td>${item.revision_no}</td><td class="wrap">${escapeHtml(item.title)}</td><td>${fmtDateTime(item.updated_at_source)}</td><td class="wrap">${escapeHtml(item.content_hash.slice(0, 16))}…</td></tr>`,
          )
          .join("")
      }</tbody></table>`
    : '<div class="muted">没有修订记录。</div>';

  resetDeleteZone();
  $("ann-detail").scrollIntoView({ behavior: "smooth" });
}

/* ---------- two-step, non-dialog hard delete ---------- */
function resetDeleteZone() {
  $("del-check").checked = false;
  $("btn-del").disabled = true;
  $("del-result").hidden = true;
}

function openDeleteZone(id) {
  openDetail(id);
}

function setupDeleteZone() {
  $("del-check").addEventListener("change", (event) => {
    $("btn-del").disabled = !event.target.checked;
  });
  $("btn-del").addEventListener("click", async () => {
    const id = state.currentAnnouncementId;
    if (!id) return;
    const result = await callApi(
      () => bridge.apiPost(`announcements/${id}/delete`, { confirm: true }),
      $("del-result"),
    );
    if (result) {
      showInfo($("del-result"), "已彻底删除。该公告不会再被检索、问答或提醒。");
      resetDeleteZone();
      $("ann-detail").hidden = true;
      loadAnnouncements();
      loadStats();
    }
  });
}

/* ---------- ongoing activities (reminder tab) ---------- */
async function loadActivities() {
  const data = await callApi(() => bridge.apiGet("activities"), null);
  if (!data) return;
  const items = data.items || [];
  $("activities-list").innerHTML = items.length
    ? items
        .map((item) => {
          const times = [];
          if (item.start_time) times.push(`开始：<span class="hl">${fmtDateTime(item.start_time)}</span>`);
          if (item.end_time) times.push(`截止：<span class="hl">${fmtDateTime(item.end_time)}</span>`);
          if (item.item_expiry) times.push(`券/道具消失：<span class="hl">${fmtDateTime(item.item_expiry)}</span>`);
          return `
          <div class="reminder-item">
            <div class="item-head">
              <span class="name">${escapeHtml(item.name)}</span>
              <span class="tag">${escapeHtml(item.category)}</span>
              ${item.pending_reminders > 0 ? `<span class="tag status-pending">待发提醒 ${item.pending_reminders}</span>` : ""}
              <button class="danger-btn" data-act-del="${item.id}">删除</button>
            </div>
            <div class="muted">
              待办：${escapeHtml(item.action || "请及时处理")}${times.length ? `<br />${times.join(" · ")}` : ""}
            </div>
            ${item.item_name ? `<div>相关物品：${escapeHtml(item.item_name)}</div>` : ""}
            ${item.explanation ? `<div>${escapeHtml(item.explanation)}</div>` : ""}
            <div class="muted">来源：${escapeHtml(item.announcement_date)}《<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${escapeHtml(item.announcement_title)}</a>》</div>
          </div>`;
        })
        .join("")
    : '<div class="muted">当前没有进行中的活动。</div>';
}

function setupActivities() {
  $("btn-act-refresh").addEventListener("click", loadActivities);
  $("activities-list").addEventListener("click", async (event) => {
    const delBtn = event.target.closest("[data-act-del]");
    if (!delBtn) return;
    await callApi(
      () => bridge.apiPost(`activities/${delBtn.dataset.actDel}/delete`, {}),
      null,
    );
    loadActivities();
    loadStats();
  });
}

/* ---------- logs ---------- */
async function loadLogs() {
  const data = await callApi(() => bridge.apiGet("logs", { limit: 50 }), null);
  if (!data) return;
  $("log-tbody").innerHTML = (data.logs || [])
    .map(
      (item) => `
      <tr>
        <td>${fmtDateTime(item.started_at)}</td>
        <td>${fmtDateTime(item.finished_at)}</td>
        <td>${Number(item.success) === 1 ? "成功" : '<span class="tag status-failed">失败</span>'}</td>
        <td>${item.fetch_limit}</td>
        <td>${item.returned_count}</td>
        <td>${item.inserted_count}</td>
        <td>${item.revised_count}</td>
        <td>${item.skipped_count}</td>
        <td class="wrap">${escapeHtml(item.error || "")}</td>
      </tr>`,
    )
    .join("");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function main() {
  await bridge.ready();
  document.title = bridge.t("pages.jx3-news.title", "剑网3新闻公告知识库");
  setupTabs();
  setupActions();
  setupAnnouncements();
  setupDeleteZone();
  setupActivities();
  await Promise.all([loadStats(), loadAnnouncements()]);
}

main();
