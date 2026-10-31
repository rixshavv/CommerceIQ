const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n || 0).toLocaleString("en-IN");
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {"Content-Type":"application/json", ...(options.headers || {})},
    ...options
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function renderRecent(items) {
  const root = $("recentReviews");
  if (!items.length) { root.innerHTML = '<div class="empty">No reviews analyzed yet. Submit one to get started.</div>'; return; }
  root.innerHTML = items.map(item => `
    <div class="recent-item">
      <span class="recent-text">${escapeHtml(item.text.length > 105 ? item.text.slice(0,105) + "…" : item.text)}</span>
      <div class="recent-meta"><span class="sentiment-badge ${escapeHtml(item.sentiment)}">${escapeHtml(item.sentiment)}</span><span>${Math.round(item.confidence*100)}% confidence</span></div>
    </div>`).join("");
}
function setDonut(dist, total) {
  const p = dist.positive || 0, n = dist.neutral || 0, neg = dist.negative || 0;
  const pct = v => total ? Math.round(v / total * 100) : 0;
  const pDeg = total ? p / total * 360 : 0, nDeg = total ? (p+n) / total * 360 : 0;
  $("donut").style.background = `conic-gradient(#18a879 0deg ${pDeg}deg,#e6a23c ${pDeg}deg ${nDeg}deg,#e56b75 ${nDeg}deg 360deg)`;
  $("positivePct").textContent = pct(p) + "%"; $("neutralPct").textContent = pct(n) + "%"; $("negativePct").textContent = pct(neg) + "%";
}
async function refresh() {
  const data = await api("/api/overview");
  const d = data.distribution || {}, total = data.total_reviews || 0;
  $("totalReviews").textContent = fmt(total); $("chartTotal").textContent = fmt(total);
  $("positiveCount").textContent = fmt(d.positive || 0); $("neutralCount").textContent = fmt(d.neutral || 0); $("negativeCount").textContent = fmt(d.negative || 0);
  setDonut(d, total); renderRecent(data.recent_reviews || []);
}
$("refreshBtn").addEventListener("click", () => refresh().catch(showError));
$("reviewText").addEventListener("input", e => $("charCount").textContent = `${e.target.value.length} / 5000`);
$("analyzeBtn").addEventListener("click", async () => {
  const text = $("reviewText").value.trim(), btn = $("analyzeBtn"), result = $("sentimentResult");
  if (text.length < 3) return showInlineError(result, "Enter at least 3 characters.");
  btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const data = await api("/api/sentiment", {method:"POST", body:JSON.stringify({text})});
    result.classList.remove("hidden");
    result.innerHTML = `<div class="result-main"><b>${escapeHtml(data.sentiment)}</b><span class="confidence">${Math.round(data.confidence*100)}% model confidence</span></div><div class="probabilities">${Object.entries(data.probabilities).map(([k,v])=>`<span class="prob">${escapeHtml(k)}: ${Math.round(v*100)}%</span>`).join("")}</div>`;
    await refresh();
  } catch(e) { showInlineError(result, e.message); }
  finally { btn.disabled = false; btn.innerHTML = 'Analyze sentiment <span>→</span>'; }
});
$("salesForm").addEventListener("submit", async e => {
  e.preventDefault();
  const btn = $("forecastBtn"), result = $("forecastResult");
  btn.disabled = true; btn.textContent = "Generating…";
  try {
    const payload = {
      price: Number($("price").value),
      discount: Number($("discount").value)/100,
      marketing_spend: Number($("marketing").value),
      rating: Number($("rating").value),
      month: Number($("month").value)
    };
    const data = await api("/api/sales-prediction", {method:"POST", body:JSON.stringify(payload)});
    result.classList.remove("hidden");
    result.innerHTML = `<small>ESTIMATED UNITS SOLD</small><strong>${fmt(data.predicted_units)} units</strong><p>${escapeHtml(data.model_note)}</p>`;
  } catch(e) { showInlineError(result, e.message); }
  finally { btn.disabled = false; btn.innerHTML = 'Generate forecast <span>→</span>'; }
});
function showInlineError(el, message) { el.classList.remove("hidden"); el.innerHTML = `<span style="color:#c34e5b;font-size:11px">${escapeHtml(message)}</span>`; }
function showError(e) { $("apiStatus").textContent = "API unavailable"; $("apiStatus").style.color = "#d75c66"; console.error(e); }
api("/health").then(() => { $("apiStatus").textContent = "API connected"; refresh(); }).catch(showError);
