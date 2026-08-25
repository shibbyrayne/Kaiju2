const $ = (id) => document.getElementById(id);

function todayStr(offsetDays = 0) {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  return d.toISOString().slice(0, 10);
}

function setStatus(el, text, kind) {
  el.textContent = text;
  el.classList.remove("ok", "bad");
  if (kind) el.classList.add(kind);
}

function money(n) {
  return "$" + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
}

async function api(path, options = {}) {
  const resp = await fetch(path, options);
  let body = null;
  try {
    body = await resp.json();
  } catch (e) {
    /* no JSON body */
  }
  if (!resp.ok) {
    const detail = body && body.detail ? body.detail : resp.statusText;
    throw new Error(detail);
  }
  return body;
}

// ---------------------------------------------------------------------------
// Status badge + defaults
// ---------------------------------------------------------------------------
async function refreshStatus() {
  const badge = $("status-badge");
  try {
    const status = await api("/api/status");
    const sales = status.models.sales;
    const guests = status.models.guest_count;
    if (sales.trained && guests.trained) {
      badge.textContent = `models trained (v${sales.model_version.slice(0, 8)})`;
      badge.className = "badge badge-ok";
      $("btn-retrain").disabled = !status.has_training_data_on_file;
    } else {
      badge.textContent = "no trained models yet -- train below";
      badge.className = "badge badge-warn";
    }
    if (!status.sample_data_available) {
      document.querySelector('input[value="sample"]').disabled = true;
    }
  } catch (e) {
    badge.textContent = "status unavailable";
    badge.className = "badge badge-bad";
  }
}

function initDefaultDates() {
  $("forecast-start").value = todayStr(1);
  $("forecast-end").value = todayStr(14);
  $("actuals-date").value = todayStr(0);
  $("push-start").value = todayStr(1);
  $("push-end").value = todayStr(14);
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------
document.querySelectorAll('input[name="data-source"]').forEach((el) => {
  el.addEventListener("change", () => {
    $("upload-row").hidden = document.querySelector('input[name="data-source"]:checked').value !== "upload";
  });
});

$("btn-train").addEventListener("click", async () => {
  const statusEl = $("train-status");
  const resultEl = $("train-result");
  const source = document.querySelector('input[name="data-source"]:checked').value;
  const fetchWeather = $("train-fetch-weather").checked;

  const form = new FormData();
  form.set("use_sample_data", source === "sample" ? "true" : "false");
  form.set("fetch_weather", fetchWeather ? "true" : "false");

  if (source === "upload") {
    const fileInput = $("csv-file");
    if (!fileInput.files.length) {
      setStatus(statusEl, "Choose a CSV file first.", "bad");
      return;
    }
    form.set("file", fileInput.files[0]);
  }

  setStatus(statusEl, "Training… this can take a little while.");
  $("btn-train").disabled = true;
  resultEl.hidden = true;

  try {
    const result = await api("/api/train?" + new URLSearchParams({
      use_sample_data: form.get("use_sample_data"),
      fetch_weather: form.get("fetch_weather"),
    }), {
      method: "POST",
      body: source === "upload" ? form : undefined,
    });

    setStatus(statusEl, "Training complete.", "ok");
    resultEl.hidden = false;
    resultEl.innerHTML = Object.entries(result)
      .map(([target, info]) => `<div><strong>${target}</strong>: v${info.model_version} (${info.training_rows} rows)</div>`)
      .join("");
    refreshStatus();
  } catch (e) {
    setStatus(statusEl, "Training failed: " + e.message, "bad");
  } finally {
    $("btn-train").disabled = false;
  }
});

$("btn-retrain").addEventListener("click", async () => {
  const statusEl = $("train-status");
  setStatus(statusEl, "Retraining on last dataset…");
  $("btn-retrain").disabled = true;
  try {
    const result = await api("/api/retrain", { method: "POST" });
    setStatus(statusEl, "Retrained.", "ok");
    $("train-result").hidden = false;
    $("train-result").innerHTML = Object.entries(result)
      .map(([target, info]) => `<div><strong>${target}</strong>: v${info.model_version} (${info.training_rows} rows)</div>`)
      .join("");
    refreshStatus();
  } catch (e) {
    setStatus(statusEl, "Retrain failed: " + e.message, "bad");
  } finally {
    $("btn-retrain").disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Forecast
// ---------------------------------------------------------------------------
let forecastChart = null;

$("btn-forecast").addEventListener("click", async () => {
  const statusEl = $("forecast-status");
  const start = $("forecast-start").value;
  const end = $("forecast-end").value;
  if (!start || !end) {
    setStatus(statusEl, "Pick a start and end date.", "bad");
    return;
  }

  setStatus(statusEl, "Generating forecast…");
  $("btn-forecast").disabled = true;

  try {
    const result = await api("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ start, end, fetch_weather: $("forecast-fetch-weather").checked }),
    });
    renderForecast(result.forecast);
    setStatus(statusEl, `${result.forecast.length} day(s) generated.`, "ok");
  } catch (e) {
    setStatus(statusEl, "Forecast failed: " + e.message, "bad");
  } finally {
    $("btn-forecast").disabled = false;
  }
});

function renderForecast(days) {
  const tbody = document.querySelector("#forecast-table tbody");
  tbody.innerHTML = "";
  for (const d of days) {
    const tr = document.createElement("tr");
    const drivers = (d.drivers || []).map((x) => `<span class="pill">${x.description}</span>`).join("");
    tr.innerHTML = `
      <td>${d.date}</td>
      <td>${money(d.projected_sales)}</td>
      <td>${money(d.conf_interval_low.sales)} - ${money(d.conf_interval_high.sales)}</td>
      <td>${d.projected_guests}</td>
      <td>${d.conf_interval_low.guest_count} - ${d.conf_interval_high.guest_count}</td>
      <td>${drivers || "-"}</td>
    `;
    tbody.appendChild(tr);
  }

  const labels = days.map((d) => d.date);
  const p50 = days.map((d) => d.projected_sales);
  const p10 = days.map((d) => d.conf_interval_low.sales);
  const p90 = days.map((d) => d.conf_interval_high.sales);

  const ctx = document.getElementById("forecast-chart").getContext("2d");
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "P90",
          data: p90,
          borderWidth: 0,
          pointRadius: 0,
          fill: "+1",
          backgroundColor: "rgba(255, 122, 69, 0.12)",
        },
        {
          label: "P10",
          data: p10,
          borderWidth: 0,
          pointRadius: 0,
          fill: false,
          backgroundColor: "rgba(255, 122, 69, 0.12)",
        },
        {
          label: "Projected sales",
          data: p50,
          borderColor: "#ff7a45",
          backgroundColor: "#ff7a45",
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { color: "#93a0b8" }, grid: { color: "#2a3247" } },
        y: { ticks: { color: "#93a0b8" }, grid: { color: "#2a3247" } },
      },
      plugins: {
        legend: { labels: { color: "#e8ecf4" } },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// Log actuals
// ---------------------------------------------------------------------------
$("btn-log-actuals").addEventListener("click", async () => {
  const statusEl = $("actuals-status");
  const resultEl = $("actuals-result");
  const date = $("actuals-date").value;
  const sales = parseFloat($("actuals-sales").value);
  const guests = parseInt($("actuals-guests").value, 10);

  if (!date || isNaN(sales) || isNaN(guests)) {
    setStatus(statusEl, "Fill in date, sales, and guests.", "bad");
    return;
  }

  setStatus(statusEl, "Logging…");
  try {
    const result = await api("/api/log-actuals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date, sales, guests }),
    });
    setStatus(statusEl, "Logged.", "ok");
    resultEl.hidden = false;
    resultEl.innerHTML = result.results
      .map((r) => {
        if (!r.had_forecast) return `<div><strong>${r.target}</strong>: no forecast on file for this date.</div>`;
        const biasLabel = r.bias > 0 ? "over-predicted" : r.bias < 0 ? "under-predicted" : "on target";
        return `<div><strong>${r.target}</strong>: actual ${r.actual}, predicted ${r.predicted.toFixed(1)}, ` +
          `error ${(r.pct_error * 100).toFixed(1)}% (${biasLabel})</div>`;
      })
      .join("") +
      result.drift
        .filter((d) => d.drift_detected)
        .map((d) => `<div class="pill" style="color:#f1b44c;background:#3a2f14;">Drift detected on ${d.target} (${(d.mean_bias_pct * 100).toFixed(1)}% bias) -- consider retraining.</div>`)
        .join("");
  } catch (e) {
    setStatus(statusEl, "Failed: " + e.message, "bad");
  }
});

// ---------------------------------------------------------------------------
// Accuracy
// ---------------------------------------------------------------------------
async function refreshAccuracy() {
  const target = $("accuracy-target").value;
  try {
    const summary = await api(`/api/accuracy/${target}`);
    const tbody = document.querySelector("#accuracy-table tbody");
    tbody.innerHTML = "";
    for (const [window, m] of Object.entries(summary)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${window.replace("trailing_", "")}</td>
        <td>${(m.wape * 100).toFixed(1)}%</td>
        <td>${(m.mape * 100).toFixed(1)}%</td>
        <td>${(m.bias_pct * 100).toFixed(1)}%</td>
        <td>${m.accuracy_pct.toFixed(1)}%</td>
        <td>${m.n_observations}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (e) {
    /* leave table as-is */
  }
}
$("btn-accuracy").addEventListener("click", refreshAccuracy);
$("accuracy-target").addEventListener("change", refreshAccuracy);

// ---------------------------------------------------------------------------
// Push projections
// ---------------------------------------------------------------------------
$("btn-push").addEventListener("click", async () => {
  const statusEl = $("push-status");
  const endpoint = $("push-endpoint").value;
  const apiKey = $("push-api-key").value;
  const start = $("push-start").value;
  const end = $("push-end").value;

  if (!endpoint || !apiKey || !start || !end) {
    setStatus(statusEl, "Fill in endpoint, API key, and dates.", "bad");
    return;
  }

  setStatus(statusEl, "Pushing…");
  $("btn-push").disabled = true;
  try {
    const result = await api("/api/push-projections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint, api_key: apiKey, start, end, fetch_weather: false }),
    });
    setStatus(statusEl, `Pushed (HTTP ${result.status_code}).`, "ok");
  } catch (e) {
    setStatus(statusEl, "Push failed: " + e.message, "bad");
  } finally {
    $("btn-push").disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
initDefaultDates();
refreshStatus();
refreshAccuracy();
