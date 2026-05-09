const healthStatus = document.querySelector("#healthStatus");
const storeStatus = document.querySelector("#storeStatus");
const providerStatus = document.querySelector("#providerStatus");
const vectorCount = document.querySelector("#vectorCount");
const sourceCount = document.querySelector("#sourceCount");
const uploadForm = document.querySelector("#uploadForm");
const documentInput = document.querySelector("#documentInput");
const fileLabel = document.querySelector("#fileLabel");
const uploadStatus = document.querySelector("#uploadStatus");
const clearButton = document.querySelector("#clearButton");
const queryForm = document.querySelector("#queryForm");
const queryInput = document.querySelector("#queryInput");
const answerOutput = document.querySelector("#answerOutput");
const verdictBadge = document.querySelector("#verdictBadge");
const traceOutput = document.querySelector("#traceOutput");

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || payload.error || `Request failed: ${response.status}`);
  }

  return payload;
}

async function refreshHealth() {
  try {
    const health = await requestJson("/health");
    healthStatus.textContent = health.status;
    storeStatus.textContent = health.vector_store_ready ? "Ready" : "Waiting";
    providerStatus.textContent = health.llm_provider;
    vectorCount.textContent = health.vector_count ?? 0;
    sourceCount.textContent = health.source_count ?? 0;
  } catch (error) {
    healthStatus.textContent = "Offline";
    storeStatus.textContent = "Unknown";
    providerStatus.textContent = "-";
    vectorCount.textContent = "0";
    sourceCount.textContent = "0";
  }
}

documentInput.addEventListener("change", () => {
  const file = documentInput.files[0];
  fileLabel.textContent = file ? file.name : "Choose or drop a document";
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = documentInput.files[0];

  if (!file) {
    uploadStatus.textContent = "Choose a document first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", file);
  uploadStatus.textContent = "Indexing document...";
  uploadForm.querySelector("button").disabled = true;

  try {
    const result = await requestJson("/documents/upload", {
      method: "POST",
      body: formData,
    });
    uploadStatus.textContent = `${result.file_name} indexed into ${result.total_chunks} chunk(s).`;
    await refreshHealth();
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    uploadForm.querySelector("button").disabled = false;
  }
});

clearButton.addEventListener("click", async () => {
  clearButton.disabled = true;
  uploadStatus.textContent = "Clearing index...";

  try {
    const result = await requestJson("/documents", { method: "DELETE" });
    uploadStatus.textContent = result.message;
    answerOutput.textContent = "Upload a document, then ask a question to start.";
    verdictBadge.textContent = "Waiting";
    traceOutput.innerHTML = "";
    await refreshHealth();
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    clearButton.disabled = false;
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();

  if (!query) {
    answerOutput.textContent = "Ask a question first.";
    return;
  }

  answerOutput.textContent = "Agents are working...";
  verdictBadge.textContent = "Running";
  traceOutput.innerHTML = "";
  queryForm.querySelector("button").disabled = true;

  try {
    const result = await requestJson("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    answerOutput.textContent = result.final_answer;
    verdictBadge.textContent = result.verdict || "Done";
    renderTrace(result.agent_logs || []);
  } catch (error) {
    answerOutput.textContent = error.message;
    verdictBadge.textContent = "Error";
  } finally {
    queryForm.querySelector("button").disabled = false;
  }
});

function renderTrace(logs) {
  traceOutput.innerHTML = "";

  if (!logs.length) {
    return;
  }

  for (const log of logs) {
    const pill = document.createElement("div");
    pill.className = "trace-pill";
    const latency = log.latency_ms ? `${log.latency_ms}ms` : log.skipped ? "skipped" : "complete";
    pill.innerHTML = `<span>${log.agent}</span><strong>${latency}</strong>`;
    traceOutput.appendChild(pill);
  }
}

refreshHealth();
