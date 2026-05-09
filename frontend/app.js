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
const answerBox = document.querySelector("#answerBox");
const chatHistory = document.querySelector("#chatHistory");
const answerOutput = document.querySelector("#answerOutput");
const verdictBadge = document.querySelector("#verdictBadge");
const reformulatedOutput = document.querySelector("#reformulatedOutput");
const chunkCountOutput = document.querySelector("#chunkCountOutput");
const traceOutput = document.querySelector("#traceOutput");
const sampleButtons = document.querySelectorAll("[data-query]");

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

sampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    queryInput.value = button.dataset.query;
    queryInput.focus();
  });
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
    resetChat();
    verdictBadge.textContent = "Waiting";
    reformulatedOutput.textContent = "-";
    chunkCountOutput.textContent = "0";
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
    setAnswer("Ask a question first.");
    return;
  }

  answerBox.classList.add("is-running");
  answerBox.classList.remove("is-idle");
  appendMessage("user", "You", query);
  queryInput.value = "";
  const pendingMessage = appendMessage(
    "assistant",
    "VaultMind",
    "VaultMind is routing your question, retrieving context, and checking the answer..."
  );
  verdictBadge.textContent = "Running";
  reformulatedOutput.textContent = "-";
  chunkCountOutput.textContent = "0";
  traceOutput.innerHTML = "";
  queryForm.querySelector("button").disabled = true;

  try {
    const result = await requestJson("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    updateMessage(pendingMessage, result.final_answer);
    verdictBadge.textContent = result.verdict || "Done";
    reformulatedOutput.textContent = result.reformulated_query || "-";
    chunkCountOutput.textContent = result.chunk_count ?? 0;
    renderTrace(result.agent_logs || []);
  } catch (error) {
    updateMessage(pendingMessage, error.message);
    verdictBadge.textContent = "Error";
  } finally {
    answerBox.classList.remove("is-running");
    queryForm.querySelector("button").disabled = false;
  }
});

function setAnswer(text) {
  const currentAnswerOutput = document.querySelector("#answerOutput");
  if (currentAnswerOutput) {
    currentAnswerOutput.innerHTML = formatAnswer(text || "No answer returned.");
    return;
  }

  appendMessage("assistant", "VaultMind", text || "No answer returned.");
}

function resetChat() {
  chatHistory.innerHTML = `
    <div class="message assistant-message">
      <div class="message-avatar">VM</div>
      <div class="message-content">
        <span class="message-label">VaultMind</span>
        <div id="answerOutput" class="answer-output">
          ${formatAnswer("Upload a document, then ask as many questions as you want. Your conversation will stay here.")}
        </div>
      </div>
    </div>
  `;
}

function appendMessage(role, label, text) {
  const message = document.createElement("div");
  message.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;
  message.innerHTML = `
    <div class="message-avatar">${role === "user" ? "You" : "VM"}</div>
    <div class="message-content">
      <span class="message-label">${escapeHtml(label)}</span>
      <div class="answer-output">${formatAnswer(text)}</div>
    </div>
  `;
  chatHistory.appendChild(message);
  chatHistory.scrollTop = chatHistory.scrollHeight;
  return message;
}

function updateMessage(message, text) {
  const output = message.querySelector(".answer-output");
  output.innerHTML = formatAnswer(text || "No answer returned.");
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function formatAnswer(text) {
  const escaped = escapeHtml(text);
  const lines = escaped.split("\n").map((line) => line.trim()).filter(Boolean);

  if (!lines.length) {
    return "<p>No answer returned.</p>";
  }

  const bulletLines = lines.filter((line) => line.startsWith("-") || line.startsWith("*"));
  if (bulletLines.length === lines.length) {
    return `<ul>${bulletLines.map((line) => `<li>${line.replace(/^[-*]\s*/, "")}</li>`).join("")}</ul>`;
  }

  return lines.map((line) => `<p>${line}</p>`).join("");
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderTrace(logs) {
  traceOutput.innerHTML = "";

  if (!logs.length) {
    return;
  }

  for (const log of logs) {
    const pill = document.createElement("div");
    pill.className = "trace-pill";
    const latency = log.latency_ms ? `${log.latency_ms}ms` : log.skipped ? "skipped" : "complete";
    pill.innerHTML = `<span class="trace-dot"></span><span>${log.agent}</span><strong>${latency}</strong>`;
    traceOutput.appendChild(pill);
  }
}

refreshHealth();
