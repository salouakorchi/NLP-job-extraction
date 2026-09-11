const runButtons = document.querySelectorAll("[data-step]");
const providerSelect = document.getElementById("llm_provider");
const modelInput = document.getElementById("model");
const endpointInput = document.getElementById("endpoint");
const customEndpoint = document.getElementById("customEndpoint");
const apiAlert = document.getElementById("apiAlert");
const apiAlertMessage = document.getElementById("apiAlertMessage");
const apiAlertDetails = document.getElementById("apiAlertDetails");

function updateMetrics(status) {
  if (!status) return;
  for (const [key, value] of Object.entries(status)) {
    const target = document.querySelector(`[data-metric="${key}"]`);
    if (target) target.textContent = value ?? "0";
  }

  const strategyList = document.getElementById("strategyList");
  if (strategyList && status.extractions) {
    strategyList.innerHTML = Object.entries(status.extractions)
      .map(([name, item]) => `
        <div class="strategy-row">
          <strong>${name}</strong>
          <span>${item.success} succes</span>
          <span>${item.errors} erreurs</span>
          <span>${item.total} total</span>
        </div>
      `)
      .join("");
  }

  updateApiAlert(status.api_alert);
}

function updateApiAlert(alert) {
  if (!apiAlert) return;
  if (!alert) {
    apiAlert.hidden = true;
    apiAlert.classList.remove("visible");
    if (apiAlertMessage) apiAlertMessage.textContent = "";
    if (apiAlertDetails) apiAlertDetails.textContent = "";
    return;
  }

  apiAlert.hidden = false;
  apiAlert.classList.add("visible");
  if (apiAlertMessage) apiAlertMessage.textContent = alert.message || "Limite API atteinte.";
  if (apiAlertDetails) {
    const context = [alert.strategy, alert.split].filter(Boolean).join(" - ");
    apiAlertDetails.textContent = context ? `Contexte: ${context}` : "";
  }
}

function syncProviderFields(overwriteModel = false) {
  if (!providerSelect) return;
  const option = providerSelect.options[providerSelect.selectedIndex];
  const isCustom = option?.dataset.custom === "1";
  if (customEndpoint) customEndpoint.classList.toggle("visible", isCustom);
  if (modelInput) {
    modelInput.placeholder = option?.dataset.model || "";
    if (overwriteModel) modelInput.value = option?.dataset.model || "";
  }
  if (endpointInput && overwriteModel) {
    endpointInput.value = option?.dataset.endpoint || "";
  }
}

if (providerSelect) {
  syncProviderFields(false);
  providerSelect.addEventListener("change", () => syncProviderFields(true));
}

function initExtractionDropdowns() {
  document.querySelectorAll("[data-extraction-picker]").forEach((picker) => {
    const select = picker.querySelector("[data-extraction-select]");
    const emptyState = picker.querySelector("[data-extraction-empty]");
    const records = Array.from(picker.querySelectorAll("[data-extraction-record]"));
    if (!select || records.length === 0) return;

    const updateVisibleRecord = () => {
      const selectedIndex = select.value;
      records.forEach((record) => {
        record.hidden = record.dataset.extractionRecord !== selectedIndex;
      });
      if (emptyState) emptyState.hidden = selectedIndex !== "";
    };

    select.addEventListener("change", updateVisibleRecord);
    updateVisibleRecord();
  });
}

initExtractionDropdowns();

runButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    const step = button.dataset.step;
    runButtons.forEach((item) => (item.disabled = true));

    try {
      const response = await fetch(`/api/run/${step}`, { method: "POST" });
      const payload = await response.json();
      updateMetrics(payload.status);
    } catch (error) {
      updateApiAlert({
        message: error.message || "Erreur pendant l'execution de l'etape.",
      });
    } finally {
      runButtons.forEach((item) => (item.disabled = false));
    }
  });
});
