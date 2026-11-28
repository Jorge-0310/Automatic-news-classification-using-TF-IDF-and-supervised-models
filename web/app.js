/* NewsScope — solo consume la API local; no guarda entradas ni entrena modelos. */
const $ = (selector) => document.querySelector(selector);
const classes = ["World", "Sports", "Business", "Sci/Tech"];
const examples = {
  World: "Leaders from several countries met to discuss a new international peace agreement.",
  Sports: "The football team won the championship after scoring three goals in the final match.",
  Business: "Shares of the bank rose after quarterly earnings beat market estimates.",
  "Sci/Tech": "Researchers launched a spacecraft to study distant planets with a new telescope."
};
const formatScore = (value) => Number(value).toLocaleString("es-CO", { minimumFractionDigits: 4, maximumFractionDigits: 4 });
const formatPercent = (value) => `${(Number(value) * 100).toLocaleString("es-CO", { maximumFractionDigits: 1 })} %`;
const formatNumber = (value) => Number(value).toLocaleString("es-CO");
const pipelineNames = {
  logistic_unigrams: "Logística · unigramas",
  logistic_bigrams: "Logística · unigramas y bigramas",
  complement_nb_bigrams: "ComplementNB · unigramas y bigramas",
  sgd_bigrams: "SGD · unigramas y bigramas"
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}
function clear(node) { node.replaceChildren(); }
async function getJSON(path, options) {
  const response = await fetch(path, { headers: { "Accept": "application/json", ...options?.headers }, ...options });
  let body;
  try { body = await response.json(); } catch { throw new Error("Respuesta no válida del servidor."); }
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Error HTTP ${response.status}.`);
  return body;
}
function activate(view, focus = false, updateHash = false) {
  const target = document.getElementById(view) ? view : "clasificar";
  document.querySelectorAll(".nav-tab").forEach((tab) => {
    const active = tab.dataset.view === target;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    if (focus && active) tab.focus();
  });
  document.querySelectorAll(".view").forEach((section) => { section.hidden = section.id !== target; });
  if (updateHash && location.hash !== `#${target}`) history.replaceState(null, "", `#${target}`);
}
document.querySelectorAll(".nav-tab").forEach((tab, index, all) => {
  tab.addEventListener("click", () => activate(tab.dataset.view, false, true));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? all.length - 1 :
      (index + (event.key === "ArrowRight" ? 1 : -1) + all.length) % all.length;
    activate(all[next].dataset.view, true, true);
  });
});
window.addEventListener("hashchange", () => activate(location.hash.slice(1)));
activate(location.hash.slice(1));

const input = $("#news-text");
const button = $("#predict-button");
const predictionStatus = $("#prediction-status");
input.addEventListener("input", () => { $("#char-count").textContent = `${input.value.length} / 5000`; });
document.querySelectorAll("[data-example]").forEach((chip) => {
  chip.addEventListener("click", () => {
    input.value = examples[chip.dataset.example];
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
});

function bar(label, value, prominent = false) {
  const wrap = element("div");
  const title = element("div", undefined, "bar-title");
  title.append(element("span", label), element("strong", formatPercent(value)));
  const track = element("div", undefined, "bar-track");
  const fill = element("div", undefined, `bar-fill${prominent ? " leading" : ""}`);
  fill.style.width = `${Math.min(100, Math.max(0, Number(value) * 100))}%`;
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", label);
  track.setAttribute("aria-valuenow", String(Math.round(Number(value) * 100)));
  track.setAttribute("aria-valuemin", "0");
  track.setAttribute("aria-valuemax", "100");
  track.append(fill);
  wrap.append(title, track);
  return wrap;
}

$("#predict-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = input.value.trim();
  clear($("#probability-bars"));
  $("#prediction-result").hidden = true;
  $("#result-placeholder").hidden = false;
  predictionStatus.classList.remove("error");
  if (!text) {
    predictionStatus.textContent = "Introduce una noticia en inglés antes de clasificar.";
    input.focus();
    return;
  }
  button.disabled = true;
  button.firstChild.textContent = "Clasificando… ";
  predictionStatus.textContent = "Analizando el texto con el modelo local…";
  try {
    const result = await getJSON("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text })
    });
    $("#predicted-class").textContent = result.predicted_class;
    const sorted = classes.slice().sort((a, b) => Number(result.probabilities[b]) - Number(result.probabilities[a]));
    sorted.forEach((name) => $("#probability-bars").append(bar(
      name, result.probabilities[name], name === result.predicted_class
    )));
    $("#prediction-result").hidden = false;
    $("#result-placeholder").hidden = true;
    predictionStatus.textContent = "Predicción completada. Revisa las cuatro probabilidades.";
  } catch (error) {
    predictionStatus.classList.add("error");
    predictionStatus.textContent = error.message || "No se pudo contactar la API.";
  } finally {
    button.disabled = false;
    button.firstChild.textContent = "Clasificar noticia ";
  }
});

function renderMetrics(data) {
  $("#kpi-accuracy").textContent = formatScore(data.test.accuracy);
  $("#kpi-f1").textContent = formatScore(data.test.f1_macro);
  const selected = data.comparison.find((row) => row.id === data.selected_id);
  $("#kpi-model").textContent = pipelineNames[data.selected_id] || (selected ? `${selected.model} · ${selected.ngram_range}` : data.selected_id);
  $("#kpi-size").textContent = `${formatNumber(data.train_count)} / ${formatNumber(data.test_count)}`;
  $("#validation-count").textContent = `${formatNumber(data.validation_count)} EJEMPLOS`;
  $("#test-count").textContent = `${formatNumber(data.test_count)} EJEMPLOS`;
  const tbody = $("#comparison-body");
  clear(tbody);
  data.comparison.forEach((row) => {
    const tr = element("tr");
    if (row.id === data.selected_id) tr.className = "selected-row";
    [pipelineNames[row.id] || row.id.replaceAll("_", " "), formatScore(row.f1_macro), formatScore(row.accuracy),
      `${Number(row.train_seconds).toLocaleString("es-CO", { maximumFractionDigits: 1 })} s`
    ].forEach((value) => tr.append(element("td", value)));
    tbody.append(tr);
  });
  const classBars = $("#class-bars");
  clear(classBars);
  data.classes.forEach((name) => classBars.append(bar(name, data.test.per_class[name]["f1-score"])));
  const head = $("#matrix-head");
  const body = $("#matrix-body");
  clear(head); clear(body);
  const headerRow = element("tr");
  headerRow.append(element("th", "Real ↓ / Predicha →"));
  data.classes.forEach((name) => headerRow.append(element("th", name)));
  head.append(headerRow);
  data.confusion_matrix.forEach((row, i) => {
    const tr = element("tr");
    tr.append(element("th", data.classes[i]));
    row.forEach((value, j) => tr.append(element(
      "td", formatNumber(value), i === j ? "correct" : Number(value) >= 50 ? "confused" : ""
    )));
    body.append(tr);
  });
  $("#method-note").textContent = data.methodology;
  $("#metrics-status").hidden = true;
  $("#metrics-content").hidden = false;
}
function renderErrors(data) {
  if (data.status !== "confirmed") {
    $("#errors-status").textContent = "Aún no existe un resumen de errores confirmados por el equipo. No se mostrarán borradores de IA como evidencia revisada.";
    return;
  }
  $("#error-count").textContent = formatNumber(data.count);
  const pairs = $("#error-pairs");
  clear(pairs);
  data.pairs.forEach((item) => {
    const row = element("tr");
    row.append(element("td", `${item.true_class} → ${item.predicted_class}`),
               element("td", formatNumber(item.count)));
    pairs.append(row);
  });
  const limitations = $("#error-limitations");
  clear(limitations);
  data.limitations.forEach((description) => limitations.append(element("li", description)));
  $("#errors-status").hidden = true;
  $("#errors-content").hidden = false;
}

async function initialize() {
  const [health, metrics, errors] = await Promise.allSettled([
    getJSON("/api/health"), getJSON("/api/metrics"), getJSON("/api/errors/summary")
  ]);
  const status = $("#system-status");
  if (health.status === "fulfilled") {
    const available = health.value.model_available;
    status.classList.add(available ? "ready" : "offline");
    status.textContent = available ? "Modelo disponible" : "Modelo no entrenado";
    if (!available) {
      predictionStatus.classList.add("error");
      predictionStatus.textContent = "Modelo no disponible. Ejecuta python src/train.py antes de clasificar.";
    }
  } else {
    status.classList.add("offline");
    status.textContent = "API no disponible";
    predictionStatus.classList.add("error");
    predictionStatus.textContent = "No se pudo conectar con la API local.";
  }
  if (metrics.status === "fulfilled") renderMetrics(metrics.value);
  else {
    $("#metrics-status").classList.add("error");
    $("#metrics-status").textContent = `No se pudieron cargar las métricas: ${metrics.reason.message}`;
  }
  if (errors.status === "fulfilled") renderErrors(errors.value);
  else {
    $("#errors-status").classList.add("error");
    $("#errors-status").textContent = `No se pudo cargar el resumen: ${errors.reason.message}`;
  }
}
initialize();
