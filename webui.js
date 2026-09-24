"use strict";
const $ = id => document.getElementById(id);
const RAW_JSON_VERSION = "__raw_json__";
let META = null;
let FILES_BY_BASE = {};
let currentBase = null;
let currentTab = "settings";
let currentLogSel = null;
let logOffset = 0;
let wasRunning = false;
let savedAudioPaths = new Set();
let runSelection = new Set();   // files ticked for a selective run
let durationTimer = null;
let changingDir = false;
let editorBase = null, savedForm = null, saving = false, editorLoading = false;
let editorSeq = 0, folderSeq = 0, filesSeq = 0, transcriptSeq = 0, logSeq = 0, logListSeq = 0;
let connected = false, lastStatus = null, runBusy = false, pollBusy = false, activeRunId = null, statusSeq = 0;
let editorScope = "file", editorMeta = null, globalRevision = null, timelineSeq = 0, exportContext = null, exporting = false;
let pendingPreview = null, dirtyChoice = null, initialized = false, pollTimer = null;
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));

function formValues() {
  const values = editorScope === "global" ? {} : { _comment: ($("commentField") || { value: "" }).value };
  document.querySelectorAll("#editor [data-key]").forEach(el => {
    const raw = el.dataset.originalValue === el.value ? el.dataset.originalRaw : el.value;
    if (raw.trim() !== "" || (editorScope === "global" && el.dataset.explicit === "1")) values[el.dataset.key] = raw.trim();
  });
  return values;
}
function formDirty() {
  return !!editorBase && savedForm !== null && (JSON.stringify(formValues()) !== savedForm || !!document.querySelector("#editor input:invalid"));
}
function updateSaveState() {
  $("saveBtn").disabled = saving || editorLoading || !formDirty() || (editorScope === "global" && lastStatus?.running);
  $("saveBtn").textContent = saving ? "Saving…" : "Save";
  if (!saving && !editorLoading && editorBase) setMsg(formDirty() ? "Unsaved changes" : "All changes saved.", "");
}
function restoreSavedForm() {
  if (savedForm === null) return;
  const values = JSON.parse(savedForm);
  if ($("commentField")) $("commentField").value = values._comment || "";
  document.querySelectorAll("#editor [data-key]").forEach(el => {
    const value = values[el.dataset.key] || "";
    el.value = value === el.dataset.originalRaw ? el.dataset.originalValue : value;
    el.dataset.explicit = Object.hasOwn(values, el.dataset.key) ? "1" : "0";
    if (el.type === "checkbox") el.checked = el.value ? ["true", "1", "yes", "on"].includes(el.value) : el.dataset.globalBool === "true";
    syncSetting(el);
  });
  updateSaveState();
}
async function resolveUnsaved() {
  if (saving) return false;
  if (!formDirty()) return true;
  if (dirtyChoice) return false;
  $("dirtyText").textContent = `Save changes to ${editorScope === "global" ? "global settings" : editorBase} before continuing?`;
  const choice = await new Promise(resolve => {
    dirtyChoice = resolve;
    $("dirtyDialog").showModal();
  });
  if (choice === "save") return saveSidecar();
  if (choice === "discard") { restoreSavedForm(); return true; }
  return false;
}
function finishDirtyDialog(choice) {
  const resolve = dirtyChoice;
  dirtyChoice = null;
  $("dirtyDialog").close();
  if (resolve) resolve(choice);
}
function runScope() {
  // Only files still visible in the list can be part of the next run.
  const known = Object.keys(FILES_BY_BASE);
  for (const base of [...runSelection]) if (!known.includes(base)) runSelection.delete(base);
  return [...runSelection];
}
function updateRunControls() {
  const running = !!(lastStatus && lastStatus.running);
  const pipeline = $("pipelineSel");
  // A reload or another tab can attach to a run with a different pipeline.
  // Show the server's active pipeline before locking the selector.
  if (running && pipeline.value !== lastStatus.pipeline &&
      [...pipeline.options].some(option => option.value === lastStatus.pipeline)) {
    pipeline.value = lastStatus.pipeline;
    filterSettings();
  }
  const chosen = runScope().length;
  $("startBtn").textContent = "Run";
  $("startBtn").title = chosen ? `Run ${chosen} selected file(s)` : "Run eligible files in this folder";
  $("clearRunSelection").hidden = !chosen;
  const scribeGate = $("pipelineSel").value === "scribe" && !scribeSelectionReady();
  $("startBtn").disabled = !connected || !initialized || running || runBusy || changingDir || saving || editorLoading || scribeGate;
  $("stopBtn").disabled = !connected || !running || runBusy;
  for (const id of ["pipelineSel", "dirBtn", "dirInput", "dirMenuBtn", "dirBrowse"]) {
    $(id).disabled = !connected || running || runBusy || changingDir;
  }
  if ($("dirMenuBtn").disabled) closeDirMenu();
  if (editorScope === "global") $("saveBtn").disabled = saving || editorLoading || !formDirty() || running;
}
function reportError(error) { setMsg(error.message, "err"); }
function handle(action) { return (...args) => Promise.resolve().then(() => action(...args)).catch(reportError); }


// A native dialog waits for the user, and the server gives up after 300 s.
const DIALOG_TIMEOUT_MS = 310000;
async function api(path, opts) {
  const res = await fetch(path, { ...opts, signal: opts?.signal || AbortSignal.timeout(opts?.timeout || 15000) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok && res.status !== 422) throw new Error(data.detail || res.statusText);
  return { status: res.status, data };
}

/* ---------- font size ---------- */

function setFontSize(px) {
  px = Math.max(14, Math.min(32, px));
  document.documentElement.style.fontSize = px + "px";
  try { localStorage.setItem("swhisper_font", px); } catch (e) {}
}
function initFontSize() {
  let px = 22;
  try { px = parseInt(localStorage.getItem("swhisper_font")) || 22; } catch (e) {}
  setFontSize(px);
}
function nudgeFont(delta) {
  const cur = parseFloat(getComputedStyle(document.documentElement).fontSize);
  setFontSize(cur + delta);
}

/* ---------- meta & file list ---------- */

async function loadMeta() {
  META = (await api("/api/meta")).data;
  $("dirInput").value = META.audio_dir;
  const paths = [...(META.audio_paths || [])];
  if (META.audio_dir && !paths.includes(META.audio_dir)) paths.push(META.audio_dir);
  savedAudioPaths = new Set(paths);
  renderDirMenu(paths);
  const sel = $("pipelineSel");
  const previous = sel.value;
  sel.innerHTML = "";
  for (const p of META.pipelines) {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = p.label;
    sel.appendChild(o);
  }
  if ([...sel.options].some(o => o.value === previous)) sel.value = previous;
}

async function changeDir() {
  if (changingDir || runBusy || lastStatus?.running) return;
  const path = $("dirInput").value.trim();
  if (!path || path === META?.audio_dir) return;
  if (!await resolveUnsaved()) { $("dirInput").value = META?.audio_dir || ""; return; }
  changingDir = true;
  updateRunControls();
  try {
    const { data } = await api("/api/audio-dir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    $("dirInput").value = data.audio_dir;
    folderSeq++; editorSeq++; transcriptSeq++; renderSeq++; logSeq++; logListSeq++;
    resetAudioReview();
    currentBase = editorBase = savedForm = null;
    txtVersionsByBase = {};
    editorLoading = false;
    $("editor").inert = false;
    $("editor").innerHTML = '<div class="placeholder">Select an audio file.</div>';
    $("transcriptText").innerHTML = '<div class="placeholder">Select an audio file to view its transcript.</div>';
    $("versionSel").innerHTML = "";
    $("versionSel").disabled = true;
    compareCanRender = {};
    loadComparison(null);
    $("logSel").innerHTML = "";
    $("logSel").disabled = true;
    $("log").textContent = "";
    $("consoleLog").textContent = "";
    currentLogSel = null;
    logOffset = 0;
    showTab("settings");
    await loadMeta();
    await loadFiles();
  } catch (e) { reportError(e); $("dirInput").value = META?.audio_dir || ""; }
  finally { changingDir = false; updateRunControls(); }
}

async function browseAudioDir() {
  if (changingDir || runBusy || lastStatus?.running) return;
  changingDir = true;
  updateRunControls();
  try {
    const { data } = await api("/api/audio-dir/pick", { method: "POST", timeout: DIALOG_TIMEOUT_MS });
    if (!data?.path) return;
    $("dirInput").value = data.path;
    changingDir = false;  // changeDir owns the busy flag once it takes over.
    if (data.path !== META?.audio_dir && await resolveUnsaved()) return changeDir();
    $("dirInput").value = META?.audio_dir || data.path;
  } catch (e) {
    if (!/canceled/i.test(e.message || "")) reportError(e);
  } finally {
    changingDir = false;
    updateRunControls();
  }
}

function changeDirIfSavedPath() {
  const path = $("dirInput").value.trim();
  if (path && savedAudioPaths.has(path) && (!META || path !== META.audio_dir)) changeDir();
}

function renderDirMenu(paths) {
  const menu = $("dirMenu");
  menu.innerHTML = "";
  for (const path of paths) {
    const option = document.createElement("button");
    option.type = "button";
    option.setAttribute("role", "menuitem");
    option.title = path;
    // The menu takes the focus when it opens, and only the popover's own show
    // algorithm can place it: beforetoggle is too early (the menu has not
    // reached the top layer, so nothing in it is focusable yet) and toggle is
    // too late (it arrives in a task of its own, after any keypress that
    // follows the click). autofocus is read during the show itself. It cannot
    // steal focus when this list is rebuilt, because document autofocus only
    // applies while the page is loading.
    if (!menu.children.length) option.autofocus = true;
    if (path === META.audio_dir) option.setAttribute("aria-current", "true");
    const name = document.createElement("span");
    name.className = "dirMenuName";
    name.textContent = path.replace(/\/+$/, "").split("/").pop() || path;
    const full = document.createElement("span");
    full.className = "dirMenuPath";
    // The path keeps its own direction inside a right-to-left line box, so the
    // leading slash stays put while the line is trimmed at the front.
    full.appendChild(document.createElement("bdi")).textContent = path;
    option.append(name, full);
    option.onclick = handle(() => {
      closeDirMenu();
      $("dirInput").value = path;
      return changeDir();
    });
    menu.appendChild(option);
  }
  if (!paths.length) {
    const empty = document.createElement("p");
    empty.className = "dirMenuEmpty";
    empty.textContent = "No saved folders yet.";
    menu.appendChild(empty);
  }
  if (dirMenuOpen()) positionDirMenu();
}

function dirMenuOpen() {
  return $("dirMenu").matches(":popover-open");
}

// The popup lives in the top layer, so it is positioned against the viewport
// rather than against #dirCombo: left edge and width come from the path field
// itself, which is what the menu has to line up with.
function positionDirMenu() {
  const menu = $("dirMenu");
  const input = $("dirInput").getBoundingClientRect();
  const arrow = $("dirMenuBtn").getBoundingClientRect();
  menu.style.left = `${input.left}px`;
  menu.style.top = `${input.bottom + 4}px`;
  menu.style.width = `${arrow.right - input.left}px`;
  menu.style.maxHeight = `${Math.max(120, window.innerHeight - input.bottom - 16)}px`;
}

function openDirMenu() {
  if (dirMenuOpen() || $("dirMenuBtn").disabled) return;
  $("dirMenu").showPopover();   // beforetoggle unhides and places the menu.
}

function closeDirMenu() {
  if (!dirMenuOpen()) return;
  const inside = $("dirMenu").contains(document.activeElement);
  $("dirMenu").hidePopover();
  if (inside) $("dirMenuBtn").focus();
}

function focusDirMenuItem(index) {
  const items = [...$("dirMenu").querySelectorAll("button")];
  if (!items.length) return;
  items[(index + items.length) % items.length].focus();
}

async function loadFiles() {
  const seq = ++filesSeq, folder = folderSeq;
  const { data } = await api("/api/files");
  if (seq !== filesSeq || folder !== folderSeq) return;
  FILES_BY_BASE = Object.fromEntries(data.files.map(f => [f.base, f]));
  $("fileNote").textContent = data.note || "";
  $("fileNote").hidden = !data.note;
  renderFiles();
  // Durations are measured in the background; pick them up once they land.
  clearTimeout(durationTimer);
  if (data.durations_pending) durationTimer = setTimeout(handle(loadFiles), 3000);
}

function fmtDuration(seconds) {
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600), minutes = Math.floor(total % 3600 / 60);
  return hours ? `${hours} h ${String(minutes).padStart(2, "0")} min`
       : minutes ? `${minutes} min`
       : `${total} s`;
}
function renderFiles() {
  const box = $("files");
  box.innerHTML = "";
  const all = Object.values(FILES_BY_BASE);
  const query = $("fileSearch").value.toLocaleLowerCase();
  const filter = $("fileFilter").value;
  const rows = all.filter(f => f.base.toLocaleLowerCase().includes(query) &&
    (filter === "all" || filter === "unfinished" && !f.has_transcript || filter === "reprocess" && f.will_reprocess ||
     filter === "ready" && f.has_transcript || filter === "overrides" && f.sidecar));
  const stateRank = f => f.will_reprocess ? 0 : !f.transcribed ? 1 : !f.has_transcript ? 2 : 3;
  rows.sort((a, b) => ($("fileSort").value === "newest" ? (b.mtime || 0) - (a.mtime || 0) :
    $("fileSort").value === "status" ? stateRank(a) - stateRank(b) : 0) || a.base.localeCompare(b.base, undefined, {numeric:true}));
  $("fileCount").textContent = `${rows.length} of ${all.length} files`;
  if (!rows.length) {
    box.textContent = all.length ? "No files match these filters." : "No audio files found.";
    return;
  }
  for (const f of rows) {
    const div = document.createElement("div");
    div.className = "file" + (f.base === currentBase ? " sel" : "");
    div.dataset.base = f.base;
    div.setAttribute("role", "group");
    div.setAttribute("aria-label", f.base);
    const badges = [];
    const primary = f.will_reprocess ? "Needs reprocessing" : f.has_transcript ? "Transcript ready" : f.transcribed ? "Awaiting diarization" : "Not transcribed";
    badges.push(`<span class="badge ${f.will_reprocess ? "on-warn" : f.has_transcript ? "on-ok" : ""}">${primary}</span>`);
    if (f.duration) badges.push(`<span class="badge" title="Recording length">${escapeHTML(fmtDuration(f.duration))}</span>`);
    if (f.sidecar) badges.push('<span class="badge on-info">overrides</span>');
    if (f.comment) badges.push('<span class="badge">comment</span>');
    if (f.has_transcript) badges.push('<span class="badge tbadge" title="Open transcript">Transcript</span>');
    if (f.transcribed) badges.push('<span class="badge jbadge" title="View raw JSON transcript">JSON</span>');
    if (f.has_logs) badges.push('<span class="badge lbadge" title="Open logs">Logs</span>');
    div.innerHTML = `<label class="runPick"><input type="checkbox"><span class="visuallyHidden">Include in run</span></label>` +
      `<button class="name fileSelect">${escapeHTML(f.base)}</button><div class="badges">${badges.join("")}</div>`;
    const pick = div.querySelector(".runPick input");
    pick.checked = runSelection.has(f.base);
    pick.setAttribute("aria-label", `Include ${f.base} in the next run`);
    pick.onclick = ev => ev.stopPropagation();
    pick.onchange = () => {
      pick.checked ? runSelection.add(f.base) : runSelection.delete(f.base);
      updateRunControls();
    };
    const fileButton = div.querySelector(".fileSelect");
    fileButton.setAttribute("aria-pressed", String(f.base === currentBase));
    fileButton.setAttribute("aria-label", `Edit settings for ${f.base}`);
    fileButton.tabIndex = f.base === currentBase || (!rows.some(row => row.base === currentBase) && f === rows[0]) ? 0 : -1;
    div.onclick = handle(() => selectFile(f.base));
    for (const [selector, tab, version] of [[".tbadge", "transcript"], [".lbadge", "logs"], [".jbadge", "transcript", RAW_JSON_VERSION]]) {
      const badge = div.querySelector(selector);
      if (!badge) continue;
      // Native buttons provide Space/Enter handling and accessible focus.
      const button = document.createElement("button");
      button.className = badge.className;
      button.textContent = badge.textContent;
      button.title = badge.title;
      button.setAttribute("aria-label", `${badge.title}: ${f.base}`);
      badge.replaceWith(button);
      button.onclick = handle(async ev => {
        ev.stopPropagation();
        if (await selectFile(f.base)) {
          showTab(tab);
          if (version) await loadTranscript(f.base, version);
        }
      });
      // Stop synchronously; the async handler runs in a microtask.
      button.addEventListener("click", ev => ev.stopPropagation());
    }
    div.addEventListener("keydown", ev => {
      if (ev.target !== fileButton) return;
      const items = [...box.querySelectorAll(".file")];
      let target = items.indexOf(div);
      if (ev.key === "ArrowDown") target++;
      else if (ev.key === "ArrowUp") target--;
      else if (ev.key === "Home") target = 0;
      else if (ev.key === "End") target = items.length - 1;
      else return;
      ev.preventDefault();
      items[Math.max(0, Math.min(items.length - 1, target))].querySelector(".fileSelect").focus();
    });
    box.appendChild(div);
  }
}

/* ---------- sidecar editor ---------- */

function syncSetting(el) {
  const box = el.closest(".fieldValue");
  if (!box) return;
  const explicit = editorScope === "global" ? el.dataset.explicit === "1" : el.value !== "";
  el.classList.toggle("set", explicit);
  const note = box.querySelector(".inheritance");
  note.textContent = explicit ? (editorScope === "global" ? "global value" : "per-file override") :
    (editorScope === "global" ? "inherited program default" : "inherited global value");
  box.querySelector(".resetSetting").disabled = el.disabled || !explicit;
  box.querySelector(".resetSetting").hidden = !explicit;
  const error = box.querySelector(".fieldError");
  error.textContent = el.validity.valid ? "" : el.validationMessage;
  el.setAttribute("aria-invalid", String(!el.validity.valid));
}
function inputFor(key, spec, value) {
  const globalVal = spec.global === null || spec.global === undefined ? "" : String(spec.global);
  const box = document.createElement("div"); box.className = "fieldValue";
  let el;
  if (spec.type === "bool") {
    el = document.createElement("input"); el.type = "checkbox";
    const raw = value !== undefined ? String(value).toLowerCase() : "";
    const normalized = ["1", "yes", "on"].includes(raw) ? "true" : ["0", "no", "off"].includes(raw) ? "false" : raw;
    el.value = normalized;
    el.checked = raw ? ["true", "1", "yes", "on"].includes(raw) : ["true", "1", "yes", "on"].includes(globalVal.toLowerCase());
    el.dataset.globalBool = String(["true", "1", "yes", "on"].includes(globalVal.toLowerCase()));
    el.dataset.originalValue = normalized; el.dataset.originalRaw = raw;
    if (raw && !["true", "false"].includes(normalized)) el.setCustomValidity("Choose true or false, or use the inherited value.");
  } else if (key.includes("PREAMBLE") || key.includes("INITIAL_PROMPT")) {
    el = document.createElement("textarea"); el.rows = 3;
  } else if (spec.choices) {
    el = document.createElement("select");
    for (const choice of ["", ...spec.choices]) {
      const option = document.createElement("option"); option.value = choice; option.textContent = choice || `Use ${editorScope === "global" ? "program default" : "global"}`; el.appendChild(option);
    }
  } else {
    el = document.createElement("input");
    el.type = ["int", "float"].includes(spec.type) ? "number" : "text";
    if (el.type === "number") { el.step = spec.type === "int" ? "1" : "any"; if (spec.min !== undefined) el.min = spec.min; if (spec.max !== undefined) el.max = spec.max; }
  }
  if (spec.type !== "bool") {
    if (el.type === "number" && value !== undefined && !Number.isFinite(Number(value))) {
      el.type = "text"; el.dataset.numericType = spec.type; el.setCustomValidity("Enter a valid number.");
    }
    el.value = value ?? ""; el.placeholder = globalVal || "Not set";
  }
  el.dataset.explicit = value !== undefined ? "1" : "0";
  el.dataset.key = key; el.id = `setting-${key}`;
  el.setAttribute("aria-describedby", `help-${key} error-${key} inherit-${key} scribe-${key}`);
  const control = document.createElement("div"); control.className = spec.type === "bool" ? "boolControl" : "";
  control.appendChild(el);
  const inheritance = document.createElement("span"); inheritance.className = "inheritance"; inheritance.id = `inherit-${key}`;
  control.appendChild(inheritance);
  const reset = document.createElement("button"); reset.className = "resetSetting"; reset.type = "button";
  reset.textContent = editorScope === "global" ? "Use program default" : "Use global";
  reset.setAttribute("aria-label", `${reset.textContent} for ${spec.label || key}`);
  reset.onclick = () => {
    el.value = ""; el.dataset.explicit = "0";
    el.setCustomValidity("");
    if (el.type === "checkbox") { el.checked = el.dataset.globalBool === "true"; el.indeterminate = editorScope === "global"; }
    syncSetting(el); if (editorScope === "global") inheritance.textContent = "program default after saving"; updateSaveState();
  };
  control.appendChild(reset); box.appendChild(control);
  const help = document.createElement("div"); help.className = "fieldHelp"; help.id = `help-${key}`;
  help.textContent = `${spec.help || ""}${spec.type !== "bool" && editorScope !== "global" ? ` Global: ${globalVal || "not set"}.` : ""}`;
  box.appendChild(help);
  const scribeHelp = document.createElement("div"); scribeHelp.className = "fieldHelp scribeHelp";
  scribeHelp.id = `scribe-${key}`;
  const applicability = document.createElement("strong"); applicability.className = "settingApplicability";
  applicability.textContent = spec.scribe?.scope || (spec.scribe?.supported ? "Local processing after Scribe" : "Unavailable for Scribe");
  scribeHelp.append(applicability, document.createTextNode(spec.scribe?.description || "Not available for Scribe.")); scribeHelp.hidden = true; box.appendChild(scribeHelp);
  const error = document.createElement("div"); error.className = "fieldError"; error.id = `error-${key}`; box.appendChild(error);
  const change = () => {
    if (el.type === "checkbox") { el.value = String(el.checked); el.indeterminate = false; el.setCustomValidity(""); }
    if (el.dataset.numericType) {
      const n = Number(el.value);
      el.setCustomValidity(el.value && (!Number.isFinite(n) || (el.dataset.numericType === "int" && !Number.isInteger(n)) || (spec.min !== undefined && n < spec.min) || (spec.max !== undefined && n > spec.max)) ? "Enter a valid number within the allowed range." : "");
    }
    el.dataset.explicit = "1"; syncSetting(el); updateSaveState();
  };
  el.addEventListener("input", change); el.addEventListener("change", change);
  return box;
}
function fieldErrors(errors) {
  if (Object.keys(errors || {}).length && $("showOtherSettings")) { $("showOtherSettings").checked = true; filterSettings(); }
  for (const [key, message] of Object.entries(errors || {})) {
    const el = $(`setting-${key}`); if (!el) continue;
    $(`error-${key}`).textContent = message; el.setAttribute("aria-invalid", "true");
    const group = el.closest("details"); if (group) group.open = true;
  }
}
function renderSettings(data, base, scope) {
  editorScope = scope; editorMeta = scope === "global" ? data : META;
  const editor = $("editor"); editor.replaceChildren();
  const title = document.createElement("h2"); title.textContent = scope === "global" ? "Global settings" : base; editor.appendChild(title);
  if (scope === "file") {
    const comment = document.createElement("textarea"); comment.id = "commentField"; comment.rows = 3; comment.value = data.comment || "";
    comment.setAttribute("aria-label", "Comments about this file"); comment.placeholder = "Comments do not trigger reprocessing."; comment.style.width = "100%";
    comment.addEventListener("input", updateSaveState); editor.appendChild(comment);
  } else {
    const note = document.createElement("p"); note.className = "settingsNote";
    note.textContent = "Defaults for future runs. Per-file overrides take precedence. Changes are saved to " + data.path + ". Credentials are not shown."; editor.appendChild(note);
  }
  const applicabilityNote = document.createElement("p"); applicabilityNote.id = "settingsApplicability"; applicabilityNote.className = "settingsNote";
  applicabilityNote.textContent = "Scribe: these settings control local processing. Backfill-only settings take effect when backfill is enabled. SUNET transcription is controlled by the Scribe language/model selector. Values marked as inherited are editable; greyed-out controls are unavailable.";
  editor.appendChild(applicabilityNote);
  const common = ["SWHISPER_LANGUAGE", "SWHISPER_NUM_SPEAKERS", "SWHISPER_MIN_SPEAKERS", "SWHISPER_MAX_SPEAKERS", "SWHISPER_INITIAL_PROMPT", "SWHISPER_FORCE_REPROCESS", "SWHISPER_DIARIZE_FORCE_REPROCESS"];
  const groups = [{name:"Common settings",keys:common.filter(k=>editorMeta.params[k])}, ...editorMeta.groups.map(g=>({...g,keys:g.keys.filter(k=>!common.includes(k))}))];
  {
    const note = document.createElement("label"); note.className = "settingsNote";
    note.innerHTML = '<input id="showOtherSettings" type="checkbox"> Show settings for other pipelines'; editor.appendChild(note);
    note.querySelector("input").onchange = filterSettings;
  }
  for (const group of groups) {
    if (!group.keys.length) continue;
    const section = document.createElement("details"); section.className = "group";
    section.open = group.name === "Common settings" || group.keys.some(k=>Object.hasOwn(data.values,k));
    const summary = document.createElement("summary"); summary.textContent = group.name; section.appendChild(summary);
    for (const key of group.keys) {
      const spec = editorMeta.params[key]; if (!spec) continue;
      const row = document.createElement("div"); row.className = "row"; row.dataset.stage = spec.stage;
      const label = document.createElement("label"); label.htmlFor = `setting-${key}`;
      label.textContent = spec.label || key.replace("SWHISPER_", "").replaceAll("_", " ").toLowerCase(); label.title = key;
      row.append(label, inputFor(key, spec, data.values[key])); section.appendChild(row);
    }
    editor.appendChild(section);
  }
  if (data.unknown && Object.keys(data.unknown).length) {
    const note = document.createElement("p"); note.className="settingsNote"; note.textContent="Additional settings are preserved: " + Object.keys(data.unknown).join(", ");editor.appendChild(note);
  }
  editorBase = scope === "global" ? "__global__" : base;
  editor.querySelectorAll("[data-key]").forEach(syncSetting);
  savedForm = JSON.stringify(formValues()); editorLoading = false; editor.inert = false;
  $("editorBar").style.display = currentTab === "settings" ? "" : "none";
  $("deleteBtn").hidden = scope === "global"; $("deleteBtn").disabled = !data.exists;
  $("globalSettingsBtn").setAttribute("aria-pressed", String(scope === "global"));
  filterSettings(); updateSaveState(); updateRunControls();
}
function filterSettings() {
  const scribe = $("pipelineSel").value === "scribe";
  $("scribeControls").hidden = !scribe;
  $("runReprocessLabel").textContent = scribe ? "Transcribe again with Scribe" : "Reprocess existing outputs";
  if ($("settingsApplicability")) $("settingsApplicability").hidden = !scribe;
  const onlyDiarize = editorScope === "file" && $("pipelineSel").value === "diarize4" && !$("showOtherSettings")?.checked;
  document.querySelectorAll("#editor .row").forEach(row => {row.hidden = onlyDiarize && row.dataset.stage === "transcribe";});
  document.querySelectorAll("#editor details.group").forEach(group => {group.hidden = ![...group.querySelectorAll(".row")].some(row=>!row.hidden);});
  document.querySelectorAll("#editor .row").forEach(row => {
    const control = row.querySelector("[data-key]");
    const key = control.dataset.key;
    const supported = editorMeta.params[key]?.scribe?.supported === true;
    if (scribe) row.hidden = !supported && !$("showOtherSettings")?.checked;
    else if (key === "SWHISPER_SCRIBE_TIMESTAMP_OFFSET") row.hidden = true;
    control.disabled = scribe && !supported;
    row.classList.toggle("settingUnavailable", control.disabled);
    row.querySelectorAll(".resetSetting").forEach(button => {button.disabled = (scribe && !supported) || button.hidden;});
    row.querySelectorAll(".scribeHelp").forEach(help => {help.hidden = !scribe;});
  });
  document.querySelectorAll("#editor details.group").forEach(group => {group.hidden = ![...group.querySelectorAll(".row")].some(row=>!row.hidden);});
}
function scribeSelectionReady() {
  return $("scribeScope").value !== "selected" || Object.values(FILES_BY_BASE).some(file => file.base === currentBase);
}
function updateScribeGate() {
  if ($("pipelineSel").value !== "scribe" || pendingPreview?.pipeline !== "scribe") return;
  $("startBtn").disabled = !scribeSelectionReady();
}
async function openGlobalSettings() {
  if (changingDir || !await resolveUnsaved()) return;
  const seq = ++editorSeq; editorLoading = true; editorBase = savedForm = null;
  $("editor").inert = true; $("editor").textContent = "Loading global settings…";
  showTab("settings"); updateSaveState();
  try {
    const {data} = await api("/api/settings/global");
    if (seq !== editorSeq) return;
    globalRevision = data.revision; renderSettings(data, null, "global");
  } catch (e) { if (seq === editorSeq) {editorLoading = false;$("editor").inert=false;reportError(e);updateSaveState();} }
}
async function saveGlobalSettings() {
  if (saving || !formDirty()) return !saving;
  if (!validateForm()) return false;
  saving = true; $("editor").inert = true; updateSaveState(); updateRunControls();
  try {
    const {status,data} = await api("/api/settings/global", {method:"PUT",headers:{"Content-Type":"application/json"},body:JSON.stringify({values:formValues(),revision:globalRevision})});
    if (status === 422) {fieldErrors(data.errors);throw new Error("Correct the highlighted settings.");}
    globalRevision = data.revision;
    await loadMeta(); await loadFiles(); renderSettings(data, null, "global");
    return true;
  } catch(e) {reportError(e);return false;}
  finally {saving=false;$("editor").inert=false;$("saveBtn").disabled=!formDirty();$("saveBtn").textContent="Save";if(!formDirty())setMsg("Global defaults saved for future runs.","ok");updateRunControls();}
}
function validateForm() {
  const invalid = document.querySelector("#editor input:invalid");
  if (invalid) {if($("showOtherSettings")){$("showOtherSettings").checked=true;filterSettings();}invalid.closest("details").open=true;invalid.reportValidity();syncSetting(invalid);return false;}
  return true;
}

async function selectFile(base, force = false) {
  if (changingDir || saving) return false;
  if (!force && base === editorBase && base === currentBase) return true;
  if (!force && !await resolveUnsaved()) return false;
  resetAudioReview();
  const seq = ++editorSeq, folder = folderSeq;
  transcriptSeq++; renderSeq++; logSeq++; logListSeq++;
  currentBase = base;
  editorBase = savedForm = null;
  editorLoading = true;
  $("editor").inert = true;
  $("editor").textContent = "Loading settings…";
  $("deleteBtn").disabled = true;
  updateSaveState(); updateRunControls();
  updateScribeGate();
  showTab(currentTab);
  document.querySelectorAll("#files .file").forEach(d => {
    d.classList.toggle("sel", d.dataset.base === base);
    d.querySelector(".fileSelect")?.setAttribute("aria-pressed", String(d.dataset.base === base));
    if (d.querySelector(".fileSelect")) d.querySelector(".fileSelect").tabIndex = d.dataset.base === base ? 0 : -1;
  });
  let data;
  try { ({ data } = await api(`/api/sidecar/${encodeURIComponent(base)}`)); }
  catch (e) {
    if (seq === editorSeq && folder === folderSeq) {
      $("editor").textContent = "Could not load settings. Select the file to retry.";
      editorLoading = false; $("editor").inert = false;
      updateSaveState(); updateRunControls(); reportError(e);
    }
    return false;
  }
  if (seq !== editorSeq || folder !== folderSeq || base !== currentBase) return false;
  renderSettings(data, base, "file");
  return true;
}

function setMsg(text, cls) {
  const m = $("editorMsg");
  m.textContent = text;
  m.className = "msg " + (cls || "");
  $("uiNotice").hidden = cls !== "err";
  if (cls === "err") $("uiNotice").textContent = text;
}

/* ---------- transcript view ---------- */

function showTab(tab) {
  // The Show settings are shared, so they follow the user back from the comparison.
  if (currentTab === "compare" && tab === "transcript") carriedRenderOptions = renderOptions();
  currentTab = tab;
  for (const [id, name] of [["tabSettings", "settings"], ["tabTranscript", "transcript"], ["tabCompare", "compare"], ["tabLogs", "logs"]]) {
    $(id).setAttribute("aria-selected", String(name === tab));
    $(id).tabIndex = name === tab ? 0 : -1;
    $(id).classList.toggle("active", name === tab);
  }
  $("editor").style.display = tab === "settings" ? "" : "none";
  $("transcriptView").style.display = tab === "transcript" ? "" : "none";
  $("compareView").style.display = tab === "compare" ? "" : "none";
  $("logsView").style.display = tab === "logs" ? "" : "none";
  $("editorBar").style.display = (tab === "settings" && editorBase) ? "" : "none";
  // One Show bar serves both transcript views; move it to the visible one.
  if (tab === "compare") $("compareBar").after($("renderBar"));
  if (tab === "transcript") $("transcriptBar").after($("renderBar"));
  $("renderSpeakers").hidden = $("renderSave").hidden = tab === "compare";
  if (tab === "transcript" && currentBase) loadTranscript(currentBase);
  if (tab === "compare") loadComparison(currentBase);
  if (tab === "logs") handle(loadLogList)();
}

function fmtTimestamp(ts) {
  return `${ts.slice(0, 4)}-${ts.slice(4, 6)}-${ts.slice(6, 8)} ${ts.slice(9, 11)}:${ts.slice(11, 13)}:${ts.slice(13, 15)}`;
}

function fillVersionSel(txtVersions, hasRawJson, selectedId) {
  const sel = $("versionSel");
  sel.innerHTML = "";
  for (const v of txtVersions) {
    const o = document.createElement("option");
    o.value = v.id;
    o.textContent = (v.timestamp ? fmtTimestamp(v.timestamp) : v.filename) + (v.pipeline ? ` · ${v.pipeline}` : "");
    sel.appendChild(o);
  }
  if (hasRawJson) {
    const o = document.createElement("option");
    o.value = RAW_JSON_VERSION;
    o.textContent = "Raw JSON transcript (pre-diarization)";
    sel.appendChild(o);
  }
  sel.value = selectedId;
  sel.disabled = false;
  // The raw JSON view has no TXT file on disk to reveal.
  $("transcriptReveal").disabled = selectedId === RAW_JSON_VERSION;
}


/* ---------- render options ----------
   Which markers and pauses a transcript shows is recoverable from the diarized
   JSON, so these are toggles over a stored run rather than settings for a new
   one. Everything here re-renders server-side in tens of milliseconds. */

const RENDER_DEFAULTS = {
  stats: true, preamble: true, disfluencies: true, pauses: true, pause_linebreaks: true,
  pause_linebreak_threshold: 1.0, min_pause: 0,
};
let renderBaseline = {...RENDER_DEFAULTS};
let txtVersionsByBase = {};   // keeps the version list available on the raw JSON view
let renderSeq = 0;          // drops responses that arrive out of order
let renderTimer = null;
let carriedRenderOptions = null;   // Show settings brought back from the comparison

function debounceRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(renderOptionsChanged, 250);
}

function renderOptionsChanged() {
  if (currentTab === "compare") refreshComparison();
  else applyRenderOptions();
}
let renderAvailable = false;

function renderOptions() {
  return {
    stats: $("optStats").checked, preamble: $("optPreamble").checked,
    disfluencies: $("optDisfluencies").checked,
    pauses: $("optPauses").checked,
    pause_linebreaks: $("optBreaks").checked,
    pause_linebreak_threshold: parseFloat($("optThreshold").value) || 0,
    min_pause: parseFloat($("optMinPause").value) || 0,
  };
}

function renderIsDefault(o) {
  return Object.keys(RENDER_DEFAULTS).every(k => o[k] === renderBaseline[k]);
}

function setRenderOptions(o) {
  $("optStats").checked = o.stats; $("optPreamble").checked = o.preamble;
  $("optDisfluencies").checked = o.disfluencies;
  $("optPauses").checked = o.pauses;
  $("optBreaks").checked = o.pause_linebreaks;
  $("optThreshold").value = o.pause_linebreak_threshold;
  $("optMinPause").value = o.min_pause;
}

function updateRenderBar(available) {
  renderAvailable = available;
  $("renderBar").style.display = available ? "" : "none";
  if (!available) setRenderOptions(RENDER_DEFAULTS);
}

async function applyRenderOptions() {
  if (!renderAvailable || !currentBase) return;
  const base = currentBase;
  const version = $("versionSel").value;
  const folder = folderSeq;
  const options = renderOptions();
  const bar = $("renderBar");
  const seq = ++renderSeq;
  bar.classList.toggle("dirty", !renderIsDefault(options));
  bar.classList.add("busy");
  try {
    const { data } = await api(`/api/render/${encodeURIComponent(base)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version, options }),
    });
    if (seq !== renderSeq || base !== currentBase || folder !== folderSeq || version !== $("versionSel").value) return;   // superseded
    $("transcriptText").textContent = data.content;
    if (data.timeline) renderTimeline(data.timeline);
    $("transcriptMsg").textContent = data.suffix
      ? `${base} — re-rendered view (${data.suffix.slice(1).replace(/_/g, ", ")})`
      : `${base} — as produced by the pipeline`;
    $("transcriptMsg").className = "msg";
  } catch (e) {
    if (seq !== renderSeq) return;
    $("transcriptMsg").textContent = e.message;
    $("transcriptMsg").className = "msg err";
  } finally {
    if (seq === renderSeq) bar.classList.remove("busy");
  }
}

/* ---------- speaker names ---------- */

let speakerContext = null;

async function openSpeakerNames() {
  if (!renderAvailable || !currentBase) return;
  const base = currentBase, version = $("versionSel").value;
  $("speakerMsg").textContent = "";
  const { data } = await api(`/api/speakers/${encodeURIComponent(base)}?version=${encodeURIComponent(version)}`);
  if (base !== currentBase) return;
  speakerContext = { base, version };
  $("speakerRows").replaceChildren(...data.detected.flatMap(label => {
    const caption = document.createElement("label");
    caption.textContent = label; caption.htmlFor = `speaker-${label}`;
    const field = document.createElement("input");
    field.type = "text"; field.id = `speaker-${label}`; field.dataset.label = label;
    field.value = data.names[label] || ""; field.placeholder = label; field.maxLength = 60;
    return [caption, field];
  }));
  // Old recording-wide names are only offered: the person decides whether they fit this run.
  const unassigned = Object.entries(data.unassigned || {});
  $("speakerUnassigned").hidden = !unassigned.length;
  $("speakerUnassignedList").textContent = unassigned.map(([label, name]) => `${label} = ${name}`).join(", ");
  $("speakerUseUnassigned").onclick = () => {
    for (const [label, name] of unassigned) {
      const field = $("speakerRows").querySelector(`input[data-label="${CSS.escape(label)}"]`);
      if (field) field.value = name;
    }
  };
  $("speakerDialog").showModal();
}

async function saveSpeakerNames() {
  if (!speakerContext) return;
  const names = {};
  for (const field of $("speakerRows").querySelectorAll("input")) names[field.dataset.label] = field.value;
  try {
    const { status, data } = await api(`/api/speakers/${encodeURIComponent(speakerContext.base)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names, version: speakerContext.version }),
    });
    // api() reports 422 rather than throwing, so validation errors land here.
    if (status === 422) { $("speakerMsg").textContent = data.detail || "Check the names."; return; }
  } catch (e) { $("speakerMsg").textContent = e.message; return; }
  $("speakerDialog").close(); speakerContext = null;
  await applyRenderOptions();
}

async function saveRenderedView() {
  if (!renderAvailable || !currentBase) return;
  exportContext = {base:currentBase, version:$("versionSel").value, options:renderOptions()};
  $("exportFormat").value = "txt";
  $("exportPaths").textContent = "Choose a format, then select Export copy to choose the filename and location.";
  $("exportDialog").showModal();
}
async function confirmExport() {
  if (!exportContext || exporting) return;
  exporting=true;$("exportConfirm").disabled=true;
  for (const id of ["exportFormat","exportCancel"]) $(id).disabled=true;
  try {
    const body = {...exportContext, format: $("exportFormat").value};
    const {data: choice}=await api(`/api/render/${encodeURIComponent(exportContext.base)}/save-as`,{method:"POST",timeout:DIALOG_TIMEOUT_MS,headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
    if (choice.canceled) { $("exportDialog").close(); exportContext=null; return; }
    const {data}=await api(`/api/render/${encodeURIComponent(exportContext.base)}/save-selected`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token:choice.token})});
    $("exportPaths").textContent="Exported:\n"+data.written.join("\n");
    $("exportDialog").close(); exportContext=null;
  } catch(e) {$("exportPaths").textContent=e.message;}
  finally {exporting=false;$("exportConfirm").disabled=false;for (const id of ["exportFormat","exportCancel"]) $(id).disabled=false;}
}
function resetAudioReview() {
  timelineSeq++; $("audioPlayer").pause(); $("audioPlayer").removeAttribute("src");$("audioPlayer").load();
  $("audioBar").hidden=true;$("timeline").hidden=true;$("timeline").replaceChildren();$("reviewAudio").checked=false;$("transcriptText").style.display="";
}
async function loadAudioReview(base, version) {
  const seq=++timelineSeq, folder=folderSeq, renderAtLoad=renderSeq;
  try {
    const {data}=await api(`/api/timeline/${encodeURIComponent(base)}?version=${encodeURIComponent(version)}`);
    if(seq!==timelineSeq||base!==currentBase||folder!==folderSeq)return;
    $("audioPlayer").src=`/api/audio/${encodeURIComponent(base)}?folder=${folderSeq}`;
    $("audioBar").hidden=false;$("reviewAudio").disabled=!data.segments.length;
    $("audioMsg").textContent=data.segments.length ? "Select a timestamp to play that segment." : "No timestamped segments for this version.";
    if (renderAtLoad === renderSeq) renderTimeline(data.segments);

  }catch(e){if(seq===timelineSeq){$("audioBar").hidden=false;$("audioMsg").textContent=e.message;$("reviewAudio").disabled=true;}}
}

function renderTimeline(segments) {
  $("timeline").replaceChildren();
  $("reviewAudio").disabled = !segments.length;
    for(const segment of segments){
      const row=document.createElement("div");row.className="segment";row.dataset.start=segment.start;row.dataset.end=segment.end;
      const button=document.createElement("button");const seconds=Math.floor(segment.start);
      button.textContent=`${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")} ${segment.speaker}`;
      button.onclick=()=>{$("audioPlayer").currentTime=segment.start;$("audioPlayer").play().catch(()=>{$("audioMsg").textContent="Press Play to start audio, or choose a browser-supported audio format.";});};
      const text=document.createElement("p");text.textContent=segment.text;row.append(button,text);$("timeline").appendChild(row);
    }
}

async function loadTranscript(base, version) {
  const seq = ++transcriptSeq, folder = folderSeq;
  resetAudioReview();
  renderSeq++;
  $("renderBar").classList.remove("busy", "dirty");
  clearTimeout(renderTimer);
  const valid = () => seq === transcriptSeq && folder === folderSeq && base === currentBase;
  const sel = $("versionSel");
  const text = $("transcriptText");
  const msg = $("transcriptMsg");
  const hasRawJson = !!(FILES_BY_BASE[base] && FILES_BY_BASE[base].transcribed);
  updateRenderBar(false);

  sel.disabled = true;
  msg.textContent = "Loading…";
  msg.className = "msg";

  if (version === RAW_JSON_VERSION) {
    try {
      const { data } = await api(`/api/json-transcript/${encodeURIComponent(base)}`);
      if (!valid()) return;
      // Offer the rendered versions alongside the raw JSON, so this view is not
      // a dead end. Fetch the list only when it is not already known.
      let txtVersions = txtVersionsByBase[base];
      if (!txtVersions) {
        try { txtVersions = (await api(`/api/transcript/${encodeURIComponent(base)}`)).data.versions || []; }
        catch (e) { txtVersions = []; }
        if (!valid()) return;
        txtVersionsByBase[base] = txtVersions;
      }
      fillVersionSel(txtVersions, hasRawJson, RAW_JSON_VERSION);
      text.textContent = data.content;
      msg.textContent = `${base} — raw JSON transcript (${data.path})`;
    } catch (e) {
      if (!valid()) return;
      text.innerHTML = '<div class="placeholder">Could not load the raw JSON transcript.</div>';
      msg.textContent = e.message;
    }
    return;
  }

  let url = `/api/transcript/${encodeURIComponent(base)}`;
  if (version) url += `?version=${encodeURIComponent(version)}`;
  msg.textContent = "Loading…";
  msg.className = "msg";
  try {
    const { data } = await api(url);
    if (!valid()) return;
    txtVersionsByBase[base] = data.versions;
    fillVersionSel(data.versions, hasRawJson, data.selected);
    text.textContent = data.content;
    msg.textContent = `${base} — ${data.versions.length} version(s)`;

    setRenderOptions({...RENDER_DEFAULTS, stats: !!data.has_stats, preamble: data.has_preamble ?? true});
    renderBaseline = renderOptions();
    updateRenderBar(!!data.can_render);
    loadAudioReview(base, data.selected);
    const carried = carriedRenderOptions;
    carriedRenderOptions = null;
    if (carried && data.can_render && !renderIsDefault(carried)) { setRenderOptions(carried); applyRenderOptions(); }
    // The stored TXT carries the original labels, so re-render to show names.
    else if (data.can_render && Object.keys(data.speaker_names || {}).length) applyRenderOptions();
  } catch (e) {
    if (!valid()) return;
    delete txtVersionsByBase[base];
    updateRenderBar(false);
    if (hasRawJson) { loadTranscript(base, RAW_JSON_VERSION); return; }
    sel.innerHTML = "";
    sel.disabled = true;
    $("transcriptReveal").disabled = true;
    text.innerHTML = '<div class="placeholder">No transcript yet — run a pipeline first.</div>';
    msg.textContent = /no transcript/i.test(e.message) ? "" : e.message;
  }
}

/* ---------- comparison tab ----------
   Two versions of one recording, side by side, rendered with the same Show
   settings. Both describe the same audio, so the panes are lined up by time
   rather than by text: the moment in the middle of one pane is found in the
   other through the paragraph anchors the server returns, and interpolated
   inside a paragraph. That keeps working however different the texts are. */

let compareSeq = 0;
let compareBase = null;
let compareSyncFrame = 0;
let compareCanRender = {};   // version id → has a diarized JSON (per folder)
const COMPARE_PANES = {left: {}, right: {}};

function comparePane(side) {
  const pane = COMPARE_PANES[side];
  if (!pane.text) {
    const suffix = side === "left" ? "Left" : "Right";
    Object.assign(pane, {sel: $(`compareSel${suffix}`), text: $(`compareText${suffix}`),
                         note: $(`compareNote${suffix}`), blocks: [], expected: NaN});
  }
  return pane;
}

function comparePlaceholder(message) {
  for (const side of ["left", "right"]) {
    const pane = comparePane(side);
    pane.blocks = pane.timed = [];
    pane.note.textContent = "";
    pane.text.innerHTML = `<div class="placeholder">${escapeHTML(message)}</div>`;
  }
}

async function loadComparison(base) {
  const seq = ++compareSeq, folder = folderSeq;
  if (!base) {
    compareBase = null;
    for (const side of ["left", "right"]) { comparePane(side).sel.innerHTML = ""; comparePane(side).sel.disabled = true; }
    comparePlaceholder("Select an audio file to compare its transcript versions.");
    return;
  }
  $("compareMsg").textContent = "Loading…";
  let versions;
  try { versions = (await api(`/api/transcript/${encodeURIComponent(base)}`)).data.versions || []; }
  catch (e) { versions = []; }
  if (seq !== compareSeq || folder !== folderSeq || base !== currentBase) return;
  // Keep the chosen versions while the same recording stays selected.
  const previous = compareBase === base ? [comparePane("left").sel.value, comparePane("right").sel.value] : [];
  compareBase = base;
  if (!versions.length) {
    for (const side of ["left", "right"]) { comparePane(side).sel.innerHTML = ""; comparePane(side).sel.disabled = true; }
    comparePlaceholder("No transcript yet — run a pipeline first.");
    $("compareMsg").textContent = "";
    return;
  }
  ["left", "right"].forEach((side, index) => {
    const sel = comparePane(side).sel;
    sel.innerHTML = "";
    for (const v of versions) {
      const o = document.createElement("option");
      o.value = v.id;
      o.textContent = (v.timestamp ? fmtTimestamp(v.timestamp) : v.filename) + (v.pipeline ? ` · ${v.pipeline}` : "");
      sel.appendChild(o);
    }
    const keep = versions.some(v => v.id === previous[index]);
    sel.value = keep ? previous[index] : versions[Math.min(index, versions.length - 1)].id;
    sel.disabled = false;
  });
  await refreshComparison();
}

async function renderComparePane(side, base, options) {
  const version = comparePane(side).sel.value;
  const query = `?version=${encodeURIComponent(version)}`;
  if (!(version in compareCanRender)) {
    const {data} = await api(`/api/transcript/${encodeURIComponent(base)}${query}`);
    compareCanRender[version] = !!data.can_render;
    if (!data.can_render) return {content: data.content, anchors: data.anchors || [], rendered: false};
  }
  if (compareCanRender[version]) {
    const {data} = await api(`/api/render/${encodeURIComponent(base)}`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({version, options}),
    });
    return {content: data.content, anchors: data.anchors || [], rendered: true};
  }
  const {data} = await api(`/api/transcript/${encodeURIComponent(base)}${query}`);
  return {content: data.content, anchors: data.anchors || [], rendered: false};
}

async function refreshComparison() {
  const base = compareBase;
  if (!base || base !== currentBase || !comparePane("left").sel.value) return;
  const seq = ++compareSeq, folder = folderSeq;
  const options = renderOptions();
  const position = compareCoordinate(comparePane("left"));
  const bar = $("renderBar");
  bar.classList.add("busy");
  bar.classList.toggle("dirty", !renderIsDefault(options));
  $("compareMsg").textContent = "Loading…";
  try {
    const results = await Promise.all(["left", "right"].map(side => renderComparePane(side, base, options)));
    if (seq !== compareSeq || folder !== folderSeq || base !== currentBase) return;
    ["left", "right"].forEach((side, index) => {
      const pane = comparePane(side), result = results[index];
      fillComparePane(pane, result.content, result.anchors);
      pane.note.textContent = result.rendered ? ""
        : "Shown as the pipeline wrote it: this version has no diarized JSON, so the Show settings do not apply.";
    });
    bar.style.display = results.some(r => r.rendered) ? "" : "none";
    const same = comparePane("left").sel.value === comparePane("right").sel.value;
    $("compareMsg").textContent = same ? "Both panes show the same version." : "";
    if (position) scrollPaneTo(comparePane("left"), position);
    syncPanes("left");
  } catch (e) {
    if (seq === compareSeq) $("compareMsg").textContent = e.message;
  } finally {
    if (seq === compareSeq) bar.classList.remove("busy");
  }
}

/* Split the text into one block per paragraph, so each block can be measured
   and mapped to the part of the recording its anchor covers. The text before
   the first paragraph (introduction, statistics) is a block without an anchor. */
function fillComparePane(pane, content, anchors) {
  const lines = content.split("\n");
  const blocks = [];
  const add = (from, to, anchor) => {
    if (to <= from) return;
    const div = document.createElement("div");
    div.textContent = lines.slice(from, to).join("\n") + (to < lines.length ? "\n" : "");
    blocks.push({el: div, anchor});
  };
  add(0, anchors.length ? anchors[0].line : lines.length, null);
  anchors.forEach((a, i) => add(a.line, i + 1 < anchors.length ? anchors[i + 1].line : lines.length, a));
  pane.blocks = blocks;
  pane.timed = blocks.filter(b => b.anchor);
  pane.text.replaceChildren(...blocks.map(b => b.el));
}

/* The position in the middle of a pane, as a moment in the recording
   ({time}), as a fraction of the introduction ({intro}), or as a fraction
   of the whole text ({ratio}) when the pane has no anchors at all. */
function compareCoordinate(pane) {
  const el = pane.text, blocks = pane.blocks || [];
  if (!blocks.length) return null;
  const middle = el.scrollTop + el.clientHeight / 2;
  if (!pane.timed.length) return {ratio: el.scrollTop / Math.max(1, el.scrollHeight - el.clientHeight)};
  let lo = 0, hi = blocks.length - 1;
  while (lo < hi) {   // last block starting at or above the middle
    const mid = (lo + hi + 1) >> 1;
    if (blocks[mid].el.offsetTop <= middle) lo = mid; else hi = mid - 1;
  }
  const block = blocks[lo];
  const fraction = Math.min(1, Math.max(0, (middle - block.el.offsetTop) / (block.el.offsetHeight || 1)));
  if (!block.anchor) return {intro: fraction};
  return {time: block.anchor.start + fraction * (block.anchor.end - block.anchor.start)};
}

// Where a coordinate lies in a pane, in scroll pixels from the top.
function compareOffset(pane, coord) {
  const el = pane.text, timed = pane.timed || [];
  if (coord.ratio !== undefined || !timed.length) {
    const ratio = coord.ratio ?? 0;
    return ratio * Math.max(0, el.scrollHeight - el.clientHeight) + el.clientHeight / 2;
  }
  if (coord.intro !== undefined) {
    const intro = pane.blocks[0];
    return intro.anchor ? 0 : intro.el.offsetTop + coord.intro * intro.el.offsetHeight;
  }
  let lo = 0, hi = timed.length - 1;
  if (coord.time < timed[0].anchor.start) return timed[0].el.offsetTop;
  while (lo < hi) {   // last paragraph starting at or before the time
    const mid = (lo + hi + 1) >> 1;
    if (timed[mid].anchor.start <= coord.time) lo = mid; else hi = mid - 1;
  }
  const {el: block, anchor} = timed[lo];
  if (anchor.end <= anchor.start) return block.offsetTop;   // a stored TXT's last paragraph has no end
  // A time after the paragraph ends falls in the pause before the next one.
  if (coord.time >= anchor.end) return block.offsetTop + block.offsetHeight;
  return block.offsetTop + (coord.time - anchor.start) / (anchor.end - anchor.start) * block.offsetHeight;
}

function scrollPaneTo(pane, coord) {
  pane.text.scrollTop = compareOffset(pane, coord) - pane.text.clientHeight / 2;
  pane.expected = pane.text.scrollTop;   // the scroll event this causes is ours
}

function syncPanes(fromSide) {
  if (!$("compareSync").checked) return;
  const coord = compareCoordinate(comparePane(fromSide));
  if (coord) scrollPaneTo(comparePane(fromSide === "left" ? "right" : "left"), coord);
}

function onCompareScroll(side) {
  const pane = comparePane(side);
  // Ignore the scroll that syncing just caused, or the panes would chase each other.
  if (Math.abs(pane.text.scrollTop - pane.expected) < 1) return;
  pane.expected = NaN;
  cancelAnimationFrame(compareSyncFrame);
  compareSyncFrame = requestAnimationFrame(() => syncPanes(side));
}

/* ---------- logs tab (historical logs only — live console lives in #consolePanel) ---------- */

function fmtLogTimestamp(ts) {
  let m = ts.match(/^(\d{4})(\d{2})(\d{2})[-_](\d{2})(\d{2})(\d{2})$/);
  if (m) return `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}`;
  m = ts.match(/^(\d{4}-\d{2}-\d{2}) (\d{2})(\d{2})(\d{2})$/);  // older log naming
  if (m) return `${m[1]} ${m[2]}:${m[3]}:${m[4]}`;
  return ts;
}

async function loadLogList() {
  const seq = ++logListSeq, folder = folderSeq, base = currentBase;
  const sel = $("logSel");
  // Both scopes are listed together. Selecting a file used to replace the run
  // logs, which left saved runs unreachable with no way to deselect the file.
  const runLogs = (await api("/api/logs")).data.versions || [];
  let fileLogs = [];
  if (base) fileLogs = (await api(`/api/logs?base=${encodeURIComponent(base)}`)).data.versions || [];
  if (seq !== logListSeq || folder !== folderSeq || base !== currentBase) return;
  sel.replaceChildren();
  const ids = [];
  const label = v => v.id === "live" ? "▶ Live console"
    : v.timestamp ? fmtLogTimestamp(v.timestamp) + (v.pipeline ? ` · ${v.pipeline}` : "")
    : v.filename;
  for (const [name, versions, scope] of [["Runs", runLogs, "run"], [`Logs for ${base || ""}`, fileLogs, "file"]]) {
    if (!versions.length) continue;
    const group = document.createElement("optgroup");
    group.label = name;
    for (const v of versions) {
      const o = document.createElement("option");
      o.value = v.id; o.textContent = label(v); o.dataset.scope = scope;
      group.appendChild(o); ids.push(v.id);
    }
    sel.appendChild(group);
  }
  if (!currentLogSel || !ids.includes(currentLogSel)) currentLogSel = ids[0] || null;
  sel.value = currentLogSel || "";
  sel.disabled = !ids.length;
  await loadLogContent();
}

async function loadLogContent() {
  const seq = ++logSeq, folder = folderSeq, base = currentBase;
  const valid = () => seq === logSeq && folder === folderSeq && base === currentBase;
  const sel = $("logSel");
  const log = $("log");
  const msg = $("logMsg");
  const id = currentLogSel;
  if (!id) { log.textContent = ""; msg.textContent = "No logs available."; return; }
  if (id === "live") {
    const { data } = await api("/api/logs?file=live");
    if (!valid()) return;
    log.textContent = (data.lines || []).join("\n") + (data.lines && data.lines.length ? "\n" : "");
    log.scrollTop = log.scrollHeight;
    msg.textContent = data.pipeline
      ? (data.running ? `running ${data.pipeline}` : `last run: ${data.pipeline} · exit ${data.returncode ?? "?"}`)
      : "No runs yet this session.";
    return;
  }
  const url = sel.selectedOptions[0]?.dataset.scope === "file"
    ? `/api/logs?base=${encodeURIComponent(currentBase)}&file=${encodeURIComponent(id)}`
    : `/api/logs?file=${encodeURIComponent(id)}`;
  const { data } = await api(url);
  if (!valid()) return;
  log.textContent = data.content;
  msg.textContent = data.selected;
}

async function saveSidecar() {
  if (editorScope === "global") return saveGlobalSettings();
  if (!validateForm()) return false;
  if (saving || editorLoading || !editorBase || editorBase !== currentBase) return false;
  if (!formDirty()) return true;
  const base = editorBase, folder = folderSeq, values = formValues();
  const deleteDisabled = $("deleteBtn").disabled;
  saving = true;
  $("editor").inert = true;
  $("deleteBtn").disabled = true;
  updateSaveState(); updateRunControls();
  try {
    const { status, data } = await api(`/api/sidecar/${encodeURIComponent(base)}`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(values),
    });
    if (status === 422) { fieldErrors(data.errors); throw new Error("Invalid values: " + Object.entries(data.errors || {}).map(([k, e]) => `${k}: ${e}`).join(" · ")); }
    if (folder !== folderSeq || base !== editorBase) return false;
    savedForm = JSON.stringify(values);
    $("deleteBtn").disabled = !data.exists;
    handle(loadFiles)();
    updateScribeGate();
    return true;
  } catch (e) {
    $("deleteBtn").disabled = deleteDisabled;
    setMsg(e.message, "err");
    return false;
  } finally {
    saving = false; $("editor").inert = false;
    $("saveBtn").disabled = !formDirty();
    $("saveBtn").textContent = "Save";
    if (!formDirty()) setMsg("All changes saved.", "ok");
    updateRunControls();
  }
}

async function deleteSidecar() {
  if (saving || editorLoading || !editorBase || !await resolveUnsaved()) return;
  const base = editorBase, folder = folderSeq;
  if (!confirm(`Delete settings AND comment for ${base}?`)) return;
  saving = true; $("editor").inert = true; updateSaveState(); updateRunControls();
  try {
    await api(`/api/sidecar/${encodeURIComponent(base)}`, { method: "DELETE" });
    if (base !== editorBase || folder !== folderSeq) return;
    saving = false;
    await selectFile(base, true);
    setMsg("Sidecar deleted — file uses global settings again.", "ok");
    await loadFiles();
  } finally { saving = false; $("editor").inert = false; updateSaveState(); updateRunControls(); }
}

/* ---------- run history ---------- */

async function showHistory() {
  const folder = folderSeq;
  const { data } = await api("/api/history");
  if (folder !== folderSeq) return;
  const body = $("historyBody");
  body.innerHTML = "";
  if (!data.entries.length) {
    body.innerHTML = '<p style="color:var(--muted)">No runs recorded yet. Every run started from this UI is logged (with the files it produced) in <code>swhisper_run_history.jsonl</code> in the audio directory.</p>';
  }
  for (const e of data.entries) {
    const det = document.createElement("details");
    det.className = "hentry";
    const mins = (e.duration_s / 60).toFixed(1);
    const result = e.stopped_by_user ? "⏹ stopped" : e.backfill_incomplete ? "⚠️ backfill incomplete" : (e.success ? "✅ ok" : `❌ exit ${e.returncode}`);
    const resultCls = e.success ? "ok" : "bad";
    det.innerHTML = `<summary>
        <span class="when">${escapeHTML((e.started || "").replace("T", " "))}</span>
        <span class="meta">${escapeHTML(e.pipeline)}</span>
        <span class="result ${resultCls}">${result}</span>
        <span class="meta">${mins} min · ${e.file_count} file(s) produced/updated</span>
      </summary>
      <div class="filelist">${escapeHTML((e.files_produced || []).join("\n") || "(no new files detected)")}</div>`;
    body.appendChild(det);
  }
  $("historyPanel").showModal();
}

/* ---------- run control ---------- */

async function startRun() {
  if (!connected || runBusy || lastStatus?.running || !await resolveUnsaved()) return;
  runBusy = true; updateRunControls();
  try {
    const pipeline = $("pipelineSel").value;
    const scribe = pipeline === "scribe";
    // Scribe carries its own single-recording scope; the local pipelines use
    // the files ticked in the list.
    const selected = scribe && $("scribeScope").value === "selected" ? (currentBase || "") : "";
    if (scribe && $("scribeScope").value === "selected" && !selected) {
      $("runStatus").textContent = "Select a recording, or choose the explicit folder scope.";
      return;
    }
    const query = new URLSearchParams({ pipeline });
    if (scribe) { query.set("selected", selected); query.set("language", $("scribeLanguage").value); }
    else for (const base of runScope()) query.append("only", base);
    // Per-run options; the preview token covers them, so the run matches the review.
    query.set("diarization_model", $("runModel").value);
    query.set("reprocess", String($("runReprocess").checked));
    const { data } = await api(`/api/run/preview?${query}`);
    pendingPreview = data;
    const label = META.pipelines.find(p => p.id === pipeline).label;
    // Consent is needed only when a recording has no saved Scribe transcript.
    const remote = scribe && data.remote;
    $("previewTitle").textContent = remote ? "Scribe remote upload preview"
      : scribe ? "Scribe rerun preview (no upload)" : "Run Preview";
    $("previewConsent").hidden = !remote;
    $("consentRow").hidden = !remote;
    $("scribeConsent").checked = false;
    $("previewStart").textContent = remote ? "Upload to Scribe and run" : scribe ? "Rerun locally" : "Run";
    $("previewFolder").textContent = `${label} · ${data.audio_dir}`
      + (scribe && selected ? ` · ${selected}` : "")
      + (data.only ? ` · ${data.only.length} selected file(s)` : "")
      + (data.diarization_model ? ` · diarization ${data.diarization_model}` : "")
      + (data.reprocess || data.retranscribe ? ` · ${scribe ? "transcribe again" : "reprocess existing outputs"}` : "");
    const counts = {};
    for (const row of data.files) counts[row.action] = (counts[row.action] || 0) + 1;
    $("previewSummary").textContent = data.files.length
      ? Object.entries(counts).map(([action, count]) => `${count} ${action === "blocked" ? "blocked" : `to ${action}`}`).join(" · ") + ". Counts are per stage."
      : "No eligible inputs found.";
    $("previewNotes").replaceChildren(...data.notes.map(note => { const li = document.createElement("li"); li.textContent = note; return li; }));
    // What will run is the decision; skipped files are reference, not noise.
    const rowFor = values => {
      const tr = document.createElement("tr");
      for (const value of values) {
        const td = document.createElement("td"); td.textContent = value; tr.appendChild(td);
      }
      return tr;
    };
    const active = data.files.filter(row => row.action !== "skip");
    const skipped = data.files.filter(row => row.action === "skip");
    $("previewTable").querySelector("tbody").replaceChildren(
      ...active.map(row => rowFor([row.file, row.stage, row.action, row.reason])));
    $("previewTable").hidden = !active.length;
    $("previewSkipped").hidden = !skipped.length;
    $("previewSkipped").open = false;
    $("previewSkippedSummary").textContent = `${skipped.length} already up to date, not part of this run`;
    $("previewSkippedTable").querySelector("tbody").replaceChildren(
      ...skipped.map(row => rowFor([row.file, row.stage, row.reason])));
    $("previewStart").disabled = !data.files.some(row => ["process", "reprocess"].includes(row.action));
    updateScribeGate();
    $("previewDialog").showModal();
  } catch (e) { $("runStatus").textContent = e.message; }
  finally { runBusy = false; updateRunControls(); }
}
async function launchPreview() {
  if (!pendingPreview || runBusy) return;
  runBusy = true; updateRunControls();
  $("previewStart").disabled = true; $("previewCancel").disabled = true;
  try {
    const body = { pipeline: pendingPreview.pipeline, preview_token: pendingPreview.token,
                   diarization_model: pendingPreview.diarization_model || "",
                   reprocess: Boolean(pendingPreview.reprocess ?? pendingPreview.retranscribe) };
    if (pendingPreview.pipeline === "scribe") {
      if (pendingPreview.remote) {
        if (!$("scribeConsent").checked) { $("previewSummary").textContent = "Consent to the remote upload first, or cancel."; return; }
        body.remote_upload = true;
      }
      body.selected = pendingPreview.selected || "";
      body.language = pendingPreview.language;
    } else if (pendingPreview.only) {
      body.only = pendingPreview.only;
    }
    const { data } = await api("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    lastStatus = data; statusSeq++;
    logOffset = 0; activeRunId = data.run_id;
    $("consoleLog").textContent = "";
    setConsoleCollapsed(false);
    $("previewDialog").close(); pendingPreview = null;
    await pollLoop();   // re-poll now and switch to the running cadence
  } catch (e) { $("previewSummary").textContent = e.message + " Close this preview and try again."; }
  finally { runBusy = false; $("previewCancel").disabled = false; updateRunControls(); }
}
async function stopRun() {
  if (runBusy) return;
  runBusy = true; updateRunControls();
  try {
    const { data } = await api("/api/run/stop", { method: "POST" });
    lastStatus = data; statusSeq++; $("runStatus").textContent = "Stopping…";
    pollLoop();
  } catch (e) { $("runStatus").textContent = e.message; }
  finally { runBusy = false; updateRunControls(); }
}
function statusText(st) {
  if (st.running) return st.stop_requested ? "Stopping…" : `Running ${st.pipeline} · ${((Date.now() / 1000 - st.started_at) / 60).toFixed(1)} min`;
  if (!st.pipeline) return "Idle";
  if (st.stop_requested) return "Stopped by you";
  if (st.returncode === null) return "Run ended; exit status unavailable";
  if (st.progress?.backfill_incomplete) return "Transcripts saved; backfill incomplete. Failed gaps are marked [-].";
  if (st.progress?.error_count) return "Run ended with reported errors. Check the console.";
  if (st.returncode !== 0) return `Run failed (exit ${st.returncode}). Check the console.`;
  return "Run finished successfully";
}
async function pollRun() {
  if (pollBusy || changingDir) return;
  pollBusy = true;
  try {
    if (!initialized) { await loadMeta(); await loadFiles(); initialized = true; }
    const seq = ++statusSeq;
    const st = (await api("/api/run/status")).data;
    // Starting or stopping a run applies its own authoritative status. A poll
    // that was already in flight must not overwrite it with the older state.
    if (seq !== statusSeq) { connected = true; return; }
    if (activeRunId !== st.run_id) {
      activeRunId = st.run_id; logOffset = 0; $("consoleLog").textContent = "";
    }
    const { data } = await api(`/api/logs?file=live&offset=${logOffset}`);
    if (data.next !== undefined) logOffset = data.next;
    if (data.lines?.length) appendConsole(data.lines);
    connected = true; lastStatus = st;
    // The folder is server-wide, so another tab can move it under this one.
    // Say so rather than silently answering for a different folder, and let
    // the reload happen on demand so unsaved edits are not discarded.
    if (st.folder && META && st.folder !== META.audio_dir && !changingDir) {
      $("folderNoticeText").textContent = `Another window switched the audio folder to ${st.folder}. This window still shows ${META.audio_dir}.`;
      $("folderNotice").hidden = false;
    } else if (!changingDir) {
      $("folderNotice").hidden = true;
    }
    $("connectionStatus").textContent = "Connected";
    $("runDot").className = "dot" + (st.running ? " run" : (st.returncode || st.progress?.error_count || st.progress?.backfill_incomplete ? " err" : ""));
    $("runStatus").textContent = statusText(st);
    $("consoleMsg").textContent = statusText(st);
    const p = st.progress || {};
    const counts = Object.entries(p.completed || {}).map(([stage, n]) => `${stage}: ${n} completed`).join(" · ");
    $("progressStatus").textContent = st.pipeline
      ? [st.audio_dir, p.stage, p.detail, p.current_file, p.total_files ? `file ${p.file_index} of ${p.total_files}` : "", counts, p.last_error].filter(Boolean).join(" · ") : "";
    if (wasRunning && !st.running) {
      txtVersionsByBase = {};   // a finished run may have written new versions
      await loadFiles();
      if (currentTab === "transcript" && currentBase) await loadTranscript(currentBase);
      if (currentTab === "compare" && currentBase) await loadComparison(currentBase);
      if (currentTab === "logs") await loadLogList();
    }
    wasRunning = st.running;
  } catch (e) {
    connected = false;
    $("connectionStatus").textContent = "Disconnected · retrying connection…";
    $("runDot").className = "dot err";
    $("runStatus").textContent = "Run status unavailable";
    $("consoleMsg").textContent = "Connection lost; transcription may still be running.";
  } finally { pollBusy = false; updateRunControls(); }
}

/* ---------- console panel ---------- */


const CONSOLE_MAX_CHUNKS = 2000;   // bounds a long run console in the DOM

function appendConsole(lines) {
  const log = $("consoleLog");
  const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
  log.appendChild(document.createTextNode(lines.join("\n") + "\n"));
  while (log.childNodes.length > CONSOLE_MAX_CHUNKS) log.removeChild(log.firstChild);
  if (atBottom) log.scrollTop = log.scrollHeight;
}

function setConsoleCollapsed(collapsed) {
  $("consolePanel").classList.toggle("collapsed", collapsed);
  $("consoleToggle").setAttribute("aria-expanded", String(!collapsed));
  $("consoleToggle").textContent = (collapsed ? "▸" : "▾") + " Live console";
  try { localStorage.setItem("swhisper_console_collapsed", collapsed ? "1" : "0"); } catch (e) {}
}

function setConsoleHeight(px, userResize = false) {
  const panel = $("consolePanel");
  const rightHeight = $("right").clientHeight;
  const max = Math.max(160, Math.min(rightHeight * 0.7, 900));
  px = Math.max(100, Math.min(max, px));
  panel.style.setProperty("--console-height", `${Math.round(px)}px`);
  if (userResize && window.innerWidth <= 760) {
    panel.style.setProperty("--console-mobile-height", `${Math.round(px)}px`);
    try { localStorage.setItem("swhisper_console_mobile_height", String(Math.round(px))); } catch {}
  }
  $("consoleResize").setAttribute("aria-valuenow", String(Math.round(px)));
  $("consoleResize").setAttribute("aria-valuemax", String(Math.round(max)));
  try { localStorage.setItem("swhisper_console_height", String(Math.round(px))); } catch (e) {}
}

function initConsole() {
  let collapsed = false;
  let height = 14 * parseFloat(getComputedStyle(document.documentElement).fontSize || "22");
  try {
    collapsed = localStorage.getItem("swhisper_console_collapsed") === "1";
    const saved = parseInt(localStorage.getItem("swhisper_console_height"), 10);
    if (Number.isFinite(saved)) height = saved;
  } catch (e) {}
  setConsoleHeight(height);
  try {
    const mobileHeight = Number(localStorage.getItem("swhisper_console_mobile_height"));
    if (mobileHeight) $("consolePanel").style.setProperty("--console-mobile-height", `${Math.max(50, Math.min(300, mobileHeight))}px`);
  } catch {}
  setConsoleCollapsed(collapsed);
}

function initConsoleResize() {
  $("consoleResize").addEventListener("pointerdown", event => {
    event.preventDefault();
    setConsoleCollapsed(false);
    const startY = event.clientY;
    const startHeight = $("consolePanel").getBoundingClientRect().height;
    const move = moveEvent => setConsoleHeight(startHeight + startY - moveEvent.clientY, true);
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  });
}

/* ---------- init ---------- */

$("saveBtn").onclick = handle(saveSidecar);
$("deleteBtn").onclick = handle(deleteSidecar);
$("tabSettings").onclick = () => showTab("settings");
$("tabTranscript").onclick = () => showTab("transcript");
$("tabLogs").onclick = () => showTab("logs");
$("logSel").onchange = handle(async () => { currentLogSel = $("logSel").value; await loadLogContent(); });
$("versionSel").onchange = () => { if (currentBase) loadTranscript(currentBase, $("versionSel").value); };
$("transcriptReload").onclick = () => { if (currentBase) loadTranscript(currentBase); };
$("transcriptCopy").onclick = handle(async () => {
  const text = $("transcriptText").textContent;
  if (!text.trim()) return;
  try {
    await navigator.clipboard.writeText(text);
    $("transcriptMsg").textContent = "Copied the transcript shown here.";
    $("transcriptMsg").className = "msg";
  } catch (e) {
    // Clipboard access needs a secure context; select the text instead.
    getSelection().selectAllChildren($("transcriptText"));
    $("transcriptMsg").textContent = "Selected the transcript — press Cmd/Ctrl+C to copy.";
    $("transcriptMsg").className = "msg";
  }
});
$("transcriptReveal").onclick = handle(async () => {
  const version = $("versionSel").value;
  if (!version || version === RAW_JSON_VERSION) return;
  const { data } = await api("/api/reveal", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: version }),
  });
  $("transcriptMsg").textContent = `Revealed ${data.revealed}`;
  $("transcriptMsg").className = "msg";
});
for (const id of ["optStats", "optPreamble", "optDisfluencies", "optPauses", "optBreaks"]) {
  $(id).onchange = renderOptionsChanged;
}
for (const id of ["optThreshold", "optMinPause"]) {
  $(id).oninput = debounceRender;
}
$("renderReset").onclick = () => {
  if (currentTab === "compare") { setRenderOptions(RENDER_DEFAULTS); refreshComparison(); }
  else if (currentBase) loadTranscript(currentBase, $("versionSel").value);
};
$("tabCompare").onclick = () => showTab("compare");
$("compareSelLeft").onchange = () => refreshComparison();
$("compareSelRight").onchange = () => refreshComparison();
$("compareSync").onchange = () => { if ($("compareSync").checked) syncPanes("left"); };
for (const side of ["left", "right"]) comparePane(side).text.addEventListener("scroll", () => onCompareScroll(side));
$("renderSave").onclick = saveRenderedView;
$("renderSpeakers").onclick = handle(openSpeakerNames);
$("speakerSave").onclick = handle(saveSpeakerNames);
$("speakerCancel").onclick = () => { $("speakerDialog").close(); speakerContext = null; };
$("speakerDialog").addEventListener("cancel", () => { speakerContext = null; });
$("startBtn").onclick = startRun;
$("clearRunSelection").onclick = () => { runSelection.clear(); renderFiles(); updateRunControls(); };
$("stopBtn").onclick = stopRun;
$("refreshBtn").onclick = handle(loadFiles);
$("clearLogBtn").onclick = () => { $("log").textContent = ""; };
$("clearConsoleBtn").onclick = () => { $("consoleLog").textContent = ""; };
$("consoleToggle").onclick = () => setConsoleCollapsed(!$("consolePanel").classList.contains("collapsed"));
$("dirBtn").onclick = handle(changeDir);
$("dirBrowse").onclick = handle(browseAudioDir);
// The arrow carries popovertarget, so the browser toggles the menu itself. A
// click handler cannot: light dismiss closes the popover on mousedown, so by
// the time the click arrives the handler reads a menu that is already closed
// and reopens the one the user was trying to dismiss. Declaring the arrow as
// the invoker also exempts it from light dismiss, so the two never race.
// beforetoggle runs synchronously, before the menu is painted.
$("dirMenu").addEventListener("beforetoggle", event => {
  const open = event.newState === "open";
  $("dirMenuBtn").setAttribute("aria-expanded", String(open));
  if (open) {
    $("dirMenu").hidden = false;   // [hidden] would outrank the top-layer display.
    positionDirMenu();
    addEventListener("resize", positionDirMenu);
    addEventListener("scroll", positionDirMenu, true);
  } else {
    removeEventListener("resize", positionDirMenu);
    removeEventListener("scroll", positionDirMenu, true);
  }
});
$("dirMenu").addEventListener("toggle", event => {
  if (event.newState === "closed") { $("dirMenu").hidden = true; return; }
  positionDirMenu();             // Scrollbars may appear once the menu is laid out.
});
$("dirMenu").addEventListener("keydown", e => {
  const items = [...$("dirMenu").querySelectorAll("button")];
  let index = items.indexOf(document.activeElement);
  if (e.key === "ArrowDown") index++;
  else if (e.key === "ArrowUp") index--;
  else if (e.key === "Home") index = 0;
  else if (e.key === "End") index = items.length - 1;
  else return;
  e.preventDefault(); focusDirMenuItem(index);
});
$("dirInput").addEventListener("input", changeDirIfSavedPath);
$("dirInput").addEventListener("change", changeDirIfSavedPath);
$("dirInput").addEventListener("keydown", e => {
  if (e.key === "Enter") return changeDir();
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault(); openDirMenu(); focusDirMenuItem(e.key === "ArrowDown" ? 0 : -1);
});
$("folderNoticeReload").onclick = handle(async () => {
  if (!lastStatus?.folder) return;   // changeDir runs the unsaved-edit guard
  $("dirInput").value = lastStatus.folder;
  $("folderNotice").hidden = true;
  await changeDir();
});
$("historyBtn").onclick = handle(showHistory);
$("historyClose").onclick = () => $("historyPanel").close();

$("fontMinus").onclick = () => nudgeFont(-2);
$("fontPlus").onclick = () => nudgeFont(2);

$("dirtyCancel").onclick = () => finishDirtyDialog("cancel");
$("dirtyDiscard").onclick = () => finishDirtyDialog("discard");
$("dirtySave").onclick = () => finishDirtyDialog("save");
$("dirtyDialog").addEventListener("cancel", e => { e.preventDefault(); finishDirtyDialog("cancel"); });
$("previewCancel").onclick = () => { pendingPreview = null; $("previewDialog").close(); };
$("previewStart").onclick = handle(launchPreview);
$("previewDialog").addEventListener("cancel", e => { if (runBusy) e.preventDefault(); else pendingPreview = null; });
window.addEventListener("beforeunload", e => {
  if (formDirty() || saving) { e.preventDefault(); e.returnValue = ""; }
});
$("tabBar").addEventListener("keydown", e => {
  const tabs = [...$("tabBar").querySelectorAll('[role="tab"]')];
  let index = tabs.indexOf(document.activeElement);
  if (e.key === "ArrowRight") index++;
  else if (e.key === "ArrowLeft") index--;
  else if (e.key === "Home") index = 0;
  else if (e.key === "End") index = tabs.length - 1;
  else return;
  e.preventDefault();
  const tab = tabs[(index + tabs.length) % tabs.length]; tab.focus(); tab.click();
});

$("consoleResize").addEventListener("keydown", e => {
  if (!["ArrowUp", "ArrowDown"].includes(e.key)) return;
  e.preventDefault(); setConsoleCollapsed(false);
  setConsoleHeight($("consolePanel").getBoundingClientRect().height + (e.key === "ArrowUp" ? 20 : -20), true);
});
$("globalSettingsBtn").onclick=handle(openGlobalSettings);
$("pipelineSel").addEventListener("change", filterSettings);
for(const id of ["fileSearch","fileFilter","fileSort"]) $(id).addEventListener(id==="fileSearch"?"input":"change",renderFiles);
$("exportConfirm").onclick=handle(confirmExport);
$("exportCancel").onclick=()=>{$("exportDialog").close();exportContext=null;};
$("exportDialog").addEventListener("cancel",e=>{if(exporting)e.preventDefault();else exportContext=null;});
$("reviewAudio").onchange=()=>{$("timeline").hidden=!$("reviewAudio").checked;$("transcriptText").style.display=$("reviewAudio").checked?"none":"";};
$("audioPlayer").addEventListener("timeupdate",()=>{
  const time=$("audioPlayer").currentTime;
  document.querySelectorAll("#timeline .segment").forEach(row=>row.classList.toggle("active",time>=Number(row.dataset.start)&&time<Number(row.dataset.end)));
});
function setSidebarWidth(width){
  width=Math.max(220,Math.min(window.innerWidth*.55,700,width));
  document.body.style.setProperty("--sidebar-width",width+"px");$("sidebarResize").setAttribute("aria-valuenow",Math.round(width));
  try{localStorage.setItem("swhisper_sidebar_width",width);}catch{}
}
function toggleFiles(force){
  const collapsed=force??!document.body.classList.contains("filesCollapsed");
  document.body.classList.toggle("filesCollapsed",collapsed);$("filesToggle").setAttribute("aria-expanded",String(!collapsed));
  try{localStorage.setItem("swhisper_files_collapsed",collapsed?"1":"0");}catch{}
}
$("filesToggle").onclick=()=>toggleFiles();
$("sidebarResize").addEventListener("keydown",e=>{if(["ArrowLeft","ArrowRight"].includes(e.key)){e.preventDefault();setSidebarWidth($("filePanel").getBoundingClientRect().width+(e.key==="ArrowRight"?20:-20));}});
$("sidebarResize").addEventListener("pointerdown",e=>{
  e.preventDefault();const start=e.clientX,width=$("filePanel").getBoundingClientRect().width;
  const move=event=>setSidebarWidth(width+event.clientX-start),stop=()=>{window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",stop);};
  window.addEventListener("pointermove",move);window.addEventListener("pointerup",stop,{once:true});
});
try{const width=Number(localStorage.getItem("swhisper_sidebar_width"));if(width)setSidebarWidth(width);toggleFiles(localStorage.getItem("swhisper_files_collapsed")==="1");}catch{}
initFontSize();
initConsole();
initConsoleResize();
function pollDelay() {
  if (document.visibilityState === "hidden") return 20000;
  return lastStatus?.running ? 1000 : 2500;
}
async function pollLoop() {
  clearTimeout(pollTimer);
  await pollRun();
  pollTimer = setTimeout(pollLoop, pollDelay());
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") pollLoop();
});
pollLoop();
