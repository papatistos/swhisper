"use strict";
const $ = id => document.getElementById(id);
const RAW_JSON_VERSION = "__raw_json__";
const UI_STORAGE_KEY = "swhisper_ui_preferences";
const uiPreferences = readUIPreferences();
let showingActivePipeline = false;

function readUIPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(UI_STORAGE_KEY));
    return saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
  } catch { return {}; }
}
function saveUIPreference(key, value) {
  // Another window may have saved a newer choice for a different control.
  Object.assign(uiPreferences, readUIPreferences());
  uiPreferences[key] = value;
  try { localStorage.setItem(UI_STORAGE_KEY, JSON.stringify(uiPreferences)); } catch {}
}
function restoreControlPreference(id) {
  const el = $(id), value = uiPreferences[id];
  if (!el) return;
  if (el.type === "checkbox") {
    if (typeof value === "boolean") el.checked = value;
  } else if (typeof value === "string" &&
      (!el.options || [...el.options].some(option => option.value === value))) el.value = value;
}
function rememberControl(id) {
  const el = $(id);
  restoreControlPreference(id);
  el.addEventListener(el.type === "search" ? "input" : "change", () => {
    saveUIPreference(id, el.type === "checkbox" ? el.checked : el.value);
  });
}
function rememberDetails(section, key) {
  if (typeof uiPreferences[key] === "boolean") section.open = uiPreferences[key];
  section.querySelector("summary").addEventListener("click", event => {
    event.preventDefault();
    section.open = !section.open;
    saveUIPreference(key, section.open);
  });
}
function rememberAudio(id) {
  const audio = $(id);
  const volume = uiPreferences[id + ".volume"], rate = uiPreferences[id + ".rate"];
  if (typeof volume === "number" && Number.isFinite(volume) && volume >= 0 && volume <= 1) audio.volume = volume;
  if (typeof uiPreferences[id + ".muted"] === "boolean") audio.muted = uiPreferences[id + ".muted"];
  if (typeof rate === "number" && Number.isFinite(rate) && rate >= 0.0625 && rate <= 16) audio.defaultPlaybackRate = audio.playbackRate = rate;
  audio.addEventListener("volumechange", () => {
    saveUIPreference(id + ".volume", audio.volume);
    saveUIPreference(id + ".muted", audio.muted);
  });
  audio.addEventListener("ratechange", () => {
    audio.defaultPlaybackRate = audio.playbackRate;
    saveUIPreference(id + ".rate", audio.playbackRate);
  });
}
let META = null;
let FILES_BY_BASE = {};
let currentBase = null;
let currentTab = "settings";
let currentLogSel = null;
let logOffset = 0;
let wasRunning = false;
let savedAudioPaths = new Set();
let runSelection = new Set();   // files selected for settings and processing
let shownBases = [], selectionBusy = false, bulkContext = null;
const expandedFiles = new Set(), fileVersionRequests = new Map();
let durationTimer = null;
let changingDir = false;
let editorBase = null, savedForm = null, saving = false, editorLoading = false;
let editorSeq = 0, folderSeq = 0, filesSeq = 0, transcriptSeq = 0, logSeq = 0, logListSeq = 0;
let connected = false, lastStatus = null, runBusy = false, pollBusy = false, activeRunId = null, statusSeq = 0;
let editorScope = "file", editorMeta = null, globalRevision = null, timelineSeq = 0, exportContext = null, exporting = false;
let pendingPreview = null, dirtyChoice = null, initialized = false, pollTimer = null;
let reuseKey = "", reuseSeq = 0, reuseTimer = null, reuseFolder = null;
// Results the selected models would reuse, by recording, and the recordings to redo instead.
let REUSABLE = {};
const retranscribeBases = new Set(), rediarizeBases = new Set();
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));

function formValues() {
  if (editorScope === "batch") {
    const changes = {};
    document.querySelectorAll("#editor [data-key]").forEach(el => {
      if (el.dataset.bulkChanged !== "1") return;
      const raw = el.value.trim();
      changes[el.dataset.key] = el.dataset.explicit !== "1" || el.value === el.dataset.inheritValue ||
        (raw === "" && el.dataset.allowEmpty !== "1") ? null : raw;
    });
    return changes;
  }
  const values = editorScope === "global" ? {} : { _comment: ($("commentField") || { value: "" }).value };
  document.querySelectorAll("#editor [data-key]").forEach(el => {
    if (el.value === el.dataset.inheritValue) return;
    const raw = el.dataset.originalValue === el.value ? el.dataset.originalRaw : el.value;
    if (raw.trim() !== "" || ((editorScope === "global" || el.dataset.allowEmpty === "1") && el.dataset.explicit === "1")) values[el.dataset.key] = raw.trim();
  });
  return values;
}
function formDirty() {
  return !!editorBase && savedForm !== null && (JSON.stringify(formValues()) !== savedForm || !!document.querySelector("#editor input:invalid"));
}
function updateSaveState() {
  $("saveBtn").disabled = saving || editorLoading || !formDirty() || (editorScope === "global" && lastStatus?.running);
  $("saveBtn").textContent = saving ? "Saving…" : editorScope === "batch" ? `Save to ${bulkContext.bases.length} files` : "Save";
  if (!saving && !editorLoading && editorBase) setMsg(formDirty() ? "Unsaved changes" : "All changes saved.", "");
}
function restoreSavedForm() {
  if (savedForm === null) return;
  if (editorScope === "batch") { renderBulkSettings(bulkContext); return; }
  const values = JSON.parse(savedForm);
  if ($("commentField")) $("commentField").value = values._comment || "";
  document.querySelectorAll("#editor [data-key]").forEach(el => {
    const value = values[el.dataset.key] ?? el.dataset.inheritValue ?? "";
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
  $("dirtyText").textContent = `Save changes to ${editorScope === "global" ? "global settings" : editorScope === "batch" ? `${bulkContext.bases.length} selected files` : editorBase} before continuing?`;
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
  // Keep filtered selections, but drop files no longer in this folder.
  const known = Object.keys(FILES_BY_BASE);
  for (const base of [...runSelection]) if (!known.includes(base)) runSelection.delete(base);
  return [...runSelection];
}
// The run preview query for the chosen models. Every pipeline runs the ticked
// files, or the whole folder when none is ticked.
function runQuery() {
  const pipeline = $("pipelineSel").value;
  const query = new URLSearchParams({ pipeline });
  if (pipeline === "scribe") query.set("language", $("scribeLanguage").value);
  for (const base of runScope()) query.append("only", base);
  query.set("diarization_model", $("runModel").value);
  return query;
}
// Existing results are reused. A recording offers "Re-transcribe" or
// "Re-diarize" only when the selected models would reuse its result. The whole
// folder is checked, so the options do not depend on the run selection.
function reuseQuery() {
  const query = new URLSearchParams({ pipeline: $("pipelineSel").value, diarization_model: $("runModel").value });
  if (query.get("pipeline") === "scribe") { query.set("selected", ""); query.set("language", $("scribeLanguage").value); }
  return query;
}
function scheduleReuseCheck() {
  if (!connected || !initialized || !META) return;
  if (reuseFolder !== folderSeq) {
    reuseFolder = folderSeq; REUSABLE = {};
    retranscribeBases.clear(); rediarizeBases.clear();
    refreshForceOptions();
  }
  const query = reuseQuery();
  const key = `${folderSeq}|${filesSeq}|${query}`;
  if (key === reuseKey) return;
  reuseKey = key;
  clearTimeout(reuseTimer);
  reuseTimer = setTimeout(() => checkReuse(query, key), 250);
}
async function checkReuse(query, key) {
  const seq = ++reuseSeq;
  let reusable = {};
  try { reusable = (await api(`/api/run/preview?${query}`)).data.reusable || {}; }
  catch {}   // The Run preview reports the problem.
  if (seq !== reuseSeq || key !== reuseKey) return;
  REUSABLE = reusable;
  // A ticked option is dropped once its result is no longer reused.
  for (const base of [...retranscribeBases]) if (!REUSABLE[base]?.transcription) retranscribeBases.delete(base);
  for (const base of [...rediarizeBases]) if (!REUSABLE[base]?.diarization) rediarizeBases.delete(base);
  refreshForceOptions();
}
function forceOptions(base) {
  const reusable = REUSABLE[base];
  if (!reusable?.transcription && !reusable?.diarization) return null;
  const box = document.createElement("div");
  box.className = "fileForce";
  box.addEventListener("click", ev => ev.stopPropagation());
  const scribe = $("pipelineSel").value === "scribe";
  const option = (set, text, title) => {
    const label = document.createElement("label"), input = document.createElement("input"), span = document.createElement("span");
    input.type = "checkbox"; input.checked = set.has(base);
    span.textContent = text; label.title = title;
    input.setAttribute("aria-label", `${text}: ${base}`);
    label.append(input, span); box.appendChild(label);
    return input;
  };
  const again = reusable.transcription && option(retranscribeBases, scribe ? "Re-transcribe (upload again)" : "Re-transcribe",
    scribe ? "Upload this recording to Scribe again in this run instead of reusing its saved transcript."
           : "Transcribe this recording again in this run instead of reusing its transcript.");
  const rediarize = reusable.diarization && option(rediarizeBases, "Re-diarize",
    "Diarize this recording again with the selected model in this run instead of reusing its result.");
  // Diarization is aligned to the transcript, so a new transcript means a new diarization.
  const sync = () => {
    if (!rediarize) return;
    rediarize.disabled = Boolean(again?.checked);
    rediarize.checked = rediarize.disabled || rediarizeBases.has(base);
  };
  if (again) again.onchange = () => { again.checked ? retranscribeBases.add(base) : retranscribeBases.delete(base); sync(); };
  if (rediarize) rediarize.onchange = () => { rediarize.checked ? rediarizeBases.add(base) : rediarizeBases.delete(base); };
  sync();
  return box;
}
function refreshForceOptions() {
  for (const row of $("files").querySelectorAll(".file")) {
    row.querySelector(".fileForce")?.remove();
    const box = forceOptions(row.dataset.base);
    if (box) row.insertBefore(box, row.querySelector(".fileTranscripts"));
  }
}
function updateRunControls() {
  const running = !!(lastStatus && lastStatus.running);
  const pipeline = $("pipelineSel");
  if (!running && showingActivePipeline) {
    restoreControlPreference("pipelineSel");
    showingActivePipeline = false;
    filterSettings();
  }
  if (running) showingActivePipeline = true;
  // A reload or another tab can attach to a run with a different pipeline.
  // Show the server's active pipeline before locking the selector.
  if (running && pipeline.value !== lastStatus.pipeline &&
      [...pipeline.options].some(option => option.value === lastStatus.pipeline)) {
    pipeline.value = lastStatus.pipeline;
    filterSettings();
  }
  const chosen = runScope().length;
  $("startBtn").textContent = chosen ? "Run selected" : "Run folder";
  $("startBtn").title = chosen ? `Run ${chosen} selected file(s)` : "Run every eligible file in this folder";
  $("clearRunSelection").hidden = !chosen;
  $("clearRunSelection").disabled = selectionBusy || saving || changingDir;
  $("selectShown").disabled = !shownBases.length || selectionBusy || saving || changingDir;
  $("startBtn").disabled = !connected || !initialized || running || runBusy || changingDir || selectionBusy || saving || editorLoading;
  $("stopBtn").disabled = !connected || !running || runBusy;
  for (const id of ["pipelineSel", "dirBtn", "dirInput", "dirMenuBtn", "dirBrowse"]) {
    $(id).disabled = !connected || running || runBusy || changingDir;
  }
  if ($("dirMenuBtn").disabled) closeDirMenu();
  if (editorScope === "global") $("saveBtn").disabled = saving || editorLoading || !formDirty() || running;
  scheduleReuseCheck();
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
  const meta = (await api("/api/meta")).data;
  if (!Array.isArray(meta.diarization_models) || !Array.isArray(meta.pipelines)) {
    throw new Error("The WebUI server is running older code. Run python webui.py --restart in your terminal. This page will retry automatically.");
  }
  META = meta;
  $("dirInput").value = META.audio_dir;
  const paths = [...(META.audio_paths || [])];
  if (META.audio_dir && !paths.includes(META.audio_dir)) paths.push(META.audio_dir);
  savedAudioPaths = new Set(paths);
  renderDirMenu(paths);
  const sel = $("pipelineSel");
  const previous = sel.value;
  sel.innerHTML = "";
  // Only the transcription models are offered; the other pipelines stay
  // available to the API and run history.
  for (const p of META.pipelines.filter(p => p.choice)) {
    const o = document.createElement("option");
    o.value = p.id; o.textContent = `Transcription: ${p.choice}`;
    sel.appendChild(o);
  }
  if ([...sel.options].some(o => o.value === previous)) sel.value = previous;
  else restoreControlPreference("pipelineSel");
  const model = $("runModel"), chosenModel = model.value;
  model.replaceChildren(...META.diarization_models.map(m => new Option(`Diarization: ${m.label}`, m.id)));
  if ([...model.options].some(o => o.value === chosenModel)) model.value = chosenModel;
  else { model.value = META.diarization_default; restoreControlPreference("runModel"); }
  filterSettings();
}

async function changeDir() {
  if (changingDir || runBusy || lastStatus?.running) return;
  const path = $("dirInput").value.trim();
  if (!path || path === META?.audio_dir) return;
  if (!await resolveUnsaved()) { $("dirInput").value = META?.audio_dir || ""; return; }
  changingDir = true;
  let folderLoaded = false;
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
    runSelection.clear(); bulkContext = null; editorScope = "file";
    txtVersionsByBase = {};
    expandedFiles.clear();
    fileVersionRequests.clear();
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
    await loadMeta();
    await loadFiles();
    showTab(currentTab, false);
    folderLoaded = true;
  } catch (e) { reportError(e); $("dirInput").value = META?.audio_dir || ""; }
  finally { changingDir = false; updateRunControls(); }
  if (folderLoaded) await restoreSelectedFile();
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
  fileVersionRequests.clear();
  for (const base of expandedFiles) if (!FILES_BY_BASE[base]) expandedFiles.delete(base);
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

async function renderFileTranscripts(list, file) {
  const folder = folderSeq;
  list.textContent = "Loading transcripts…";
  let request = fileVersionRequests.get(file.base);
  if (!request) {
    request = api(`/api/transcript/${encodeURIComponent(file.base)}/versions`);
    fileVersionRequests.set(file.base, request);
  }
  const valid = () => folder === folderSeq && list.isConnected && fileVersionRequests.get(file.base) === request;
  try {
    const { data } = await request;
    if (!valid()) return;
    list.replaceChildren();
    const versions = [...data.versions];
    if (file.transcribed) versions.push({id: RAW_JSON_VERSION, filename: "Raw JSON transcript"});
    for (const version of versions) {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.className = "fileTranscript";
      button.dataset.version = version.id;
      button.textContent = (version.timestamp ? fmtTimestamp(version.timestamp) : version.filename)
        + (version.pipeline ? ` · ${version.pipeline}` : "");
      button.title = version.id === RAW_JSON_VERSION ? "Before diarization" : version.id;
      button.setAttribute("aria-label", `Open ${button.textContent} for ${file.base}`);
      button.onclick = handle(async () => {
        if (folder !== folderSeq || !await selectFile(file.base) || folder !== folderSeq) return;
        showTab("transcript");
        await loadTranscript(file.base, version.id);
      });
      item.appendChild(button);
      list.appendChild(item);
    }
    if (!versions.length) list.textContent = "No transcripts yet.";
  } catch (error) {
    if (!valid()) return;
    fileVersionRequests.delete(file.base);
    list.textContent = `Could not load transcripts: ${error.message} `;
    const retry = document.createElement("button");
    retry.textContent = "Retry";
    retry.onclick = () => renderFileTranscripts(list, file);
    list.appendChild(retry);
  }
}

function visibleFiles() {
  const all = Object.values(FILES_BY_BASE);
  const query = $("fileSearch").value.toLocaleLowerCase();
  const filter = $("fileFilter").value;
  const rows = all.filter(f => f.base.toLocaleLowerCase().includes(query) &&
    (filter === "all" || filter === "unfinished" && !f.has_transcript || filter === "reprocess" && f.will_reprocess ||
     filter === "ready" && f.has_transcript || filter === "overrides" && f.sidecar));
  const stateRank = f => f.will_reprocess ? 0 : !f.transcribed ? 1 : !f.has_transcript ? 2 : 3;
  rows.sort((a, b) => ($("fileSort").value === "newest" ? (b.mtime || 0) - (a.mtime || 0) :
    $("fileSort").value === "status" ? stateRank(a) - stateRank(b) : 0) || a.base.localeCompare(b.base, undefined, {numeric:true}));
  return rows;
}

function renderFiles() {
  const box = $("files");
  box.innerHTML = "";
  const all = Object.values(FILES_BY_BASE), rows = visibleFiles();
  shownBases = rows.map(f => f.base);
  const chosen = runScope(), hidden = chosen.filter(base => !shownBases.includes(base)).length;
  $("fileCount").textContent = `${rows.length} of ${all.length} files`;
  $("selectionCount").textContent = chosen.length ? `${chosen.length} selected${hidden ? ` (${hidden} hidden by filters)` : ""}` : "";
  $("selectShown").textContent = shownBases.length && shownBases.every(base => runSelection.has(base)) ? "Deselect shown" : "Select all shown";
  updateRunControls();
  if (!rows.length) {
    box.textContent = all.length ? "No files match these filters." : "No audio files found.";
    return;
  }
  for (const [index, f] of rows.entries()) {
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
    div.innerHTML = `<label class="runPick"><input type="checkbox"><span class="visuallyHidden">Select file</span></label>` +
      `<button class="fileExpand" type="button"></button>` +
      `<button class="name fileSelect">${escapeHTML(f.base)}</button><div class="badges">${badges.join("")}</div>`;
    const expand = div.querySelector(".fileExpand");
    const transcripts = document.createElement("ul");
    transcripts.className = "fileTranscripts";
    transcripts.id = `file-transcripts-${index}`;
    transcripts.setAttribute("aria-label", `Transcripts for ${f.base}`);
    transcripts.setAttribute("aria-live", "polite");
    expand.setAttribute("aria-controls", transcripts.id);
    const updateExpansion = () => {
      const open = expandedFiles.has(f.base);
      expand.textContent = open ? "▾" : "▸";
      expand.setAttribute("aria-expanded", String(open));
      expand.setAttribute("aria-label", `${open ? "Hide" : "Show"} transcripts for ${f.base}`);
      expand.title = open ? "Hide transcripts" : "Show transcripts";
      transcripts.hidden = !open;
    };
    expand.onclick = ev => {
      ev.stopPropagation();
      expandedFiles.has(f.base) ? expandedFiles.delete(f.base) : expandedFiles.add(f.base);
      updateExpansion();
      if (!transcripts.hidden) renderFileTranscripts(transcripts, f);
    };
    transcripts.addEventListener("click", ev => ev.stopPropagation());
    updateExpansion();
    const force = forceOptions(f.base);
    if (force) div.appendChild(force);
    div.appendChild(transcripts);
    const pick = div.querySelector(".runPick input");
    pick.checked = runSelection.has(f.base);
    pick.setAttribute("aria-label", `Select ${f.base} for settings and processing`);
    pick.onclick = ev => ev.stopPropagation();
    pick.disabled = selectionBusy || saving || changingDir;
    pick.onchange = handle(async () => {
      const next = new Set(runSelection);
      pick.checked ? next.add(f.base) : next.delete(f.base);
      pick.checked = runSelection.has(f.base);
      await changeFileSelection(next);
    });
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
    if (!transcripts.hidden) renderFileTranscripts(transcripts, f);
  }
}

/* ---------- sidecar editor ---------- */

function syncSetting(el) {
  const box = el.closest(".fieldValue");
  if (!box) return;
  const explicit = editorScope === "global" || el.dataset.allowEmpty === "1" ? el.dataset.explicit === "1" : el.value !== "";
  el.classList.toggle("set", explicit);
  const note = box.querySelector(".inheritance");
  note.textContent = explicit ? (editorScope === "global" ? "global value" : "per-file override") :
    (editorScope === "global" ? "inherited program default" : "inherited global value");
  box.querySelector(".resetSetting").disabled = el.disabled || !explicit;
  box.querySelector(".resetSetting").hidden = !explicit;
  const error = box.querySelector(".fieldError");
  error.textContent = el.validity.valid ? "" : el.validationMessage;
  el.setAttribute("aria-invalid", String(!el.validity.valid));
  if (editorScope === "batch") {
    note.textContent = el.dataset.bulkChanged === "1" ? (explicit ? "apply to all selected files" : "use global for all selected files") :
      el.dataset.mixed === "1" ? "mixed values — unchanged unless edited" : "same setting in all selected files";
    box.querySelector(".resetSetting").hidden = false;
    box.querySelector(".resetSetting").disabled = el.disabled;
  }
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
    const inheritValue = editorScope !== "global" && spec.allow_empty ? "__inherit__" : "";
    if (inheritValue) el.dataset.inheritValue = inheritValue;
    for (const choice of new Set([inheritValue, ...spec.choices])) {
      const option = document.createElement("option"); option.value = choice; option.textContent = choice === "__inherit__" ? "Use global" : (spec.choice_labels?.[choice] ?? (choice || `Use ${editorScope === "global" ? "program default" : "global"}`)); el.appendChild(option);
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
    el.value = value ?? el.dataset.inheritValue ?? ""; el.placeholder = globalVal || "Not set";
  }
  el.dataset.allowEmpty = spec.allow_empty ? "1" : "0";
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
    el.dataset.bulkChanged = "1"; el.dataset.mixed = "0";
    el.value = el.dataset.inheritValue ?? ""; el.dataset.explicit = "0";
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
    el.dataset.bulkChanged = "1"; el.dataset.mixed = "0";
    if (el.type === "checkbox") { el.value = String(el.checked); el.indeterminate = false; el.setCustomValidity(""); }
    if (el.dataset.numericType) {
      const n = Number(el.value);
      el.setCustomValidity(el.value && (!Number.isFinite(n) || (el.dataset.numericType === "int" && !Number.isInteger(n)) || (spec.min !== undefined && n < spec.min) || (spec.max !== undefined && n > spec.max)) ? "Enter a valid number within the allowed range." : "");
    }
    el.dataset.explicit = el.value === el.dataset.inheritValue ? "0" : "1"; syncSetting(el); updateSaveState();
  };
  el.addEventListener("input", change); el.addEventListener("change", change);
  return box;
}

function renderBulkSettings(context) {
  bulkContext = context;
  const values = {}, mixed = [];
  for (const key of Object.keys(META.params)) {
    const first = context.files[context.bases[0]].values[key];
    if (context.bases.every(base => context.files[base].values[key] === first)) {
      if (first !== undefined) values[key] = first;
    } else mixed.push(key);
  }
  renderSettings({values, mixed}, null, "batch");
}

async function changeFileSelection(next) {
  if (selectionBusy || saving || changingDir) return false;
  selectionBusy = true; updateRunControls();
  try {
    if (!await resolveUnsaved()) return false;
    runSelection = new Set(next);
    const base = currentBase || [...runSelection][0];
    if (base) {
      await selectFile(base, true);
      showTab("settings");
    }
    return true;
  } finally { selectionBusy = false; renderFiles(); updateRunControls(); }
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
  const title = document.createElement("h2"); title.textContent = scope === "global" ? "Global settings" : scope === "batch" ? `Settings for ${bulkContext.bases.length} selected files` : base; editor.appendChild(title);
  if (scope === "file") {
    const comment = document.createElement("textarea"); comment.id = "commentField"; comment.rows = 3; comment.value = data.comment || "";
    comment.setAttribute("aria-label", "Comments about this file"); comment.placeholder = "Comments do not trigger reprocessing."; comment.style.width = "100%";
    comment.addEventListener("input", updateSaveState); editor.appendChild(comment);
  } else if (scope === "batch") {
    const note = document.createElement("p"); note.className = "settingsNote";
    note.textContent = "Only fields you edit are applied to every selected file. Mixed values, other settings and individual comments are preserved."; editor.appendChild(note);
    const files = document.createElement("details"), summary = document.createElement("summary");
    summary.textContent = "Selected files"; files.appendChild(summary);
    const list = document.createElement("p"); list.textContent = bulkContext.bases.join(", "); files.appendChild(list); editor.appendChild(files);
  } else {
    const note = document.createElement("p"); note.className = "settingsNote";
    note.textContent = "Defaults for future runs. Per-file overrides take precedence. Changes are saved to " + data.path + ". Credentials are not shown."; editor.appendChild(note);
  }
  const applicabilityNote = document.createElement("p"); applicabilityNote.id = "settingsApplicability"; applicabilityNote.className = "settingsNote";
  applicabilityNote.textContent = "Scribe: these settings control local processing. Backfill-only settings take effect when backfill is enabled. SUNET transcription is controlled by the Scribe language/model selector. Values marked as inherited are editable; greyed-out controls are unavailable.";
  editor.appendChild(applicabilityNote);
  const common = ["RECORDING_TYPE", "INTERVIEWER_NAME", "INTERVIEWEE_NAME", "SWHISPER_LANGUAGE", "SWHISPER_NUM_SPEAKERS", "SWHISPER_MIN_SPEAKERS", "SWHISPER_MAX_SPEAKERS", "SWHISPER_INITIAL_PROMPT", "SWHISPER_FORCE_REPROCESS", "SWHISPER_DIARIZE_FORCE_REPROCESS"];
  const groups = [{name:"Common settings",keys:common.filter(k=>editorMeta.params[k])}, ...editorMeta.groups.map(g=>({...g,keys:g.keys.filter(k=>!common.includes(k))}))];
  {
    const note = document.createElement("label"); note.className = "settingsNote";
    note.innerHTML = '<input id="showOtherSettings" type="checkbox"> Show settings for other pipelines'; editor.appendChild(note);
    rememberControl("showOtherSettings");
    note.querySelector("input").onchange = filterSettings;
  }
  for (const group of groups) {
    if (!group.keys.length) continue;
    const section = document.createElement("details"); section.className = "group";
    section.open = group.name === "Common settings" || group.keys.some(k=>Object.hasOwn(data.values,k));
    const summary = document.createElement("summary"); summary.textContent = group.name; section.appendChild(summary);
    rememberDetails(section, `group.${scope}.${group.name}`);
    for (const key of group.keys) {
      const spec = editorMeta.params[key]; if (!spec) continue;
      const row = document.createElement("div"); row.className = "row"; row.dataset.stage = spec.stage;
      const label = document.createElement("label"); label.htmlFor = `setting-${key}`;
      label.textContent = spec.label || key.replace("SWHISPER_", "").replaceAll("_", " ").toLowerCase(); label.title = key;
      row.append(label, inputFor(key, spec, data.values[key])); section.appendChild(row);
      if (scope === "batch" && data.mixed.includes(key)) {
        const el = row.querySelector("[data-key]"); el.dataset.mixed = "1";
        if (el.type === "checkbox") el.indeterminate = true;
        else if (el.tagName === "SELECT") {
          const option = document.createElement("option"); option.value = "__mixed__"; option.textContent = "Mixed values"; option.disabled = true;
          el.appendChild(option); el.value = "__mixed__";
        } else { el.value = ""; el.placeholder = "Mixed values"; }
      }
    }
    editor.appendChild(section);
  }
  if (data.unknown && Object.keys(data.unknown).length) {
    const note = document.createElement("p"); note.className="settingsNote"; note.textContent="Additional settings are preserved: " + Object.keys(data.unknown).join(", ");editor.appendChild(note);
  }
  editorBase = scope === "global" ? "__global__" : scope === "batch" ? "__batch__" : base;
  editor.querySelectorAll("[data-key]").forEach(syncSetting);
  savedForm = JSON.stringify(formValues()); editorLoading = false; editor.inert = false;
  $("editorBar").style.display = currentTab === "settings" ? "" : "none";
  $("deleteBtn").hidden = scope !== "file"; $("deleteBtn").disabled = !data.exists;
  $("globalSettingsBtn").setAttribute("aria-pressed", String(scope === "global"));
  filterSettings(); updateSaveState(); updateRunControls();
}
function filterSettings() {
  const scribe = $("pipelineSel").value === "scribe";
  $("scribeControls").hidden = !scribe;
  // Scribe runs local community-1 or Nemotron diarization only.
  for (const option of $("runModel").options) option.disabled = scribe && option.value === "precision";
  if ($("runModel").selectedOptions[0]?.disabled) $("runModel").value = META?.diarization_default || "community-1";
  if ($("settingsApplicability")) $("settingsApplicability").hidden = !scribe;
  document.querySelectorAll("#editor .row").forEach(row => {
    row.hidden = false;
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

async function restoreSelectedFile() {
  if (!META || currentBase || editorBase || editorLoading || changingDir || saving) return;
  const key = `selectedFile.${META.audio_dir}`;
  const base = readUIPreferences()[key] ?? uiPreferences[key];
  if (typeof base === "string" && Object.hasOwn(FILES_BY_BASE, base)) await selectFile(base);
}

async function selectFile(base, force = false) {
  if (changingDir || saving) return false;
  if (!force && base === editorBase && base === currentBase) return true;
  if (!force && !await resolveUnsaved()) return false;
  resetAudioReview();
  if (base !== currentBase) setCompareAudio(null);
  const seq = ++editorSeq, folder = folderSeq;
  transcriptSeq++; renderSeq++; logSeq++; logListSeq++;
  currentBase = base;
  saveUIPreference(`selectedFile.${META.audio_dir}`, base);
  editorBase = savedForm = null;
  editorLoading = true;
  $("editor").inert = true;
  $("editor").textContent = "Loading settings…";
  $("deleteBtn").disabled = true;
  updateSaveState(); updateRunControls();
  showTab(currentTab, false);
  document.querySelectorAll("#files .file").forEach(d => {
    d.classList.toggle("sel", d.dataset.base === base);
    d.querySelector(".fileSelect")?.setAttribute("aria-pressed", String(d.dataset.base === base));
    if (d.querySelector(".fileSelect")) d.querySelector(".fileSelect").tabIndex = d.dataset.base === base ? 0 : -1;
  });
  const bases = runScope().length ? runScope() : [base];
  let files;
  try {
    files = Object.fromEntries(await Promise.all(bases.map(async name => {
      const {data} = await api(`/api/sidecar/${encodeURIComponent(name)}`);
      return [name, data];
    })));
  }
  catch (e) {
    if (seq === editorSeq && folder === folderSeq) {
      $("editor").textContent = "Could not load settings. Select the file to retry.";
      editorLoading = false; $("editor").inert = false;
      updateSaveState(); updateRunControls(); reportError(e);
    }
    return false;
  }
  if (seq !== editorSeq || folder !== folderSeq || base !== currentBase) return false;
  if (bases.length > 1) renderBulkSettings({bases, files});
  else renderSettings(files[bases[0]], bases[0], "file");
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

function showTab(tab, remember = true) {
  if (remember) saveUIPreference("activeTab", tab);
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
  // Audio keeps playing when its view is hidden, so stop the one left behind.
  if (tab !== "compare") $("compareAudio").pause();
  if (tab !== "transcript") $("audioPlayer").pause();
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
const RENDER_STORAGE_KEY = "swhisper_render_options";
let renderPreferences = loadRenderPreferences();
if (renderPreferences) setRenderOptions(renderPreferences);

function loadRenderPreferences() {
  try {
    const saved = JSON.parse(localStorage.getItem(RENDER_STORAGE_KEY));
    if (!saved || typeof saved !== "object" || Array.isArray(saved)) return null;
    const options = {...RENDER_DEFAULTS};
    for (const [key, fallback] of Object.entries(RENDER_DEFAULTS)) {
      const value = saved[key];
      if (typeof fallback === "boolean" ? typeof value === "boolean"
          : typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 60) {
        options[key] = value;
      }
    }
    return options;
  } catch { return null; }
}

function saveRenderPreferences() {
  if (!["optThreshold", "optMinPause"].every(id => $(id).value !== "" && $(id).validity.valid)) return;
  renderPreferences = renderOptions();
  try { localStorage.setItem(RENDER_STORAGE_KEY, JSON.stringify(renderPreferences)); } catch {}
}

function debounceRender() {
  // Save immediately, even if the user closes the page before rendering starts.
  saveRenderPreferences();
  clearTimeout(renderTimer);
  renderTimer = setTimeout(renderOptionsChanged, 250);
}

function renderOptionsChanged() {
  saveRenderPreferences();
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
    field.value = data.names[label] || (data.suggested || {})[label] || ""; field.placeholder = label; field.maxLength = 60;
    return [caption, field];
  }));
  $("speakerSuggestion").hidden = !Object.keys(data.suggested || {}).length;
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
  const context = {...speakerContext, folder: folderSeq};
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
  if (context.folder === folderSeq && context.base === currentBase && context.version === $("versionSel").value) {
    await loadAudioReview(context.base, context.version, true);
  }
}

async function saveRenderedView() {
  if (!renderAvailable || !currentBase) return;
  exportContext = {base:currentBase, version:$("versionSel").value, options:renderOptions()};
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
/* ---------- waveform playback ---------- */

function waveformTime(seconds) {
  const tenths = Math.round(Math.max(0, seconds || 0) * 10);
  const hours = Math.floor(tenths / 36000), minutes = Math.floor(tenths / 600) % 60;
  return `${hours ? hours + ":" + String(minutes).padStart(2, "0") : minutes}:${String(Math.floor(tenths / 10) % 60).padStart(2, "0")}.${tenths % 10}`;
}

// Each tab owns a player; both use the same interaction and stop logic.
class WaveformPlayer {
  constructor(host, audio) {
    this.host = host;
    this.audio = audio;
    this.sequence = 0;
    this.duration = 0;
    this.viewStart = this.viewEnd = 0;
    this.detailSequence = 0;
    this.detailCache = new Map();
    this.range = null;
    host.innerHTML = `<div class="waveformTrack" role="slider" tabindex="0" aria-label="Audio playhead"
        aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" aria-disabled="true" aria-describedby="${host.id}Hint"
        title="Arrow keys seek; Shift + arrows select. Space plays or pauses. Escape clears the selection.">
        <svg viewBox="0 0 2048 100" preserveAspectRatio="none" aria-hidden="true"><path></path></svg>
        <div class="waveformRange" hidden></div><div class="waveformPlayhead" hidden></div>
      </div>
      <div class="waveformSpeakers"></div>
      <div class="waveformScale" aria-hidden="true"></div>
      <div class="waveformZoom" role="group" aria-label="Waveform zoom">
        <button class="waveformZoomOut" aria-label="Zoom out waveform" title="Zoom out">−</button>
        <output class="waveformZoomLevel" aria-label="Waveform zoom level">1×</output>
        <button class="waveformZoomIn" aria-label="Zoom in waveform" title="Zoom in around the playhead">+</button>
        <button class="waveformFit">Fit selection</button><button class="waveformFull">Full audio</button>
        <span class="waveformDetailStatus" role="status"></span>
      </div>
      <div class="waveformPan" hidden>
        <button class="waveformEarlier" aria-label="Show earlier audio">‹</button>
        <input class="waveformPosition" type="range" min="0" max="0" step="any" value="0" aria-label="Waveform position">
        <button class="waveformLater" aria-label="Show later audio">›</button>
      </div>
      <div class="waveformFooter"><span class="waveformStatus" role="status"></span>
        <button class="waveformClear" hidden>Clear selection</button><button class="waveformRetry" hidden>Retry waveform</button>
        <span class="waveformHint" id="${host.id}Hint">Click to seek. Drag to select, then press Play.</span></div>`;
    const find = selector => host.querySelector(selector);
    this.track = find(".waveformTrack");
    this.path = find("path");
    this.overlay = find(".waveformRange");
    this.playhead = find(".waveformPlayhead");
    this.status = find(".waveformStatus");
    this.clear = find(".waveformClear");
    this.retry = find(".waveformRetry");
    this.speakers = new WaveformSpeakers(find(".waveformSpeakers"));
    this.zoomIn = find(".waveformZoomIn");
    this.zoomOut = find(".waveformZoomOut");
    this.fit = find(".waveformFit");
    this.full = find(".waveformFull");
    this.position = find(".waveformPosition");
    this.detailStatus = find(".waveformDetailStatus");
    this.zoomIn.onclick = () => this.zoomBy(2);
    this.zoomOut.onclick = () => this.zoomBy(.5);
    this.full.onclick = () => this.setViewport(0, this.duration);
    this.fit.onclick = () => { if (this.range) this.setViewport(this.range.start, this.range.end - this.range.start); };
    this.position.oninput = () => this.setViewport(Number(this.position.value), this.viewEnd - this.viewStart);
    find(".waveformEarlier").onclick = () => this.pan(-.75);
    find(".waveformLater").onclick = () => this.pan(.75);
    this.clear.onclick = () => { this.clearRange(); this.track.focus(); };
    this.retry.onclick = () => this.load(this.base, true);
    this.track.onpointerdown = e => this.pointerDown(e);
    this.track.onpointermove = e => this.pointerMove(e);
    this.track.onpointerup = e => this.pointerUp(e);
    this.track.onpointercancel = this.track.onlostpointercapture = () => this.cancelDrag();
    this.track.onkeydown = e => this.keyDown(e);
    audio.addEventListener("play", () => {
      if (this.range && (audio.currentTime < this.range.start || audio.currentTime >= this.range.end - .01)) {
        audio.currentTime = this.range.start;
      }
      this.followView = true;
      this.revealTime(audio.currentTime);
      this.armStop();
      this.animate();
    });
    for (const type of ["pause", "ended"]) audio.addEventListener(type, () => {
      clearTimeout(this.stopTimer); cancelAnimationFrame(this.frame); this.paint();
    });
    audio.addEventListener("timeupdate", () => {
      this.enforceEnd();
      if (!audio.paused && this.followView) this.revealTime(audio.currentTime);
      this.paint();
    });
    audio.addEventListener("seeking", () => {
      // Native controls may seek outside a selected range. That starts a new seek.
      if (this.range && (audio.currentTime < this.range.start - .01 || audio.currentTime > this.range.end + .01)) this.clearRange();
      this.armStop(); this.paint();
    });
    for (const type of ["seeked", "ratechange", "playing"]) audio.addEventListener(type, () => this.armStop());
  }

  async load(base, retry = false) {
    const src = base ? `/api/audio/${encodeURIComponent(base)}?folder=${folderSeq}` : "";
    if (src === this.source && !retry) return;
    const sameSource = src === this.source;
    const sequence = ++this.sequence;
    this.abort?.abort();
    this.detailAbort?.abort();
    clearTimeout(this.detailTimer);
    this.detailSequence++;
    this.detailCache.clear();
    this.cancelDrag();
    this.audio.pause();
    this.base = base;
    this.source = src;
    this.duration = 0;
    this.viewStart = this.viewEnd = 0;
    this.overview = null;
    this.detailStatus.textContent = "";
    if (!sameSource) this.speakers.clear();
    else this.speakers.setViewport(0, 0);
    this.range = null;
    this.keyAnchor = null;
    this.path.setAttribute("d", "");
    this.host.querySelector(".waveformScale").replaceChildren();
    this.retry.hidden = true;
    if (src) this.audio.src = src; else this.audio.removeAttribute("src");
    this.audio.load();
    this.paint();
    this.status.textContent = base ? "Loading waveform…" : "";
    if (!base) return;
    const controller = this.abort = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 130000);
    try {
      const {data} = await api(`/api/waveform/${encodeURIComponent(base)}?folder=${folderSeq}`, {signal: this.abort.signal});
      if (sequence !== this.sequence) return;
      if (!(data.duration > 0) || !Number.isFinite(data.duration) || !Array.isArray(data.peaks) || !data.peaks.length
          || !data.peaks.every(p => Number.isFinite(p) && p >= 0 && p <= 1)) throw Error("Invalid waveform");
      this.duration = data.duration;
      this.overview = {...data, start: 0, end: data.duration};
      // A single gain across the whole file keeps quiet passages proportionally quiet.
      this.gain = 44 / Math.max(.01, ...data.peaks);
      this.setViewport(0, data.duration);
    } catch (e) {
      if (sequence !== this.sequence) return;
      this.status.textContent = "Waveform unavailable. Audio controls still work.";
      this.retry.hidden = false;
    } finally { clearTimeout(timeout); }
  }

  get minimumSpan() { return Math.min(this.duration, Math.max(.25, this.duration / 4096)); }

  setSpeakerTracks(tracks) {
    this.speakers.setTracks(tracks);
    this.speakers.setViewport(this.viewStart, this.viewEnd);
  }

  zoomBy(factor) {
    if (!this.duration) return;
    const span = this.viewEnd - this.viewStart;
    const time = this.audio.currentTime;
    const anchor = time >= this.viewStart && time <= this.viewEnd ? time : (this.viewStart + this.viewEnd) / 2;
    const fraction = (anchor - this.viewStart) / span;
    const next = Math.max(this.minimumSpan, Math.min(this.duration, span / factor));
    this.setViewport(anchor - next * fraction, next);
  }

  pan(fraction) {
    const span = this.viewEnd - this.viewStart;
    this.setViewport(this.viewStart + span * fraction, span);
  }

  revealTime(time) {
    if (!this.duration || (time >= this.viewStart - .001 && time < this.viewEnd) || (time === this.duration && this.viewEnd === this.duration)) return;
    const span = this.viewEnd - this.viewStart;
    this.setViewport(time - span * .1, span, true);
  }

  setViewport(start, span, follow = false) {
    if (!this.duration) return;
    span = Math.max(this.minimumSpan, Math.min(this.duration, span));
    start = Math.max(0, Math.min(this.duration - span, start));
    this.cancelDrag();
    this.followView = follow;
    this.viewStart = start;
    this.viewEnd = start + span;
    this.track.querySelector("svg").setAttribute("viewBox", `${start} 0 ${span} 100`);
    this.path.style.strokeWidth = span / 2048;
    const scale = this.host.querySelector(".waveformScale");
    scale.replaceChildren();
    for (let i = 0; i <= 4; i++) {
      const tick = document.createElement("span");
      tick.textContent = waveformTime(start + span * i / 4);
      scale.append(tick);
    }
    this.speakers.setViewport(start, this.viewEnd);
    this.position.max = this.duration - span;
    this.position.value = start;
    this.position.setAttribute("aria-valuetext", `${waveformTime(start)} to ${waveformTime(this.viewEnd)}`);
    this.host.querySelector(".waveformPan").hidden = span >= this.duration - .000001;
    const zoom = this.duration / span;
    this.host.querySelector(".waveformZoomLevel").textContent = `${Number(zoom.toFixed(1))}×`;
    this.host.querySelector(".waveformEarlier").disabled = start <= 0;
    this.host.querySelector(".waveformLater").disabled = this.viewEnd >= this.duration;
    this.drawPeaks(this.overview);
    this.paint();
    clearTimeout(this.detailTimer);
    this.detailAbort?.abort();
    const seq = ++this.detailSequence;
    if (span >= this.duration - .000001) { this.detailStatus.textContent = ""; return; }
    this.detailStatus.textContent = "Loading detail…";
    this.detailTimer = setTimeout(() => this.loadDetail(seq), 100);
  }

  drawPeaks(data) {
    if (!data) return;
    const width = (data.end - data.start) / data.peaks.length;
    this.path.setAttribute("d", data.peaks.map((peak, i) => {
      const x = data.start + (i + .5) * width, height = Math.max(.35, Math.min(48, peak * this.gain));
      return `M${x.toFixed(6)},${(50 - height).toFixed(2)}V${(50 + height).toFixed(2)}`;
    }).join(""));
  }

  async loadDetail(sequence) {
    const key = `${this.viewStart.toFixed(6)},${this.viewEnd.toFixed(6)}`;
    const controller = this.detailAbort = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 130000);
    try {
      let data = this.detailCache.get(key);
      if (!data) {
        ({data} = await api(`/api/waveform/${encodeURIComponent(this.base)}?folder=${folderSeq}&start=${this.viewStart}&end=${this.viewEnd}`, {signal: controller.signal}));
      }
      if (sequence !== this.detailSequence) return;
      if (!Number.isFinite(data.start) || !Number.isFinite(data.end) || data.end <= data.start || !Array.isArray(data.peaks)
          || !data.peaks.length || !data.peaks.every(p => Number.isFinite(p) && p >= 0 && p <= 1)) throw Error("Invalid waveform detail");
      this.detailCache.set(key, data);
      if (this.detailCache.size > 16) this.detailCache.delete(this.detailCache.keys().next().value);
      this.drawPeaks(data);
      this.detailStatus.textContent = "";
    } catch (e) {
      if (sequence === this.detailSequence) this.detailStatus.textContent = "Detail unavailable; showing overview.";
    } finally { clearTimeout(timeout); }
  }

  paint() {
    const duration = this.duration, time = Math.min(duration, this.audio.currentTime || 0);
    const span = this.viewEnd - this.viewStart;
    this.track.setAttribute("aria-disabled", String(!duration));
    this.track.setAttribute("aria-valuemax", String(duration));
    this.track.setAttribute("aria-valuenow", String(time));
    this.track.setAttribute("aria-valuetext", waveformTime(time));
    this.playhead.hidden = !span || time < this.viewStart - .001 || time > this.viewEnd + .001;
    this.playhead.style.left = `${span ? Math.max(0, Math.min(1, (time - this.viewStart) / span)) * 100 : 0}%`;
    this.clear.hidden = !this.range;
    const left = Math.max(this.viewStart, this.range?.start ?? 0), right = Math.min(this.viewEnd, this.range?.end ?? 0);
    this.overlay.hidden = !this.range || right <= left;
    if (this.range && span) {
      this.overlay.style.left = `${(left - this.viewStart) / span * 100}%`;
      this.overlay.style.width = `${Math.max(0, right - left) / span * 100}%`;
    }
    this.zoomIn.disabled = !duration || span <= this.minimumSpan + .000001;
    this.zoomOut.disabled = this.full.disabled = !duration || span >= duration - .000001;
    this.fit.disabled = !duration || !this.range;
    if (!duration) {
      this.host.querySelector(".waveformPan").hidden = true;
      this.host.querySelector(".waveformZoomLevel").textContent = "1×";
    }
    if (duration) {
      const message = this.range
        ? `Selected ${waveformTime(this.range.start)} – ${waveformTime(this.range.end)} · ${waveformTime(this.range.end - this.range.start)}`
        : `Duration ${waveformTime(duration)}`;
      if (this.status.textContent !== message) this.status.textContent = message;
    }
  }

  clearRange() {
    this.range = null; this.keyAnchor = null;
    clearTimeout(this.stopTimer);
    this.paint();
  }

  seek(time) {
    this.clearRange();
    this.audio.currentTime = time;
    this.revealTime(time);
    this.paint();
  }

  select(start, end) {
    this.range = {start: Math.min(start, end), end: Math.max(start, end)};
    if (this.range.end - this.range.start < .001) this.range = null;
    this.paint();
  }

  pointerTime(e) {
    const box = this.track.getBoundingClientRect();
    return this.viewStart + Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) * (this.viewEnd - this.viewStart);
  }

  pointerDown(e) {
    if (!this.duration || e.button !== 0 || this.drag) return;
    e.preventDefault();
    this.track.focus(); this.audio.pause(); this.keyAnchor = null;
    this.drag = {id: e.pointerId, x: e.clientX, start: this.pointerTime(e), range: this.range, moved: false};
    this.track.setPointerCapture(e.pointerId);
  }

  pointerMove(e) {
    if (!this.drag || this.drag.id !== e.pointerId) return;
    this.drag.moved ||= Math.abs(e.clientX - this.drag.x) >= 4;
    if (this.drag.moved) this.select(this.drag.start, this.pointerTime(e));
  }

  pointerUp(e) {
    if (!this.drag || this.drag.id !== e.pointerId) return;
    this.pointerMove(e);
    const drag = this.drag;
    this.drag = null;
    this.track.releasePointerCapture(e.pointerId);
    if (drag.moved && this.range) this.audio.currentTime = this.range.start;
    else this.seek(this.pointerTime(e));
    this.paint();
  }

  cancelDrag() {
    if (!this.drag) return;
    const drag = this.drag;
    this.drag = null;
    this.range = drag.range;
    if (this.track.hasPointerCapture(drag.id)) this.track.releasePointerCapture(drag.id);
    this.paint();
  }

  keyDown(e) {
    if (e.key === "Escape") { e.preventDefault(); this.cancelDrag(); this.clearRange(); return; }
    if (!this.duration || !["ArrowLeft", "ArrowRight", "Home", "End", " "].includes(e.key)) return;
    e.preventDefault();
    if (e.key === " ") {
      if (this.audio.paused) this.audio.play().catch(() => {}); else this.audio.pause();
      return;
    }
    this.audio.pause();
    const step = Math.min(.1, (this.viewEnd - this.viewStart) / 10) || .1;
    const position = e.shiftKey && this.keyAnchor != null ? this.keyEnd : this.audio.currentTime;
    const target = e.key === "Home" ? 0 : e.key === "End" ? this.duration
      : Math.max(0, Math.min(this.duration, position + (e.key === "ArrowLeft" ? -step : step)));
    if (e.shiftKey) {
      this.keyAnchor ??= this.audio.currentTime;
      this.keyEnd = target;
      this.select(this.keyAnchor, target);
      this.audio.currentTime = this.range?.start ?? target;
      this.revealTime(target);
    } else this.seek(target);
    this.paint();
  }

  enforceEnd() {
    if (!this.range || this.audio.paused || this.audio.currentTime < this.range.end) return;
    this.audio.pause();
    this.audio.currentTime = this.range.end;
  }

  armStop() {
    clearTimeout(this.stopTimer);
    if (!this.range || this.audio.paused) return;
    this.enforceEnd();
    if (!this.audio.paused) this.stopTimer = setTimeout(() => this.armStop(),
      Math.max(10, (this.range.end - this.audio.currentTime) / this.audio.playbackRate * 1000));
  }

  animate() {
    cancelAnimationFrame(this.frame);
    this.enforceEnd(); this.paint();
    if (!this.audio.paused) this.frame = requestAnimationFrame(() => this.animate());
  }
}

function resetAudioReview(keepAudio = false) {
  timelineSeq++;
  transcriptWaveform.speakers.clear();
  if (!keepAudio) transcriptWaveform.load(null);
  $("audioBar").hidden=true;$("timeline").hidden=true;$("timeline").replaceChildren();$("transcriptText").style.display="";
}
async function loadAudioReview(base, version, speakersOnly = false) {
  const seq=++timelineSeq, folder=folderSeq, renderAtLoad=renderSeq;
  try {
    const {data}=await api(`/api/timeline/${encodeURIComponent(base)}?version=${encodeURIComponent(version)}`);
    if(seq!==timelineSeq||base!==currentBase||folder!==folderSeq)return;
    transcriptWaveform.setSpeakerTracks([{label: "Speakers", segments: data.segments}]);
    if (speakersOnly) return;
    transcriptWaveform.load(base);
    $("audioBar").hidden=false;$("reviewAudio").disabled=!data.segments.length;
    $("audioMsg").textContent=data.segments.length ? "" : "No timestamped segments for this version.";
    if (renderAtLoad === renderSeq) renderTimeline(data.segments);

  }catch(e){if(seq===timelineSeq && !speakersOnly){$("audioBar").hidden=false;$("audioMsg").textContent=e.message;$("reviewAudio").disabled=true;}}
}

function renderTimeline(segments) {
  $("timeline").replaceChildren();
  $("reviewAudio").disabled = !segments.length;
    for(const segment of segments){
      const row=document.createElement("div");row.className="segment";row.dataset.start=segment.start;row.dataset.end=segment.end;
      const button=document.createElement("button");const seconds=Math.floor(segment.start);
      button.textContent=`${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,"0")} ${segment.speaker}`;
      button.onclick=()=>{transcriptWaveform.seek(segment.start);$("audioPlayer").play().catch(()=>{$("audioMsg").textContent="Press Play to start audio, or choose a browser-supported audio format.";});};
      const text=document.createElement("p");text.textContent=segment.text;row.append(button,text);$("timeline").appendChild(row);
    }
  updateReviewView();
}

function updateReviewView() {
  const review = $("reviewAudio").checked && !$("reviewAudio").disabled;
  $("timeline").hidden = !review;
  $("transcriptText").style.display = review ? "none" : "";
}

async function loadTranscript(base, version) {
  const seq = ++transcriptSeq, folder = folderSeq;
  resetAudioReview(true);
  transcriptWaveform.load(base);
  $("audioBar").hidden = !base;
  $("reviewAudio").disabled = true;
  $("audioMsg").textContent = "";
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

    renderBaseline = {...RENDER_DEFAULTS, stats: !!data.has_stats, preamble: data.has_preamble ?? true};
    if (!renderPreferences && data.can_render) {
      setRenderOptions(renderBaseline);
      saveRenderPreferences();
    }
    updateRenderBar(!!data.can_render);
    loadAudioReview(base, data.selected);
    // Re-render saved name changes, including removal of automatic names.
    if (data.can_render && (!renderIsDefault(renderOptions()) || data.speaker_names_edited || Object.keys(data.speaker_names || {}).length)) applyRenderOptions();
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
    pane.result = pane.active = null;
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
    setCompareAudio(null);
    return;
  }
  setCompareAudio(base);
  $("compareMsg").textContent = "Loading…";
  let versions;
  try { versions = (await api(`/api/transcript/${encodeURIComponent(base)}`)).data.versions || []; }
  catch (e) { versions = []; }
  if (seq !== compareSeq || folder !== folderSeq || base !== currentBase) return;
  // Keep the chosen versions while the same recording stays selected.
  const previous = compareBase === base ? [comparePane("left").sel.value, comparePane("right").sel.value] : [];
  compareBase = base;
  if (!versions.length) {
    comparisonWaveform.speakers.clear();
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
    if (!data.can_render) return {content: data.content, anchors: data.anchors || [], timeline: [], rendered: false};
  }
  if (compareCanRender[version]) {
    const {data} = await api(`/api/render/${encodeURIComponent(base)}`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({version, options}),
    });
    return {content: data.content, anchors: data.anchors || [], timeline: data.timeline || [], rendered: true};
  }
  // Without a diarized JSON there are no segment times to review with.
  const {data} = await api(`/api/transcript/${encodeURIComponent(base)}${query}`);
  return {content: data.content, anchors: data.anchors || [], timeline: [], rendered: false};
}

async function refreshComparison() {
  const base = compareBase;
  if (!base || base !== currentBase || !comparePane("left").sel.value) return;
  const seq = ++compareSeq, folder = folderSeq;
  const options = renderOptions();
  const position = compareCoordinate(comparePane("left"));
  comparisonWaveform.speakers.clear();
  const bar = $("renderBar");
  bar.classList.add("busy");
  bar.classList.toggle("dirty", !renderIsDefault(options));
  $("compareMsg").textContent = "Loading…";
  try {
    const [results, speakerTracks] = await Promise.all([
      Promise.all(["left", "right"].map(side => renderComparePane(side, base, options))),
      Promise.all(["left", "right"].map(async side => {
        const version = comparePane(side).sel.value;
        try {
          const {data} = await api(`/api/timeline/${encodeURIComponent(base)}?version=${encodeURIComponent(version)}`);
          return {label: side === "left" ? "Left" : "Right", segments: data.segments || []};
        } catch { return {label: side, segments: []}; }
      })),
    ]);
    if (seq !== compareSeq || folder !== folderSeq || base !== currentBase) return;
    comparisonWaveform.setSpeakerTracks(speakerTracks);
    ["left", "right"].forEach((side, index) => { comparePane(side).result = results[index]; });
    showComparePanes(position);
    bar.style.display = results.some(r => r.rendered) ? "" : "none";
    const same = comparePane("left").sel.value === comparePane("right").sel.value;
    $("compareMsg").textContent = same ? "Both panes show the same version." : "";
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
    const rest = lines.slice(anchor ? from + 1 : from, to).join("\n") + (to < lines.length ? "\n" : "");
    if (anchor) {
      // The speaker heading plays the paragraph. Its start is an exact time;
      // positions inside the paragraph are only interpolated, so they do not seek.
      div.className = "compareParagraph";
      const heading = document.createElement("span");
      heading.className = "compareSeek";
      heading.tabIndex = 0;
      heading.setAttribute("role", "button");
      heading.title = "Play from here";
      heading.textContent = lines[from];
      heading.onclick = () => compareSeek(anchor.start);
      heading.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); compareSeek(anchor.start); } };
      div.append(heading, "\n" + rest);
    } else div.textContent = rest;
    blocks.push({el: div, anchor});
  };
  add(0, anchors.length ? anchors[0].line : lines.length, null);
  anchors.forEach((a, i) => add(a.line, i + 1 < anchors.length ? anchors[i + 1].line : lines.length, a));
  setCompareBlocks(pane, blocks);
}

// The review list: one row per segment, each anchored to the segment's own times.
function fillCompareTimeline(pane, segments) {
  setCompareBlocks(pane, segments.map(segment => {
    const row = document.createElement("div");
    row.className = "segment";
    const button = document.createElement("button");
    const seconds = Math.floor(segment.start);
    button.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")} ${segment.speaker}`;
    button.onclick = () => compareSeek(segment.start);
    const text = document.createElement("p");
    text.textContent = segment.text;
    row.append(button, text);
    return {el: row, anchor: {start: segment.start, end: segment.end}};
  }));
}

function setCompareBlocks(pane, blocks) {
  pane.blocks = blocks;
  pane.timed = blocks.filter(b => b.anchor);
  pane.active = null;
  pane.text.replaceChildren(...blocks.map(b => b.el));
}

// Show each pane's last result as text or as a review list, keeping the reading position.
function showComparePanes(position = compareCoordinate(comparePane("left"))) {
  const review = $("compareReview").checked;
  for (const side of ["left", "right"]) {
    const pane = comparePane(side), result = pane.result;
    if (!result) continue;
    const asList = review && result.timeline.length > 0;
    if (asList) fillCompareTimeline(pane, result.timeline);
    else fillComparePane(pane, result.content, result.anchors);
    pane.note.textContent = !result.rendered
      ? "Shown as the pipeline wrote it: this version has no diarized JSON, so the Show settings do not apply."
      : review && !asList ? "This version has no timestamped segments, so it is shown as text." : "";
  }
  if (position) scrollPaneTo(comparePane("left"), position);
  syncPanes("left");
  highlightComparePlayback();
}

/* ---------- comparison playback ----------
   One player serves both panes, because both versions describe the same
   recording. While it plays, both panes keep the moment being played in the
   middle and mark the paragraph or segment it falls in. */

let compareFollowSuspended = false;

function compareSeek(time) {
  const audio = $("compareAudio");
  compareFollowSuspended = false;
  comparisonWaveform.seek(time);
  audio.play().catch(() => {
    $("compareAudioMsg").textContent = "Press Play to start audio, or choose a browser-supported audio format.";
  });
}

function highlightComparePlayback() {
  const audio = $("compareAudio");
  const time = audio.currentTime;
  const playing = audio.src && (time > 0 || !audio.paused);
  for (const side of ["left", "right"]) {
    const pane = comparePane(side);
    const active = playing ? (pane.timed || []).find(b => time >= b.anchor.start && time < b.anchor.end) : null;
    if (active === pane.active) continue;
    pane.active?.el.classList.remove("active");
    active?.el.classList.add("active");
    pane.active = active || null;
  }
}

function followComparePlayback() {
  highlightComparePlayback();
  const audio = $("compareAudio");
  if (audio.paused || compareFollowSuspended || !$("compareFollow").checked) return;
  for (const side of ["left", "right"]) scrollPaneTo(comparePane(side), {time: audio.currentTime});
}

function setCompareAudio(base) {
  comparisonWaveform.load(base);
  compareFollowSuspended = false;
  $("compareAudioBar").hidden = !base;
  $("compareAudioMsg").textContent = "";
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
  // Scrolling by hand while audio plays means the reader wants to look elsewhere.
  if (!$("compareAudio").paused && $("compareFollow").checked && !compareFollowSuspended) {
    compareFollowSuspended = true;
    $("compareAudioMsg").textContent = "Following paused because you scrolled. Select a speaker heading or a timestamp, or pause and play again, to resume.";
  }
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
  if (editorScope === "batch") return saveBulkSettings();
  if (!validateForm()) return false;
  if (saving || editorLoading || !editorBase) return false;
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

async function saveBulkSettings() {
  if (saving || editorLoading || !bulkContext || !validateForm()) return false;
  if (!formDirty()) return true;
  const context = bulkContext, folder = folderSeq, changes = formValues();
  let failure = null;
  saving = true; $("editor").inert = true; updateSaveState(); updateRunControls();
  try {
    const revisions = Object.fromEntries(context.bases.map(base => [base, context.files[base].revision ?? null]));
    const {status, data} = await api("/api/sidecars/batch", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({bases:context.bases, changes, revisions}),
    });
    if (status === 422) { fieldErrors(data.errors); throw new Error("Correct the highlighted settings. No files were changed."); }
    if (folder !== folderSeq || context !== bulkContext) return false;
    renderBulkSettings({bases:context.bases, files:data.files});
    await loadFiles();
    return true;
  } catch (e) {failure = e;return false;}
  finally {
    saving = false; $("editor").inert = false;
    updateSaveState(); updateRunControls();
    if (failure) setMsg(failure.message, "err");
    else if (!formDirty()) setMsg(`Settings saved to ${context.bases.length} files.`, "ok");
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
    const query = runQuery();
    const pipeline = query.get("pipeline"), scribe = pipeline === "scribe";
    // Per-run options; the preview token covers them, so the run matches the review.
    for (const base of retranscribeBases) query.append("retranscribe", base);
    for (const base of rediarizeBases) if (!retranscribeBases.has(base)) query.append("rediarize", base);
    const { data } = await api(`/api/run/preview?${query}`);
    pendingPreview = data;
    const label = META.pipelines.find(p => p.id === pipeline).label;
    // Consent is needed only for recordings never uploaded to Scribe before.
    const remote = scribe && data.remote, consent = scribe && data.consent_required;
    $("previewTitle").textContent = remote ? "Scribe remote upload preview"
      : scribe ? "Scribe rerun preview (no upload)" : "Run Preview";
    $("previewConsent").hidden = !consent;
    $("consentRow").hidden = !consent;
    $("scribeConsent").checked = false;
    $("previewStart").textContent = remote ? "Upload to Scribe and run" : scribe ? "Rerun locally" : "Run";
    $("previewFolder").textContent = `${label} · ${data.audio_dir}`
      + (data.only ? ` · ${data.only.length} selected file(s)` : "")
      + (data.diarization_model ? ` · diarization ${data.diarization_model}` : "")
      + (data.reprocess || data.retranscribe === true ? " · reprocess existing outputs" : "")
      + (data.retranscribe?.length ? ` · ${data.retranscribe.length} to transcribe again` : "")
      + (data.rediarize?.length ? ` · ${data.rediarize.length} to diarize again` : "");
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
    $("previewSkippedSummary").textContent = `${skipped.length} already up to date, not part of this run`;
    $("previewSkippedTable").querySelector("tbody").replaceChildren(
      ...skipped.map(row => rowFor([row.file, row.stage, row.reason])));
    $("previewStart").disabled = !data.files.some(row => ["process", "reprocess"].includes(row.action));
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
                   reprocess: Boolean(pendingPreview.reprocess),
                   retranscribe: Array.isArray(pendingPreview.retranscribe) ? pendingPreview.retranscribe : [],
                   rediarize: Array.isArray(pendingPreview.rediarize) ? pendingPreview.rediarize : [] };
    if (pendingPreview.pipeline === "scribe") {
      if (pendingPreview.consent_required && !$("scribeConsent").checked) {
        $("previewSummary").textContent = "Consent to the remote upload first, or cancel."; return;
      }
      if (pendingPreview.remote) body.remote_upload = true;
      body.language = pendingPreview.language;
    }
    if (pendingPreview.only) body.only = pendingPreview.only;
    const { data } = await api("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    lastStatus = data; statusSeq++;
    logOffset = 0; activeRunId = data.run_id;
    $("consoleLog").textContent = "";
    $("previewDialog").close(); pendingPreview = null;
    // The launched run produces fresh results; its redo choices are spent.
    retranscribeBases.clear(); rediarizeBases.clear(); refreshForceOptions();
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
    if (!initialized) {
      await loadMeta(); await loadFiles();
      await restoreSelectedFile();
      initialized = true;
    }
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
    $("connectionStatus").textContent = initialized ? "Disconnected · retrying connection…" : "Could not load WebUI · retrying…";
    $("runDot").className = "dot err";
    $("runStatus").textContent = initialized ? "Run status unavailable" : e.message;
    $("consoleMsg").textContent = initialized ? "Connection lost; transcription may still be running." : "Waiting for the WebUI to load.";
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

const transcriptWaveform = new WaveformPlayer($("transcriptWaveform"), $("audioPlayer"));
const comparisonWaveform = new WaveformPlayer($("comparisonWaveform"), $("compareAudio"));

$("saveBtn").onclick = handle(saveSidecar);
$("deleteBtn").onclick = handle(deleteSidecar);
$("tabSettings").onclick = () => showTab("settings", true);
$("tabTranscript").onclick = () => showTab("transcript", true);
$("tabLogs").onclick = () => showTab("logs", true);
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
  clearTimeout(renderTimer);
  setRenderOptions(currentTab === "compare" ? RENDER_DEFAULTS : renderBaseline);
  saveRenderPreferences();
  if (currentTab === "compare") refreshComparison();
  else if (currentBase) loadTranscript(currentBase, $("versionSel").value);
};
$("tabCompare").onclick = () => showTab("compare", true);
$("compareSelLeft").onchange = () => refreshComparison();
$("compareSelRight").onchange = () => refreshComparison();
$("compareSync").onchange = () => { if ($("compareSync").checked) syncPanes("left"); };
$("compareReview").onchange = () => showComparePanes();
$("compareFollow").onchange = () => { compareFollowSuspended = false; followComparePlayback(); };
$("compareAudio").addEventListener("timeupdate", followComparePlayback);
// Playing again, or moving the playhead, asks for the view to follow once more.
for (const type of ["play", "seeked"]) $("compareAudio").addEventListener(type, () => {
  compareFollowSuspended = false;
  $("compareAudioMsg").textContent = "";
  followComparePlayback();
});
for (const side of ["left", "right"]) comparePane(side).text.addEventListener("scroll", () => onCompareScroll(side));
$("renderSave").onclick = saveRenderedView;
$("renderSpeakers").onclick = handle(openSpeakerNames);
$("speakerSave").onclick = handle(saveSpeakerNames);
$("speakerCancel").onclick = () => { $("speakerDialog").close(); speakerContext = null; };
$("speakerDialog").addEventListener("cancel", () => { speakerContext = null; });
$("startBtn").onclick = startRun;
$("clearRunSelection").onclick = handle(() => changeFileSelection(new Set()));
$("selectShown").onclick = handle(() => {
  const next = new Set(runSelection), deselect = shownBases.every(base => next.has(base));
  for (const base of shownBases) deselect ? next.delete(base) : next.add(base);
  return changeFileSelection(next);
});
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
for (const id of ["pipelineSel", "runModel", "scribeLanguage"]) rememberControl(id);
for (const id of ["fileSearch", "fileFilter", "fileSort", "reviewAudio", "compareSync", "compareReview", "compareFollow", "exportFormat"]) rememberControl(id);
rememberDetails($("previewSkipped"), "previewSkipped");
for (const id of ["audioPlayer", "compareAudio"]) rememberAudio(id);
$("pipelineSel").addEventListener("change", () => { filterSettings(); updateRunControls(); });
for (const id of ["runModel", "scribeLanguage"]) $(id).addEventListener("change", updateRunControls);
for(const id of ["fileSearch","fileFilter","fileSort"]) $(id).addEventListener(id==="fileSearch"?"input":"change",renderFiles);
$("exportConfirm").onclick=handle(confirmExport);
$("exportCancel").onclick=()=>{$("exportDialog").close();exportContext=null;};
$("exportDialog").addEventListener("cancel",e=>{if(exporting)e.preventDefault();else exportContext=null;});
$("reviewAudio").onchange = updateReviewView;
$("audioPlayer").addEventListener("timeupdate",()=>{
  const time=$("audioPlayer").currentTime;
  document.querySelectorAll("#timeline .segment").forEach(row=>row.classList.toggle("active",time>=Number(row.dataset.start)&&time<Number(row.dataset.end)));
});
function setSidebarWidth(width, remember = true){
  width=Math.max(220,Math.min(window.innerWidth*.55,700,width));
  document.body.style.setProperty("--sidebar-width",width+"px");$("sidebarResize").setAttribute("aria-valuenow",Math.round(width));
  if (remember) try{localStorage.setItem("swhisper_sidebar_width",width);}catch{}
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
try{const width=Number(localStorage.getItem("swhisper_sidebar_width"));if(width)setSidebarWidth(width, false);toggleFiles(localStorage.getItem("swhisper_files_collapsed")==="1");}catch{}
initFontSize();
initConsole();
initConsoleResize();
// Restore the view before waiting for metadata or the first status poll.
if (["settings", "transcript", "compare", "logs"].includes(uiPreferences.activeTab)) showTab(uiPreferences.activeTab, false);
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
