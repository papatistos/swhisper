// Exercise the real status polling and controls without starting a browser or models.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';

const source = await fs.readFile(new URL('../webui.js', import.meta.url), 'utf8');
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {
    value: '', textContent: '', disabled: false, hidden: false, dataset: {},
    setAttribute() {},
    matches: () => false,
    querySelectorAll: () => [],
  });
  return elements.get(id);
};
const selector = element('pipelineSel');
selector.options = ['transcribe4', 'transcribe5', 'diarize4', 'scribe'].map(value => ({value}));
selector.value = 'transcribe4';
Object.assign(element('runModel'), {options: [], selectedOptions: []});
let status = {running: true, pipeline: 'transcribe5', run_id: 'existing-run',
              started_at: Date.now() / 1000, progress: {}};
const logRequests = [];
let readLog = async () => ({lines: ['Previous run output'], next: 1});
const context = vm.createContext({
  document: {getElementById: element, querySelectorAll: () => []},
  AbortSignal,
  fetch: async path => {
    if (path === '/api/run/status') return {ok: true, status: 200, json: async () => status};
    logRequests.push(path);
    const data = await readLog();
    return {ok: true, status: 200, json: async () => data};
  },
});
// Skip event binding and automatic startup; call the actual poll entry point below.
vm.runInContext(source.split('/* ---------- init ---------- */')[0], context);
vm.runInContext(`initialized = true;
  scheduleReuseCheck = () => {};
  appendConsole = lines => { $("consoleLog").textContent += lines.join("\\n"); };
`, context);

await vm.runInContext('pollRun()', context);
assert.equal(selector.value, 'transcribe5', 'a reattached run must replace the default selection');
assert.equal(selector.disabled, true);
assert.match(element('runStatus').textContent, /Running transcribe5/);

status = {...status, pipeline: 'scribe', run_id: 'other-tab-run'};
await vm.runInContext('pollRun()', context);
assert.equal(selector.value, 'scribe');
assert.equal(vm.runInContext('transcription().pipeline', context), 'scribe', 'settings follow the active pipeline');

// When idle, preserve the user's next-run choice rather than selecting the last run.
status = {...status, running: false, returncode: 0};
selector.value = 'diarize4';
vm.runInContext('wasRunning = false;', context);
await vm.runInContext('pollRun()', context);
assert.equal(selector.value, 'diarize4');
assert.equal(selector.disabled, false);

// A completed warning from another folder must not leak into this folder.
status = {...status, audio_dir: '/interviews', progress: {
  backfill_incomplete: 1, detail: 'Transcript saved; backfill incomplete',
  current_file: 'interview.wav', completed: {Diarization: 1},
}};
vm.runInContext('META = {audio_dir: "/recordings"};', context);
const beforeHidden = logRequests.length;
await vm.runInContext('pollRun()', context);
assert.equal(element('runStatus').textContent, 'Idle');
assert.equal(element('consoleMsg').textContent, 'Idle');
assert.equal(element('progressStatus').textContent, '');
assert.equal(element('runDot').className, 'dot');
assert.equal(element('consoleLog').textContent, '');
assert.equal(logRequests.length, beforeHidden, 'hidden runs must not fetch old console output');

// Returning to the run's folder shows the real warning and re-reads its log.
vm.runInContext('META.audio_dir = "/interviews";', context);
await vm.runInContext('pollRun()', context);
assert.match(element('runStatus').textContent, /backfill incomplete/);
assert.match(element('consoleMsg').textContent, /backfill incomplete/);
assert.match(element('progressStatus').textContent, /interview.wav/);
assert.equal(element('runDot').className, 'dot err');
assert.equal(logRequests.at(-1), '/api/logs?file=live&offset=0');
assert.match(element('consoleLog').textContent, /Previous run output/);

// Metadata refresh clears all old results immediately, before the next poll.
vm.runInContext('META.audio_dir = "/recordings"; renderRunStatus(lastStatus);', context);
assert.equal(element('runStatus').textContent, 'Idle');
assert.equal(element('progressStatus').textContent, '');
assert.equal(element('consoleLog').textContent, '');

// A running job in another folder must still be visible and stoppable.
status = {...status, running: true, returncode: null};
await vm.runInContext('pollRun()', context);
assert.match(element('runStatus').textContent, /Running scribe/);
assert.match(element('progressStatus').textContent, /\/interviews/);
assert.equal(element('stopBtn').disabled, false);
assert.equal(element('startBtn').disabled, true);
assert.equal(selector.disabled, true);

// A folder switch while a log request is pending discards the stale response.
status = {...status, running: false, returncode: 0};
vm.runInContext('META.audio_dir = "/interviews"; wasRunning = false;', context);
let releaseLog, logStarted;
const pendingLog = new Promise(resolve => { logStarted = resolve; });
readLog = () => new Promise(resolve => { releaseLog = resolve; logStarted(); });
const polling = vm.runInContext('pollRun()', context);
await pendingLog;
vm.runInContext('folderSeq++; META.audio_dir = "/recordings"; renderRunStatus(lastStatus = {...lastStatus, running: false});', context);
releaseLog({lines: ['Stale response'], next: 99});
await polling;
assert.equal(element('runStatus').textContent, 'Idle');
assert.equal(element('consoleLog').textContent, '');
assert.equal(vm.runInContext('logOffset', context), 0);

console.log('Passed: active run controls, directory-scoped results and console, and folder-switch race');
