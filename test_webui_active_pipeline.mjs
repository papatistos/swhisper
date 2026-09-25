// Exercise the real status polling and controls without starting a browser or models.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';

const source = await fs.readFile(new URL('./webui.js', import.meta.url), 'utf8');
const elements = new Map();
const element = id => {
  if (!elements.has(id)) elements.set(id, {
    value: '', textContent: '', disabled: false, hidden: false,
    setAttribute() {},
    matches: () => false,
  });
  return elements.get(id);
};
const selector = element('pipelineSel');
selector.options = ['transcribe4', 'transcribe5', 'diarize4', 'scribe'].map(value => ({value}));
selector.value = 'transcribe4';
Object.assign(element('runModel'), {options: [], selectedOptions: []});
let status = {running: true, pipeline: 'transcribe5', run_id: 'existing-run',
              started_at: Date.now() / 1000, progress: {}};
const context = vm.createContext({
  document: {getElementById: element, querySelectorAll: () => []},
  AbortSignal,
  fetch: async path => ({ok: true, status: 200, json: async () =>
    path === '/api/run/status' ? status : {lines: [], next: 0}}),
});
// Skip event binding and automatic startup; call the actual poll entry point below.
vm.runInContext(source.split('/* ---------- init ---------- */')[0], context);
vm.runInContext('initialized = true;', context);

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

console.log('Passed: active run selection, attached run settings, and idle selection preservation');
