// Run with: node test_waveform_speakers.cjs
// A minimal DOM tests timing geometry without a browser dependency.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
class Node {
  constructor(tag) {
    this.tag = tag; this.children = []; this.attributes = {}; this.hidden = false;
    this.classList = {add() {}};
    this.style = {setProperty(key, value) {this[key] = value;}};
  }
  append(...nodes) {for (const node of nodes) this.children.push(...(node.tag === 'fragment' ? node.children : [node]));}
  replaceChildren(...nodes) {this.children = []; this.append(...nodes);}
  setAttribute(key, value) {this.attributes[key] = value;}
}
const context = {window: {}, document: {createElement: tag => new Node(tag), createDocumentFragment: () => new Node('fragment')}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '..', 'webui_waveform_speakers.js'), 'utf8'), context);
const host = new Node('div');
const strips = new context.window.WaveformSpeakers(host);
assert.equal(host.hidden, true);
strips.setViewport(10, 20);
strips.setTracks([
  {label: 'Left', segments: [
    {start: 0, end: 12, speaker: 'A'}, {start: 12, end: 15, speaker: 'A'},
    {start: 11, end: 19, speaker: 'B'}, {start: 18, end: 25, speaker: 'A'},
    {start: 14, end: 14, speaker: 'invalid'}, {start: null, end: 18, speaker: 'invalid'},
    {start: 1, end: Infinity, speaker: 'invalid'}, {start: 'no', end: 18, speaker: 'invalid'},
    {start: 20, end: 22, speaker: 'C'}]},
  {label: 'Right', segments: [{start: 12, end: 16, speaker: 'A'}]}
]);
assert.equal(host.hidden, false);
assert.equal(strips.tracks.length, 2);
assert.equal(strips.tracks[0].lanes.length, 3);
const [a,b,c] = strips.tracks[0].lanes;
assert.equal(a.intervals.length, 2);
assert.equal(a.node.children.length, 2);
assert.equal(a.node.children[0].style.left, '0%');
assert.equal(a.node.children[0].style.width, '50%');
assert.equal(a.node.children[1].style.left, '80%');
assert.equal(a.node.children[1].style.width, '20%');
assert.equal(b.node.children[0].style.left, '10%');
assert.equal(b.node.children[0].style.width, '80%');
assert.equal(c.node.children.length, 0);
assert.notEqual(a.node, b.node);
assert.equal(strips.tracks[1].lanes[0].node.children[0].style.left, '20%');
assert.equal(a.node.style['--speaker-color'], strips.tracks[1].lanes[0].node.style['--speaker-color']);
assert.match(a.node.children[0].attributes['aria-label'], /Left, A: 0:00.0 to 0:15.0/);
strips.setViewport(21, 23);
assert.equal(a.node.children.length, 1);
assert.equal(a.node.children[0].style.width, '100%');
assert.equal(b.node.children.length, 0);
strips.setViewport(5, 5);
assert.equal(host.hidden, true);
assert.equal(a.node.children.length, 0);
strips.setViewport(0, 30);
strips.setTracks([{segments: [{start: 1, end: 2}]}]);
assert.equal(strips.tracks[0].lanes[0].name, 'Unknown speaker');
// Silence between turns stays blank, including when both turns have one speaker.
strips.setViewport(0, 10);
strips.setTracks([
  {label: 'Left', segments: [
    {start: 0, end: 2, speaker: 'A'},
    {start: 2, end: 5, speaker: 'SILENCE'},
    {start: 5, end: 8, speaker: 'A'},
    {start: 8, end: 10, is_silence_marker: true},
  ]},
  {label: 'Right', segments: [{start: 0, end: 10, speaker: ' silence '}]},
]);
assert.equal(strips.tracks.length, 1, 'A silence-only comparison track is omitted');
assert.equal(strips.tracks[0].lanes.length, 1, 'Silence creates no speaker or legend entry');
assert.equal(strips.tracks[0].lanes[0].name, 'A');
assert.deepEqual(strips.tracks[0].lanes[0].node.children.map(line => [line.style.left, line.style.width]),
  [['0%', '20%'], ['50%', '30%']], 'The pause remains a gap between the two lines');
strips.setTracks([{segments: [{start: 0, end: 10, speaker: 'SILENCE'}]}]);
assert.equal(host.hidden, true);
assert.equal(host.children.length, 0);
strips.setTracks([{segments: [{start: null, end: 3}]}]);
assert.equal(host.hidden, true);
assert.equal(host.children.length, 0);
strips.clear();
assert.equal(host.children.length, 0);
assert.equal(strips.viewport, null);
console.log('Speaker-strip geometry and silence checks passed');
