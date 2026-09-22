// Run against an isolated headless Chrome started with --remote-debugging-port=9334.
// Node 22+ only; no dependencies. All API requests use in-memory fixtures.
import http from 'node:http';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {setTimeout as delay} from 'node:timers/promises';

const html = await fs.readFile(new URL('./webui.html', import.meta.url), 'utf8');
const css = await fs.readFile(new URL('./webui.css', import.meta.url), 'utf8');
const js = await fs.readFile(new URL('./webui.js', import.meta.url), 'utf8');
let audioDir = '/mock/audio', serverFolder = null, offline = false, failSave = false, sidecarDelay = {}, versionDelay = {}, renderDelay = {}, logDelay = {};
let status = {running:false, pipeline:null, returncode:null, run_id:null, progress:{completed:{}}};
const saved = {a:{values:{SWHISPER_NUM_SPEAKERS:'2',SWHISPER_CONDITION_ON_PREVIOUS:'yes'},comment:'original'}, b:{values:{},comment:'B'}};
const writes = [], launches = [], errors = [], exports = [];
let globalValues={}, globalRevision="g1", exportCanceled=false;
let speakerNames={};
const meta = () => ({audio_dir:audioDir,audio_paths:['/mock/audio','/mock/other'],
  pipelines:[{id:'transcribe4',label:'Standard pipeline'},{id:'diarize4',label:'Diarization only'},{id:'scribe',label:'SUNET Scribe + local diarization'}],groups:[{name:'Speakers',keys:['SWHISPER_NUM_SPEAKERS','SWHISPER_CONDITION_ON_PREVIOUS','SWHISPER_VAD']}],
  params:{SWHISPER_VAD:{type:'bool',global:true,stage:'transcribe',scribe:{supported:false,scope:'Unavailable for Scribe',description:'Not applied to Scribe.'}},SWHISPER_NUM_SPEAKERS:{type:'int',min:1,label:'Number of speakers',global:2,stage:'diarize',help:'Speaker count',scribe:{supported:true,description:'Local speaker count'}},SWHISPER_CONDITION_ON_PREVIOUS:{type:'bool',global:true,stage:'transcribe',help:'Use previous text',scribe:{supported:true,scope:'Local backfill only',description:'Local Whisper backfill only'}}}});
const server = http.createServer(async (req,res) => {
  const u = new URL(req.url, 'http://localhost');
  let body = ''; for await (const chunk of req) body += chunk;
  const data = body ? JSON.parse(body) : {};
  const send = (value, code=200) => {res.writeHead(code, {'Content-Type':'application/json'});res.end(JSON.stringify(value));};
  if (u.pathname === '/') {res.writeHead(200, {'Content-Type':'text/html','Cache-Control':'no-store'});res.end(html);return;}
  if (u.pathname === '/webui.css') {res.writeHead(200, {'Content-Type':'text/css','Cache-Control':'no-store'});res.end(css);return;}
  if (u.pathname === '/webui.js') {res.writeHead(200, {'Content-Type':'text/javascript','Cache-Control':'no-store'});res.end(js);return;}
  if (u.pathname === '/api/meta') return send(meta());
  if (u.pathname === '/api/files') return send({files:['a','b','<unsafe>'].map(base=>({base,transcribed:base!=='<unsafe>',diarized:base!=='<unsafe>',has_transcript:base!=='<unsafe>',has_logs:true,will_reprocess:base==='b',mtime:base==='b'?20:10,duration:base==='a'?3202:base==='b'?95:null}))});
  if (u.pathname === '/api/run/status') return offline ? send({detail:'Offline'},503) : send({...status, folder:serverFolder??audioDir});
  if (u.pathname === '/api/logs') {
    const file = u.searchParams.get('file');
    await delay(logDelay[file] || 0);
    if (file) return send({lines:[],next:0,content:`Log ${file}`,selected:file});
    if (u.searchParams.get('base')) return send({base:u.searchParams.get('base'),versions:[{id:'logs/a.log',filename:'a.log'}]});
    return send({versions:[{id:'live',filename:'Live console'},{id:'first',filename:'First log'},{id:'second',filename:'Second log'}]});
  }
  if (u.pathname === '/api/settings/global') {
    if(req.method==='PUT') {if(data.revision!==globalRevision)return send({detail:'stale'},409);globalValues=data.values;globalRevision+='x';}
    const gm=meta();gm.params.SWHISPER_MIN_SPEAKERS=gm.params.SWHISPER_NUM_SPEAKERS;delete gm.params.SWHISPER_NUM_SPEAKERS;gm.groups=gm.groups.map(g=>({...g,keys:g.keys.map(k=>k==='SWHISPER_NUM_SPEAKERS'?'SWHISPER_MIN_SPEAKERS':k)}));return send({...gm,values:globalValues,revision:globalRevision,path:'/mock/.swhisper.env'});
  }
  if (u.pathname.startsWith('/api/timeline/')) return send({audio:'a.wav',segments:[{start:1,end:3,speaker:'S01',text:'Hello'},{start:3,end:6,speaker:'S02',text:'World'}]});
  if (u.pathname.startsWith('/api/audio/')) {
    const pcm=Buffer.alloc(44+160000);pcm.write('RIFF');pcm.writeUInt32LE(pcm.length-8,4);pcm.write('WAVEfmt ',8);pcm.writeUInt32LE(16,16);pcm.writeUInt16LE(1,20);pcm.writeUInt16LE(1,22);pcm.writeUInt32LE(8000,24);pcm.writeUInt32LE(16000,28);pcm.writeUInt16LE(2,32);pcm.writeUInt16LE(16,34);pcm.write('data',36);pcm.writeUInt32LE(160000,40);
    const range=/bytes=(\d+)-(\d*)/.exec(req.headers.range||'');const start=range?Number(range[1]):0,end=range&&range[2]?Number(range[2]):pcm.length-1;res.writeHead(range?206:200,{'Content-Type':'audio/wav','Accept-Ranges':'bytes','Content-Length':end-start+1,...(range?{'Content-Range':`bytes ${start}-${end}/${pcm.length}`}:{})});res.end(pcm.subarray(start,end+1));return;
  }
  if (u.pathname.endsWith('/save-as')) return send(exportCanceled ? {canceled:true} : {token:'native-save-selection',path:'/mock/chosen/review.txt',format:data.format});
  if (u.pathname.endsWith('/save-selected')) {exports.push(data);return send({written:['/mock/chosen/review.txt']});}
  if (u.pathname.startsWith('/api/render/') && u.pathname.endsWith('/save')) {exports.push(data);return send({written:['/mock/audio/exports/review.txt']});}
  if (u.pathname.startsWith('/api/render/')) {
    await delay(renderDelay[data.options?.min_pause] || 0);
    return send({content:`Rendered ${data.options.min_pause}`,suffix:'_view',timeline:[{start:1,end:3,speaker:'S01',text:`Filtered ${data.options.min_pause}`}]});
  }
  if (u.pathname === '/api/history') return send({entries:[]});
  if (u.pathname === '/api/audio-dir') {audioDir=data.path;return send({audio_dir:audioDir,audio_paths:['/mock/audio','/mock/other']});}
  if (u.pathname === '/api/audio-dir/pick') return send({path:'/mock/picked'});
  if (u.pathname === '/api/run/preview') {const only=u.searchParams.getAll('only');return send({audio_dir:audioDir,pipeline:'transcribe4',token:'preview-1',only:only.length?only:null,notes:['Folder scope'],files:[{file:'a.wav',stage:'Transcription',action:'process',reason:'No output'},{file:'b.wav',stage:'Transcription',action:'skip',reason:'Existing output'},{file:'c.wav',stage:'Transcription',action:'skip',reason:'Existing output'}]});}
  if (u.pathname === '/api/run' && req.method === 'POST') {launches.push(data);status={running:true,pipeline:'transcribe4',run_id:'run1',started_at:Date.now()/1000,audio_dir:audioDir,progress:{stage:'Transcription',current_file:'a.wav',file_index:1,total_files:2,completed:{Transcription:0}}};return send(status);}
  if (u.pathname === '/api/run/stop') {status={...status,running:false,stop_requested:true,returncode:0};return send(status);}
  if (u.pathname.startsWith('/api/sidecar/')) {
    const base=decodeURIComponent(u.pathname.split('/').at(-1));
    if (req.method==='PUT') {writes.push({base,data});if(failSave)return send({errors:{SWHISPER_NUM_SPEAKERS:'invalid'}},422);saved[base]={values:Object.fromEntries(Object.entries(data).filter(([k])=>k!=='_comment')),comment:data._comment};}
    if(req.method==='DELETE') saved[base]={values:{},comment:''};
    await delay(sidecarDelay[base] || 0);
    return send({exists:true,unknown:{FUTURE:'preserve'},...(saved[base]||{values:{},comment:''})});
  }
  if (u.pathname.startsWith('/api/speakers/')) {
    if (req.method==='PUT') {if(data.names.S01==='bad')return send({detail:'Two speakers cannot share that name'},422);speakerNames=Object.fromEntries(Object.entries(data.names).filter(([,v])=>v));}
    return send({base:'a',detected:['S01','S02'],names:speakerNames});
  }
  if (u.pathname.startsWith('/api/json-transcript/')) return send({base:'a',path:'whisper-json-output/a.json',content:'{"raw": true}'});
  if (u.pathname.startsWith('/api/transcript/')) {
    const version=u.searchParams.get('version')||'v2';
    await delay(versionDelay[version]||0);
    return send({versions:[{id:'v2',filename:'new'},{id:'v1',filename:'old'}],selected:version,content:`Transcript ${version}`,can_render:true,speaker_names:speakerNames});
  }
  send({detail:'Unknown fixture route'},404);
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const port = process.env.CHROME_DEBUG_PORT || '9334';
let socket, target;
try {
  target = await (await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method:'PUT'})).json();
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  let sequence=0;const calls=new Map();
  socket.onmessage=event=>{const msg=JSON.parse(event.data);if(msg.id){const c=calls.get(msg.id);calls.delete(msg.id);msg.error?c.reject(msg.error):c.resolve(msg.result);}else if(msg.method==='Runtime.exceptionThrown')errors.push(msg.params.exceptionDetails);};
  const cdp=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;calls.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const r=await cdp('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
  const wait=async(expression)=>{for(let i=0;i<400;i++){if(await evaluate(expression))return;await delay(20);}throw Error('Timed out: '+expression);};
  const click=async id=>evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
  // el.click() dispatches a bare click event. Popover light dismiss runs on
  // mousedown, so only a real pointer sequence exercises the arrow's toggle.
  const mouseClick=async id=>{
    const at=await evaluate(`(()=>{const r=document.getElementById(${JSON.stringify(id)}).getBoundingClientRect();return{x:(r.left+r.right)/2,y:(r.top+r.bottom)/2}})()`);
    for(const type of ['mousePressed','mouseReleased'])
      await cdp('Input.dispatchMouseEvent',{type,x:at.x,y:at.y,button:'left',buttons:type==='mousePressed'?1:0,clickCount:1});
  };
  const select=async base=>evaluate(`document.querySelector('[data-base="${base}"] .fileSelect').click()`);
  const input=async value=>evaluate(`document.getElementById('commentField').value=${JSON.stringify(value)};document.getElementById('commentField').dispatchEvent(new Event('input',{bubbles:true}))`);
  const key=async(key,code)=>{await cdp('Input.dispatchKeyEvent',{type:'keyDown',key,code,windowsVirtualKeyCode:({Escape:27,Enter:13,ArrowLeft:37,ArrowRight:39,ArrowUp:38,ArrowDown:40})[key]});await cdp('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:({Escape:27,Enter:13,ArrowLeft:37,ArrowRight:39,ArrowUp:38,ArrowDown:40})[key]});};
  const check=async(name,fn)=>{await fn();console.log('PASS '+name);};
  await cdp('Runtime.enable');await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:`http://127.0.0.1:${server.address().port}/`});
  await wait('typeof initialized !== "undefined" && initialized && connected');
  await check('Directory controls share aligned edges at different font sizes',async()=>{
    for (const fontSize of [16,22,28]) {
      await evaluate(`document.documentElement.style.fontSize='${fontSize}px'`);
      const bounds = await evaluate(`['dirInput','dirMenuBtn'].map(id=>{const r=$(id).getBoundingClientRect();return {top:r.top,bottom:r.bottom,left:r.left,right:r.right}})`);
      const typography = await evaluate(`['dirInput','dirMenuBtn'].map(id=>{const s=getComputedStyle($(id));return {fontSize:s.fontSize,lineHeight:s.lineHeight}})`);
      assert.equal(typography[0].fontSize,typography[1].fontSize);
      assert.equal(typography[0].lineHeight,typography[1].lineHeight);
      assert.equal(bounds[0].top,bounds[1].top);
      assert.equal(bounds[0].bottom,bounds[1].bottom);
      assert.equal(bounds[0].right-1,bounds[1].left,'Arrow sits on the right edge of the editable field');
      assert.deepEqual(await evaluate(`(()=>{const input=$('dirInput').getBoundingClientRect(), arrow=$('dirMenuBtn').getBoundingClientRect(); return [document.elementFromPoint(input.left+10,input.top+10).id,document.elementFromPoint((arrow.left+arrow.right)/2,(arrow.top+arrow.bottom)/2).id]})()`),['dirInput','dirMenuBtn']);
    }
    await evaluate("document.documentElement.style.fontSize='22px'");
  });
  // The opened menu is what users see misplaced, so it is measured while open:
  // a native select would hand that menu to the platform and leave nothing here
  // to measure.
  await check('Opened directory menu is anchored to the path field at every font size',async()=>{
    for (const fontSize of [16,22,28]) {
      await evaluate(`document.documentElement.style.fontSize='${fontSize}px'`);
      await click('dirMenuBtn');
      await wait('$("dirMenu").matches(":popover-open")');
      assert.equal(await evaluate('$("dirMenuBtn").getAttribute("aria-expanded")'),'true');
      const geometry = await evaluate(`(()=>{const input=$('dirInput').getBoundingClientRect(), arrow=$('dirMenuBtn').getBoundingClientRect(), menu=$('dirMenu').getBoundingClientRect(), item=$('dirMenu').querySelector('button').getBoundingClientRect(), hit=document.elementFromPoint((item.left+item.right)/2,(item.top+item.bottom)/2); return {left:menu.left-input.left,top:menu.top-input.bottom,right:menu.right-arrow.right,width:menu.width,covered:$('dirMenu').contains(hit),inViewport:menu.bottom<=innerHeight}})()`);
      assert.equal(geometry.left,0,'Menu starts at the left edge of the path field');
      assert.equal(geometry.top,4,'Menu drops directly below the path field');
      assert.equal(geometry.right,0,'Menu ends at the right edge of the arrow');
      assert.ok(geometry.width>0,'Menu has a measurable box of its own');
      assert.equal(geometry.covered,true,'Nothing paints over the opened menu');
      assert.equal(geometry.inViewport,true,'Menu fits below the field');
      await key('Escape','Escape');
      await wait('!$("dirMenu").matches(":popover-open")');
      assert.equal(await evaluate('$("dirMenuBtn").getAttribute("aria-expanded")'),'false');
    }
    await evaluate("document.documentElement.style.fontSize='22px'");
  });
  // The arrow is a toggle, and a real pointer is the only way to prove it:
  // light dismiss closes the popover on mousedown, so a handler that reads the
  // open state during the click sees a menu the browser has already closed.
  await check('Arrow closes the menu it opened under a real pointer',async()=>{
    await mouseClick('dirMenuBtn');
    await wait('$("dirMenu").matches(":popover-open")');
    await mouseClick('dirMenuBtn');
    await wait('!$("dirMenu").matches(":popover-open")');
    assert.equal(await evaluate('$("dirMenuBtn").getAttribute("aria-expanded")'),'false');
    await mouseClick('dirMenuBtn');
    await wait('$("dirMenu").matches(":popover-open")');
    // Clicking a menu item must survive the same pointer sequence.
    const item=await evaluate(`(()=>{const b=[...$('dirMenu').querySelectorAll('button')].find(b=>b.title==='/mock/other');const r=b.getBoundingClientRect();return{x:(r.left+r.right)/2,y:(r.top+r.bottom)/2}})()`);
    for(const type of ['mousePressed','mouseReleased'])
      await cdp('Input.dispatchMouseEvent',{type,x:item.x,y:item.y,button:'left',buttons:type==='mousePressed'?1:0,clickCount:1});
    await wait('META.audio_dir==="/mock/other"');
    assert.equal(await evaluate('$("dirMenu").matches(":popover-open")'),false,'Choosing a folder closes the menu');
    await evaluate('$("dirInput").value="/mock/audio";$("dirInput").dispatchEvent(new Event("input"))');
    await wait('META.audio_dir==="/mock/audio"');
  });
  if (process.env.DIRECTORY_ALIGNMENT_ONLY) { console.log('PASS focused directory alignment'); process.exit(0); }
  await check('Directory menu lists saved folders and keeps keyboard access',async()=>{
    assert.deepEqual(await evaluate('[...$("dirMenu").querySelectorAll("button")].map(item=>item.title)'),['/mock/audio','/mock/other']);
    assert.deepEqual(await evaluate('[...$("dirMenu").querySelectorAll(".dirMenuName")].map(name=>name.textContent)'),['audio','other']);
    assert.equal(await evaluate('$("dirMenu").querySelector("[aria-current]").title'),'/mock/audio');
    await evaluate('$("dirInput").focus();$("dirInput").dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowDown",bubbles:true,cancelable:true}))');
    await wait('$("dirMenu").matches(":popover-open")');
    assert.equal(await evaluate('document.activeElement.title'),'/mock/audio');
    await key('ArrowDown','ArrowDown');
    assert.equal(await evaluate('document.activeElement.title'),'/mock/other');
    await key('Escape','Escape');
    await wait('!$("dirMenu").matches(":popover-open")');
    assert.equal(await evaluate('document.activeElement.id'),'dirInput','Escape returns focus to the field the menu was opened from');
    await evaluate('$("dirMenuBtn").focus();$("dirMenuBtn").click()');
    await wait('$("dirMenu").matches(":popover-open")');
    await key('End','End');
    assert.equal(await evaluate('document.activeElement.title'),'/mock/other');
    await key('Home','Home');
    assert.equal(await evaluate('document.activeElement.title'),'/mock/audio');
    await key('Escape','Escape');
    await wait('!$("dirMenu").matches(":popover-open")');
    assert.equal(await evaluate('document.activeElement.id'),'dirMenuBtn','Escape returns focus to the arrow');
  });
  await check('Directory menu selection switches folders and manual editing still works',async()=>{
    await click('dirMenuBtn');
    await wait('$("dirMenu").matches(":popover-open")');
    await evaluate('[...$("dirMenu").querySelectorAll("button")].find(item=>item.title==="/mock/other").click()');
    await wait('META.audio_dir==="/mock/other"');
    assert.equal(await evaluate('$("dirMenu").matches(":popover-open")'),false);
    assert.equal(await evaluate('$("dirInput").value'),'/mock/other');
    await evaluate('$("dirInput").value="/mock/audio";$("dirInput").dispatchEvent(new Event("input"))');
    await wait('META.audio_dir==="/mock/audio"');
  });
  await check('Save disabled, enabled by edit, disabled on revert',async()=>{await select('a');await wait('editorBase==="a"');assert.equal(await evaluate('$("saveBtn").disabled'),true);await input('changed');assert.equal(await evaluate('$("saveBtn").disabled'),false);await input('original');assert.equal(await evaluate('$("saveBtn").disabled'),true);});
  await check('Boolean alias display and discard preserve inherited semantics',async()=>{
    assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").value'),'true');
    await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").checked=false;$("setting-SWHISPER_CONDITION_ON_PREVIOUS").dispatchEvent(new Event("input"))');
    await select('b');await wait('$("dirtyDialog").open');await click('dirtyDiscard');await wait('editorBase==="b"');
    await select('a');await wait('editorBase==="a"');assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").value'),'true');
  });
  await check('Cancel navigation preserves draft',async()=>{await input('draft');await select('b');await wait('$("dirtyDialog").open');await click('dirtyCancel');await wait('!$("dirtyDialog").open');assert.equal(await evaluate('editorBase'), 'a');assert.equal(await evaluate('$("commentField").value'),'draft');});
  await check('Save and continue uses original target',async()=>{await select('b');await wait('$("dirtyDialog").open');await click('dirtySave');await wait('editorBase==="b"');assert.equal(writes.at(-1).base,'a');assert.equal(saved.a.comment,'draft');assert.equal(writes.at(-1).data.SWHISPER_CONDITION_ON_PREVIOUS,'yes');});
  await check('Discard navigation does not save',async()=>{const n=writes.length;await input('discard');await select('a');await wait('$("dirtyDialog").open');await click('dirtyDiscard');await wait('editorBase==="a"');assert.equal(writes.length,n);});
  await check('Failed save keeps draft and blocks navigation',async()=>{failSave=true;await input('invalid');await select('b');await wait('$("dirtyDialog").open');await click('dirtySave');await wait('!saving && $("editorMsg").classList.contains("err")');assert.equal(await evaluate('editorBase'),'a');assert.equal(await evaluate('$("saveBtn").disabled'),false);failSave=false;await click('saveBtn');await wait('!formDirty()');});
  await check('Out-of-order settings response cannot replace new selection',async()=>{sidecarDelay.b=250;await select('b');await wait('editorLoading');await select('a');await wait('editorBase==="a"');await delay(300);assert.equal(await evaluate('editorBase'),'a');assert.equal(await evaluate('$("commentField").value'),saved.a.comment);sidecarDelay.b=0;});
  await check('Out-of-order transcript version response ignored',async()=>{await click('tabTranscript');await wait('$("transcriptText").textContent==="Transcript v2"');versionDelay.v1=250;await evaluate('loadTranscript(currentBase,"v1"); void 0');await evaluate('loadTranscript(currentBase,"v2"); void 0');await delay(300);assert.equal(await evaluate('$("transcriptText").textContent'),'Transcript v2');versionDelay.v1=0;});
  await check('Run logs stay reachable while a file is selected',async()=>{
    await select('a');await wait('editorBase==="a"');await click('tabLogs');
    await wait('$("logSel").options.length===4');
    assert.deepEqual(await evaluate('[...$("logSel").querySelectorAll("optgroup")].map(g=>g.label)'),['Runs','Logs for a']);
    assert.deepEqual(await evaluate('[...$("logSel").options].map(o=>o.value)'),['live','first','second','logs/a.log']);
    await evaluate('$("logSel").value="second";$("logSel").dispatchEvent(new Event("change"))');
    await wait('$("log").textContent==="Log second"');
    await evaluate('$("logSel").value="logs/a.log";$("logSel").dispatchEvent(new Event("change"))');
    await wait('$("log").textContent==="Log logs/a.log"');
    await click('tabTranscript');
  });
  await check('Speakers can be named without touching the stored transcript',async()=>{
    await select('a');await wait('editorBase==="a"');await click('tabTranscript');
    await wait('$("transcriptText").textContent==="Transcript v2" && renderAvailable');
    await click('renderSpeakers');await wait('$("speakerDialog").open');
    assert.deepEqual(await evaluate('[...$("speakerRows").querySelectorAll("input")].map(i=>i.dataset.label)'),['S01','S02']);
    await evaluate('$("speaker-S01").value="bad"');await click('speakerSave');
    await wait('$("speakerMsg").textContent.includes("share")');
    assert.equal(await evaluate('$("speakerDialog").open'),true);   // stays open on error
    await evaluate('$("speaker-S01").value="Interviewer";$("speaker-S02").value="Christoph"');
    await click('speakerSave');await wait('!$("speakerDialog").open');
    assert.deepEqual(speakerNames,{S01:'Interviewer',S02:'Christoph'});
    await wait('$("transcriptText").textContent.startsWith("Rendered")');
    speakerNames={};
  });
  await check('Run preview separates what runs from what is skipped',async()=>{
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('$("previewTable").querySelectorAll("tbody tr").length'),1);
    assert.equal(await evaluate('$("previewSkipped").hidden'),false);
    assert.equal(await evaluate('$("previewSkipped").open'),false);
    assert.match(await evaluate('$("previewSkippedSummary").textContent'),/2 already up to date/);
    assert.equal(await evaluate('$("previewSkippedTable").querySelectorAll("tbody tr").length'),2);
    await click('previewCancel');await wait('!$("previewDialog").open');
  });
  await check('Recording length is shown for files that have one',async()=>{
    assert.equal(await evaluate('[...document.querySelectorAll(\'[data-base="a"] .badge\')].map(b=>b.textContent).includes("53 min")'),true);
    assert.equal(await evaluate('[...document.querySelectorAll(\'[data-base="b"] .badge\')].map(b=>b.textContent).includes("1 min")'),true);
    assert.equal(await evaluate('[...document.querySelectorAll(\'[data-base="<unsafe>"] .badge\')].some(b=>/min|h /.test(b.textContent))'),false);
  });
  await check('Selected files narrow the run and reach the launch call',async()=>{
    assert.equal(await evaluate('$("startBtn").textContent'),'Run folder…');
    assert.equal(await evaluate('$("clearRunSelection").hidden'),true);
    await evaluate('document.querySelector(\'[data-base="b"] .runPick input\').click()');
    await wait('$("startBtn").textContent==="Run 1 selected…"');
    assert.equal(await evaluate('$("clearRunSelection").hidden'),false);
    const before=launches.length;
    await click('startBtn');await wait('$("previewDialog").open');
    assert.match(await evaluate('$("previewFolder").textContent'),/1 selected file/);
    await click('previewStart');
    for(let i=0;i<200 && launches.length===before;i++) await delay(20);
    assert.deepEqual(launches.at(-1).only,['b']);
    await wait('lastStatus && lastStatus.running');await click('stopBtn');await wait('lastStatus && !lastStatus.running');
    await click('clearRunSelection');await wait('$("startBtn").textContent==="Run folder…"');
    assert.equal(await evaluate('document.querySelector(\'[data-base="b"] .runPick input\').checked'),false);
  });
  await check('Folder moved by another window is reported, not applied silently',async()=>{
    await select('a');await wait('editorBase==="a"');await input('keep me');
    serverFolder='/mock/elsewhere';
    await wait('!$("folderNotice").hidden');
    assert.match(await evaluate('$("folderNoticeText").textContent'),/\/mock\/elsewhere/);
    assert.equal(await evaluate('$("commentField").value'),'keep me');   // edits survive
    assert.equal(await evaluate('META.audio_dir'),'/mock/audio');
    await click('folderNoticeReload');await wait('$("dirtyDialog").open');await click('dirtyDiscard');
    await wait('META.audio_dir==="/mock/elsewhere"');
    await wait('$("folderNotice").hidden');
    serverFolder=null;audioDir='/mock/audio';
    await evaluate('$("dirInput").value="/mock/audio";changeDir()');await wait('META.audio_dir==="/mock/audio"');
  });
  await check('Raw JSON view keeps every version selectable',async()=>{
    await select('a');await wait('editorBase==="a"');await click('tabTranscript');
    await wait('$("transcriptText").textContent==="Transcript v2"');
    assert.deepEqual(await evaluate('[...$("versionSel").options].map(o=>o.value)'),['v2','v1','__raw_json__']);
    await evaluate('$("versionSel").value="__raw_json__";$("versionSel").dispatchEvent(new Event("change"))');
    await wait('$("transcriptText").textContent.includes("raw")');
    assert.deepEqual(await evaluate('[...$("versionSel").options].map(o=>o.value)'),['v2','v1','__raw_json__']);
    assert.equal(await evaluate('$("versionSel").value'),'__raw_json__');
    await evaluate('$("versionSel").value="v1";$("versionSel").dispatchEvent(new Event("change"))');
    await wait('$("transcriptText").textContent==="Transcript v1"');
  });
  await check('Out-of-order render response ignored',async()=>{
    renderDelay[1]=250;
    await evaluate('$("optMinPause").value="1"; applyRenderOptions(); void 0');
    await evaluate('$("optMinPause").value="2"; applyRenderOptions(); void 0');
    await wait('$("transcriptText").textContent==="Rendered 2"');
    await delay(300);assert.equal(await evaluate('$("transcriptText").textContent'),'Rendered 2');assert.equal(await evaluate('$("timeline").querySelector("p").textContent'),'Filtered 2');renderDelay={};
  });
  await check('Out-of-order log response ignored',async()=>{
    await click('tabLogs');await wait('$("logSel").options.length>=3');logDelay.first=250;
    await evaluate('currentLogSel="first"; loadLogContent(); void 0');
    await evaluate('currentLogSel="second"; loadLogContent(); void 0');
    await wait('$("log").textContent==="Log second"');await delay(300);
    assert.equal(await evaluate('$("log").textContent'),'Log second');logDelay={};await click('tabTranscript');
  });
  await check('Filename text is not interpreted as HTML',async()=>{assert.equal(await evaluate('document.querySelector("#files unsafe")'),null);assert.equal(await evaluate(`document.querySelector('[data-base="<unsafe>"] .fileSelect').textContent`),'<unsafe>');});
  await check('Settings remain hidden after load on transcript tab',async()=>{await select('b');await wait('editorBase==="b"');assert.equal(await evaluate('$("editorBar").style.display'),'none');});
  await check('Keyboard tabs and field labels',async()=>{await evaluate('$("tabTranscript").focus()');await key('ArrowLeft','ArrowLeft');assert.equal(await evaluate('currentTab'),'settings');assert.equal(await evaluate('document.querySelector("label[for=setting-SWHISPER_NUM_SPEAKERS]").control.id'),'setting-SWHISPER_NUM_SPEAKERS');});
  await check('Keyboard file navigation',async()=>{await evaluate('document.querySelector("[data-base=a] .fileSelect").focus()');await key('ArrowDown','ArrowDown');assert.equal(await evaluate('document.activeElement.closest(".file").dataset.base'),'b');await key('Enter','Enter');await wait('editorBase==="b"');});
  await check('History modal Escape restores focus',async()=>{await evaluate('$("historyBtn").focus()');await click('historyBtn');await wait('$("historyPanel").open');await key('Escape','Escape');await wait('!$("historyPanel").open');assert.equal(await evaluate('document.activeElement.id'),'historyBtn');});
  await check('Keyboard console resize',async()=>{const before=await evaluate('Number($("consoleResize").getAttribute("aria-valuenow"))');await evaluate('$("consoleResize").focus()');await key('ArrowUp','ArrowUp');assert.equal(await evaluate('Number($("consoleResize").getAttribute("aria-valuenow"))'),before+20);});
  await check('Search and status filters preserve current editor',async()=>{
    await evaluate('$("fileSearch").value="unsafe";$("fileSearch").dispatchEvent(new Event("input"))');
    assert.equal(await evaluate('document.querySelectorAll(".fileSelect").length'),1);assert.equal(await evaluate('editorBase'),'b');
    await evaluate('$("fileSearch").value="";$("fileFilter").value="reprocess";renderFiles()');
    assert.equal(await evaluate('document.querySelector(".file").dataset.base'),'b');
    await evaluate('$("fileFilter").value="all";$("fileSort").value="newest";renderFiles()');
    assert.equal(await evaluate('document.querySelector(".file").dataset.base'),'b');await evaluate('$("fileSort").value="name";renderFiles()');
  });
  await check('Inherited checkbox and Use global reset',async()=>{
    const id='setting-SWHISPER_CONDITION_ON_PREVIOUS';
    assert.equal(await evaluate(`$("${id}").type`),'checkbox');assert.equal(await evaluate(`$("${id}").checked`),true);
    assert.equal(await evaluate(`$("${id}").closest(".fieldValue").querySelector(".inheritance").textContent`),'inherited global value');
    await click(id);assert.equal(await evaluate('formDirty()'),true);
    await evaluate(`$("${id}").closest(".fieldValue").querySelector(".resetSetting").click()`);
    assert.equal(await evaluate('formDirty()'),false);assert.equal(await evaluate(`$("${id}").checked`),true);
  });
  await check('Global defaults save through dedicated endpoint',async()=>{
    await click('globalSettingsBtn');await wait('editorScope==="global" && !editorLoading');
    assert.equal(await evaluate('$("deleteBtn").hidden'),true);
    await evaluate('$("setting-SWHISPER_MIN_SPEAKERS").value="4";$("setting-SWHISPER_MIN_SPEAKERS").dispatchEvent(new Event("input"))');
    await click('saveBtn');await wait('!saving && !formDirty()');assert.equal(globalValues.SWHISPER_MIN_SPEAKERS,'4');
    await select('b');await wait('editorScope==="file" && editorBase==="b"');
  });
  await check('Global draft navigation uses edit guard',async()=>{
    await click('globalSettingsBtn');await wait('editorScope==="global" && !editorLoading');
    await evaluate('$("setting-SWHISPER_MIN_SPEAKERS").value="5";$("setting-SWHISPER_MIN_SPEAKERS").dispatchEvent(new Event("input"))');
    await select('b');await wait('$("dirtyDialog").open');await click('dirtyCancel');assert.equal(await evaluate('editorScope'),'global','cancelling the dirty dialog keeps the global editor open');
    await select('b');await wait('$("dirtyDialog").open');await click('dirtyDiscard');await wait('editorBase==="b"');
  });
  await check('Pipeline-specific fields and inline number validation',async()=>{
    await evaluate('$("pipelineSel").value="diarize4";filterSettings()');
    assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").closest(".row").hidden'),true);
    await evaluate('$("showOtherSettings").checked=true;filterSettings()');
    assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").closest(".row").hidden'),false);
    await evaluate('$("setting-SWHISPER_NUM_SPEAKERS").value="0";$("setting-SWHISPER_NUM_SPEAKERS").dispatchEvent(new Event("input"))');
    assert.notEqual(await evaluate('$("error-SWHISPER_NUM_SPEAKERS").textContent'),'');
    await evaluate('restoreSavedForm();$("pipelineSel").value="transcribe4";filterSettings()');
  });
  await check('Sidebar width is adjustable and independent of text size',async()=>{
    const before=await evaluate('$("filePanel").getBoundingClientRect().width');
    await evaluate('$("sidebarResize").focus()');await key('ArrowRight','ArrowRight');
    assert.equal(await evaluate('$("filePanel").getBoundingClientRect().width'),before+20);
    await click('fontPlus');assert.equal(await evaluate('$("filePanel").getBoundingClientRect().width'),before+20);await click('fontMinus');
  });
  await check('Sidebar collapse and responsive width',async()=>{
    await click('filesToggle');assert.equal(await evaluate('getComputedStyle($("filePanel")).display'),'none');await click('filesToggle');
    await cdp('Emulation.setDeviceMetricsOverride',{width:680,height:900,deviceScaleFactor:1,mobile:false});
    await delay(50);assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),true);assert.ok(await evaluate('$("editor").clientHeight>=120'));
    const shot=await cdp('Page.captureScreenshot',{format:'png'});await fs.writeFile('/private/tmp/swhisper-narrow.png',Buffer.from(shot.data,'base64'));
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  });
  await check('Console remains resizable on narrow screens',async()=>{
    await cdp('Emulation.setDeviceMetricsOverride',{width:680,height:900,deviceScaleFactor:1,mobile:false});
    const before=await evaluate('$("consolePanel").getBoundingClientRect().height');await evaluate('$("consoleResize").focus()');await key('ArrowUp','ArrowUp');
    assert.equal(await evaluate('$("consolePanel").getBoundingClientRect().height'),before+20);
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  });
  await check('Audio timestamps seek and highlight segments',async()=>{
    await click('tabTranscript');await wait('$("timeline").children.length===2');await click('reviewAudio');
    await evaluate('$("audioPlayer").load()');await wait('$("audioPlayer").readyState>=3');
    await evaluate('$("timeline").querySelector("button").click()');await wait('$("audioPlayer").currentTime>=1');
    await evaluate('$("audioPlayer").pause();$("audioPlayer").currentTime=3.5;$("audioPlayer").dispatchEvent(new Event("timeupdate"))');
    await wait('$("timeline").querySelector(".segment.active")?.dataset.start==="3"');
    await click('reviewAudio');
  });
  await check('Export uses native Save As and cancellation writes nothing',async()=>{
    await click('renderSave');await wait('$("exportDialog").open');
    assert.equal(await evaluate('$("exportFormat").value'),'txt');assert.equal(exports.length,0);
    await click('exportConfirm');await wait('!exporting && !$("exportDialog").open');
    assert.equal(exports.at(-1).token,'native-save-selection');
    exportCanceled=true;const before=exports.length;await click('renderSave');await wait('$("exportDialog").open');await click('exportConfirm');await wait('!exporting && !$("exportDialog").open');assert.equal(exports.length,before);exportCanceled=false;await click('tabSettings');
  });
  await check('Folder change cancel preserves folder and edits',async()=>{await input('folder draft');await evaluate('$("dirInput").value="/mock/other"; $("dirBtn").click()');await wait('$("dirtyDialog").open');await click('dirtyCancel');await wait('!$("dirtyDialog").open');assert.equal(audioDir,'/mock/audio');assert.equal(await evaluate('$("dirInput").value'),'/mock/audio');});
  await check('Dirty page requests unload warning',async()=>{
    assert.equal(await evaluate('(()=>{const e=new Event("beforeunload",{cancelable:true});window.dispatchEvent(e);return e.defaultPrevented})()'),true);
  });
  await check('Save error is visible from Transcript tab',async()=>{
    const n=launches.length;
    await click('tabTranscript');failSave=true;await click('startBtn');await wait('$("dirtyDialog").open');await click('dirtySave');
    await wait('!saving && !$("uiNotice").hidden');assert.equal(await evaluate('currentTab'),'transcript');assert.equal(launches.length,n);failSave=false;
  });
  await check('Run asks to save, then previews folder before launch',async()=>{const launchedBefore=launches.length;await click('startBtn');await wait('$("dirtyDialog").open');await click('dirtySave');await wait('$("previewDialog").open');assert.equal(launches.length,launchedBefore);assert.equal(saved.b.comment,'folder draft');assert.equal(await evaluate('$("previewTable").querySelectorAll("tbody tr").length'),1);const shot=await cdp('Page.captureScreenshot',{format:'png'});await fs.writeFile('/private/tmp/swhisper-run-preview.png',Buffer.from(shot.data,'base64'));await click('previewStart');await wait('lastStatus?.running && !runBusy');assert.equal(launches.at(-1).preview_token,'preview-1');assert.equal(await evaluate('$("dirMenuBtn").disabled'),true);});
  await check('Progress current file and stop outcome',async()=>{await evaluate('pollRun()');assert.match(await evaluate('$("progressStatus").textContent'),/a.wav.*file 1 of 2/);await click('stopBtn');await wait('lastStatus?.stop_requested && !runBusy');await evaluate('pollRun()');await wait('$("runStatus").textContent==="Stopped by you"');});
  await check('Disconnection disables launch and recovery restores it',async()=>{offline=true;await evaluate('pollRun()');await wait('!connected');assert.match(await evaluate('$("connectionStatus").textContent'),/Disconnected/);assert.equal(await evaluate('$("startBtn").disabled'),true);offline=false;await evaluate('pollRun()');await wait('connected');assert.equal(await evaluate('$("startBtn").disabled'),false);});
  await check('Exit zero with reported errors is not success',async()=>{status={...status,stop_requested:false,progress:{error_count:1,last_error:'Failed: 1',completed:{}}};await evaluate('pollRun()');await wait('$("runStatus").textContent.includes("reported errors")');});
  await check('Folder switch invalidates pending same-name file response',async()=>{
    await click('tabSettings');sidecarDelay.a=250;await select('a');await wait('editorLoading');
    await evaluate('$("dirInput").value="/mock/other"; $("dirBtn").click()');await wait('META.audio_dir==="/mock/other" && !changingDir');
    await delay(300);assert.equal(await evaluate('editorBase'),null);assert.equal(await evaluate('$("saveBtn").disabled'),true);sidecarDelay.a=0;
  });
  await check('Light and dark themes both render',async()=>{
    await select('a');await wait('editorBase==="a"');
    for (const scheme of ['dark','light']) {
      await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:scheme}]});
      await delay(150);
      const shot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(`/private/tmp/swhisper-theme-${scheme}.png`,Buffer.from(shot.data,'base64'));
    }
    await cdp('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:'dark'}]});
  });
  await check('Clean page has no unload warning',async()=>{
    assert.equal(await evaluate('(()=>{const e=new Event("beforeunload",{cancelable:true});window.dispatchEvent(e);return e.defaultPrevented})()'),false);
  });
  await check('Browse opens the native folder dialog and applies the picked path',async()=>{
    await evaluate('$("dirInput").value="/mock/audio"');
    await click('dirBrowse');
    await wait('!changingDir && $("dirInput").value==="/mock/picked"');
    await wait('META.audio_dir==="/mock/picked"');
    audioDir='/mock/audio';
  });
  await check('Scribe defaults to explicit selected scope and Swedish verbatim without changing local defaults',async()=>{
    assert.equal(await evaluate('$("pipelineSel").value'),'transcribe4');
    await evaluate('$("pipelineSel").value="scribe";$("pipelineSel").dispatchEvent(new Event("change"))');
    assert.equal(await evaluate('$("scribeControls")?.hidden'),false);
    assert.equal(await evaluate('$("scribeLanguage").value'),'Swedish (verbatim)');
    assert.equal(await evaluate('$("scribeScope").value'),'selected');
    assert.match(await evaluate('$("scribeNotice").textContent'),/Local Whisper backfill/);
    await evaluate('currentBase=null;updateRunControls()');
    assert.equal(await evaluate('$("startBtn").disabled'),true,'Selected scope without a recording must not start');
    await select('a');await wait('editorBase==="a"');
    assert.equal(await evaluate('$("startBtn").disabled'),false);
    assert.equal(await evaluate('$("setting-SWHISPER_NUM_SPEAKERS").closest(".row").hidden'),false);
    assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").disabled'),false);
    assert.match(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").closest(".row").querySelector(".scribeHelp").textContent'),/backfill only/);
  });
  await check('Scribe availability and inherited value contrast in file and global settings',async()=>{
    for (const scope of ['file','global']) {
      if (scope==='global') {await click('globalSettingsBtn');await wait('editorScope==="global" && !editorLoading');}
      const speaker = scope==='global' ? 'SWHISPER_MIN_SPEAKERS' : 'SWHISPER_NUM_SPEAKERS';
      assert.equal(await evaluate(`(()=>{const el=$('setting-${speaker}');return !el.disabled && getComputedStyle(el,'::placeholder').color===getComputedStyle(el).color})()`),true);
      assert.equal(await evaluate('$(' + JSON.stringify('setting-SWHISPER_VAD') + ').closest(".row").hidden'),true);
      await click('showOtherSettings');
      assert.equal(await evaluate('$("setting-SWHISPER_VAD").closest(".row").hidden'),false);
      assert.equal(await evaluate('$("setting-SWHISPER_VAD").disabled'),true);
      assert.equal(await evaluate('(()=>{const el=$("setting-SWHISPER_VAD");const before=el.checked;el.click();return el.checked===before})()'),true);
      assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").closest(".row").querySelector(".settingApplicability").textContent'),'Local backfill only');
      await evaluate('$("pipelineSel").value="transcribe4";filterSettings()');
      assert.equal(await evaluate('$("setting-SWHISPER_VAD").disabled'),false);
      await evaluate('$("pipelineSel").value="scribe";filterSettings()');
    }
  });
  assert.deepEqual(errors,[],'No uncaught browser errors');
  console.log('All browser workflow checks passed.');
} finally {
  socket?.close();
  if(target)await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`).catch(()=>{});
  server.close();server.closeAllConnections();
}
