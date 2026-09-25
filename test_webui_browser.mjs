// Run against an isolated headless Chrome started with --remote-debugging-port=9334.
// Node 22+ only; no dependencies. All API requests use in-memory fixtures.
import http from 'node:http';
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {setTimeout as delay} from 'node:timers/promises';

const html = await fs.readFile(new URL('./webui.html', import.meta.url), 'utf8');
const css = await fs.readFile(new URL('./webui.css', import.meta.url), 'utf8');
const js = await fs.readFile(new URL('./webui.js', import.meta.url), 'utf8');
const speakerJS = await fs.readFile(new URL('./webui_waveform_speakers.js', import.meta.url), 'utf8');
const speakerCSS = await fs.readFile(new URL('./webui_waveform_speakers.css', import.meta.url), 'utf8');
let audioDir = '/mock/audio', serverFolder = null, offline = false, failSave = false, sidecarDelay = {}, versionDelay = {}, renderDelay = {}, logDelay = {}, pickDelay = 0;
let status = {running:false, pipeline:null, returncode:null, run_id:null, progress:{completed:{}}};
const saved = {a:{values:{SWHISPER_NUM_SPEAKERS:'2',SWHISPER_CONDITION_ON_PREVIOUS:'yes'},comment:'original'}, b:{values:{},comment:'B'}};
const writes = [], launches = [], errors = [], exports = [];
let globalValues={}, globalRevision="g1", exportCanceled=false, consentFixture=false;
let reusableFixture={a:{transcription:true,diarization:true},b:{transcription:true,diarization:false}};
let previewAllSkipped=false;
let legacyMetadata=false;
let speakerNames={};
let speakerSuggestions={};
let waveformDelay = {}, waveformFailure = false, noTranscript = false, detailDelay = 0;
let versionListFailure = false, versionListDelay = 0;
let listedVersions = [{id:'v2',filename:'new',timestamp:'20260924-120000',pipeline:'Scribe'}, {id:'v1',filename:'old'}];
const waveformRequests = [];
// Two versions of one recording that split it into paragraphs differently:
// v2 has short paragraphs every 10 s, v1 long ones every 25 s behind an introduction.
let compareFixture=null;const compareRenders=[];
const compareDoc=version=>{
  const [step,count,intro]=version==='v1'?[25,40,['Introduction','','Statistics','']]:[10,100,[]];
  const lines=[...intro],anchors=[],timeline=[];
  for(let i=0;i<count;i++){
    const start=i*step,end=start+step-2,middle=(start+end)/2;
    anchors.push({line:lines.length,start,end});
    lines.push(`S0${i%2} [${start}]:`,`${version} words `.repeat(version==='v1'?60:15),'');
    timeline.push({start,end:middle,speaker:`S0${i%2}`,text:`${version} first half ${i}`},{start:middle,end,speaker:`S0${i%2}`,text:`${version} second half ${i}`});
  }
  return {content:lines.join('\n'),anchors,timeline};
};
const interviewParams = {
      RECORDING_TYPE:{allow_empty:true,type:'str',stage:'diarize',label:'Recording type',global:'',choices:['','interview'],choice_labels:{'':'Not specified',interview:'Interview'},help:'Interview enables automatic speaker naming by word count after diarization and name suggestions in the Speaker names dialog. Not specified disables both.',scribe:{supported:true}},
      INTERVIEWER_NAME:{type:'str',stage:'diarize',label:'Interviewer name',global:'INTERVIEWER',help:'Name for the speaker with the second-highest word count when recording type is Interview. Default: INTERVIEWER.',scribe:{supported:true}},
      INTERVIEWEE_NAME:{type:'str',stage:'diarize',label:'Interviewee name',global:'INTERVIEWEE',help:'Name for the speaker with the highest word count when recording type is Interview. Default: INTERVIEWEE.',scribe:{supported:true}},
};
const meta = () => ({audio_dir:audioDir,audio_paths:['/mock/audio','/mock/other'],
  pipelines:[{id:'transcribe4',label:'Standard pipeline',choice:'Local Whisper'},{id:'diarize4',label:'Diarization only',choice:null},{id:'scribe',label:'SUNET Scribe + local diarization',choice:'SUNET Scribe (remote)'}],
  diarization_models:[{id:'community-1',label:'pyannote community-1'},{id:'nemotron',label:'Nemotron 3 (up to 8 speakers)'},{id:'precision',label:'pyannote Precision-2 (hosted)'}],diarization_default:'community-1',groups:[{name:'Speakers',keys:['SWHISPER_NUM_SPEAKERS','SWHISPER_CONDITION_ON_PREVIOUS','SWHISPER_VAD']}],
  params:{...interviewParams,SWHISPER_VAD:{type:'bool',global:true,stage:'transcribe',scribe:{supported:false,scope:'Unavailable for Scribe',description:'Not applied to Scribe.'}},SWHISPER_NUM_SPEAKERS:{type:'int',min:1,label:'Number of speakers',global:2,stage:'diarize',help:'Speaker count',scribe:{supported:true,description:'Local speaker count'}},SWHISPER_CONDITION_ON_PREVIOUS:{type:'bool',global:true,stage:'transcribe',help:'Use previous text',scribe:{supported:true,scope:'Local backfill only',description:'Local Whisper backfill only'}}}});
const server = http.createServer(async (req,res) => {
  const u = new URL(req.url, 'http://localhost');
  let body = ''; for await (const chunk of req) body += chunk;
  const data = body ? JSON.parse(body) : {};
  const send = (value, code=200) => {res.writeHead(code, {'Content-Type':'application/json'});res.end(JSON.stringify(value));};
  if (u.pathname === '/') {res.writeHead(200, {'Content-Type':'text/html','Cache-Control':'no-store'});res.end(html);return;}
  if (u.pathname === '/webui.css') {res.writeHead(200, {'Content-Type':'text/css','Cache-Control':'no-store'});res.end(css);return;}
  if (u.pathname === '/webui.js') {res.writeHead(200, {'Content-Type':'text/javascript','Cache-Control':'no-store'});res.end(js);return;}
  if (u.pathname === '/webui_waveform_speakers.js') {res.writeHead(200, {'Content-Type':'text/javascript'});res.end(speakerJS);return;}
  if (u.pathname === '/webui_waveform_speakers.css') {res.writeHead(200, {'Content-Type':'text/css'});res.end(speakerCSS);return;}
  if (u.pathname === '/api/meta') {
    const metadata = meta();
    if (legacyMetadata) { delete metadata.diarization_models; delete metadata.diarization_default; }
    return send(metadata);
  }
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
    const gm=meta();gm.params.SWHISPER_MIN_SPEAKERS=gm.params.SWHISPER_NUM_SPEAKERS;delete gm.params.SWHISPER_NUM_SPEAKERS;gm.groups=gm.groups.map(g=>({...g,keys:g.keys.map(k=>k==='SWHISPER_NUM_SPEAKERS'?'SWHISPER_MIN_SPEAKERS':k)}));
    return send({...gm,values:globalValues,revision:globalRevision,path:'/mock/.swhisper.env'});
  }
  if (u.pathname.startsWith('/api/timeline/')) {
    if (noTranscript) return send({detail:'No transcript'},404);
    const version=u.searchParams.get('version') || 'v2';
    return send({audio:'a.wav',segments:compareFixture
      ? version===compareFixture.stored ? [] : compareDoc(version).timeline
      : [{start:1,end:3,speaker:speakerNames.S01 || 'S01',text:'Hello'},{start:3,end:6,speaker:speakerNames.S02 || 'S02',text:'World'}]});
  }
  if (u.pathname.startsWith('/api/waveform/')) {
    const base = decodeURIComponent(u.pathname.split('/').at(-1));
    const start = Number(u.searchParams.get('start') || 0), end = Number(u.searchParams.get('end') || 10);
    waveformRequests.push({base,start,end});
    await delay(u.searchParams.has('start') ? detailDelay : waveformDelay[base] || 0);
    if (waveformFailure) return send({detail:'Cannot decode fixture'},422);
    return send({duration:10,start,end,peaks:Array.from({length:2048},(_,i)=>i%400>300?.01:(.2+.6*Math.abs(Math.sin(i*.032)))*(.3+.7*Math.abs(Math.sin(i*.37))))});
  }
  if (u.pathname.startsWith('/api/audio/')) {
    const pcm=Buffer.alloc(44+160000);pcm.write('RIFF');pcm.writeUInt32LE(pcm.length-8,4);pcm.write('WAVEfmt ',8);pcm.writeUInt32LE(16,16);pcm.writeUInt16LE(1,20);pcm.writeUInt16LE(1,22);pcm.writeUInt32LE(8000,24);pcm.writeUInt32LE(16000,28);pcm.writeUInt16LE(2,32);pcm.writeUInt16LE(16,34);pcm.write('data',36);pcm.writeUInt32LE(160000,40);
    const range=/bytes=(\d+)-(\d*)/.exec(req.headers.range||'');const start=range?Number(range[1]):0,end=range&&range[2]?Number(range[2]):pcm.length-1;res.writeHead(range?206:200,{'Content-Type':'audio/wav','Accept-Ranges':'bytes','Content-Length':end-start+1,...(range?{'Content-Range':`bytes ${start}-${end}/${pcm.length}`}:{})});res.end(pcm.subarray(start,end+1));return;
  }
  if (u.pathname.endsWith('/save-as')) return send(exportCanceled ? {canceled:true} : {token:'native-save-selection',path:'/mock/chosen/review.txt',format:data.format});
  if (u.pathname.endsWith('/save-selected')) {exports.push(data);return send({written:['/mock/chosen/review.txt']});}
  if (u.pathname.startsWith('/api/render/') && u.pathname.endsWith('/save')) {exports.push(data);return send({written:['/mock/audio/exports/review.txt']});}
  if (u.pathname.startsWith('/api/render/') && compareFixture) {
    compareRenders.push({version:data.version,options:data.options});
    return send(compareDoc(data.version));
  }
  if (u.pathname.startsWith('/api/render/')) {
    await delay(renderDelay[data.options?.min_pause] || 0);
    return send({content:`Rendered ${data.options.min_pause}`,suffix:'_view',timeline:[{start:1,end:3,speaker:'S01',text:`Filtered ${data.options.min_pause}`}]});
  }
  if (u.pathname === '/api/history') return send({entries:[]});
  if (u.pathname === '/api/audio-dir') {audioDir=data.path;return send({audio_dir:audioDir,audio_paths:['/mock/audio','/mock/other']});}
  if (u.pathname === '/api/audio-dir/pick') {await delay(pickDelay);return send({path:'/mock/picked'});}
  if (u.pathname === '/api/run/preview') {
    const only=u.searchParams.getAll('only'), reprocess=u.searchParams.get('reprocess')==='true';
    const retranscribe=u.searchParams.getAll('retranscribe'), rediarize=u.searchParams.getAll('rediarize');
    const forced=reprocess||retranscribe.length>0||rediarize.length>0, scribe=u.searchParams.get('pipeline')==='scribe';
    const files=previewAllSkipped
      ? [{file:'a.json',stage:'Diarization',action:forced?'reprocess':'skip',reason:forced?'Reprocessing requested for this run':'Existing output; no reprocessing requested'}]
      : [{file:'a.wav',stage:'Transcription',action:'process',reason:'No output'},{file:'b.wav',stage:'Transcription',action:'skip',reason:'Existing output'},{file:'c.wav',stage:'Transcription',action:'skip',reason:'Existing output'}];
    const notes=previewAllSkipped && !forced
      ? ['Run is disabled because existing outputs are being skipped. To run them again, cancel this preview, tick "Re-transcribe" or "Re-diarize" under the recording, then select Run again. Selecting a recording alone does not enable reprocessing.']
      : ['Folder scope'];
    const reusable=previewAllSkipped?{a:{transcription:false,diarization:true}}:reusableFixture;
    const remote=scribe&&(retranscribe.length>0||consentFixture);
    return send({audio_dir:audioDir,pipeline:u.searchParams.get('pipeline'),token:`preview-${reprocess}-${retranscribe}-${rediarize}`,
                 diarization_model:u.searchParams.get('diarization_model'),reprocess,retranscribe,rediarize,reusable,remote,
                 consent_required:scribe&&consentFixture,language:u.searchParams.get('language'),selected:u.searchParams.get('selected'),
                 only:only.length?only:null,notes,files});
  }
  if (u.pathname === '/api/run' && req.method === 'POST') {launches.push(data);status={running:true,pipeline:'transcribe4',run_id:'run1',started_at:Date.now()/1000,audio_dir:audioDir,progress:{stage:'Transcription',current_file:'a.wav',file_index:1,total_files:2,completed:{Transcription:0}}};return send(status);}
  if (u.pathname === '/api/run/stop') {status={...status,running:false,stop_requested:true,returncode:0};return send(status);}
  if (u.pathname === '/api/sidecars/batch') {
    if(failSave)return send({errors:{RECORDING_TYPE:'Invalid fixture setting'}},422);
    for (const base of data.bases) {
      const values={...(saved[base]?.values||{})}, comment=saved[base]?.comment||'';
      for(const [key,value] of Object.entries(data.changes)) value===null?delete values[key]:values[key]=value;
      saved[base]={values,comment};writes.push({base,data:{...values,_comment:comment}});
    }
    return send({files:Object.fromEntries(data.bases.map(base=>[base,{exists:true,...saved[base],revision:JSON.stringify(saved[base])}]))});
  }
  if (u.pathname.startsWith('/api/sidecar/')) {
    const base=decodeURIComponent(u.pathname.split('/').at(-1));
    if (req.method==='PUT') {writes.push({base,data});if(failSave)return send({errors:{SWHISPER_NUM_SPEAKERS:'invalid'}},422);saved[base]={values:Object.fromEntries(Object.entries(data).filter(([k])=>k!=='_comment')),comment:data._comment};}
    if(req.method==='DELETE') saved[base]={values:{},comment:''};
    await delay(sidecarDelay[base] || 0);
    return send({exists:true,unknown:{FUTURE:'preserve'},...(saved[base]||{values:{},comment:''}),revision:JSON.stringify(saved[base]||{})});
  }
  if (u.pathname.startsWith('/api/speakers/')) {
    if (req.method==='PUT') {if(data.names.S01==='bad')return send({detail:'Two speakers cannot share that name'},422);speakerNames=Object.fromEntries(Object.entries(data.names).filter(([,v])=>v));}
    return send({base:'a',detected:['S01','S02'],names:speakerNames,suggested:speakerSuggestions});
  }
  if (u.pathname.startsWith('/api/json-transcript/')) return send({base:'a',path:'whisper-json-output/a.json',content:'{"raw": true}'});
  if (u.pathname.startsWith('/api/transcript/') && u.pathname.endsWith('/versions')) {
    await delay(versionListDelay);
    if (versionListFailure) return send({detail:'Version list unavailable'},503);
    return send({versions:decodeURIComponent(u.pathname.split('/')[3])==='<unsafe>'?[]:listedVersions});
  }
  if (u.pathname.startsWith('/api/transcript/')) {
    if (noTranscript) return send({detail:'No transcript'},404);
    const version=u.searchParams.get('version')||'v2';
    await delay(versionDelay[version]||0);
    if (compareFixture) return send({versions:[{id:'v2',filename:'new'},{id:'v1',filename:'old'}],selected:version,
      ...compareDoc(version),can_render:version!==compareFixture.stored,speaker_names:{}});
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
  const waveformDrag = async (id, from, to = from) => {
    await wait(`$(${JSON.stringify(id)}).offsetWidth>0`);
    await evaluate(`$(${JSON.stringify(id)}).querySelector('.waveformTrack').scrollIntoView({block:'nearest'})`);
    const rect = await evaluate(`(()=>{const r=$(${JSON.stringify(id)}).querySelector('.waveformTrack').getBoundingClientRect();return {x:r.left,y:r.top+r.height/2,width:r.width}})()`);
    await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:rect.x+rect.width*from,y:rect.y,button:'left',buttons:1,clickCount:1});
    if (to !== from) await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',x:rect.x+rect.width*to,y:rect.y,button:'left',buttons:1});
    await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:rect.x+rect.width*to,y:rect.y,button:'left',buttons:0,clickCount:1});
  };
  const select=async base=>evaluate(`document.querySelector('[data-base="${base}"] .fileSelect').click()`);
  const input=async value=>evaluate(`document.getElementById('commentField').value=${JSON.stringify(value)};document.getElementById('commentField').dispatchEvent(new Event('input',{bubbles:true}))`);
  const key=async(key,code)=>{await cdp('Input.dispatchKeyEvent',{type:'keyDown',key,code,...(key==='Enter'?{text:'\r'}:{}),windowsVirtualKeyCode:({Escape:27,Enter:13,ArrowLeft:37,ArrowRight:39,ArrowUp:38,ArrowDown:40})[key]});await cdp('Input.dispatchKeyEvent',{type:'keyUp',key,code,windowsVirtualKeyCode:({Escape:27,Enter:13,ArrowLeft:37,ArrowRight:39,ArrowUp:38,ArrowDown:40})[key]});};
  const check=async(name,fn)=>{
    try { await fn(); console.log('PASS '+name); }
    catch (error) {
      const shot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile('/private/tmp/swhisper-browser-failure.png',Buffer.from(shot.data,'base64'));
      throw error;
    }
  };
  await cdp('Runtime.enable');await cdp('Page.enable');
  await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
  await cdp('Page.navigate',{url:`http://127.0.0.1:${server.address().port}/`});
  await wait('typeof initialized !== "undefined" && initialized && connected');
  const expandFile = base => evaluate(`document.querySelector('[data-base="${base}"] .fileExpand').click()`);
  await check('File expansion lists transcript versions without selecting or ticking the recording',async()=>{
    assert.equal(await evaluate('document.querySelectorAll(".fileExpand[aria-expanded=false]").length'),3);
    await expandFile('a');
    await wait('document.querySelectorAll("[data-base=a] .fileTranscript").length===3');
    assert.equal(await evaluate('currentBase'),null);
    assert.equal(await evaluate('runSelection.size'),0);
    assert.deepEqual(await evaluate('[...document.querySelectorAll("[data-base=a] .fileTranscript")].map(b=>b.textContent)'),
      ['2026-09-24 12:00:00 · Scribe','old','Raw JSON transcript']);
    await evaluate('document.querySelector("[data-base=a] .fileExpand").focus()');
    await key('Enter','Enter');
    assert.equal(await evaluate('document.querySelector("[data-base=a] .fileTranscripts").hidden'),true);
    await key('Enter','Enter');
    await wait('document.querySelectorAll("[data-base=a] .fileTranscript").length===3');
    await expandFile('<unsafe>');
    await wait('document.querySelectorAll(".fileTranscripts")[0].textContent==="No transcripts yet."');
    assert.equal(await evaluate('document.querySelector("#files unsafe")'),null);
  });
  await check('Expanded transcript lists survive filtering and refresh with newly available versions',async()=>{
    await evaluate('$("fileSearch").value="b";renderFiles();$("fileSearch").value="";renderFiles()');
    await wait('document.querySelectorAll("[data-base=a] .fileTranscript").length===3');
    listedVersions.push({id:'v0',filename:'earlier'});
    await evaluate('loadFiles()');
    await wait('document.querySelectorAll("[data-base=a] .fileTranscript").length===4');
    assert.equal(await evaluate('document.querySelector("[data-base=a] .fileExpand").getAttribute("aria-expanded")'),'true');
    const shot=await cdp('Page.captureScreenshot',{format:'png'});
    await fs.writeFile('/private/tmp/swhisper-expanded-files.png',Buffer.from(shot.data,'base64'));
    await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
    assert.equal(await evaluate('$("files").scrollWidth <= $("files").clientWidth'),true);
    const mobile=await cdp('Page.captureScreenshot',{format:'png'});
    await fs.writeFile('/private/tmp/swhisper-expanded-files-mobile.png',Buffer.from(mobile.data,'base64'));
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    listedVersions.pop();
  });
  await check('Failed transcript lists offer a working retry',async()=>{
    versionListFailure=true;
    await expandFile('b');
    await wait('document.querySelector("[data-base=b] .fileTranscripts").textContent.includes("Retry")');
    versionListFailure=false;
    await evaluate('document.querySelector("[data-base=b] .fileTranscripts button").click()');
    await wait('document.querySelectorAll("[data-base=b] .fileTranscript").length===3');
  });
  await check('Transcript entries open the chosen version and respect unsaved settings',async()=>{
    await evaluate('document.querySelector("[data-base=a] [data-version=v1]").click()');
    await wait('$("versionSel").value==="v1" && !$("versionSel").disabled');
    assert.equal(await evaluate('currentTab'),'transcript');
    await click('tabSettings');await input('unsaved transcript navigation');
    await expandFile('b');await expandFile('b');
    await wait('document.querySelectorAll("[data-base=b] .fileTranscript").length===3');
    assert.equal(await evaluate('$("dirtyDialog").open'),false);
    await evaluate('document.querySelector("[data-base=b] [data-version=v1]").click()');
    await wait('$("dirtyDialog").open');await click('dirtyCancel');
    await wait('!$("dirtyDialog").open');
    assert.equal(await evaluate('currentBase'),'a');
    assert.equal(await evaluate('$("commentField").value'),'unsaved transcript navigation');
    await input(saved.a.comment);
    await evaluate('document.querySelector("[data-base=b] [data-version=__raw_json__]").click()');
    await wait('currentBase==="b" && $("versionSel").value==="__raw_json__" && !$("versionSel").disabled');
  });
  await check('Folder changes clear expanded lists and ignore responses from the previous folder',async()=>{
    versionListDelay=300;
    await evaluate('fileVersionRequests.clear();renderFiles()');
    await evaluate('$("dirInput").value="/mock/other";changeDir()');
    await wait('META.audio_dir==="/mock/other" && !changingDir');
    await delay(350);
    assert.equal(await evaluate('expandedFiles.size'),0);
    assert.equal(await evaluate('document.querySelectorAll(".fileTranscript").length'),0);
    versionListDelay=0;
    await evaluate('$("dirInput").value="/mock/audio";changeDir()');
    await wait('META.audio_dir==="/mock/audio" && !changingDir');
    await click('tabSettings');
  });
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
    // A menu opened by pointer must be navigable by keyboard straight away,
    // with no wait: the keypress a user makes next arrives in the very next
    // task, so any focus move that is merely queued has already lost.
    assert.equal(await evaluate('$("dirMenu").contains(document.activeElement)'),true,'Opening focuses an item, not the arrow');
    await key('End','End');
    assert.equal(await evaluate('document.activeElement.title'),'/mock/other','End reaches the last folder');
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
    await wait('$("transcriptWaveform").querySelector(".waveformSpeakerLegend")?.textContent.includes("Interviewer")');
    assert.equal(await evaluate('$("timeline").children.length'),1,'Renaming preserves the filtered review list');
    speakerNames={};
  });
  await check('Interview suggestions fill the naming modal and remain editable',async()=>{
    speakerSuggestions={S01:'INTERVIEWER',S02:'Anna Svensson'};
    speakerNames={S01:'Chris'};
    await click('renderSpeakers');await wait('$("speakerDialog").open');
    assert.equal(await evaluate('$("speaker-S01").value'),'Chris');
    assert.equal(await evaluate('$("speaker-S02").value'),'Anna Svensson');
    assert.equal(await evaluate('$("speakerSuggestion").hidden'),false);
    if (process.env.SWHISPER_INTERVIEW_SCREENSHOT) {
      const shot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(process.env.SWHISPER_INTERVIEW_SCREENSHOT,Buffer.from(shot.data,'base64'));
    }
    await click('speakerCancel');await wait('!$("speakerDialog").open');
    assert.deepEqual(speakerNames,{S01:'Chris'});
    speakerNames={};
    await click('renderSpeakers');await wait('$("speakerDialog").open');
    await click('speakerSave');await wait('!$("speakerDialog").open');
    assert.deepEqual(speakerNames,speakerSuggestions);
    speakerSuggestions={};speakerNames={};
  });
  await check('Run preview separates what runs from what is skipped',async()=>{
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('$("previewTitle").textContent'),'Run Preview');
    assert.equal(await evaluate('$("previewStart").textContent'),'Run');
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
  await check('Skipped output disables Run until reprocessing is requested',async()=>{
    previewAllSkipped=true;
    await evaluate('$("pipelineSel").value="transcribe4";$("runModel").value="nemotron";retranscribeBases.clear();rediarizeBases.clear();reuseKey="";updateRunControls()');
    await wait('document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\') && !document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\')');
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('$("previewStart").disabled'),true);
    assert.match(await evaluate('$("previewNotes").textContent'),/Re-diarize/);
    if (process.env.SWHISPER_TEST_SCREENSHOT) {
      const screenshot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(process.env.SWHISPER_TEST_SCREENSHOT,Buffer.from(screenshot.data,'base64'));
    }
    await click('previewCancel');await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\').click()');
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('$("previewStart").disabled'),false);
    assert.match(await evaluate('$("previewTable").textContent'),/Reprocessing requested/);
    assert.deepEqual(await evaluate('[pendingPreview.retranscribe,pendingPreview.rediarize]'),[[],['a']]);
    assert.equal(await evaluate('pendingPreview.diarization_model'),'nemotron');
    await click('previewCancel');
    await evaluate('$("runModel").value="community-1";rediarizeBases.clear()');
    previewAllSkipped=false;
    await evaluate('updateRunControls()');
  });
  await check('Files offer Re-transcribe and Re-diarize only for reusable results',async()=>{
    reusableFixture={};
    await evaluate('reuseKey="";updateRunControls()');
    await wait('!document.querySelector(".fileForce")');
    reusableFixture={a:{transcription:true,diarization:true},b:{transcription:true,diarization:false}};
    await evaluate('reuseKey="";updateRunControls()');
    await wait('document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\') && document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\') && document.querySelector(\'[data-base="b"] .fileForce input[aria-label^="Re-transcribe"]\') && !document.querySelector(\'[data-base="b"] .fileForce input[aria-label="Re-diarize: b"]\')');
    assert.deepEqual(await evaluate('[...document.querySelectorAll(\'[data-base="a"] .fileForce span\')].map(s=>s.textContent)'),['Re-transcribe','Re-diarize']);
    const editing=await evaluate('editorBase');
    await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\').click()');
    assert.equal(await evaluate('editorBase'),editing,'Ticking an option does not open the recording');
    assert.equal(await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\').checked && document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\').disabled'),true,'A new transcript means a new diarization');
    await click('startBtn');await wait('$("previewDialog").open');
    assert.deepEqual(await evaluate('[pendingPreview.retranscribe,pendingPreview.rediarize]'),[['a'],[]]);
    await click('previewCancel');
    await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\').click()');
    assert.equal(await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\').checked || document.querySelector(\'[data-base="a"] .fileForce input[aria-label="Re-diarize: a"]\').disabled'),false);
    reusableFixture={a:{transcription:false,diarization:true}};
    await evaluate('document.querySelector(\'[data-base="b"] .fileForce input[aria-label^="Re-transcribe"]\').click()');
    await evaluate('reuseKey="";updateRunControls()');
    await wait('!document.querySelector(\'[data-base="b"] .fileForce input[aria-label^="Re-transcribe"]\')');
    assert.equal(await evaluate('retranscribeBases.has("b")'),false,'A choice is dropped when its result is no longer reused');
    reusableFixture={a:{transcription:true,diarization:true},b:{transcription:true,diarization:false}};
    await evaluate('reuseKey="";updateRunControls()');
    await evaluate('$("runModel").value="precision";$("pipelineSel").value="scribe";$("pipelineSel").dispatchEvent(new Event("change"))');
    assert.equal(await evaluate('[...$("runModel").options].find(o=>o.value==="precision").disabled'),true);
    assert.equal(await evaluate('$("runModel").value'),'community-1','Scribe cannot use Precision');
    await evaluate('$("pipelineSel").value="transcribe4";$("pipelineSel").dispatchEvent(new Event("change"))');
    assert.deepEqual(await evaluate('[...$("pipelineSel").options].map(o=>o.textContent)'),
                     ['Transcription: Local Whisper','Transcription: SUNET Scribe (remote)']);
  });
  await check('Scribe uploads a recording again without new consent, but asks before a first upload',async()=>{
    await evaluate('$("pipelineSel").value="scribe";$("pipelineSel").dispatchEvent(new Event("change"));$("scribeScope").value="folder";$("scribeScope").dispatchEvent(new Event("change"))');
    await wait('document.querySelector(\'[data-base="a"] .fileForce span\')?.textContent===\'Re-transcribe (upload again)\'');
    await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\').click()');
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('$("consentRow").hidden'),true,'A recording uploaded before needs no new consent');
    assert.equal(await evaluate('$("previewTitle").textContent'),'Scribe remote upload preview');
    const before=launches.length;
    await click('previewStart');
    for(let i=0;i<200 && launches.length===before;i++) await delay(20);
    assert.deepEqual([launches.at(-1).retranscribe,launches.at(-1).remote_upload],[['a'],true]);
    await wait('lastStatus && lastStatus.running');await click('stopBtn');await wait('lastStatus && !lastStatus.running');
    assert.equal(await evaluate('retranscribeBases.size'),0,'A launched run spends its choices');
    consentFixture=true;
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('$("consentRow").hidden'),false);
    await click('previewStart');
    assert.match(await evaluate('$("previewSummary").textContent'),/Consent to the remote upload first/);
    await click('previewCancel');
    consentFixture=false;
    await evaluate('$("scribeScope").value="selected";$("scribeScope").dispatchEvent(new Event("change"));$("pipelineSel").value="transcribe4";$("pipelineSel").dispatchEvent(new Event("change"))');
  });
  await check('Selected files narrow the run and reach the launch call',async()=>{
    assert.equal(await evaluate('$("startBtn").textContent'),'Run');
    assert.equal(await evaluate('$("clearRunSelection").hidden'),true);
    await evaluate('document.querySelector(\'[data-base="b"] .runPick input\').click()');
    await wait('!$("startBtn").disabled && $("startBtn").title==="Run 1 selected file(s)"');
    assert.equal(await evaluate('$("startBtn").textContent'),'Run');
    assert.equal(await evaluate('$("clearRunSelection").hidden'),false);
    const before=launches.length;
    await click('startBtn');await wait('$("previewDialog").open');
    assert.match(await evaluate('$("previewFolder").textContent'),/1 selected file/);
    await click('previewStart');
    for(let i=0;i<200 && launches.length===before;i++) await delay(20);
    assert.deepEqual(launches.at(-1).only,['b']);
    await wait('lastStatus && lastStatus.running');await click('stopBtn');await wait('lastStatus && !lastStatus.running');
    await click('clearRunSelection');await wait('!selectionBusy && $("startBtn").title==="Run eligible files in this folder"');
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
    await evaluate('$("fileSearch").value="";$("fileSearch").dispatchEvent(new Event("input"));$("fileFilter").value="reprocess";renderFiles()');
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
  await check('Interview settings appear once in Common settings and save their values',async()=>{
    await click('globalSettingsBtn');await wait('editorScope==="global" && !editorLoading');
    for (const name of ['RECORDING_TYPE','INTERVIEWER_NAME','INTERVIEWEE_NAME']) {
      assert.equal(await evaluate(`$("setting-${name}").closest("details").querySelector("summary").textContent`),'Common settings');
      assert.equal(await evaluate(`document.querySelectorAll('#setting-${name}').length`),1);
    }
    assert.deepEqual(await evaluate('[...$("setting-RECORDING_TYPE").options].map(o=>[o.value,o.textContent])'),[['','Not specified'],['interview','Interview']]);
    await evaluate('$("setting-RECORDING_TYPE").value="interview";$("setting-RECORDING_TYPE").dispatchEvent(new Event("change"));$("setting-INTERVIEWER_NAME").value="Researcher";$("setting-INTERVIEWER_NAME").dispatchEvent(new Event("input"));$("setting-INTERVIEWEE_NAME").value="Participant";$("setting-INTERVIEWEE_NAME").dispatchEvent(new Event("input"))');
    if (process.env.SWHISPER_COMMON_SCREENSHOT) {
      const shot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(process.env.SWHISPER_COMMON_SCREENSHOT,Buffer.from(shot.data,'base64'));
    }
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(globalValues.RECORDING_TYPE,'interview');
    assert.equal(globalValues.INTERVIEWER_NAME,'Researcher');
    assert.equal(globalValues.INTERVIEWEE_NAME,'Participant');
    await evaluate('$("setting-RECORDING_TYPE").value="";$("setting-RECORDING_TYPE").dispatchEvent(new Event("change"))');
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(globalValues.RECORDING_TYPE,'');
  });
  await check('Per-file interview overrides distinguish inheritance from disabled naming',async()=>{
    await select('a');await wait('editorScope==="file" && editorBase==="a" && !editorLoading');
    for (const name of ['RECORDING_TYPE','INTERVIEWER_NAME','INTERVIEWEE_NAME']) {
      assert.equal(await evaluate(`$("setting-${name}").closest("details").querySelector("summary").textContent`),'Common settings');
    }
    assert.deepEqual(await evaluate('[...$("setting-RECORDING_TYPE").options].map(o=>o.textContent)'),['Use global','Not specified','Interview']);
    assert.equal(await evaluate('$("setting-RECORDING_TYPE").value'),'__inherit__');
    assert.equal(await evaluate('Object.hasOwn(formValues(),"RECORDING_TYPE")'),false);
    await evaluate('$("setting-RECORDING_TYPE").value="interview";$("setting-RECORDING_TYPE").dispatchEvent(new Event("change"));$("setting-INTERVIEWER_NAME").value="Chris";$("setting-INTERVIEWER_NAME").dispatchEvent(new Event("input"));$("setting-INTERVIEWEE_NAME").value="Anna";$("setting-INTERVIEWEE_NAME").dispatchEvent(new Event("input"))');
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(saved.a.values.RECORDING_TYPE,'interview');
    assert.equal(saved.a.values.INTERVIEWER_NAME,'Chris');
    assert.equal(saved.a.values.INTERVIEWEE_NAME,'Anna');
    await evaluate('$("setting-RECORDING_TYPE").value="";$("setting-RECORDING_TYPE").dispatchEvent(new Event("change"))');
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(saved.a.values.RECORDING_TYPE,'');
    await select('b');await wait('editorBase==="b"');
    await select('a');await wait('editorBase==="a"');
    assert.equal(await evaluate('$("setting-RECORDING_TYPE").value'),'');
    assert.equal(await evaluate('$("setting-RECORDING_TYPE").closest(".fieldValue").querySelector(".inheritance").textContent'),'per-file override');
    await evaluate('$("setting-RECORDING_TYPE").closest(".fieldValue").querySelector(".resetSetting").click()');
    assert.equal(await evaluate('Object.hasOwn(formValues(),"RECORDING_TYPE")'),false);
    await evaluate('restoreSavedForm()');
    assert.equal(await evaluate('$("setting-RECORDING_TYPE").value'),'');
    await evaluate('$("setting-RECORDING_TYPE").value="__inherit__";$("setting-RECORDING_TYPE").dispatchEvent(new Event("change"))');
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(Object.hasOwn(saved.a.values,'RECORDING_TYPE'),false);
    await select('b');await wait('editorBase==="b"');
    assert.equal(await evaluate('$("setting-INTERVIEWEE_NAME").value'),'');
  });
  await check('Select all shown edits only touched settings across the selection',async()=>{
    saved.a={values:{SWHISPER_NUM_SPEAKERS:'2',RECORDING_TYPE:'',SWHISPER_CONDITION_ON_PREVIOUS:'true'},comment:'First file'};
    saved.b={values:{SWHISPER_NUM_SPEAKERS:'4',RECORDING_TYPE:'interview',SWHISPER_CONDITION_ON_PREVIOUS:'false'},comment:'Second file'};
    await evaluate('$("fileSearch").value="";$("fileFilter").value="ready";renderFiles()');
    await click('selectShown');await wait('editorScope==="batch" && !editorLoading && !selectionBusy');
    assert.deepEqual(await evaluate('[...runSelection].sort()'),['a','b']);
    assert.equal(await evaluate('$("setting-SWHISPER_NUM_SPEAKERS").placeholder'),'Mixed values');
    assert.equal(await evaluate('$("setting-RECORDING_TYPE").value'),'__mixed__');
    assert.equal(await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").indeterminate'),true);
    assert.equal(await evaluate('$("commentField")===null && $("deleteBtn").hidden'),true);
    assert.equal(await evaluate('$("saveBtn").disabled'),true);
    await evaluate('$("setting-RECORDING_TYPE").value="interview";$("setting-RECORDING_TYPE").dispatchEvent(new Event("change"))');
    if (process.env.SWHISPER_BULK_SCREENSHOT) {
      await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1300,deviceScaleFactor:1,mobile:false});
      const shot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(process.env.SWHISPER_BULK_SCREENSHOT,Buffer.from(shot.data,'base64'));
      await cdp('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});
      const mobile=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(process.env.SWHISPER_BULK_SCREENSHOT.replace('.png','-mobile.png'),Buffer.from(mobile.data,'base64'));
      assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'),true);
      await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    }
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(saved.a.values.RECORDING_TYPE,'interview');assert.equal(saved.b.values.RECORDING_TYPE,'interview');
    assert.equal(saved.a.values.SWHISPER_NUM_SPEAKERS,'2');assert.equal(saved.b.values.SWHISPER_NUM_SPEAKERS,'4');
    assert.equal(saved.a.comment,'First file');assert.equal(saved.b.comment,'Second file');
    await evaluate('$("setting-INTERVIEWEE_NAME").value="Participant";$("setting-INTERVIEWEE_NAME").dispatchEvent(new Event("input"))');
    await click('clearRunSelection');await wait('$("dirtyDialog").open');await click('dirtyCancel');await wait('!selectionBusy');
    assert.equal(await evaluate('runSelection.size'),2);assert.equal(await evaluate('formDirty()'),true);
    failSave=true;await click('saveBtn');await wait('!saving && $("editorMsg").textContent.includes("No files")');
    assert.equal(saved.a.values.INTERVIEWEE_NAME,undefined);failSave=false;
    await click('clearRunSelection');await wait('$("dirtyDialog").open');await click('dirtySave');await wait('!selectionBusy && runSelection.size===0');
    assert.equal(saved.a.values.INTERVIEWEE_NAME,'Participant');assert.equal(saved.b.values.INTERVIEWEE_NAME,'Participant');
    await click('selectShown');await wait('editorScope==="batch" && !selectionBusy');
    await evaluate('$("setting-SWHISPER_CONDITION_ON_PREVIOUS").closest(".fieldValue").querySelector(".resetSetting").click()');
    assert.deepEqual(await evaluate('formValues()'),{SWHISPER_CONDITION_ON_PREVIOUS:null});
    await click('saveBtn');await wait('!saving && !formDirty()');
    assert.equal(Object.hasOwn(saved.a.values,'SWHISPER_CONDITION_ON_PREVIOUS'),false);
    assert.equal(Object.hasOwn(saved.b.values,'SWHISPER_CONDITION_ON_PREVIOUS'),false);
    await evaluate('$("fileSearch").value="a";renderFiles()');
    assert.equal(await evaluate('$("selectionCount").textContent'),'2 selected (1 hidden by filters)');
    await click('selectShown');await wait('!selectionBusy && editorScope==="file"');
    assert.deepEqual(await evaluate('[...runSelection]'),['b']);assert.equal(await evaluate('editorBase'),'b');
    await click('clearRunSelection');await wait('!selectionBusy && runSelection.size===0');
    await evaluate('$("fileSearch").value="no matching files";renderFiles()');
    assert.equal(await evaluate('$("selectShown").disabled'),true);
    await evaluate('$("fileSearch").value="";$("fileFilter").value="all";renderFiles()');
    await click('selectShown');await wait('editorScope==="batch" && !selectionBusy');
    await evaluate('$("dirInput").value="/mock/bulk-test";changeDir()');await wait('META.audio_dir==="/mock/bulk-test" && !changingDir');
    assert.equal(await evaluate('runSelection.size'),0);
    assert.equal(await evaluate('editorBase'),null);
    await evaluate('$("dirInput").value="/mock/audio";changeDir()');await wait('META.audio_dir==="/mock/audio" && !changingDir');
  });
  await check('Global draft navigation uses edit guard',async()=>{
    await click('globalSettingsBtn');await wait('editorScope==="global" && !editorLoading');
    await evaluate('$("setting-SWHISPER_MIN_SPEAKERS").value="5";$("setting-SWHISPER_MIN_SPEAKERS").dispatchEvent(new Event("input"))');
    await select('b');await wait('$("dirtyDialog").open');await click('dirtyCancel');assert.equal(await evaluate('editorScope'),'global','cancelling the dirty dialog keeps the global editor open');
    await select('b');await wait('$("dirtyDialog").open');await click('dirtyDiscard');await wait('editorBase==="b"');
  });
  await check('Inline number validation',async()=>{
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
  await check('Comparison tab keeps the same moment in the middle of both panes',async()=>{
    compareFixture={};
    await select('a');await wait('currentBase==="a"');
    await click('tabCompare');
    await wait('COMPARE_PANES.left.blocks?.length===100 && COMPARE_PANES.right.blocks?.length===41');
    assert.deepEqual(await evaluate('[$("compareSelLeft").value,$("compareSelRight").value]'),['v2','v1']);
    assert.equal(await evaluate('$("renderBar").parentElement.id'),'compareView');
    assert.equal(await evaluate('$("renderSpeakers").hidden && $("renderSave").hidden'),true);
    const times='(()=>{const l=compareCoordinate(comparePane("left")),r=compareCoordinate(comparePane("right"));return [l.time,r.time]})()';
    for (const [side,top] of [['Left',4000],['Right',2500],['Left',9000]]) {
      await evaluate(`$("compareText${side}").scrollTop=${top}`);
      await wait(`(()=>{const [l,r]=${times};return l>50 && Math.abs(l-r)<1.5})()`);
    }
    const [before]=await evaluate(times);
    compareRenders.length=0;
    await click('optPauses');
    await wait(`!$("renderBar").classList.contains("busy") && $("renderBar").classList.contains("dirty")`);
    assert.deepEqual(compareRenders.map(r=>r.version).sort(),['v1','v2']);
    assert.deepEqual(compareRenders[0].options,compareRenders[1].options,'Both panes use the same Show settings');
    assert.equal(compareRenders[0].options.pauses,false);
    const [after,right]=await evaluate(times);
    assert.ok(Math.abs(after-before)<1.5 && Math.abs(after-right)<1.5,'Re-rendering keeps the reading position');
    await click('tabTranscript');
    await wait('$("renderBar").parentElement.id==="transcriptView" && $("versionSel").value==="v2" && !$("renderBar").classList.contains("busy")');
    assert.equal(await evaluate('$("optPauses").checked'),false,'Show settings follow the user back to the Transcript tab');
    compareFixture.stored='v1';await evaluate('compareCanRender={}');
    await click('tabCompare');
    await wait('$("compareNoteRight").textContent.includes("no diarized JSON") && $("compareNoteLeft").textContent===""');
    compareFixture=null;
    await evaluate('setRenderOptions(RENDER_DEFAULTS);compareCanRender={}');
    await click('tabSettings');
  });
  await check('Comparison playback follows the audio in both panes and in the review lists',async()=>{
    compareFixture={};
    await click('tabCompare');
    await wait('COMPARE_PANES.left.blocks?.length===100 && COMPARE_PANES.right.blocks?.length===41 && !$("renderBar").classList.contains("busy")');
    assert.equal(await evaluate('!$("compareAudioBar").hidden && $("compareAudio").getAttribute("src").startsWith("/api/audio/a")'),true);
    // A speaker heading seeks to its paragraph's start and plays.
    await evaluate('$("compareAudio").load()');await wait('$("compareAudio").readyState>=3');
    await evaluate('$("compareTextLeft").querySelectorAll(".compareSeek")[0].click()');
    await wait('!$("compareAudio").paused && $("compareAudio").currentTime>0.2');
    await evaluate('$("compareAudio").pause()');
    // The fixture spans 1000 s but the mock audio lasts 10 s, so pretend the player is at 400 s.
    await evaluate('Object.defineProperty($("compareAudio"),"currentTime",{get:()=>400,set:()=>{},configurable:true});Object.defineProperty($("compareAudio"),"paused",{get:()=>false,configurable:true})');
    const middles='[compareCoordinate(comparePane("left")).time,compareCoordinate(comparePane("right")).time]';
    const tick='$("compareAudio").dispatchEvent(new Event("timeupdate"))';
    await evaluate(tick);
    let [left,right]=await evaluate(middles);
    assert.ok(Math.abs(left-400)<2 && Math.abs(right-400)<2,`Both panes follow playback (${left}, ${right})`);
    assert.deepEqual(await evaluate('[COMPARE_PANES.left.active.anchor.start,COMPARE_PANES.right.active.anchor.start]'),[400,400]);
    assert.equal(await evaluate('document.querySelectorAll("#compareView .compareParagraph.active").length'),2);
    // Scrolling by hand stops the view from jumping back until playback is resumed.
    await evaluate('$("compareTextLeft").scrollTop=0');
    await wait('$("compareAudioMsg").textContent.includes("Following paused")');
    await evaluate(tick);
    [left]=await evaluate(middles);
    assert.ok(left<100,'A suspended follow leaves the reader where they scrolled');
    await evaluate('$("compareAudio").dispatchEvent(new Event("play"))');
    [left,right]=await evaluate(middles);
    assert.ok(Math.abs(left-400)<2 && Math.abs(right-400)<2,'Play resumes following');
    // The review lists replace the text in both panes and follow the same way.
    await click('compareReview');
    assert.deepEqual(await evaluate('[...document.querySelectorAll("#compareTextLeft .segment, #compareTextRight .segment")].length'),280);
    await evaluate(tick);
    [left,right]=await evaluate(middles);
    assert.ok(Math.abs(left-400)<2 && Math.abs(right-400)<2,'Review lists follow playback');
    assert.deepEqual(await evaluate('[...document.querySelectorAll("#compareView .segment.active p")].map(p=>p.textContent)'),['v2 first half 40','v1 first half 16']);
    // A version without a diarized JSON has no segments and keeps its text.
    compareFixture.stored='v1';
    await evaluate('compareCanRender={};refreshComparison()');
    await wait('$("compareNoteRight").textContent.includes("no diarized JSON") && $("compareTextRight").querySelector(".compareParagraph") && $("compareTextLeft").querySelector(".segment")');
    await evaluate('delete $("compareAudio").currentTime;delete $("compareAudio").paused');
    await click('compareReview');
    compareFixture=null;
    await evaluate('compareCanRender={}');
    await click('tabSettings');
    assert.equal(await evaluate('$("compareAudio").paused'),true,'Leaving the tab stops its audio');
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
    await click('tabTranscript');await wait('!$("versionSel").disabled && renderAvailable');
    // Earlier Show choices now persist; explicitly restore the unfiltered timeline.
    await click('renderReset');await wait('$("timeline").children.length===2');await click('reviewAudio');
    await evaluate('$("audioPlayer").load()');await wait('$("audioPlayer").readyState>=3');
    await evaluate('$("timeline").querySelector("button").click()');await wait('$("audioPlayer").currentTime>=1');
    await evaluate('$("audioPlayer").pause();$("audioPlayer").currentTime=3.5;$("audioPlayer").dispatchEvent(new Event("timeupdate"))');
    await wait('$("timeline").querySelector(".segment.active")?.dataset.start==="3"');
    await click('reviewAudio');
  });
  for (const [tab, host, controller, audio] of [
    ['tabTranscript','transcriptWaveform','transcriptWaveform','audioPlayer'],
    ['tabCompare','comparisonWaveform','comparisonWaveform','compareAudio'],
  ]) {
    await check(`${tab}: waveform click seeks and forward/reverse drags select`,async()=>{
      await click(tab);await wait(`${controller}.duration===10 && $(${JSON.stringify(audio)}).readyState>=3`);
      await waveformDrag(host,.3);
      assert.ok(Math.abs(await evaluate(`$(${JSON.stringify(audio)}).currentTime`)-3)<.04);
      assert.equal(await evaluate(`${controller}.range`),null);
      for (const [from,to] of [[.2,.6],[.6,.2]]) {
        await waveformDrag(host,from,to);
        const range = await evaluate(`${controller}.range`);
        assert.ok(Math.abs(range.start-2)<.04 && Math.abs(range.end-6)<.04,JSON.stringify(range));
        assert.ok(Math.abs(await evaluate(`$(${JSON.stringify(audio)}).currentTime`)-2)<.04);
        assert.equal(await evaluate(`$(${JSON.stringify(audio)}).paused`),true);
      }
      const shot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(`/private/tmp/swhisper-${host}.png`,Buffer.from(shot.data,'base64'));
    });
    await check(`${tab}: selected audio stops, replays, and honors playback speed`,async()=>{
      await waveformDrag(host,.15,.21);
      const end = await evaluate(`${controller}.range.end`);
      for (const rate of [1,2]) {
        await evaluate(`$(${JSON.stringify(audio)}).playbackRate=${rate};$(${JSON.stringify(audio)}).play()`);
        await wait(`!$(${JSON.stringify(audio)}).paused`);
        await wait(`$(${JSON.stringify(audio)}).paused`);
        assert.ok(Math.abs(await evaluate(`$(${JSON.stringify(audio)}).currentTime`)-end)<.035);
      }
      await evaluate(`$(${JSON.stringify(audio)}).playbackRate=1`);
      await waveformDrag(host,.5);
      assert.equal(await evaluate(`${controller}.range`),null,'Click clears the previous stop boundary');
      await evaluate(`$(${JSON.stringify(audio)}).play()`);
      await wait(`$(${JSON.stringify(audio)}).currentTime>5.3`);
      await evaluate(`$(${JSON.stringify(audio)}).pause()`);
    });
    await check(`${tab}: keyboard selection and clear control`,async()=>{
      await waveformDrag(host,.3);
      for (let i=0;i<3;i++) await cdp('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',modifiers:8,windowsVirtualKeyCode:39});
      const range=await evaluate(`${controller}.range`);
      assert.ok(Math.abs(range.end-range.start-.3)<.01);
      await evaluate(`$(${JSON.stringify(host)}).querySelector('.waveformClear').click()`);
      assert.equal(await evaluate(`${controller}.range`),null);
      await waveformDrag(host,.2,.4);await key('Escape','Escape');
      assert.equal(await evaluate(`${controller}.range`),null);
    });
    await check(`${tab}: canceled drag restores selection and native seek clears it`,async()=>{
      await waveformDrag(host,.2,.4);
      const original=await evaluate(`${controller}.range`);
      const rect=await evaluate(`(()=>{const r=$(${JSON.stringify(host)}).querySelector('.waveformTrack').getBoundingClientRect();return {x:r.left,y:r.top+20,width:r.width}})()`);
      await cdp('Input.dispatchMouseEvent',{type:'mousePressed',x:rect.x+rect.width*.6,y:rect.y,button:'left',buttons:1,clickCount:1});
      await cdp('Input.dispatchMouseEvent',{type:'mouseMoved',x:rect.x+rect.width*.8,y:rect.y,button:'left',buttons:1});
      await evaluate(`${controller}.track.dispatchEvent(new PointerEvent('pointercancel'))`);
      await cdp('Input.dispatchMouseEvent',{type:'mouseReleased',x:rect.x+rect.width*.8,y:rect.y,button:'left',buttons:0,clickCount:1});
      assert.deepEqual(await evaluate(`${controller}.range`),original);
      await evaluate(`$(${JSON.stringify(audio)}).currentTime=8`);
      await wait(`${controller}.range===null`);
    });
    await check(`${tab}: zoom loads detail, preserves time and maps selection correctly`,async()=>{
      await waveformDrag(host,.4);
      const time=await evaluate(`$(${JSON.stringify(audio)}).currentTime`);
      await evaluate(`${controller}.zoomIn.click()`);
      await wait(`${controller}.detailStatus.textContent===""`);
      let [start,end]=await evaluate(`[${controller}.viewStart,${controller}.viewEnd]`);
      assert.ok(Math.abs(start-2)<.02 && Math.abs(end-7)<.02);
      assert.equal(await evaluate(`$(${JSON.stringify(audio)}).currentTime`),time,'Zoom does not seek');
      assert.ok(waveformRequests.some(r=>Math.abs(r.start-start)<.001 && Math.abs(r.end-end)<.001),'Server receives the visible interval');
      await waveformDrag(host,.2,.6);
      let range=await evaluate(`${controller}.range`);
      assert.ok(Math.abs(range.start-3)<.03 && Math.abs(range.end-5)<.03);
      await evaluate(`${controller}.fit.click()`);await wait(`${controller}.detailStatus.textContent===""`);
      [start,end]=await evaluate(`[${controller}.viewStart,${controller}.viewEnd]`);
      assert.ok(Math.abs(start-range.start)<.001 && Math.abs(end-range.end)<.001);
      assert.deepEqual(await evaluate(`${controller}.range`),range);
      assert.equal(await evaluate(`${controller}.overlay.style.width`),'100%');
      const screenshot=await cdp('Page.captureScreenshot',{format:'png'});
      await fs.writeFile(`/private/tmp/swhisper-${host}-zoom.png`,Buffer.from(screenshot.data,'base64'));
      await waveformDrag(host,.1,.35);
      range=await evaluate(`${controller}.range`);
      await evaluate(`$(${JSON.stringify(audio)}).play()`);
      await wait(`$(${JSON.stringify(audio)}).paused`);
      assert.ok(Math.abs(await evaluate(`$(${JSON.stringify(audio)}).currentTime`)-range.end)<.035,'Zoomed selection stops at its own end');
      await evaluate(`${controller}.full.click()`);
      assert.deepEqual(await evaluate(`[${controller}.viewStart,${controller}.viewEnd]`),[0,10]);
      assert.deepEqual(await evaluate(`${controller}.range`),range,'Full audio retains selection');
    });
    await check(`${tab}: panning, clipping and stale detail cannot move the playhead`,async()=>{
      await waveformDrag(host,.2,.4);
      const selected=await evaluate(`${controller}.range`);
      await evaluate(`${controller}.zoomIn.click()`);await wait(`${controller}.detailStatus.textContent===""`);
      await evaluate(`${controller}.position.value=5;${controller}.position.dispatchEvent(new Event('input'))`);
      assert.deepEqual(await evaluate(`[${controller}.viewStart,${controller}.viewEnd]`),[5,10]);
      assert.equal(await evaluate(`${controller}.overlay.hidden && ${controller}.playhead.hidden`),true);
      assert.deepEqual(await evaluate(`${controller}.range`),selected);
      assert.ok(Math.abs(await evaluate(`$(${JSON.stringify(audio)}).currentTime`)-selected.start)<.02);
      await waveformDrag(host,.5);
      assert.ok(Math.abs(await evaluate(`$(${JSON.stringify(audio)}).currentTime`)-7.5)<.03);
      detailDelay=300;
      const count=waveformRequests.length;
      await evaluate(`${controller}.zoomIn.click()`);
      for(let i=0;i<30 && waveformRequests.length===count;i++) await delay(20);
      assert.ok(waveformRequests.length>count);
      await evaluate(`${controller}.full.click()`);
      await delay(350);
      assert.deepEqual(await evaluate(`[${controller}.viewStart,${controller}.viewEnd]`),[0,10]);
      assert.equal(await evaluate(`${controller}.track.querySelector('svg').getAttribute('viewBox')`),'0 0 10 100');
      assert.equal(await evaluate(`${controller}.detailStatus.textContent`),'');
      detailDelay=0;
    });
  }
  await check('Speaker strips clip to zoom and preserve separate comparison versions',async()=>{
    await click('tabTranscript');await wait('!$("versionSel").disabled && transcriptWaveform.speakers.tracks.length===1');
    assert.equal(await evaluate('transcriptWaveform.speakers.tracks[0].lanes.length'),2);
    await click('optPauses');await wait('!$("renderBar").classList.contains("busy")');
    assert.equal(await evaluate('transcriptWaveform.speakers.tracks[0].lanes.length'),2,'Text filters do not erase speaker timing');
    await evaluate('transcriptWaveform.setViewport(2,2)');
    const lines=await evaluate('[...$("transcriptWaveform").querySelectorAll(".waveformSpeakerLine")].map(el=>[el.style.left,el.style.width])');
    assert.deepEqual(lines,[['0%','50%'],['50%','50%']]);
    compareFixture={stored:'v1'};await evaluate('compareCanRender={}');
    await click('tabCompare');await wait('COMPARE_PANES.right.result?.rendered===false && !$("renderBar").classList.contains("busy")');
    assert.deepEqual(await evaluate('comparisonWaveform.speakers.tracks.map(t=>t.label)'),['Left']);
    delete compareFixture.stored;
    await evaluate('compareCanRender={};refreshComparison()');await wait('comparisonWaveform.speakers.tracks.length===2');
    assert.deepEqual(await evaluate('comparisonWaveform.speakers.tracks.map(t=>t.label)'),['Left','Right']);
    noTranscript=true;
    await evaluate('loadComparison(currentBase)');
    assert.equal(await evaluate('comparisonWaveform.speakers.tracks.length'),0,'Removed versions clear stale strips');
    noTranscript=false;
    compareFixture=null;
    await evaluate('compareCanRender={};comparisonWaveform.full.click()');
    await click('tabTranscript');await wait('!$("versionSel").disabled');
    await evaluate('transcriptWaveform.full.click()');
    await click('tabCompare');await wait('!$("renderBar").classList.contains("busy")');
  });
  await check('Waveform selection survives re-rendering and switching tabs',async()=>{
    await waveformDrag('comparisonWaveform',.2,.4);
    await evaluate('comparisonWaveform.setViewport(1,5)');
    await click('optPauses');await wait('!$("renderBar").classList.contains("busy")');
    assert.ok(await evaluate('comparisonWaveform.range?.end>3.9'));
    assert.deepEqual(await evaluate('[comparisonWaveform.viewStart,comparisonWaveform.viewEnd]'),[1,6]);
    await click('tabTranscript');await wait('transcriptWaveform.duration===10 && !$("versionSel").disabled');
    await waveformDrag('transcriptWaveform',.3,.6);
    await evaluate('transcriptWaveform.setViewport(2,5)');
    await evaluate('$("versionSel").value="v1";$("versionSel").dispatchEvent(new Event("change"))');
    await wait('!$("versionSel").disabled && $("versionSel").value==="v1"');
    assert.ok(await evaluate('transcriptWaveform.range?.end>5.9'));
    assert.deepEqual(await evaluate('[transcriptWaveform.viewStart,transcriptWaveform.viewEnd]'),[2,7]);
    await click('tabCompare');await wait('!$("compareAudioBar").hidden');
    assert.ok(await evaluate('comparisonWaveform.range?.end>3.9'));
    await click('tabTranscript');await wait('!$("versionSel").disabled');
    assert.ok(await evaluate('transcriptWaveform.range?.end>5.9'));
    await wait('$("timeline").querySelector("button")');
    await evaluate('transcriptWaveform.full.click()');
    await evaluate('$("timeline").querySelector("button").click()');
    assert.equal(await evaluate('transcriptWaveform.range'),null,'Timestamp playback clears the stop boundary');
    await evaluate('$("audioPlayer").pause()');
  });
  await check('Waveform resets on file changes, ignores stale loads and offers retry',async()=>{
    waveformDelay.b=350;
    await select('b');await wait('currentBase==="b"');
    assert.equal(await evaluate('comparisonWaveform.range'),null);
    await select('a');await wait('currentBase==="a" && transcriptWaveform.base==="a" && transcriptWaveform.duration===10');
    await delay(400);
    assert.equal(await evaluate('transcriptWaveform.base'),'a');
    assert.equal(await evaluate('transcriptWaveform.range'),null);
    assert.deepEqual(await evaluate('[transcriptWaveform.viewStart,transcriptWaveform.viewEnd]'),[0,10]);
    waveformDelay={};waveformFailure=true;
    await select('b');await wait('!transcriptWaveform.retry.hidden');
    assert.match(await evaluate('transcriptWaveform.status.textContent'),/Audio controls still work/);
    assert.equal(await evaluate('$("audioPlayer").getAttribute("src").startsWith("/api/audio/b")'),true);
    waveformFailure=false;
    await evaluate('transcriptWaveform.retry.click()');await wait('transcriptWaveform.duration===10');
    assert.equal(await evaluate('transcriptWaveform.retry.hidden'),true);
    assert.equal(await evaluate('transcriptWaveform.speakers.tracks.length'),1,'Retry keeps speaker data');
  });
  await check('Zoom bounds and unavailable detail preserve working playback controls',async()=>{
    waveformFailure=true;
    await evaluate('transcriptWaveform.zoomIn.click()');
    await wait('transcriptWaveform.detailStatus.textContent.includes("Detail unavailable")');
    await waveformDrag('transcriptWaveform',.5);
    assert.ok(Math.abs(await evaluate('$("audioPlayer").currentTime')-2.5)<.03);
    waveformFailure=false;
    await evaluate('transcriptWaveform.full.click();transcriptWaveform.zoomIn.click()');
    await wait('transcriptWaveform.detailStatus.textContent===""');
    await evaluate('for(let i=0;i<20;i++) transcriptWaveform.zoomIn.click()');
    assert.equal(await evaluate('transcriptWaveform.zoomIn.disabled'),true);
    assert.ok(Math.abs(await evaluate('transcriptWaveform.viewEnd-transcriptWaveform.viewStart')-.25)<.001);
    await evaluate('transcriptWaveform.full.click()');
    assert.equal(await evaluate('transcriptWaveform.zoomOut.disabled'),true);
  });
  await check('Waveform fits narrow tabs and pointer capture clamps at the edges',async()=>{
    await cdp('Emulation.setDeviceMetricsOverride',{width:680,height:1000,deviceScaleFactor:1,mobile:false});
    await waveformDrag('transcriptWaveform',.7,-.1);
    const range=await evaluate('transcriptWaveform.range');
    assert.equal(range.start,0);assert.ok(Math.abs(range.end-7)<.05);
    assert.equal(await evaluate('document.documentElement.scrollWidth<=window.innerWidth'),true);
    await evaluate('$("audioBar").scrollIntoView({block:"end"})');
    const shot=await cdp('Page.captureScreenshot',{format:'png'});
    await fs.writeFile('/private/tmp/swhisper-waveform-narrow.png',Buffer.from(shot.data,'base64'));
    await cdp('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate('transcriptWaveform.clearRange()');
  });
  await check('Audio waveform remains usable in raw JSON and without a transcript',async()=>{
    await evaluate('$("versionSel").value="__raw_json__";$("versionSel").dispatchEvent(new Event("change"))');
    await wait('!$("versionSel").disabled');
    assert.equal(await evaluate('!$("audioBar").hidden && transcriptWaveform.duration===10'),true);
    noTranscript=true;
    await select('<unsafe>');await wait('$("transcriptText").textContent.includes("No transcript yet")');
    await wait('transcriptWaveform.duration===10');
    await waveformDrag('transcriptWaveform',.1,.2);
    assert.ok(await evaluate('transcriptWaveform.range?.end>1.9'));
    await click('tabCompare');await wait('$("compareTextLeft").textContent.includes("No transcript yet")');
    await wait('comparisonWaveform.duration===10');
    assert.equal(await evaluate('$("compareAudioBar").hidden'),false);
    noTranscript=false;
    await select('a');await wait('editorBase==="a"');
    await click('tabTranscript');await wait('!$("versionSel").disabled && renderAvailable');
  });
  await check('Export uses native Save As and cancellation writes nothing',async()=>{
    await click('renderSave');await wait('$("exportDialog").open');
    assert.equal(await evaluate('$("exportFormat").value'),'txt');assert.equal(exports.length,0);
    await click('exportConfirm');await wait('!exporting && !$("exportDialog").open');
    assert.equal(exports.at(-1).token,'native-save-selection');
    exportCanceled=true;const before=exports.length;await click('renderSave');await wait('$("exportDialog").open');await click('exportConfirm');await wait('!exporting && !$("exportDialog").open');assert.equal(exports.length,before);exportCanceled=false;await click('tabSettings');
  });
  await check('Folder change cancel preserves folder and edits',async()=>{await select('b');await wait('editorBase==="b"');await input('folder draft');await evaluate('$("dirInput").value="/mock/other"; $("dirBtn").click()');await wait('$("dirtyDialog").open');await click('dirtyCancel');await wait('!$("dirtyDialog").open');assert.equal(audioDir,'/mock/audio');assert.equal(await evaluate('$("dirInput").value'),'/mock/audio');});
  await check('Dirty page requests unload warning',async()=>{
    assert.equal(await evaluate('(()=>{const e=new Event("beforeunload",{cancelable:true});window.dispatchEvent(e);return e.defaultPrevented})()'),true);
  });
  await check('Save error is visible from Transcript tab',async()=>{
    const n=launches.length;
    await click('tabTranscript');failSave=true;await click('startBtn');await wait('$("dirtyDialog").open');await click('dirtySave');
    await wait('!saving && !$("uiNotice").hidden');assert.equal(await evaluate('currentTab'),'transcript');assert.equal(launches.length,n);failSave=false;
  });
  await check('Run asks to save, then previews folder before launch',async()=>{const launchedBefore=launches.length;await click('startBtn');await wait('$("dirtyDialog").open');await click('dirtySave');await wait('$("previewDialog").open');assert.equal(launches.length,launchedBefore);assert.equal(saved.b.comment,'folder draft');assert.equal(await evaluate('$("previewTable").querySelectorAll("tbody tr").length'),1);const shot=await cdp('Page.captureScreenshot',{format:'png'});await fs.writeFile('/private/tmp/swhisper-run-preview.png',Buffer.from(shot.data,'base64'));await click('previewStart');await wait('lastStatus?.running && !runBusy');assert.equal(launches.at(-1).preview_token,'preview-false--');assert.equal(await evaluate('$("dirMenuBtn").disabled'),true);});
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
  await check('Browse waits for a folder chosen after the normal request timeout',async()=>{
    // Scale every client timeout down 1000x, so 15 s becomes 15 ms, and answer
    // the dialog after 200 ms, i.e. 200 s in real time.
    await evaluate('META.audio_dir="/mock/audio";$("dirInput").value="/mock/audio";window.__realTimeout=AbortSignal.timeout;AbortSignal.timeout=ms=>window.__realTimeout.call(AbortSignal,ms/1000)');
    pickDelay=200;
    try {
      await click('dirBrowse');
      await wait('!changingDir && $("dirInput").value==="/mock/picked"');
      await wait('META.audio_dir==="/mock/picked"');
    } finally {
      pickDelay=0;
      await evaluate('AbortSignal.timeout=window.__realTimeout');
      audioDir='/mock/audio';
    }
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
      if (await evaluate('$("showOtherSettings").checked')) await click('showOtherSettings');
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
  const reopen = async () => {
    await cdp('Page.navigate',{url:'about:blank'});
    await wait('location.href==="about:blank"');
    await cdp('Page.navigate',{url:`http://127.0.0.1:${server.address().port}/`});
    await wait('typeof initialized !== "undefined" && initialized && connected');
  };
  const showPreferences = {stats:true,preamble:false,disfluencies:false,pauses:false,pause_linebreaks:false,pause_linebreak_threshold:2.4,min_pause:0.6};
  await check('Show choices survive files, versions, raw JSON, folders and tabs',async()=>{
    await select('a');await wait('editorBase==="a"');await click('tabTranscript');
    await wait('!$("versionSel").disabled && renderAvailable');
    for (const [id,checked] of [['optStats',true],['optPreamble',false],['optDisfluencies',false],['optPauses',false],['optBreaks',false]]) {
      if (await evaluate(`$(${JSON.stringify(id)}).checked!==${checked}`)) await click(id);
    }
    for (const [id,value] of [['optThreshold',2.4],['optMinPause',0.6]]) {
      await evaluate(`$(${JSON.stringify(id)}).value=${value};$(${JSON.stringify(id)}).dispatchEvent(new Event('input'))`);
    }
    assert.deepEqual(await evaluate('JSON.parse(localStorage.getItem("swhisper_render_options"))'),showPreferences,'Numeric changes persist before the debounce fires');
    for (const version of ['__raw_json__','v1']) {
      await evaluate(`$("versionSel").value=${JSON.stringify(version)};$("versionSel").dispatchEvent(new Event("change"))`);
      await wait(`!$("versionSel").disabled && $("versionSel").value===${JSON.stringify(version)}`);
      assert.deepEqual(await evaluate('renderOptions()'),showPreferences);
    }
    await select('b');await wait('editorBase==="b" && !$("versionSel").disabled');
    assert.deepEqual(await evaluate('renderOptions()'),showPreferences);
    await click('tabCompare');await wait('COMPARE_PANES.left.result && !$("renderBar").classList.contains("busy")');
    assert.deepEqual(await evaluate('renderOptions()'),showPreferences);
    await click('tabTranscript');await wait('!$("versionSel").disabled');
    await evaluate('$("dirInput").value="/mock/other";changeDir()');await wait('META.audio_dir==="/mock/other" && !changingDir');
    assert.equal(await evaluate('currentTab'),'transcript','Changing folders preserves the chosen view');
    await select('a');await wait('editorBase==="a"');await click('tabTranscript');await wait('!$("versionSel").disabled');
    assert.deepEqual(await evaluate('renderOptions()'),showPreferences);
  });
  await check('Show choices restore in a fresh page and are used for rendering',async()=>{
    await reopen();
    assert.deepEqual(await evaluate('renderOptions()'),showPreferences);
    await select('a');await wait('editorBase==="a"');await click('tabTranscript');
    await wait('$("transcriptText").textContent==="Rendered 0.6"');
    assert.deepEqual(await evaluate('renderOptions()'),showPreferences);
  });
  await check('Reset persists the original view as the new Show preference',async()=>{
    await click('renderReset');await wait('!$("versionSel").disabled && !$("renderBar").classList.contains("busy")');
    const reset = await evaluate('renderOptions()');
    assert.deepEqual(reset,await evaluate('renderBaseline'));
    assert.deepEqual(await evaluate('JSON.parse(localStorage.getItem("swhisper_render_options"))'),reset);
    await reopen();
    assert.deepEqual(await evaluate('renderOptions()'),reset);
  });
  await check('Invalid saved Show settings safely fall back to valid defaults',async()=>{
    await evaluate('localStorage.setItem("swhisper_render_options", "broken JSON")');
    await reopen();
    assert.equal(await evaluate('editorBase'),'a','The selected file is restored too');
    assert.deepEqual(await evaluate('renderOptions()'),await evaluate('renderBaseline'));
    await evaluate('localStorage.setItem("swhisper_render_options", JSON.stringify({stats:false,pauses:"false",min_pause:-1,pause_linebreak_threshold:100}))');
    await reopen();
    assert.deepEqual(await evaluate('renderOptions()'),await evaluate('({...RENDER_DEFAULTS,stats:false})'));
  });
  await check('Run preferences survive metadata refresh and a fresh session',async()=>{
    for (const [id,value] of [['pipelineSel','scribe'],['runModel','nemotron'],['scribeScope','folder'],['scribeLanguage','Swedish']]) {
      await evaluate(`$(${JSON.stringify(id)}).value=${JSON.stringify(value)};$(${JSON.stringify(id)}).dispatchEvent(new Event('change'))`);
    }
    await wait('document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\')');
    await evaluate('document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\').checked || document.querySelector(\'[data-base="a"] .fileForce input[aria-label^="Re-transcribe"]\').click()');
    await evaluate('loadMeta()');
    assert.equal(await evaluate('$("pipelineSel").value'),'scribe');
    await reopen();
    assert.deepEqual(await evaluate('["pipelineSel","runModel","scribeScope","scribeLanguage"].map(id=>$(id).value)'),['scribe','nemotron','folder','Swedish']);
    assert.equal(await evaluate('retranscribeBases.size'),0,'Re-transcribe belongs to one run and one session');
    assert.equal(await evaluate('$("scribeControls").hidden'),false);
    assert.equal(await evaluate('$("startBtn").disabled'),false,'Restored folder scope can run without selecting a recording');
    consentFixture=true;
    await click('startBtn');await wait('$("previewDialog").open');
    assert.equal(await evaluate('pendingPreview.diarization_model'),'nemotron');
    consentFixture=false;
    assert.equal(await evaluate('$("scribeConsent").checked'),false,'Upload consent still belongs to each individual run');
    await click('previewCancel');
  });
  await check('An attached run does not replace the preferred pipeline',async()=>{
    status={...status,running:true,pipeline:'transcribe4',run_id:'other-window',started_at:Date.now()/1000};
    await reopen();
    assert.equal(await evaluate('$("pipelineSel").value'),'transcribe4');
    assert.equal(await evaluate('$("pipelineSel").disabled'),true);
    assert.equal(await evaluate('JSON.parse(localStorage.getItem("swhisper_ui_preferences")).pipelineSel'),'scribe');
    status={...status,running:false,returncode:0};
    await evaluate('pollRun()');
    assert.equal(await evaluate('$("pipelineSel").value'),'scribe');
    assert.equal(await evaluate('$("pipelineSel").disabled'),false);
    await evaluate('localStorage.setItem("swhisper_ui_preferences", JSON.stringify({pipelineSel:"removed-pipeline",runModel:"removed-model"}))');
    await reopen();
    assert.equal(await evaluate('$("pipelineSel").value'),'transcribe4');
    assert.equal(await evaluate('$("runModel").value'),'community-1','An unknown saved model falls back to the saved default');
  });
  await check('File filters, view toggles, export format and active tab survive reopening',async()=>{
    for (const [id,value] of [['fileSearch','a'],['fileFilter','ready'],['fileSort','newest'],['exportFormat','rtf']]) {
      await evaluate(`$(${JSON.stringify(id)}).value=${JSON.stringify(value)};$(${JSON.stringify(id)}).dispatchEvent(new Event(${JSON.stringify(id==='fileSearch'?'input':'change')}))`);
    }
    for (const [id,checked] of [['compareSync',false],['compareFollow',false],['compareReview',true]]) {
      if (await evaluate(`$(${JSON.stringify(id)}).checked!==${checked}`)) await click(id);
    }
    await click('tabTranscript');
    await reopen();
    assert.deepEqual(await evaluate('["fileSearch","fileFilter","fileSort","exportFormat"].map(id=>$(id).value)'),['a','ready','newest','rtf']);
    assert.equal(await evaluate('document.querySelectorAll(".fileSelect").length'),1);
    assert.deepEqual(await evaluate('["compareSync","compareFollow","compareReview"].map(id=>$(id).checked)'),[false,false,true]);
    assert.equal(await evaluate('currentTab'),'transcript');
    await select('a');await wait('editorBase==="a" && renderAvailable && !$("versionSel").disabled');
    await click('renderSave');await wait('$("exportDialog").open');
    assert.equal(await evaluate('$("exportFormat").value'),'rtf','Opening Export does not reset the format');
    await click('exportCancel');
    await evaluate('$("fileSearch").value="";$("fileSearch").dispatchEvent(new Event("input"))');
  });
  await check('Settings visibility and group expansion persist across files and sessions',async()=>{
    await click('tabSettings');
    if (!await evaluate('$("showOtherSettings").checked')) await click('showOtherSettings');
    await evaluate('(()=>{const group=document.querySelector("#editor details.group");if(group.open)group.querySelector("summary").click()})()');
    await select('b');await wait('editorBase==="b"');
    assert.equal(await evaluate('$("showOtherSettings").checked'),true);
    assert.equal(await evaluate('document.querySelector("#editor details.group").open'),false);
    await reopen();await select('a');await wait('editorBase==="a"');
    assert.equal(await evaluate('$("showOtherSettings").checked'),true);
    assert.equal(await evaluate('document.querySelector("#editor details.group").open'),false);
  });
  await check('Timestamp review remains selected through transcript and raw JSON navigation',async()=>{
    await click('tabTranscript');await wait('!$("reviewAudio").disabled');
    if (!await evaluate('$("reviewAudio").checked')) await click('reviewAudio');
    await select('b');await wait('editorBase==="b" && !$("versionSel").disabled && !$("reviewAudio").disabled');
    assert.equal(await evaluate('$("reviewAudio").checked && !$("timeline").hidden && $("transcriptText").style.display==="none"'),true);
    await evaluate('$("versionSel").value="__raw_json__";$("versionSel").dispatchEvent(new Event("change"))');
    await wait('$("transcriptText").textContent.includes("raw")');
    assert.equal(await evaluate('$("transcriptText").style.display'),'', 'Versions without a timeline stay readable');
    await reopen();await select('a');await wait('editorBase==="a" && !$("reviewAudio").disabled');
    assert.equal(await evaluate('$("reviewAudio").checked && !$("timeline").hidden'),true);
  });
  await check('Playback volume, mute and speed survive new audio and fresh sessions',async()=>{
    await evaluate('$("audioPlayer").volume=0.4;$("audioPlayer").muted=true;$("audioPlayer").playbackRate=1.5;$("compareAudio").volume=0.6;$("compareAudio").playbackRate=0.75');
    await wait('uiPreferences["audioPlayer.rate"]===1.5 && uiPreferences["compareAudio.rate"]===0.75 && uiPreferences["audioPlayer.muted"]===true');
    await reopen();await select('a');await wait('editorBase==="a" && !$("reviewAudio").disabled');
    assert.deepEqual(await evaluate('[ $("audioPlayer").volume,$("audioPlayer").muted,$("audioPlayer").playbackRate,$("compareAudio").volume,$("compareAudio").playbackRate ]'),[0.4,true,1.5,0.6,0.75]);
  });
  const reloadPage = async () => {
    await evaluate('window.__beforeReload=true');
    await cdp('Page.reload');
    await wait('!window.__beforeReload && typeof initialized !== "undefined" && initialized && connected');
  };
  await check('Tab persistence: file shortcuts and global settings survive reload',async()=>{
    await evaluate('$("fileSearch").value="";$("fileFilter").value="all";renderFiles()');
    await select('a');await wait('editorBase==="a"');
    for (const [selector,tab] of [['.tbadge','transcript'],['.lbadge','logs'],['.jbadge','transcript']]) {
      await click('tabSettings');
      await evaluate(`document.querySelector('[data-base="a"] ${selector}').click()`);
      await wait(`currentTab===${JSON.stringify(tab)}`);
      assert.equal(await evaluate('JSON.parse(localStorage.getItem("swhisper_ui_preferences")).activeTab'),tab);
      await reloadPage();
      assert.equal(await evaluate('currentTab'),tab);
      await select('a');await wait('editorBase==="a"');
    }
    await click('globalSettingsBtn');await wait('editorScope==="global" && !editorLoading');
    await reloadPage();
    assert.equal(await evaluate('currentTab'),'settings');
  });
  await check('Tab persistence: every tab button and keyboard navigation survive reload',async()=>{
    for (const [button,tab] of [['tabCompare','compare'],['tabLogs','logs'],['tabTranscript','transcript'],['tabSettings','settings']]) {
      await click(button);await reloadPage();
      assert.equal(await evaluate('currentTab'),tab);
      assert.equal(await evaluate(`$(${JSON.stringify(button)}).getAttribute('aria-selected')`),'true');
    }
    await evaluate('$("tabSettings").focus()');await key('ArrowRight','ArrowRight');
    await reloadPage();assert.equal(await evaluate('currentTab'),'transcript');
  });
  await check('Tab persistence: unrelated writes keep the latest tab saved by another window',async()=>{
    await click('tabSettings');
    // Simulate another window choosing Comparison while this document still has Settings in memory.
    await evaluate('localStorage.setItem(UI_STORAGE_KEY, JSON.stringify({...readUIPreferences(),activeTab:"compare"}));saveUIPreference("fileSort","name")');
    await reloadPage();
    assert.equal(await evaluate('currentTab'),'compare');
  });
  await check('Tab persistence: the saved view restores while the server is unavailable',async()=>{
    await click('tabCompare');
    await cdp('Network.enable');
    await cdp('Network.setBlockedURLs',{urls:['*/api/meta*']});
    try {
      await evaluate('window.__beforeReload=true');
      await cdp('Page.reload');
      await wait('!window.__beforeReload && typeof currentTab !== "undefined" && document.querySelector("#tabCompare").getAttribute("aria-selected")==="true"');
      assert.equal(await evaluate('currentTab'),'compare');
      assert.equal(await evaluate('initialized'),false);
    } finally {
      await cdp('Network.setBlockedURLs',{urls:[]});
      await evaluate('pollLoop()');await wait('initialized && connected');
    }
  });
  await check('Older server metadata explains the restart and recovers without a page reload',async()=>{
    legacyMetadata=true;
    try {
      await evaluate('window.__beforeReload=true');
      await cdp('Page.reload');
      await wait('!window.__beforeReload && document.querySelector("#runStatus").textContent.includes("python webui.py --restart")');
      assert.equal(await evaluate('initialized'),false);
      assert.equal(await evaluate('$("startBtn").disabled'),true);
    } finally { legacyMetadata=false; }
    await wait('initialized && connected');
    assert.equal(await evaluate('Object.keys(FILES_BY_BASE).length'),3);
    assert.equal(await evaluate('$("connectionStatus").textContent'),'Connected');
    assert.doesNotMatch(await evaluate('$("runStatus").textContent'),/older code|--restart/);
  });
  await check('File persistence: reload restores the file and tab without requiring a visible search result',async()=>{
    await evaluate('$("fileSearch").value="";$("fileSearch").dispatchEvent(new Event("input"));$("fileFilter").value="all";$("fileFilter").dispatchEvent(new Event("change"))');
    await select('b');await wait('editorBase==="b"');await click('tabTranscript');
    await evaluate('$("fileSearch").value="a";$("fileSearch").dispatchEvent(new Event("input"))');
    await reloadPage();
    assert.equal(await evaluate('currentBase'),'b');
    assert.equal(await evaluate('editorBase'),'b');
    assert.equal(await evaluate('currentTab'),'transcript');
    await wait('!$("versionSel").disabled');
    assert.equal(await evaluate('document.querySelector("[data-base=b]")'),null,'The filter is preserved independently of the open file');
    await evaluate('$("fileSearch").value="";$("fileSearch").dispatchEvent(new Event("input"))');
    await click('tabSettings');await input('unsaved selection guard');await select('a');await wait('$("dirtyDialog").open');
    await click('dirtyCancel');
    assert.equal(await evaluate('readUIPreferences()[`selectedFile.${META.audio_dir}`]'),'b','Canceled navigation does not change the saved file');
    await select('a');await wait('$("dirtyDialog").open');await click('dirtyDiscard');await wait('editorBase==="a"');
  });
  await check('File persistence: each folder restores its own selected recording',async()=>{
    await evaluate('$("dirInput").value="/mock/audio";changeDir()');
    await select('b');await wait('editorBase==="b"');
    await evaluate('$("dirInput").value="/mock/other";changeDir()');
    await select('a');await wait('editorBase==="a"');
    await evaluate('$("dirInput").value="/mock/audio";changeDir()');
    assert.equal(await evaluate('currentBase'),'b');
    assert.equal(await evaluate('editorBase'),'b');
    await reloadPage();assert.equal(await evaluate('editorBase'),'b');
    await evaluate('$("dirInput").value="/mock/other";changeDir()');
    assert.equal(await evaluate('editorBase'),'a');
  });
  await check('File persistence: a user selection supersedes slow restoration',async()=>{
    await select('b');await wait('editorBase==="b"');sidecarDelay.b=300;
    await evaluate('window.__beforeReload=true');await cdp('Page.reload');
    await wait('!window.__beforeReload && typeof currentBase !== "undefined" && currentBase==="b" && editorLoading');
    await select('a');await wait('editorBase==="a" && initialized && connected');
    sidecarDelay.b=0;
    assert.equal(await evaluate('currentBase'),'a');
    assert.equal(await evaluate('readUIPreferences()[`selectedFile.${META.audio_dir}`]'),'a');
    await reloadPage();assert.equal(await evaluate('editorBase'),'a');
  });
  await check('File persistence: deleted files and invalid saved values are ignored',async()=>{
    for (const value of ['missing-recording','__proto__',42]) {
      await evaluate(`saveUIPreference('selectedFile.'+META.audio_dir,${JSON.stringify(value)})`);
      await reloadPage();
      assert.equal(await evaluate('currentBase'),null);
      assert.equal(await evaluate('editorBase'),null);
    }
  });
  assert.deepEqual(errors,[],'No uncaught browser errors');
  console.log('All browser workflow checks passed.');
} finally {
  socket?.close();
  if(target)await fetch(`http://127.0.0.1:${port}/json/close/${target.id}`).catch(()=>{});
  server.close();server.closeAllConnections();
}
