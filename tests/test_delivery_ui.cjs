const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const dist=path.join(__dirname,'..','dist');
const nodes=new Map(),timers=new Map(),intervals=new Map(),sources=[];let mono=0,seq=0,polls=0;
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',hidden:false,disabled:false,textContent:'',innerHTML:'',dataset:{},classList:{toggle(){}},setAttribute(){},addEventListener(){},replaceChildren(){},append(){},focus(){}});return nodes.get(id)}
class FakeEventSource{
 constructor(url){this.url=url;this.events={};sources.push(this)}
 addEventListener(name,fn){this.events[name]=fn}
 close(){this.closed=true}
 emit(data){this.onmessage({data:JSON.stringify(data)})}
}
const ctx={console,document:{documentElement:{},getElementById:node,querySelector:node,querySelectorAll:()=>[]},
 localStorage:{getItem:()=>null,setItem(){},removeItem(){}},EventSource:FakeEventSource,
 setTimeout:(fn,delay)=>{const id=++seq;timers.set(id,{fn,delay});return id},clearTimeout:id=>timers.delete(id),
 setInterval:fn=>{const id=++seq;intervals.set(id,fn);return id},clearInterval:id=>intervals.delete(id),
 performance:{now:()=>mono},Date:{now:()=>1000000+mono},URL,Blob,AbortController,
 fetch:()=>{polls++;return new Promise(()=>{})}};
ctx.window=ctx;vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(dist,'i18n.js'),'utf8'),ctx);
let source=fs.readFileSync(path.join(dist,'app.js'),'utf8');
source=source.replace('translate();controls();init();',`window.begin=()=>{info={streaming:true,gpu_configured:true};active='a'.repeat(32);startClock();pending();watchJob()};window.delivery=()=>result?.delivery;`);
vm.runInContext(source,ctx);
function advance(ms){mono+=ms;for(const fn of intervals.values())fn()}
const engine={solver_ms:776.4,bundle_wall_ms:783.4,median_ms:257.7,energy:2,decoded:{metrics:{},feasible:true},bits:[0,1]};
const result={problem:{variables:2,data_source:'synthetic'},hardware:{},engines:{quantum:engine},complete:false};
ctx.begin();const stream=sources.at(-1);
assert(stream.url.endsWith('/events'));assert.equal(polls,0);
stream.emit({status:'running',stage:'waiting_worker'});
assert.match(node('status').textContent,/calculation has not started/);
advance(5000);
stream.emit({status:'running',stage:'waiting_cpu',partial:result,worker_timeline:[{stage:'running',worker_ms:20}]});
assert.match(node('quantum').innerHTML,/783\.4 ms/);
assert.match(node('status').textContent,/CPU/);
assert.equal(ctx.delivery().first_visible_ms.quantum,5000);
assert.equal(ctx.delivery().transport,'server_sent_events');
assert.equal(polls,0);
advance(3000);
stream.emit({status:'completed',stage:'completed',result:{...result,complete:true,
 engines:{...result.engines,traditional:{...engine,bundle_wall_ms:4560}},comparison:{sb_speed_ratio:5.82}}});
assert.match(node('status').textContent,/Completed · Total request time: 8 s/);
assert(stream.closed);assert.equal(intervals.size,0);
assert.equal(ctx.delivery().first_visible_ms.quantum,5000);
assert.equal(ctx.delivery().first_visible_ms.traditional,8000);
advance(60000);assert.match(node('status').textContent,/8 s$/);
// A broken stream falls back to the existing authenticated GET endpoint.
ctx.begin();const broken=sources.at(-1);broken.onerror();
assert(broken.closed);assert.equal(polls,1);
broken.emit({status:'completed',result:{}}); // Late callbacks must be ignored.
assert.match(node('status').textContent,/0 s$/);
const css=fs.readFileSync(path.join(dist,'style.css'),'utf8');
assert.match(css,/#status\{font-size:14px;font-weight:750;/);
console.log('PASS: GPU streamed before CPU, truthful startup phase, first-visible timings, frozen final clock, stream failure fallback, larger bold status.');
