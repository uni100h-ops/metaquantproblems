// Contract test for the actual renderer; no browser or fabricated benchmark.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path'),dist=path.join(__dirname,'..','dist');
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{value:'',hidden:false,disabled:false,textContent:'',innerHTML:'',dataset:{},classList:{toggle(){}},setAttribute(){},addEventListener(){},replaceChildren(){},append(){},focus(){}});return nodes.get(id)}
const context={console,document:{documentElement:{},getElementById:node,querySelector:node,querySelectorAll:()=>[]},
  localStorage:{getItem:()=>null,setItem(){},removeItem(){}},setTimeout,clearTimeout,setInterval,clearInterval,
  URL,Blob,AbortController,performance,Date};
context.window=context;vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(dist,'i18n.js'),'utf8'),context);
// Expose only the render entry point in this isolated VM; disable startup HTTP.
let source=fs.readFileSync(path.join(dist,'app.js'),'utf8');
assert(source.includes('translate();controls();init();'));
source=source.replace('translate();controls();init();',
  'window.renderFixture=(value,language="en")=>{result=value;lang=language;renderResult()};');
vm.runInContext(source,context);
const engine=(sum,wall)=>({solver_ms:sum,bundle_wall_ms:wall,median_ms:5000,energy:1,
  decoded:{metrics:{},feasible:true},bits:[0,1]});
const fixture={problem:{variables:2,data_source:'synthetic'},hardware:{},complete:false,
  engines:{quantum:engine(1000,1100)}};
context.renderFixture(fixture);
assert.match(node('quantum').innerHTML,/1\.1 s/);
assert.match(node('comparison').textContent,/CPU/); // GPU result visible while CPU runs.
const complete={...fixture,complete:true,engines:{...fixture.engines,traditional:engine(15000,5500)},
  comparison:{sb_speed_ratio:15}};
context.renderFixture(complete);
assert.match(node('traditional').innerHTML,/5\.5 s/); // Actual wall time.
assert.match(node('traditional').innerHTML,/Sum of call durations: 15 s/); // Retained raw sum.
assert.match(node('comparison').textContent,/5×/); // Wall ratio, not overlapping duration sum.
for(const language of ['en','es','fr','de']){
  context.renderFixture(complete,language);
  assert(node('traditional').innerHTML.includes(context.MQ_I18N[language].elapsedTrials));
  assert(node('traditional').innerHTML.includes(context.MQ_I18N[language].sumTrials));
}
const legacy={...complete,engines:{traditional:engine(15000,undefined),quantum:engine(1000,undefined)}};
context.renderFixture(legacy);
assert.match(node('traditional').innerHTML,/15 s/);
assert.match(node('traditional').innerHTML,/Solver time/);
assert.match(node('comparison').textContent,/15×/);
console.log('PASS: elapsed vs summed time, partial GPU result, consistent ratio, four languages, legacy fallback.');
