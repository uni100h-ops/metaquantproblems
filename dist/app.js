(()=>{'use strict';
const $=id=>document.getElementById(id),D=window.MQ_I18N;
const store={get(k){try{return localStorage.getItem(k)}catch{return null}},set(k,v){try{localStorage.setItem(k,v)}catch{}},del(k){try{localStorage.removeItem(k)}catch{}}};
let lang=store.get('mq-language')||'en';if(!D[lang])lang='en';
let problemInput='';
let info=null,category='channels',preset=null,result=null,active=null,lastJob=null,working=false,started=0,timer=null;
let eventStream=null,streamWatchdog=null,firstSeen={};
let statusCode='ready',clockOrigin=null,clockOffset=0,elapsedMs=null,clockTimer=null;
const t=k=>D[lang][k]||D.en[k]||k,loc=o=>o?.[lang]||o?.en||'',esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,d=2)=>Number(x).toLocaleString({en:'en-GB',es:'es-ES',fr:'fr-FR',de:'de-DE'}[lang],{maximumFractionDigits:d}),ms=x=>x<1000?num(x,1)+' ms':num(x/1000,2)+' s';
function error(code,issues=[]){$('error').hidden=false;$('error').textContent=t(code)===code?t('error')+' ('+code+')':t(code);if(issues.length)$('error').textContent+=' '+issues.map(x=>x.field+': '+x.code).join('; ')}
function clearError(){$('error').hidden=true}
async function api(url,body,method){const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),75000);try{const r=await fetch(url,{method:method||(body?'POST':'GET'),credentials:'same-origin',headers:{'Content-Type':'application/json','X-Metaquant-Request':'1'},body:body?JSON.stringify(body):undefined,signal:controller.signal});let d;try{d=await r.json()}catch{const e=new Error('server_unavailable');e.http=r.status;throw e}if(!r.ok){let e=new Error(typeof d.detail==='string'?d.detail:d.detail?.code||'invalid_problem_data');e.issues=d.detail?.issues||[];e.providerStatus=d.detail?.provider_status;e.http=r.status;throw e}return d}finally{clearTimeout(timeout)}}
function requestId(){if(typeof crypto.randomUUID==='function')return crypto.randomUUID();const bytes=crypto.getRandomValues(new Uint8Array(16));bytes[6]=(bytes[6]&15)|64;bytes[8]=(bytes[8]&63)|128;const hex=Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');return `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`}
function controls(){const busy=working||!!active;const hasSpec=!!problemInput.trim(),hasText=!!$('intent').value.trim();$('run').disabled=busy||!info?.gpu_configured||!(hasSpec||hasText);$('intent').disabled=busy;$('problemSelect').disabled=busy;$('cancel').hidden=!active;$('explain').disabled=busy||!result?.complete||!info?.meta_configured;$('export').disabled=!result;$('csv').disabled=!result?.engines?.traditional||!result?.engines?.quantum;$('showEvidence').disabled=!result}
function planMessage(text=''){$('planMessage').textContent=text;$('planMessage').hidden=!text}
function resetResult(){resetClock();result=null;lastJob=null;for(const n of ['traditional','quantum']){$(n).innerHTML='<div class="empty">'+esc(t('empty'))+'</div>';document.querySelector('[data-download="'+n+'"]').hidden=true}$('cpuState').textContent='';$('gpuState').textContent='';$('comparison').textContent='';$('runSummary').textContent=t('same');$('explanation').textContent='';$('evidence').textContent='';$('hardware').textContent='';}
function stopPending(code){for(const name of ['traditional','quantum']){if(!result?.engines?.[name]){$(name).innerHTML='<div class="empty">'+esc(t(code))+'</div>';$(name==='traditional'?'cpuState':'gpuState').textContent=t(code)}}if(result&&!result.complete)$('comparison').textContent=t('incomplete')}
function renderStatus(){const suffix=elapsedMs===null?'':' · '+t('elapsed')+': '+num(Math.floor(elapsedMs/1000),0)+' s';$('status').textContent=t(statusCode)+suffix;$('status').classList.toggle('busy',working||!!active)}
function tickClock(){if(clockOrigin!==null)elapsedMs=clockOffset+performance.now()-clockOrigin;renderStatus()}
function startClock(wallStarted=Date.now()){clearInterval(clockTimer);started=Number.isFinite(wallStarted)?wallStarted:Date.now();clockOffset=Math.max(0,Date.now()-started);clockOrigin=performance.now();tickClock();clockTimer=setInterval(tickClock,250)}
function stopClock(){if(clockOrigin!==null)tickClock();clockOrigin=null;clearInterval(clockTimer);clockTimer=null;renderStatus()}
function resetClock(){stopClock();elapsedMs=null;renderStatus()}
function status(code){statusCode=code;renderStatus()}
function choose(p){if(!p||active||working)return;resetResult();preset=p;category=p.category;$('intent').value=presetPrompt(p);problemInput=JSON.stringify(p.spec,null,2);planMessage();clearError();renderCatalog();status('ready')}
function presetPrompt(p){
 const c=info.categories.find(c=>c.id===p.category);
 return [loc(c.title),num(p.spec.size,0)+' '+t('variables'),loc(c.description),loc(c.limits)].join(' · ');
}
function chooseOwn(){
 if(active||working)return;
 resetResult();clearError();preset=null;category='own';problemInput='';
 $('intent').value='';planMessage();renderCatalog();status('ready');$('intent').focus();
}
function renderCatalog(){
 if(!info)return;
 const select=$('problemSelect');select.replaceChildren();
 for(const c of info.categories){
  if(!info.presets.some(p=>p.category===c.id))continue;
  const option=document.createElement('option');option.value=c.id;option.textContent=t('category_'+c.id);select.append(option);
 }
 const own=document.createElement('option');own.value='own';own.textContent=t('own');select.append(own);
 select.value=preset?category:'own';
 const c=preset?info.categories.find(c=>c.id===category):null;
 $('categoryTitle').textContent=c?t('category_'+c.id):t('own');
 $('description').textContent=c?loc(c.description):t('ownHint');
 $('intent').placeholder=t('ownHint');
 $('exampleExplanation').textContent=c?t('example_'+c.id):t('ownExplanation');
 $('limits').textContent=c?loc(c.limits):'';$('limits').hidden=!c;
 $('scale').textContent=preset?num(preset.spec.size,0)+' '+t('variables')+' · '+t('synthetic'):'';
 $('scale').hidden=!preset;
 controls();
}
$('problemSelect').onchange=()=>{
 if(active||working)return;
 if($('problemSelect').value==='own')chooseOwn();
 else choose(info.presets.find(p=>p.category===$('problemSelect').value));
};
function translate(){document.documentElement.lang=lang;document.querySelectorAll('[data-t]').forEach(e=>e.textContent=t(e.dataset.t));document.querySelectorAll('[data-lang]').forEach(b=>b.classList.toggle('active',b.dataset.lang===lang));if(preset&&!active)$('intent').value=presetPrompt(preset);renderCatalog();if(result)renderResult();else if(!active&&!working)resetResult();if(!active&&!working)status('ready');renderConnection();renderStatus()}
function renderConnection(){if(!info)return;$('connection').textContent=(info.gpu_configured?t('configured'):t('noGPU'))+' · '+(info.meta_configured?t('aiAvailable'):t('aiUnavailable'));if(!info.gpu_configured&&!active)status('missing')}
function download(name,data,type='application/json'){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([typeof data==='string'?data:JSON.stringify(data,null,2)],{type}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000)}
function renderResult(){if(!result)return;const r=result;$('runSummary').textContent=num(r.problem.variables,0)+' '+t('variables')+' · '+(r.problem.data_source==='synthetic'?t('synthetic'):'User data');for(const name of ['traditional','quantum']){const e=r.engines[name];if(!e)continue;$(name==='traditional'?'cpuState':'gpuState').textContent=t('done');let metrics=Object.entries(e.decoded.metrics).map(([k,v])=>`<div class="metric-detail"><span>${esc(metricLabel(k))}</span><strong>${esc(num(v,3))}</strong></div>`).join('');$(name).innerHTML=`<div class="metrics"><div><span class="value">${esc(ms(e.bundle_wall_ms??e.solver_ms))}</span><span class="label">${esc(t(e.bundle_wall_ms!=null?'elapsedTrials':'time'))}</span></div><div><span class="value">${esc(num(e.energy,3))}</span><span class="label">${esc(t('cost'))}</span></div></div><p class="small muted">${esc(t('sumTrials'))}: ${esc(ms(e.solver_ms))} · ${esc(t('median'))}: ${esc(ms(e.median_ms))}</p><span class="valid ${e.decoded.feasible?'':'invalid'}">${esc(t(e.decoded.feasible?'feasible':'infeasible'))}</span>${metrics}<p class="small">${esc(t(e.at_proven_optimum?'optimum':'unknown'))}</p>`;document.querySelector('[data-download="'+name+'"]').hidden=false}if(r.complete){const cpuWall=r.engines.traditional?.bundle_wall_ms,gpuWall=r.engines.quantum?.bundle_wall_ms;const ratio=Number.isFinite(cpuWall)&&Number.isFinite(gpuWall)?cpuWall/Math.max(gpuWall,1e-9):r.comparison.sb_speed_ratio,fast=ratio>=1?t('gpu'):t('cpu');$('comparison').textContent=fast+' · '+num(ratio>=1?ratio:1/ratio)+'× '+t('faster')+'.'}else $('comparison').textContent=t(r.engines.quantum&&r.engines.traditional?'verifying':r.engines.quantum?'waiting_cpu':r.engines.traditional?'waiting_gpu':'running');$('evidence').textContent=JSON.stringify({problem:r.problem,reference:r.reference,protocol:r.protocol,comparison:r.comparison,hardware:r.hardware,delivery:r.delivery,solutions:Object.fromEntries(Object.entries(r.engines).map(([name,e])=>[name,e.decoded]))},null,2);$('hardware').textContent=JSON.stringify(r.hardware,null,2);controls()}
function metricLabel(k){const labels={remaining_interference:['Audience conflict cost','Coste de conflicto de público'],avoided_interference:['Avoided conflicts','Conflictos evitados'],channel_1_count:['Session A events','Eventos sesión A'],channel_2_count:['Session B events','Eventos sesión B'],load_a:['Zone A load','Carga zona A'],load_b:['Zone B load','Carga zona B'],load_difference:['Load difference','Diferencia de carga'],contention_cost:['Resource contention','Competencia por recursos'],balance_cost:['Balance cost','Coste de equilibrio'],benefit:['Restoration value','Valor de restauración'],pair_cost:['Interaction cost','Coste de interacción'],net_value:['Net value','Valor neto'],selected_count:['Selected actions','Actuaciones seleccionadas'],execution_cost:['Handling cost','Coste de manipulación'],interaction_cost:['Pairwise transport cost','Coste entre parejas'],pool_a_count:['Hub A shipments','Pedidos centro A'],pool_b_count:['Hub B shipments','Pedidos centro B']};return labels[k]?.[lang==='es'?1:0]||k.replaceAll('_',' ')}
function pendingStage(stage){for(const name of ['traditional','quantum']){if(result?.engines?.[name])continue;const code=['connecting','waiting_worker','initializing','warming_up','preparing_problem','receiving_result','cancelling'].includes(stage)?stage:'running';$(name).innerHTML='<div class="empty">'+esc(t(code))+'</div>';$(name==='traditional'?'cpuState':'gpuState').textContent=''}}
function pending(){firstSeen={};result=null;lastJob=null;for(const n of ['traditional','quantum']){$(n).innerHTML='<div class="empty">'+esc(t('connecting'))+'</div>';document.querySelector('[data-download="'+n+'"]').hidden=true}$('cpuState').textContent='';$('gpuState').textContent='';$('comparison').textContent='';$('explanation').textContent=t('explanationHint');$('evidence').textContent='';$('export').disabled=true;$('csv').disabled=true}
function finish(){closeStream();clearTimeout(timer);timer=null;lastJob=active;active=null;store.del('mq-active');stopClock();controls()}
function closeStream(){if(eventStream){eventStream.close();eventStream=null}clearTimeout(streamWatchdog);streamWatchdog=null}
function applyJob(job){
 if(!job||!job.status)return;
 const incoming=job.status==='completed'?job.result:job.partial;
 if(incoming){
  const now=clockOrigin===null?(elapsedMs||0):clockOffset+performance.now()-clockOrigin;
  for(const name of Object.keys(incoming.engines||{}))if(firstSeen[name]===undefined)firstSeen[name]=Math.round(now);
  result={...incoming,delivery:{gateway_timeline:job.gateway_timeline||[],worker_timeline:job.worker_timeline||[],gateway_elapsed_ms:job.gateway_elapsed_ms,first_visible_ms:{...firstSeen},transport:eventStream?'server_sent_events':'polling'}};
  renderResult();
 }
 if(job.status==='completed'){finish();status('completed')}
 else if(['failed','cancelled'].includes(job.status)){finish();status(job.status==='failed'?'failed':job.stage);stopPending(job.status==='failed'?'failed':'cancelled');if(job.status==='failed')error(job.error||'compute_failed')}
 else{const stage=job.stage==='running'&&result?.engines?.quantum&&!result?.engines?.traditional?'waiting_cpu':job.stage==='running'&&result?.engines?.traditional&&!result?.engines?.quantum?'waiting_gpu':job.stage;status(stage);pendingStage(stage)}
}
async function poll(){if(!active)return;const id=active;try{const job=await api('/api/jobs/'+id);if(active!==id)return;applyJob(job);if(active)timer=setTimeout(poll,1000)}catch(e){if(active!==id)return;if(e.http===404){finish();stopPending('job_expired');status('job_expired');error(e.message)}else{status('error');timer=setTimeout(poll,4000)}}}
function watchJob(){
 closeStream();clearTimeout(timer);timer=null;
 if(!active)return;
 if(!info?.streaming||typeof EventSource!=='function'){poll();return}
 const id=active;
 const fallback=()=>{if(active!==id)return;closeStream();poll()};
 const arm=()=>{clearTimeout(streamWatchdog);streamWatchdog=setTimeout(fallback,45000)};
 try{
  const source=new EventSource('/api/jobs/'+id+'/events');eventStream=source;
  streamWatchdog=setTimeout(fallback,8000);
  source.onopen=()=>{if(eventStream===source)arm()};
  source.addEventListener('heartbeat',()=>{if(eventStream===source)arm()});
  source.onmessage=event=>{if(active!==id||eventStream!==source)return;arm();try{applyJob(JSON.parse(event.data))}catch{fallback()}};
  source.onerror=()=>{if(eventStream===source)fallback()};
 }catch{fallback()}
}

$('intent').addEventListener('input',()=>{preset=null;category='own';problemInput='';resetResult();clearError();planMessage();renderCatalog();status('ready')});
document.querySelectorAll('[data-close]').forEach(b=>b.onclick=()=>$(b.dataset.close).close());
$('showEvidence').onclick=()=>$('evidenceDialog').showModal();
$('run').onclick=async()=>{
 if(working||active)return;
 clearError();working=true;startClock();controls();status('validating');
 try{
  let input=problemInput.trim();
  if(!input){
   const question=$('intent').value.trim();
   if(question.startsWith('{'))input=question;
   else{
    if(!info?.meta_configured)throw new Error('meta_not_configured');
    status('planning');
    const proposal=await api('/api/plan',{question,language:lang});
    planMessage(proposal.message||'');
    if(proposal.status!=='ready'){problemInput='';status('clarify');return}
    input=JSON.stringify(proposal.spec,null,2);problemInput=input;preset=null;
   }
  }
  const checked=await api('/api/validate',{question:input,language:lang});
  problemInput=JSON.stringify(checked.spec,null,2);
  const id=requestId();pending();active=id.replaceAll('-','');
  store.set('mq-active',JSON.stringify({id:active,started}));controls();status('connecting');
  const submitted=await api('/api/jobs',{request_id:id,spec:checked.spec,language:lang});applyJob(submitted);if(active)watchJob();
 }catch(e){
  if(e.http){if(active){finish();stopPending('failed')}error(e.message,e.issues);status('failed')}
  else if(active){watchJob()}
  else{error(e.message,e.issues);status('failed')}
 }finally{working=false;if(!active)stopClock();controls();renderStatus()}
};
$('cancel').onclick=async()=>{if(!active)return;try{await api('/api/jobs/'+active+'/cancel',{});status('cancelling')}catch(e){error(e.message)}};
$('explain').onclick=async()=>{
 if(!lastJob||working)return;
 clearError();working=true;controls();
 const box=$('explanation');
 box.textContent=t('explaining');
 box.classList.toggle('loading',true);
 box.setAttribute('aria-busy','true');
 try{
  $('explanationDialog').showModal();
  const d=await api('/api/jobs/'+lastJob+'/explain',{});
  box.textContent=d.text;result.meta_explanation=d;
 }catch(e){
  box.textContent=t(e.message)+(Number.isInteger(e.providerStatus)?' (HTTP '+e.providerStatus+')':'');
  error(e.message);
 }finally{
  box.classList.toggle('loading',false);
  box.setAttribute('aria-busy','false');
  working=false;controls();
 }
};
$('export').onclick=()=>{if(result)download('metaquantproblems-measurement.json',result)};
$('csv').onclick=()=>{if(!result?.engines.traditional||!result?.engines.quantum)return;const a=result.engines.traditional.bits,b=result.engines.quantum.bits;download('metaquantproblems-decisions.csv','decision,cpu_binary,gpu_binary\n'+a.map((v,i)=>`${i+1},${v},${b[i]}`).join('\n')+'\n','text/csv;charset=utf-8')};
document.querySelectorAll('[data-download]').forEach(b=>b.onclick=()=>{const name=b.dataset.download;if(result?.engines[name])download('metaquantproblems-'+name+'.json',{problem:result.problem,hardware:result.hardware,engine:result.engines[name],complete:result.complete})});
document.querySelectorAll('[data-lang]').forEach(b=>b.onclick=()=>{lang=b.dataset.lang;store.set('mq-language',lang);translate()});
$('technical').onclick=()=>$('technicalDialog').showModal();$('closeDialog').onclick=()=>$('technicalDialog').close();
async function init(){try{info=await api('/api/info');choose(info.presets[0]);translate();let saved;try{saved=JSON.parse(store.get('mq-active'))}catch{}if(saved&&/^[a-f0-9]{32}$/.test(saved.id)){active=saved.id;startClock(saved.started);pending();controls();status('connecting');watchJob()}}catch(e){error(e.message);status('error')}}
translate();controls();init();
})();
