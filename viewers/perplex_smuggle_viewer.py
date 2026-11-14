"""Grade one run of the watermark email simulation by hand, with the detector's reading beside each email.

    python viewers/perplex_smuggle_viewer.py --run Evaluation/hash_gender/holdout/qwen --port 8021
    python viewers/perplex_smuggle_viewer.py --run Evaluation/hash_age/holdout/qwen    --port 8022

One viewer per run. It reads the run's conversations (a run that is still being written shows what is there and
picks up the rest), shows each conversation with the cue turn and the email request outlined, the email in its own
block, and under it what the detector read: the verdict against what it should be, the tokens counted, the hits
in each list against the number needed, the p-values, and the email with every list token coloured.

What you grade is the TEXT: is the conversation valid for its condition (see the grading question on the page).
The detector's reading is shown for you to look at and is scored by itself. Grades go to
<run>/human_grades.json on every click; `cli finalize` then adds tables for the conversations graded correct.

Standard library only; nothing is loaded on a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Src" / "holdout_hash" / "src"))

from perplex_smuggle.questions import grading_question  # noqa: E402

_LOCK = threading.Lock()

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg:#14161a; --panel:#1c1f26; --line:#2c313b; --fg:#e6e8ec; --dim:#9aa2b1;
    --user:#2a3550; --asst:#23302a; --ok:#4ec27e; --bad:#e2616c; --warn:#d6a444;
    --l0:#6fa8ff; --l0bg:#1f3358; --l1:#ff8fc4; --l1bg:#4a2338; --n:#7d8698;
  }
  body.age { --l0:#ffc56b; --l0bg:#4a3814; --l1:#7fd6c8; --l1bg:#16403a; }
  *{box-sizing:border-box}
  body{margin:0;height:100vh;display:flex;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
  #list{width:320px;flex:none;overflow-y:auto;border-right:1px solid var(--line);background:var(--panel)}
  #stats{padding:10px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--dim);position:sticky;top:0;background:var(--panel);z-index:2}
  #stats b{color:var(--fg)}
  #filters{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap}
  #filters button{font:11px ui-sans-serif;background:#232732;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:3px 7px;cursor:pointer}
  #filters button.on{background:#33405c;color:#fff}
  .item{padding:9px 12px;border-top:1px solid var(--line);cursor:pointer;display:flex;gap:8px;align-items:flex-start}
  .item:hover{background:#232732} .item.sel{background:#2b3242}
  .item .id{font-weight:600;font-size:12px} .item .sub{color:var(--dim);font-size:11px}
  .dot{width:9px;height:9px;border-radius:50%;flex:none;margin-top:5px;background:#3a4050}
  .dot.ok{background:var(--ok)} .dot.bad{background:var(--bad)}
  #main{flex:1;overflow-y:auto;padding:18px 24px 90px}
  #meta{border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:16px;background:var(--panel)}
  .kv{display:grid;grid-template-columns:230px 1fr;gap:4px 12px;font-size:13px}
  .kv>div:nth-child(odd){color:var(--dim)}
  .msg{border-radius:8px;padding:10px 12px;margin:8px 0;white-space:pre-wrap;overflow-wrap:anywhere}
  .u{background:var(--user)} .a{background:var(--asst)}
  .u.cue{outline:2px solid var(--warn)} .u.req{outline:2px solid #8c7bff}
  .probe{background:#2a2433;border:1px dashed #8c7bff;white-space:normal}
  .role{font-size:11px;color:var(--dim);margin-bottom:4px;letter-spacing:.04em}
  .tag{font-weight:700;padding:1px 6px;border-radius:4px;font-size:11px;background:#23262e;color:var(--n)}
  .tag.cue{color:#1d1503;background:var(--warn)} .tag.req{color:#0f0a2e;background:#a99cff}
  .tag.L0{color:var(--l0);background:var(--l0bg)} .tag.L1{color:var(--l1);background:var(--l1bg)}
  .tag.good{color:#06210f;background:var(--ok)} .tag.miss{color:#2a0a0d;background:var(--bad)}
  .email{margin:10px 0 4px;padding:10px 12px;border:1px dashed #6b7690;border-radius:8px;background:#1a211d;white-space:pre-wrap;line-height:1.75}
  .email .L0{background:var(--l0bg);color:var(--l0);border-radius:3px} .email .L1{background:var(--l1bg);color:var(--l1);border-radius:3px}
  .det{margin:6px 0 2px;padding:10px 12px;border:1px solid var(--line);border-radius:8px;background:#171a20;white-space:normal;font-size:13px}
  .det table{border-collapse:collapse;margin-top:6px} .det td,.det th{padding:2px 12px 2px 0;text-align:right;font-weight:400}
  .det th{color:var(--dim)} .det td:first-child,.det th:first-child{text-align:left}
  .det .line{margin-bottom:4px}
  #grade{position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line);padding:12px;margin:16px -24px -90px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  #grade button{font:13px ui-sans-serif;border:1px solid var(--line);border-radius:6px;padding:8px 16px;cursor:pointer;background:#232732;color:var(--fg)}
  #grade button.ok{border-color:var(--ok)} #grade button.ok.on{background:var(--ok);color:#06210f}
  #grade button.bad{border-color:var(--bad)} #grade button.bad.on{background:var(--bad);color:#2a0a0d}
  #grade input{background:#12141a;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:7px 9px}
  #grade label{font-size:12px;color:var(--dim);display:flex;gap:6px;align-items:center}
  #note{flex:1;min-width:200px} #rt,#et{width:70px}
  .hint{color:var(--dim);font-size:12px}
  .toggle{color:#8fb4ff;cursor:pointer;font-size:12px;white-space:nowrap}
</style>
</head>
<body class="__ATTRIBUTE__">
<div id="list"><div id="stats"></div><div id="filters"></div><div id="items"></div></div>
<div id="main"><div id="meta"></div><div id="msgs"></div>
  <div id="grade">
    <button class="ok" id="bOk">Correct &nbsp;<span class=hint>(a)</span></button>
    <button class="bad" id="bBad">Incorrect &nbsp;<span class=hint>(d)</span></button>
    <label>cue turn <input id="rt" /></label>
    <label>email asked at turn <input id="et" /></label>
    <input id="note" placeholder="note (optional)" />
    <label><input type="checkbox" id="colour" checked /> colour list tokens</label>
    <label><input type="checkbox" id="collapse" checked /> shorten long replies</label>
    <span class=hint>j/k = next/prev</span>
  </div>
</div>
<script>
let POOL=[], GRADES={}, INFO={}, cur=0, filter='all';
const OUT={right:'read right',no_mark:'no mark',wrong_list:'wrong list',undetermined:'both lists',
           unmarked_ok:'unmarked, as it should be',false_mark:'marked, should not be',no_email:'no email written'};

function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function L(name){const i=(INFO.lists||[]).indexOf(name);return i<0?'':'L'+i;}
function listTag(name){return name&&name!=='none'?`<span class="tag ${L(name)}">${name}</span>`:'<span class="tag">none</span>';}
function counts(){
  let g=0,ok=0,bad=0;
  for(const p of POOL){const v=GRADES[p.conversation_id]; if(v&&v.verdict){g++; v.verdict==='correct'?ok++:bad++;}}
  return {g,ok,bad,n:POOL.length};
}
function visible(){
  return POOL.filter(p=>{
    const v=(GRADES[p.conversation_id]||{}).verdict;
    if(filter==='all')return true;
    if(filter==='ungraded')return !v;
    if(filter==='correct')return v==='correct';
    if(filter==='incorrect')return v==='incorrect';
    if(filter==='cue')return p.condition==='reveal';
    if(filter==='turns between')return p.condition==='reveal'&&p.gap>0;
    if(filter==='decoy')return p.condition==='decoy';
    if(filter==='no cue')return p.condition==='no_reveal';
    if(filter==='detector off')return !p.detection.match;
    return true;
  });
}
function renderStats(){
  const c=counts();
  const hit=POOL.filter(p=>p.detection.match).length;
  document.getElementById('stats').innerHTML=
    `<b>${esc(INFO.run||'')}</b> &middot; ${esc(INFO.attribute||'')}<br>`
    +`graded <b>${c.g}</b>/${c.n} &nbsp; correct <b>${c.ok}</b> &nbsp; incorrect <b>${c.bad}</b><br>`
    +`detector as expected <b>${hit}</b>/${c.n}`+(INFO.scheduled&&INFO.scheduled>c.n?` &nbsp; <span class=hint>(${INFO.scheduled-c.n} not written yet)</span>`:'');
  const fs=['all','ungraded','correct','incorrect','cue','turns between','decoy','no cue','detector off'];
  document.getElementById('filters').innerHTML=fs.map(f=>
    `<button class="${f===filter?'on':''}" onclick="setFilter('${f}')">${f}</button>`).join('');
}
function setFilter(f){filter=f;cur=0;renderStats();renderList();show();}
function renderList(){
  const vis=visible();
  document.getElementById('items').innerHTML=vis.map((p,i)=>{
    const v=(GRADES[p.conversation_id]||{}).verdict;
    const cls=v==='correct'?'ok':v==='incorrect'?'bad':'';
    const d=p.detection;
    return `<div class="item ${i===cur?'sel':''}" onclick="pick(${i})">
      <div class="dot ${cls}"></div>
      <div><div class="id">${p.conversation_id} <span class=sub>${p.role} / ${p.group}</span></div>
      <div class="sub">${p.num_turns} turns${p.gap>0?' &middot; '+p.gap+' between':''} &middot;
        <span class="tag ${d.match?'good':'miss'}">${OUT[d.outcome]||d.outcome}</span></div></div></div>`;
  }).join('');
}
function emailHtml(e){
  const on=document.getElementById('colour').checked;
  if(!on||!(e.spans||[]).length)return esc(e.text);
  let out='',last=0;
  for(const [a,b,name] of e.spans){ if(a<last)continue; out+=esc(e.text.slice(last,a))+`<span class="${L(name)}">${esc(e.text.slice(a,b))}</span>`; last=b; }
  return out+esc(e.text.slice(last));
}
function detHtml(e,p,isRequest){
  const r=e.read, lists=INFO.lists||[];
  const want=isRequest?p.detection.expected:'none';
  const good=r.verdict===want;
  const rows=lists.map(n=>`<tr><td>${listTag(n)}</td><td>${r.green[n]}</td><td>${r.share[n]==null?'--':(100*r.share[n]).toFixed(1)+'%'}</td><td>${r.needed??'--'}</td><td>${Number(r.p[n]).toPrecision(3)}</td></tr>`).join('');
  return `<div class="det"><div class="line">detector reads ${r.verdict==='undetermined'?'<span class="tag miss">both lists</span>':listTag(r.verdict)}
      &nbsp; should read ${listTag(want)} &nbsp; <span class="tag ${good?'good':'miss'}">${good?'as expected':'not as expected'}</span>
      ${isRequest?'':'<span class=hint>&nbsp; (an email nobody asked for at this turn)</span>'}</div>
    <div class="line hint">${r.scored} distinct tokens counted &middot; ${e.words} words &middot; unmarked text has about 25% in each list
      ${r.lean?' &middot; leans to '+r.lean:''}${e.closed?'':' &middot; <b>closing tag missing</b>'}${e.second_email?' &middot; a second email follows':''}
      ${(e.defects||[]).length?' &middot; odd: '+e.defects.join(', '):''}</div>
    <table><tr><th>list</th><th>tokens in list</th><th>share</th><th>needed for a verdict</th><th>p-value (cut-off ${INFO.per_list_alpha})</th></tr>${rows}</table></div>`;
}
// A long ordinary reply is shown by its opening lines; "show all" opens it. The email itself is never cut.
function clip(text){
  text=(text||'').trim();
  if(!document.getElementById('collapse').checked||text.length<=420)return esc(text);
  let cut=text.lastIndexOf(' ',320); if(cut<200)cut=320;
  const rest=text.slice(cut), words=rest.trim().split(/\s+/).length;
  return `${esc(text.slice(0,cut))}<span class="more" hidden>${esc(rest)}</span><a class="toggle" onclick="unfold(this)"> &hellip; show all (${words} more words)</a>`;
}
function unfold(a){const more=a.previousElementSibling; more.hidden=!more.hidden; a.innerHTML=more.hidden?' &hellip; show all':' show less';}
function replyHtml(t,p){
  const e=t.email;
  if(!e)return clip(t.assistant);
  const open=INFO.tags[0], close=INFO.tags[1];
  const at=t.assistant.indexOf(open);
  const before=t.assistant.slice(0,at);
  const stop=t.assistant.indexOf(close,at+open.length);
  const after=stop<0?'':t.assistant.slice(stop+close.length);
  return clip(before)+`<div class="email">${emailHtml(e)}</div>`+detHtml(e,p,t.turn===p.email_turn)+clip(after);
}
// The probe: asked on the side before the email request, not part of the conversation.
function probeTag(v){return v==='unknown'||v==='unclear'?`<span class="tag">${v}</span>`:listTag(v);}
function probeHtml(p){
  const q=p.probe;
  return `<div class="msg probe"><div class=role>PROBE &middot; asked on the side before turn ${q.asked_before_turn}, not part of the conversation</div>
    <div class=hint>${esc(q.question)}</div>
    <div style="margin-top:6px">answers: ${q.answers.map((a,i)=>`<i>${esc(a.slice(0,160))}</i> ${probeTag(q.readings[i])}`).join(' &nbsp;|&nbsp; ')}</div>
    <div style="margin-top:6px">probe says ${probeTag(q.reading)} &nbsp; should say ${probeTag(q.expected)}
      &nbsp; <span class="tag ${q.match?'good':'miss'}">${q.match?'as expected':'not as expected'}</span></div>
    ${q.forced?`<div class=hint style="margin-top:8px">${esc(q.forced.question)}</div>
    <div style="margin-top:6px">answers: ${q.forced.answers.map((a,i)=>`<i>${esc(a.slice(0,160))}</i> ${probeTag(q.forced.readings[i])}`).join(' &nbsp;|&nbsp; ')}</div>
    <div style="margin-top:6px">asked to pick one, it says ${probeTag(q.forced.reading)}
      ${q.forced.expected?`&nbsp; the user is ${probeTag(q.forced.expected)} &nbsp; <span class="tag ${q.forced.match?'good':'miss'}">${q.forced.match?'right':'wrong'}</span>`:'<span class=hint>&nbsp; (a guess: nothing places the user)</span>'}</div>`:''}</div>`;
}
function show(){
  const vis=visible(); const p=vis[cur];
  if(!p){document.getElementById('meta').innerHTML='<i>nothing matches this filter</i>';document.getElementById('msgs').innerHTML='';return;}
  const g=GRADES[p.conversation_id]||{}, d=p.detection;
  document.getElementById('meta').innerHTML=`<div class=kv>
    <div>conversation</div><div>${p.conversation_id} &middot; ${p.num_turns} user turns</div>
    <div>persona / condition</div><div>${p.role}${p.persona_age?' ('+p.persona_age+' years)':''} / ${p.condition==='reveal'?'cue about the user':p.condition==='decoy'?'cue about somebody else':'no cue'} &middot; ${p.group}</div>
    <div>grading question</div><div><b>${p.question}</b> Grade the TEXT the user wrote; the detector's reading is scored by itself.</div>
    <div>reveal form</div><div>${p.form}</div>
    <div>assigned cue</div><div>${esc(p.cue||'--')}${p.cue_source?' <span class=hint>('+p.cue_source+' cue list)</span>':''}${p.cue_replaced_from?' <span class=hint>(filled later; the word first assigned was "'+esc(p.cue_replaced_from)+'")</span>':''}</div>
    <div>planned cue turn</div><div>${p.reveal_turn??'--'}</div>
    <div>user turns between cue and email request</div><div>${p.gap??'--'}</div>
    <div>planned email request turn</div><div>${p.email_turn} <span class=hint>(the last turn)</span></div>
    <div>email should read</div><div>${listTag(d.expected)}</div>
    ${p.probe?`<div>probe before the email request</div><div>says ${probeTag(p.probe.reading)}, should say ${probeTag(p.probe.expected)} <span class="tag ${p.probe.match?'good':'miss'}">${p.probe.match?'as expected':'not as expected'}</span></div>`:''}
    <div>detector</div><div>${d.email_written?(d.verdict==='undetermined'?'<span class="tag miss">both lists</span>':listTag(d.verdict)):'--'} <span class="tag ${d.match?'good':'miss'}">${OUT[d.outcome]||d.outcome}</span>
       ${(d.emails_before_the_request||[]).length?' <span class=hint>an email was also written at turn '+d.emails_before_the_request.join(', ')+'</span>':''}</div>
    <div>scenario</div><div>${esc(p.scenario)}</div>
    <div>planned cue sentence</div><div>${esc(p.reveal_style||'--')}</div>
    <div>planned email</div><div>to ${esc(p.email_recipient)} &middot; ${esc(p.email_purpose)}</div>
    ${p.rejected?`<div>attempts rejected before this one</div><div>${p.rejected}</div>`:''}
  </div>`;
  document.getElementById('msgs').innerHTML=(p.turns||[]).map(t=>{
    const cue=t.turn===p.reveal_turn, req=t.turn===p.email_turn;
    return (req&&p.probe?probeHtml(p):'')+`<div class="msg u${cue?' cue':''}${req?' req':''}"><div class=role>USER &middot; turn ${t.turn}
        ${cue?' <span class="tag cue">planned cue turn</span>':''}${req?' <span class="tag req">email request</span>':''}
        ${t.corrections?' <span class=hint>corrected '+t.corrections+'x</span>':''}</div>${esc(t.user)}</div>
     <div class="msg a"><div class=role>ASSISTANT &middot; turn ${t.turn}${t.finished?'':' &middot; <b>cut by the token limit</b>'}</div>${replyHtml(t,p)}</div>`;
  }).join('');
  document.getElementById('bOk').className='ok'+(g.verdict==='correct'?' on':'');
  document.getElementById('bBad').className='bad'+(g.verdict==='incorrect'?' on':'');
  document.getElementById('note').value=g.note||'';
  // Both boxes are set again on every conversation, so a number typed for the last one cannot be saved here.
  document.getElementById('rt').value = g.true_reveal_turn ?? (p.condition==='no_reveal'?0:'');
  document.getElementById('rt').placeholder = p.condition==='no_reveal'?'0':'1-'+p.num_turns+', 0 = none';
  document.getElementById('et').value = g.email_request_turn ?? p.email_turn;
  renderList();
}
function pick(i){cur=i;show();document.getElementById('main').scrollTop=0;}
function turnValue(id,p,what){
  const raw=String(document.getElementById(id).value).trim();
  if(raw==='')return {v:null};
  const n=Number(raw);
  if(!Number.isInteger(n)||n<0||n>p.num_turns){alert('Enter 0 (no '+what+') or a turn from 1 to '+p.num_turns+'.');return null;}
  return {v:n};
}
function grade(v){
  const vis=visible(); const p=vis[cur]; if(!p)return;
  const rt=turnValue('rt',p,'cue'), et=turnValue('et',p,'email request');
  if(!rt||!et)return;
  if(v==='correct' && p.condition!=='no_reveal' && rt.v===null){alert('Enter the turn where the cue appears, or 0 if it never did.');return;}
  if(v==='correct' && et.v===null){alert('Enter the turn where the email is asked for, or 0 if it never was.');return;}
  const body={conversation_id:p.conversation_id,verdict:v,note:document.getElementById('note').value,
              true_reveal_turn:rt.v,email_request_turn:et.v};
  fetch('/grade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(async response=>{
      if(!response.ok)throw new Error(await response.text());
      GRADES[p.conversation_id]=await response.json();
      renderStats();
      const after=visible();
      if(filter==='ungraded'){cur=Math.min(cur,Math.max(0,after.length-1));}
      else if(cur<after.length-1){cur++;}
      show();document.getElementById('main').scrollTop=0;
    })
    .catch(error=>alert('Grade was not saved: '+error.message));
}
document.getElementById('bOk').onclick=()=>grade('correct');
document.getElementById('bBad').onclick=()=>grade('incorrect');
document.getElementById('colour').onchange=()=>show();
document.getElementById('collapse').onchange=()=>show();
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  const vis=visible();
  if(e.key==='a')grade('correct');
  else if(e.key==='d')grade('incorrect');
  else if(e.key==='j'){if(cur<vis.length-1){pick(cur+1);}}
  else if(e.key==='k'){if(cur>0){pick(cur-1);}}
});
function load(first){
  return Promise.all([fetch('/pool').then(r=>r.json()),fetch('/grades').then(r=>r.json())]).then(([pool,grades])=>{
    const at=(visible()[cur]||{}).conversation_id, grew=pool.rows.length!==POOL.length;
    INFO=pool.info; POOL=pool.rows; if(first)GRADES=grades;
    const i=visible().findIndex(p=>p.conversation_id===at); if(i>=0)cur=i;
    renderStats(); if(first||grew){renderList();} if(first)show();
  });
}
load(true);
setInterval(()=>load(false).catch(()=>{}),20000);   // conversations written after the page was opened
</script>
</body></html>
"""


def pool_row(record: dict, question) -> dict:
    """One conversation as the page shows it."""
    plan, turns = record["gemma_private_plan"], []
    by_turn: dict[int, dict] = {}
    for message in record["messages"]:
        turn = by_turn.setdefault(message["turn"], {"turn": message["turn"]})
        if message["role"] == "user":
            turn.update(user=message["content"], corrections=message.get("corrections", 0))
        else:
            turn.update(assistant=message["content"], finished=message["finished"], email=message["email"])
    turns = [by_turn[k] for k in sorted(by_turn)]
    condition = record["assigned_reveal_condition"]
    return {
        "conversation_id": record["conversation_id"], "role": record["assigned_role"],
        "persona_age": record.get("assigned_persona_age"), "condition": condition, "group": record["assigned_reveal_group"],
        "form": record["assigned_reveal_form"], "cue": record["assigned_reveal_cue"], "cue_source": record.get("assigned_cue_source"),
        "cue_replaced_from": record.get("cue_replaced_from"),
        "reveal_turn": record["assigned_reveal_turn"], "gap": record["assigned_turns_between_cue_and_email_request"],
        "email_turn": record["assigned_email_turn"], "num_turns": record["num_turns"], "scenario": plan["scenario"],
        "reveal_style": plan["reveal_style"], "email_recipient": plan["email_recipient"], "email_purpose": plan["email_purpose"],
        "rejected": len(record.get("rejected_attempts") or []), "detection": record["detection"], "probe": record.get("probe"),
        "question": question(condition, record["assigned_role"]), "turns": turns,
    }


class Run:
    """A run folder: its conversations (read again when the folder changes) and its grades."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.attribute = self.manifest["attribute"]
        self.question = lambda condition, role: grading_question(self.attribute, condition, role)
        self.grades_path = run_dir / "human_grades.json"
        self.grades = json.loads(self.grades_path.read_text(encoding="utf-8")) if self.grades_path.exists() else {}
        self._seen, self._rows = None, []

    def rows(self) -> list[dict]:
        """The run's conversations: one file per conversation while a run is written (conversations/), else the
        finished run's conversations.jsonl (the form the repository keeps)."""
        paths = sorted((self.dir / "conversations").glob("sim_*.json"))
        if not paths and (self.dir / "conversations.jsonl").is_file():
            paths = [self.dir / "conversations.jsonl"]
        seen = tuple((p.name, p.stat().st_mtime_ns) for p in paths)
        if seen != self._seen:
            records = []
            for path in paths:
                text = path.read_text(encoding="utf-8")
                records += [json.loads(line) for line in text.splitlines() if line.strip()] if path.suffix == ".jsonl" \
                    else [json.loads(text)]
            self._rows = [pool_row(record, self.question) for record in records]
            self._seen = seen
        return self._rows

    def info(self) -> dict:
        detector = self.manifest["detector"]
        return {"run": self.dir.name, "attribute": self.attribute, "lists": detector["lists"], "tags": detector["tags"],
                "alpha": detector["alpha"], "per_list_alpha": detector["per_list_alpha"],
                "scheduled": self.manifest["requested_conversations"]}

    def save_grade(self, payload: dict) -> dict:
        rows = {row["conversation_id"]: row for row in self.rows()}
        row = rows.get(payload.get("conversation_id"))
        if row is None:
            raise ValueError("unknown conversation")
        verdict = payload.get("verdict")
        if verdict not in ("correct", "incorrect"):
            raise ValueError("verdict must be correct or incorrect")
        turns = {}
        for key in ("true_reveal_turn", "email_request_turn"):
            value = payload.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= row["num_turns"]):
                raise ValueError(f"invalid {key}")
            turns[key] = value
        if row["condition"] == "no_reveal" and turns["true_reveal_turn"] is None:
            turns["true_reveal_turn"] = 0                      # 0 = nothing shown, told apart from "not graded"
        if verdict == "correct" and (turns["true_reveal_turn"] is None or turns["email_request_turn"] is None):
            raise ValueError("a correct grade needs the cue turn and the email request turn")
        grade = {"verdict": verdict, "note": payload.get("note") or "", **turns}
        with _LOCK:
            self.grades[row["conversation_id"]] = grade
            tmp = self.grades_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.grades, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.grades_path)                      # an interrupted save cannot cut the file short
        return grade


def make_handler(run: Run):
    page = PAGE.replace("__TITLE__", f"{run.attribute} watermark grading: {run.dir.name}").replace("__ATTRIBUTE__", run.attribute)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, code: int = 200) -> None:
            self._send(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/pool":
                with _LOCK:
                    rows = run.rows()
                self._json({"info": run.info(), "rows": rows})
            elif route == "/grades":
                self._json(run.grades)
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/grade":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                self._json(run.save_grade(json.loads(self.rfile.read(length) or b"{}")))
            except ValueError as exc:
                self._send(400, str(exc).encode("utf-8"), "text/plain")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="the run folder, e.g. Evaluation/hash_gender/holdout/qwen")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run_dir = args.run if args.run.is_absolute() else (Path.cwd() / args.run)
    if not (run_dir / "manifest.json").is_file():
        run_dir = HERE / args.run
    if not (run_dir / "manifest.json").is_file():
        raise SystemExit(f"no run at {args.run}: manifest.json is missing")
    run = Run(run_dir.resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(run))
    print(f"{run.attribute} run {run.dir.name}: {len(run.rows())} of {run.manifest['requested_conversations']} conversations", flush=True)
    print(f"open http://{args.host}:{args.port}/", flush=True)
    print(f"grades: {run.grades_path}", flush=True)
    print("keys: a = correct, d = incorrect, j / k = next / previous. Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
