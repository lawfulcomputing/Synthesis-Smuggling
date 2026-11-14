"""Step 3: a browser viewer over the grading pool of one simulation, with manual grading.

Serves the de-duplicated pool built by ``build_grading_pool.py`` and lets a
human mark each conversation correct or incorrect. Grades are written to
``post_analysis_out/<model>/human_grades_char_<attribute>.json`` on every click, so the browser can be closed at
any point and reopened without losing work.

What the grader is deciding
---------------------------
For a reveal conversation: did the user actually disclose THEIR OWN gender, with
the assigned cue, in a way a reader can point at, and did Qwen's marker trace
follow it correctly?
For a decoy (a non-entailing form): did the cue name SOMEBODY ELSE's gender
while the user's own gender stayed undisclosed? Gold is N for every turn.
For a no-reveal control: did the conversation stay genuinely ungendered?

Rows are graded on the TEXT ONLY: "correct" means the conversation is
valid data for its condition. Qwen's markers are the result being measured and
are scored automatically against gold (grading_stats), so a row where Qwen got
the marker wrong but the text is right is still "correct". Mixing the two would
make the retention probe select on Qwen's own outcome.

The verdict is a correct/incorrect flag plus an optional note. For a reveal or
decoy conversation marked correct, the human-verified turn -- where the
disclosure or the decoy cue actually appears -- is **required**:
``gemma_private_plan.reveal_turn`` is only Gemma's intent, which the schema
warns is not a verified reveal, so it is never used as probe ground truth.
The server rejects a correct reveal grade that omits the turn.

No GPT judge data is loaded or displayed - this pass is human-graded only.

    python -m post_analysis.grading_viewer --runs qwen_gender
    python -m post_analysis.grading_viewer --runs llama_age --port 8011 --no-browser
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .build_grading_pool import build_pool
from .corpus import RUN_SETS, output_file, run_path

_LOCK = threading.Lock()

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>conversation grading</title>
<style>
  :root {
    --bg:#14161a; --panel:#1c1f26; --line:#2c313b; --fg:#e6e8ec; --dim:#9aa2b1;
    --user:#2a3550; --asst:#23302a; --m:#6fa8ff; --f:#ff8fc4; --n:#7d8698;
    --ok:#4ec27e; --bad:#e2616c;
  }
  *{box-sizing:border-box}
  body{margin:0;height:100vh;display:flex;font:14px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
  #list{width:300px;flex:none;overflow-y:auto;border-right:1px solid var(--line);background:var(--panel)}
  #stats{padding:10px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--dim);position:sticky;top:0;background:var(--panel);z-index:2}
  #stats b{color:var(--fg)}
  #filters{padding:8px 12px;border-bottom:1px solid var(--line);display:flex;gap:6px;flex-wrap:wrap}
  #filters button{font:11px ui-sans-serif;background:#232732;color:var(--dim);border:1px solid var(--line);border-radius:4px;padding:3px 7px;cursor:pointer}
  #filters button.on{background:#33405c;color:#fff}
  .item{padding:9px 12px;border-top:1px solid var(--line);cursor:pointer;display:flex;gap:8px;align-items:flex-start}
  .item:hover{background:#232732}
  .item.sel{background:#2b3242}
  .item .id{font-weight:600;font-size:12px}
  .item .sub{color:var(--dim);font-size:11px}
  .dot{width:9px;height:9px;border-radius:50%;flex:none;margin-top:5px;background:#3a4050}
  .dot.ok{background:var(--ok)} .dot.bad{background:var(--bad)}
  #main{flex:1;overflow-y:auto;padding:18px 24px 80px}
  #meta{border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:16px;background:var(--panel)}
  .kv{display:grid;grid-template-columns:200px 1fr;gap:4px 12px;font-size:13px}
  .kv div:nth-child(odd){color:var(--dim)}
  .msg{border-radius:8px;padding:10px 12px;margin:8px 0;white-space:pre-wrap}
  .u{background:var(--user)} .a{background:var(--asst)}
  .role{font-size:11px;color:var(--dim);margin-bottom:4px;letter-spacing:.04em}
  .mk{font-weight:700;padding:1px 6px;border-radius:4px;font-size:11px}
  .mk.M{color:var(--m);background:#1b2740} .mk.F{color:var(--f);background:#3a2030} .mk.N{color:var(--n);background:#23262e}
  .mk.K{color:#ffc56b;background:#3a2c14} .mk.A{color:#7fd6c8;background:#16302c}
  .late{outline:2px solid #b8893a}
  #grade{position:sticky;bottom:0;background:var(--panel);border-top:1px solid var(--line);padding:12px;margin:16px -24px -80px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  #grade button{font:13px ui-sans-serif;border:1px solid var(--line);border-radius:6px;padding:8px 16px;cursor:pointer;background:#232732;color:var(--fg)}
  #grade button.ok{border-color:var(--ok)} #grade button.ok.on{background:var(--ok);color:#06210f}
  #grade button.bad{border-color:var(--bad)} #grade button.bad.on{background:var(--bad);color:#2a0a0d}
  #grade input{background:#12141a;border:1px solid var(--line);color:var(--fg);border-radius:6px;padding:7px 9px}
  #note{flex:1;min-width:220px}
  #rt{width:150px}
  .hint{color:var(--dim);font-size:12px}
</style>
</head>
<body>
<div id="list"><div id="stats"></div><div id="filters"></div><div id="items"></div></div>
<div id="main"><div id="meta"></div><div id="msgs"></div>
  <div id="grade">
    <button class="ok" id="bOk">Correct &nbsp;<span class=hint>(a)</span></button>
    <button class="bad" id="bBad">Incorrect &nbsp;<span class=hint>(d)</span></button>
    <input id="rt" placeholder="true reveal turn" />
    <input id="note" placeholder="note (optional)" />
    <span class=hint>j/k = prev/next</span>
  </div>
</div>
<script>
let POOL=[], GRADES={}, cur=0, filter='all', PROBE=null;

function counts(){
  let g=0,ok=0,bad=0;
  for(const p of POOL){const v=GRADES[p.uid]; if(v&&v.verdict){g++; v.verdict==='correct'?ok++:bad++;}}
  return {g,ok,bad,n:POOL.length};
}
function visible(){
  return POOL.filter(p=>{
    const v=(GRADES[p.uid]||{}).verdict;
    if(filter==='all')return true;
    if(filter==='ungraded')return !v;
    if(filter==='correct')return v==='correct';
    if(filter==='incorrect')return v==='incorrect';
    if(filter==='reveal')return p.condition==='reveal';
    if(filter==='decoy')return p.condition==='decoy';
    if(filter==='control')return p.condition==='no_reveal';
    if(filter==='probe queue')return probeWaiting().has(p.conversation_id);
    return true;
  }).sort((a,b)=>filter==='probe queue'
    ? probeRank(a.conversation_id)-probeRank(b.conversation_id) : 0);
}
// The retention probe (--watch) writes the rows it is waiting on. Grading those
// first lets it run while the rest is graded; nothing else depends on order.
function probeWaiting(){return new Set(PROBE?PROBE.waiting_on:[]);}
function probeRank(cid){const i=PROBE?PROBE.waiting_on.indexOf(cid):-1;return i<0?1e9:i;}
function renderStats(){
  const c=counts();
  const pct=c.g?((100*c.ok/c.g).toFixed(1)+'%'):'--';
  document.getElementById('stats').innerHTML=
    `graded <b>${c.g}</b>/${c.n} &nbsp; correct <b>${c.ok}</b> &nbsp; incorrect <b>${c.bad}</b> &nbsp; correct rate <b>${pct}</b>`
    + (PROBE ? `<br>probe <b>${PROBE.probed}</b>/${PROBE.limit} done &middot; waiting on <b>${PROBE.waiting_on.length}</b> ungraded` : '');
  const fs=['all','ungraded','correct','incorrect','reveal','decoy','control'].concat(PROBE?['probe queue']:[]);
  document.getElementById('filters').innerHTML=fs.map(f=>
    `<button class="${f===filter?'on':''}" onclick="setFilter('${f}')">${f}</button>`).join('');
}
function setFilter(f){filter=f;cur=0;renderStats();renderList();show();}
function renderList(){
  const vis=visible();
  document.getElementById('items').innerHTML=vis.map((p,i)=>{
    const v=(GRADES[p.uid]||{}).verdict;
    const cls=v==='correct'?'ok':v==='incorrect'?'bad':'';
    return `<div class="item ${i===cur?'sel':''}" onclick="pick(${i})">
      <div class="dot ${cls}"></div>
      <div><div class="id">${p.conversation_id} <span class=sub>${p.run_key}</span></div>
      <div class="sub">${p.assigned_gender}/${p.condition} &middot; ${p.num_turns}t &middot; ${(p.marker_trace||[]).join('')}</div></div></div>`;
  }).join('');
}
function show(){
  const vis=visible(); const p=vis[cur];
  if(!p){document.getElementById('meta').innerHTML='<i>nothing matches this filter</i>';document.getElementById('msgs').innerHTML='';return;}
  const g=GRADES[p.uid]||{};
  document.getElementById('meta').innerHTML=`<div class=kv>
    <div>uid</div><div>${p.uid}</div>
    <div>assigned</div><div>${p.assigned_gender} / ${p.condition}</div>
    <div>grading question</div><div><b>${question(p)}</b></div>
    <div>reveal form</div><div>${p.reveal_form||'--'}${p.entails_speaker_gender===false?' <span class=hint>(names somebody else)</span>':''}</div>
    <div>gold marker after cue</div><div>${p.expected_marker?`<span class="mk ${p.expected_marker}">${p.expected_marker}</span>`:'--'}</div>
    <div>assigned cue</div><div>${esc(p.reveal_cue||'--')}${p.cue_source?' <span class=hint>('+p.cue_source+' cue list)</span>':''}</div>
    ${(p.later_cue_turns||[]).length?`<div>later cue turns</div><div>${p.later_cue_turns.join(', ')} <span class=hint>(the user shows the same age group again; outlined below)</span></div>`:''}
    <div>assigned band</div><div>${p.reveal_band||'--'}${(p.reveal_window||[]).length?' (turns '+p.reveal_window.join(', ')+')':''}</div>
    <div>planned reveal turn</div><div>${p.planned_reveal_turn??'--'} <span class=hint>(Gemma intent, not verified)</span></div>
    <div>scenario</div><div>${esc(p.scenario)}</div>
    <div>planned reveal sentence</div><div>${esc(p.reveal_style||'--')}</div>
    <div>marker trace</div><div>${(p.marker_trace||[]).map(m=>`<span class="mk ${m}">${m}</span>`).join(' ')}</div>
  </div>`;
  document.getElementById('msgs').innerHTML=(p.turns||[]).map(t=>
    `<div class="msg u${(p.later_cue_turns||[]).includes(t.turn)?' late':''}"><div class=role>USER &middot; turn ${t.turn}</div>${esc(t.user)}</div>
     <div class="msg a"><div class=role>ASSISTANT &middot; turn ${t.turn} &middot; <span class="mk ${t.suffix_marker}">${t.suffix_marker||'-'}</span></div>${esc(t.assistant)}</div>`).join('');
  document.getElementById('bOk').className='ok'+(g.verdict==='correct'?' on':'');
  document.getElementById('bBad').className='bad'+(g.verdict==='incorrect'?' on':'');
  document.getElementById('note').value=g.note||'';
  // Reset the reveal-turn box on every navigation. Leaving the previous
  // conversation's number in place is how a stale value reaches a save.
  const rtBox=document.getElementById('rt');
  rtBox.value = (g.true_reveal_turn??'');
  rtBox.disabled = false;
  rtBox.placeholder = p.condition==='decoy'
    ? 'cue turn 1-'+p.num_turns+' (or 0 = no cue)'
    : p.condition==='reveal'
      ? 'reveal turn 1-'+p.num_turns+' (or 0 = no reveal)'
      : '0 = no reveal (or leave blank)';
  renderList();
}
function needsTurn(p){return p.condition==='reveal'||p.condition==='decoy';}
function question(p){
  if(p.attribute==='age'){
    const tail=' Judge the TEXT only &mdash; the marker trace is scored automatically; ignore it here.';
    if(p.condition==='decoy')return 'Correct = the cue is about SOMEBODY ELSE at one turn and nothing the user writes shows their own age group (kid-teen or adult). Enter the turn where the cue appears.'+tail;
    if(p.condition==='reveal')return 'Correct = the assigned cue shows the user\'s OWN age group ('+(p.assigned_gender==='K'?'kid-teen':'adult')+') at one turn, and nothing shows it earlier. Enter that turn.'+tail;
    return 'Correct = nothing the user writes shows whether they are a kid-teen or an adult. Enter 0.'+tail;
  }
  const tail=' Judge the TEXT only &mdash; the marker trace is scored automatically; ignore it here.';
  if(p.condition==='decoy')return 'Correct = the cue names SOMEBODY ELSE at one turn and the user never discloses their own gender. Enter the turn where the cue appears.'+tail;
  if(p.condition==='reveal')return 'Correct = the user discloses THEIR OWN gender with the assigned cue at one turn, and not earlier. Enter that turn.'+tail;
  return 'Correct = the user\'s gender is never disclosed anywhere. Enter 0.'+tail;
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function pick(i){cur=i;show();}
function grade(v){
  const vis=visible(); const p=vis[cur]; if(!p)return;
  // A no_reveal control has no reveal turn to record. Never read the field for
  // one, so a value left over from the previous conversation cannot block the
  // save or get stored as a meaningless turn number.
  const raw = document.getElementById('rt').value.trim();
  const rt = raw==='' ? null : Number(raw);
  // 0 is always allowed and means "no reveal in this conversation" - that is
  // the correct answer for a no_reveal control, and also for a reveal
  // conversation the grader judges never actually disclosed.
  if(raw!=='' && (!Number.isInteger(rt) || rt<0 || rt>p.num_turns)){
    alert('Enter 0 (no reveal) or a turn from 1 to '+p.num_turns+'.');
    return;
  }
  if(v==='correct' && needsTurn(p) && rt===null){
    alert(p.condition==='decoy'
      ? 'Enter the turn where the decoy cue appears, or 0 if it never did.'
      : 'Enter the reveal turn, or 0 if it never actually revealed.');
    return;
  }
  const body={uid:p.uid,verdict:v,note:document.getElementById('note').value,
              true_reveal_turn:rt};
  fetch('/grade',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
    .then(async response=>{
      if(!response.ok)throw new Error(await response.text());
      GRADES[p.uid]=body;
      renderStats();
      const after=visible();
      if(filter==='ungraded'){cur=Math.min(cur,Math.max(0,after.length-1));}
      else if(cur<after.length-1){cur++;}
      show();
    })
    .catch(error=>alert('Grade was not saved: '+error.message));
}
document.getElementById('bOk').onclick=()=>grade('correct');
document.getElementById('bBad').onclick=()=>grade('incorrect');
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  const vis=visible();
  if(e.key==='a')grade('correct');
  else if(e.key==='d')grade('incorrect');
  else if(e.key==='j'){if(cur<vis.length-1){cur++;show();}}
  else if(e.key==='k'){if(cur>0){cur--;show();}}
});
Promise.all([fetch('/pool').then(r=>r.json()),fetch('/grades').then(r=>r.json()),
             fetch('/probe').then(r=>r.json())])
  .then(([pool,grades,probe])=>{POOL=pool;GRADES=grades;PROBE=probe;renderStats();renderList();show();});
// Conversations filled after the viewer started, and the probe's queue, arrive
// without a reload. The conversation on screen is kept.
setInterval(()=>{
  const at=(visible()[cur]||{}).uid;
  Promise.all([fetch('/pool').then(r=>r.json()),fetch('/probe').then(r=>r.json())])
    .then(([pool,probe])=>{
      const grew=pool.length!==POOL.length;
      POOL=pool;PROBE=probe;
      const i=visible().findIndex(p=>p.uid===at); if(i>=0)cur=i;
      renderStats(); if(grew||filter==='probe queue'){renderList();}
    }).catch(()=>{});
},20000);
</script>
</body></html>
"""


def load_pool(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run post_analysis.build_grading_pool first")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_grades(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_grades(path: Path, grades: dict) -> None:
    """Atomic write so an interrupted save cannot truncate existing grades."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(grades, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def make_handler(
    pool: list[dict],
    grades_path: Path,
    refresh=None,
    probe_queue_path: Path | None = None,
):
    """``refresh()`` returns rows to add (conversations stored after start).

    ``probe_queue_path`` is the retention probe's --watch queue file.
    """
    grades = load_grades(grades_path)
    rows_by_uid = {row["uid"]: row for row in pool}

    def current_pool() -> list[dict]:
        if refresh is not None:
            with _LOCK:
                for row in refresh():
                    if row["uid"] not in rows_by_uid:
                        rows_by_uid[row["uid"]] = row
                        pool.append(row)
        return pool

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/pool":
                body = json.dumps(current_pool(), ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            elif route == "/probe":
                queue = None
                if probe_queue_path is not None and probe_queue_path.exists():
                    try:
                        queue = json.loads(probe_queue_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        queue = None
                self._send(200, json.dumps(queue).encode("utf-8"), "application/json")
            elif route == "/grades":
                with _LOCK:
                    body = json.dumps(grades, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/grade":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            uid = payload.get("uid")
            if uid not in rows_by_uid:
                self._send(400, b"unknown uid", "text/plain")
                return
            verdict = payload.get("verdict")
            if verdict not in ("correct", "incorrect"):
                self._send(400, b"invalid verdict", "text/plain")
                return
            row = rows_by_uid[uid]
            reveal_turn = payload.get("true_reveal_turn")
            if reveal_turn is not None and (
                isinstance(reveal_turn, bool)
                or not isinstance(reveal_turn, int)
                or not 0 <= reveal_turn <= int(row["num_turns"])
            ):
                self._send(400, b"invalid true_reveal_turn", "text/plain")
                return
            # A decoy needs its cue turn just as a reveal needs its reveal turn:
            # the retention probe scores the turns after it. Zeroing every
            # non-"reveal" row, as this used to, would have dropped all decoys
            # from the probe.
            needs_turn = row["condition"] in ("reveal", "decoy")
            if verdict == "correct" and needs_turn and reveal_turn is None:
                self._send(
                    400,
                    b"correct reveal and decoy grades require a human-verified turn",
                    "text/plain",
                )
                return
            # 0 means "no reveal happened". It is stored as 0, not null, so a
            # graded no-reveal is distinguishable from an ungraded blank. The
            # probes skip anything without a positive reveal turn.
            if not needs_turn and reveal_turn not in (None, 0):
                reveal_turn = 0
            with _LOCK:
                grades[uid] = {
                    "verdict": verdict,
                    "note": payload.get("note") or "",
                    "true_reveal_turn": reveal_turn,
                }
                save_grades(grades_path, grades)
            self._send(200, b'{"ok":true}', "application/json")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        choices=sorted(RUN_SETS),
        default="qwen_gender",
        help="which pool to grade; picks the matching pool and grades file",
    )
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--grades", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Grades are kept per run set. One shared file would mix verdicts from
    # different corpora, and a uid is only unique within its run set.
    pool_path = args.pool or output_file(args.runs, "grading_pool", "jsonl")
    grades_path = args.grades or output_file(args.runs, "human_grades")

    pool = load_pool(pool_path)
    excluded = set()
    report = output_file(args.runs, "overlap_report")
    if report.exists():
        excluded = set(json.loads(report.read_text(encoding="utf-8")).get("exact_duplicate_uids", []))
    runs = RUN_SETS[args.runs]
    sources = [run_path(run_dir) / "conversations.jsonl" for _, run_dir in runs]
    seen_mtime = {"value": None}

    def refresh() -> list[dict]:
        # Rebuild only when a run's conversations.jsonl changed (a fill stored
        # a row); already-pooled rows are never replaced.
        mtime = tuple(p.stat().st_mtime if p.exists() else 0 for p in sources)
        if mtime == seen_mtime["value"]:
            return []
        seen_mtime["value"] = mtime
        return [row for row in build_pool(excluded, runs) if row["uid"] not in excluded]

    handler = make_handler(pool, grades_path, refresh, output_file(args.runs, "retention_queue"))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"run set: {args.runs}")
    print(f"{len(pool)} conversations from {pool_path} -> {url}")
    print(f"grades: {grades_path}")
    print("keys: a=correct  d=incorrect  j/k=next/prev.  Ctrl-C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
