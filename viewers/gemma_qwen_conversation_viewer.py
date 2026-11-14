"""Read-only viewer for the character-based simulations (Src/holdout_char).

With no --file argument, the viewer finds the most recently modified
Evaluation/char_<attribute>/holdout/<model>/conversations.jsonl on every
browser refresh, so a run in progress becomes the active dataset automatically.
Pass --file to pin the viewer to one run.

The view includes assigned reveal metadata, planned gold versus predicted
labels, private-plan provenance, privacy counters, and the exact marker-free
assistant text Gemma saw. Split-run placeholders are hidden by default.

Nothing here writes to disk or loads a model — it just renders the JSONL.

    python viewers/gemma_qwen_conversation_viewer.py
    python viewers/gemma_qwen_conversation_viewer.py --file Evaluation/char_gender/holdout/qwen/conversations.jsonl
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[1] / "Evaluation"   # runs: char_<attribute>/holdout/<model>

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Simulated conversations</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  /* Palette and card/bubble layout shared with viewer_unicode_data.py and
     checkpoint_viewer.py. */
  :root {
    --bg: #f4f4f5; --fg: #18181b; --panel: #fff; --border: #e4e4e7; --code-bg: #fafafa;
    --muted: #71717a; --faint: #a1a1aa; --user-bg: #dbeafe; --asst-bg: #f4f4f5;
    --m: #1d4ed8; --m-bg: #dbeafe; --f: #be185d; --f-bg: #fce7f3; --n: #52525b; --n-bg: #e4e4e7;
    --ok: #15803d; --ok-bg: #dcfce7; --bad: #b91c1c; --bad-bg: #fee2e2;
    --warn2: #a16207; --warn2-bg: #fef9c3;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #18181b; --fg: #f4f4f5; --panel: #27272a; --border: #3f3f46; --code-bg: #1f1f23;
      --muted: #a1a1aa; --faint: #71717a; --user-bg: #1e3a5f; --asst-bg: #3f3f46;
      --m: #93c5fd; --m-bg: #1e3a5f; --f: #f9a8d4; --f-bg: #500724; --n: #d4d4d8; --n-bg: #3f3f46;
      --ok: #4ade80; --ok-bg: #14321f; --bad: #fca5a5; --bad-bg: #450a0a;
      --warn2: #facc15; --warn2-bg: #3f3313;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 system-ui, -apple-system, sans-serif;
         background: var(--bg); color: var(--fg); }
  .layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 16px;
            max-width: 1320px; margin: 0 auto; padding: 16px; align-items: start; }
  @media (max-width: 860px) { .layout { grid-template-columns: minmax(0, 1fr); } }
  #list { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow-y: auto;
          display: flex; flex-direction: column; gap: 6px; padding: 10px;
          background: var(--panel); border: 1px solid var(--border); border-radius: 8px; }
  #list h1 { font-size: 13px; margin: 0 0 4px; font-weight: 600; overflow-wrap: anywhere; }
  #list h1 .src { display: block; font-size: 11px; font-weight: 400; color: var(--muted); }
  .item { padding: 7px 9px; border: 1px solid var(--border); border-radius: 6px;
          background: var(--bg); cursor: pointer; }
  .item:hover { border-color: var(--muted); }
  .item.sel { border-color: var(--m); }
  .item .id { font-size: 13px; font-weight: 700; overflow-wrap: anywhere; }
  .item .sub { color: var(--muted); font-size: 11px; }
  #list .flags { font-size: 11px; margin-top: 3px; display: flex; gap: 4px; flex-wrap: wrap; }
  #main { min-width: 0; max-width: 900px; width: 100%; margin: 0 auto; }
  #meta { border: 1px solid var(--border); border-radius: 8px; padding: 14px; margin-bottom: 16px;
          background: var(--panel); }
  #meta h2 { margin: 0 0 10px; font-size: 18px; text-align: center; }
  .kv { display: grid; grid-template-columns: 190px 1fr; gap: 4px 12px; font-size: 13px; }
  .kv div:nth-child(odd) { color: var(--muted); }
  details { margin-top: 10px; }
  summary { cursor: pointer; color: var(--muted); font-size: 12px; }
  pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0 0; background: var(--code-bg);
        border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 12px; }
  .chat { display: flex; flex-direction: column; gap: 8px; padding: 12px; background: var(--panel);
          border: 1px solid var(--border); border-radius: 8px; }
  .turn-no { margin-top: 4px; font-size: 11px; color: var(--faint); text-align: center; }
  .msg { max-width: 85%; padding: 8px 12px; border-radius: 12px; }
  .msg.user { align-self: flex-start; background: var(--user-bg); }
  .msg.assistant { align-self: flex-end; background: var(--asst-bg); }
  .msg.bad { outline: 2px solid var(--bad); }
  .msg .body { white-space: pre-wrap; overflow-wrap: anywhere; }
  .msg .hdr { font-size: 11px; color: var(--muted); margin-top: 6px;
              display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; }
  .msg pre { background: var(--panel); }
  .tag { border-radius: 4px; padding: 0 6px; font: 600 11px/1.6 ui-monospace, monospace; }
  .tag.M { color: var(--m); background: var(--m-bg); }
  .tag.F { color: var(--f); background: var(--f-bg); }
  .tag.N { color: var(--n); background: var(--n-bg); }
  .trace { display: flex; gap: 4px; flex-wrap: wrap; }
  .trace span { min-width: 22px; text-align: center; }
  .trace .miss { outline: 2px solid var(--bad); }
  .empty { color: var(--muted); font-style: italic; }
  .pill { border-radius: 4px; padding: 0 6px; font-size: 11px; font-weight: 700; }
  .pass { color: var(--ok); background: var(--ok-bg); }
  .fail { color: var(--bad); background: var(--bad-bg); }
  .decoy { color: var(--warn2); background: var(--warn2-bg); }
  .entail { color: var(--m); background: var(--m-bg); }
  .no-reveal { color: var(--n); background: var(--n-bg); }
  .gold { color: var(--muted); }
</style>
</head>
<body>
<div class="layout">
<aside id="list"></aside>
<main id="main"></main>
</div>
<script>
const DATA = __DATA__;
const SOURCE = __SOURCE__;
let sel = 0;

const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const tag = m => `<span class="tag ${esc(m)}">${esc(m)}</span>`;
const revealClass = c => c.assigned_reveal_form === "none" ? "no-reveal" : (c.reveal_form_entails_speaker_gender === false ? "decoy" : "entail");
const revealNote = c => c.assigned_reveal_form === "none" ? " &mdash; own gender stays undisclosed" : (c.reveal_form_entails_speaker_gender === false ? " &mdash; names somebody else, so the correct marker is N" : "");

// Gold marker per turn: N before the cue, expected_marker_after_reveal from the
// cue onward. For a decoy form that expected value is N for the whole thing.
function goldSeq(c) {
  const n = (c.marker_trace || []).length;
  const cue = (c.gemma_private_plan || {}).reveal_turn;
  const after = c.expected_marker_after_reveal ?? "N";
  if (!cue) return Array(n).fill("N");
  return Array.from({length: n}, (_, i) => (i + 1 < cue ? "N" : after));
}
function verdict(c) {
  const gold = goldSeq(c), trace = c.marker_trace || [];
  if (!trace.length) return {ok: 0, n: 0, pass: false};
  const ok = trace.filter((t, i) => t === gold[i]).length;
  return {ok, n: trace.length, pass: ok === trace.length};
}

function renderList() {
  const items = DATA.map((c, i) => {
    const v = verdict(c);
    const entails = c.reveal_form_entails_speaker_gender;
    const flags = [
      `<span class="pill ${v.pass ? "pass" : "fail"}">${v.ok}/${v.n}</span>`,
      c.assigned_reveal_form
        ? `<span class="pill ${revealClass(c)}">${esc(c.assigned_reveal_form)}</span>`
        : "",
    ].join("");
    return `
    <div class="item ${i === sel ? "sel" : ""}" data-i="${i}">
      <div class="id">${esc(c.conversation_id)}</div>
      <div class="sub">${esc(c.assigned_role_gender)} · ${esc(c.num_turns)} turns</div>
      <div class="flags">${flags}</div>
    </div>`;
  }).join("");
  const list = document.getElementById("list");
  const tot = DATA.length;
  const passed = DATA.filter(c => verdict(c).pass).length;
  list.innerHTML = `<h1>${passed}/${tot} conversations fully correct<span class="src">${esc(SOURCE)}</span></h1>` + items;
  list.querySelectorAll(".item").forEach(el =>
    el.onclick = () => { sel = +el.dataset.i; render(); window.scrollTo(0, 0); });
}

function renderMain() {
  const c = DATA[sel];
  if (!c) { document.getElementById("main").innerHTML = '<p class="empty">No conversations.</p>'; return; }
  const plan = c.gemma_private_plan || {};
  const models = c.models || {};
  const gold = goldSeq(c);
  const v = verdict(c);
  const trace = (c.marker_trace || [])
    .map((m, i) => `<span class="tag ${esc(m)}${m === gold[i] ? "" : " miss"}">${esc(m)}</span>`)
    .join("");
  const goldTrace = gold.map(m => tag(m)).join("");
  const entails = c.reveal_form_entails_speaker_gender;

  const meta = `
    <div id="meta">
      <h2>${esc(c.conversation_id)}</h2>
      <div class="kv">
        <div>verdict</div><div><span class="pill ${v.pass ? "pass" : "fail"}">${v.ok}/${v.n} turns correct</span></div>
        <div>assigned role gender</div><div>${tag(c.assigned_role_gender)}</div>
        <div>assigned reveal form</div><div>${c.assigned_reveal_form
            ? `<span class="pill ${revealClass(c)}">${esc(c.assigned_reveal_form)}</span>${revealNote(c)}`
            : "<i>none recorded</i>"}</div>
        <div>expected marker after cue</div><div>${tag(c.expected_marker_after_reveal ?? "N")}</div>
        <div>gold trace</div><div class="trace">${goldTrace}</div>
        <div>status / seed</div><div>${esc(c.status)} · seed ${esc(c.seed)}</div>
        <div>marker trace</div><div class="trace">${trace}</div>
        <div>scenario</div><div>${esc(plan.scenario)}</div>
        <div>planned turns / reveal turn</div><div>${esc(plan.planned_turns)} · reveal at ${esc(plan.reveal_turn)}</div>
        <div>reveal style</div><div>${esc(plan.reveal_style)}</div>
        <div>opening approach</div><div>${esc(plan.opening_approach)}</div>
        <div>plan attempts</div><div>${esc(c.gemma_plan_attempts)}</div>
        <div>models</div><div>${esc(models.gemma)}<br>${esc(models.qwen_base)}<br>${esc(models.qwen_adapter)}</div>
      </div>
      <details><summary>privacy counters</summary><pre>${esc(JSON.stringify(c.privacy, null, 2))}</pre></details>
      <details><summary>raw plan</summary><pre>${esc(c.gemma_raw_plan)}</pre></details>
    </div>`;

  const msgs = (c.messages || []).map(m => {
    const isUser = m.role === "user";
    const body = isUser ? m.content : m.content_raw;
    const bits = [];
    let wrong = false;
    if (!isUser) {
      const g = gold[m.turn - 1];
      wrong = g !== undefined && m.suffix_marker !== g;
      bits.push(`marker ${tag(m.suffix_marker)}`);
      if (g !== undefined) bits.push(`<span class="gold">gold ${tag(g)}</span>`);
      bits.push(`count ${esc(m.marker_count_anywhere)}`);
      bits.push(`${esc(m.num_tokens)} tok`);
      if (!m.terminated) bits.push("UNTERMINATED");
    }
    // The two assistant fields differ only when hidden codepoints were stripped
    // before handing the turn back to gemma; show the stripped copy when so.
    const gemmaView = (!isUser && m.content_visible_to_gemma !== m.content_raw)
      ? `<details><summary>as seen by gemma (${esc(m.hidden_codepoints_removed_for_gemma)} codepoints removed)</summary><pre>${esc(m.content_visible_to_gemma)}</pre></details>`
      : "";
    const turnLabel = isUser ? `<div class="turn-no">Turn ${esc(m.turn)}</div>` : "";
    const hdr = bits.length ? `<div class="hdr">${bits.join(" &middot; ")}</div>` : "";
    return turnLabel + `<div class="msg ${esc(m.role)}${wrong ? " bad" : ""}">
      <div class="body">${esc(body)}</div>${hdr}${gemmaView}</div>`;
  }).join("");

  document.getElementById("main").innerHTML = meta + `<div class="chat">${msgs}</div>`;
}

function render() { renderList(); renderMain(); }
render();
document.addEventListener("keydown", e => {
  if (e.key === "j" && sel < DATA.length - 1) { sel++; render(); }
  if (e.key === "k" && sel > 0) { sel--; render(); }
});
</script>
</body>
</html>
"""


def find_latest_conversations(runs_dir: Path = RUNS_DIR) -> Path:
    candidates = [
        path for pattern in ("*/conversations.jsonl", "char_*/holdout/*/conversations.jsonl")
        for path in runs_dir.glob(pattern) if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No conversations.jsonl of a simulation found under {runs_dir}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def load(path: Path, show_placeholders: bool = False) -> list[dict]:
    """Parse the JSONL, dropping the split-run placeholder stubs by default.

    Splitting a 100-conversation run across two GPUs seeds one worker's output
    directory with stub records so it skips the slots the other worker owns.
    Those stubs have status "placeholder" and no messages; showing them would
    pad the conversation count with blanks.
    """
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"skipping {path.name}:{lineno}: {exc}")
            continue
        if not show_placeholders and (
            record.get("status") == "placeholder" or not record.get("messages")
        ):
            continue
        records.append(record)
    return records


def build_page(records: list[dict], source: Path | str = "") -> bytes:
    payload = json.dumps(records).replace("<", "\\u003c")
    source_payload = json.dumps(str(source)).replace("<", "\\u003c")
    page = PAGE.replace("__SOURCE__", source_payload).replace("__DATA__", payload)
    return page.encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, help="pin one JSONL instead of following latest")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--port", type=int, default=8006)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--show-placeholders", action="store_true")
    args = ap.parse_args()

    fixed_path = args.file.resolve() if args.file else None
    runs_dir = args.runs_dir.resolve()

    def current_path() -> Path:
        return fixed_path or find_latest_conversations(runs_dir)

    path = current_path()
    records = load(path, args.show_placeholders)
    print(f"{path}: {len(records)} conversations")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            # Re-resolve latest and re-read it on every browser refresh.
            selected = current_path()
            body = build_page(load(selected, args.show_placeholders), selected)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # quiet
            pass

    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {url}  (ctrl-c to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
