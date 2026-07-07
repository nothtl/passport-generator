"""Generate clean HTML reports — monochromatic indigo palette, bento grid, no emojis."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "reports", "tingli_deepseek")
OUT_DIR = IN_DIR

CSS = """
:root {
  --ink: #1a1a2e; --muted: #4a4a6a; --soft: #6b6b8a;
  --accent: #5b5fef; --accent-glow: #7c7ff8;
  --surface: #ffffff; --surface-alt: #f5f4fA;
  --border: #e8e8f0; --border-focus: #d0d0e0;
  --ready: #2d8a56; --ready-bg: #eaf5ef;
  --goal: #4a6cf7; --goal-bg: #eef1fe;
  --gap-core: #c44545; --gap-bridge: #c4820e; --gap-stretch: #3b7a3b;
  --radius: 10px; --radius-sm: 6px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font:15px/1.6 "Inter","Segoe UI",system-ui,sans-serif;color:var(--ink);background:var(--surface-alt);min-height:100vh}
.container{max-width:960px;margin:0 auto;padding:2rem 1.5rem 4rem}

.header{padding:2rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:2rem}
.header h1{font-size:1.8rem;font-weight:700;letter-spacing:-0.02em;color:var(--ink)}
.header .meta{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:.4rem;font-size:.88rem;color:var(--muted)}
.header .meta span{display:flex;align-items:center;gap:.3rem}
.header .meta .tag{display:inline-block;padding:.15rem .55rem;border-radius:99px;font-size:.76rem;font-weight:600;letter-spacing:.01em}
.tag-ready{background:var(--ready-bg);color:var(--ready)}
.tag-goal{background:var(--goal-bg);color:var(--goal)}
.tag-review{background:#fef3cd;color:#856404}
.needs-review{border-left:3px solid #e2a300}

/* Skills bar */
.skills-bar{display:flex;flex-wrap:wrap;gap:.4rem;margin:1rem 0}
.skill{display:inline-flex;align-items:center;gap:.25rem;padding:.25rem .6rem;border-radius:99px;font-size:.78rem;font-weight:500;background:var(--surface);border:1px solid var(--border);color:var(--ink)}
.skill-implicit{background:var(--surface-alt);border-style:dashed;color:var(--muted)}

/* Summary card */
.summary-card{background:linear-gradient(135deg,#f0f1ff 0%,#f5f4fA 100%);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.4rem;margin:1.2rem 0}
.summary-card p{color:var(--ink);font-size:.92rem;line-height:1.6}
.summary-card .careers{font-weight:600;color:var(--accent)}

/* Bento grid */
.bento{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem;margin:1.5rem 0}
.bento-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem;transition:box-shadow .2s}
.bento-card:hover{box-shadow:0 2px 12px rgba(0,0,0,.06)}
.bento-card h3{font-size:.82rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:.6rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}
.bento-card.full{grid-column:1/-1}

/* Job cards */
.job-list{display:flex;flex-direction:column;gap:.8rem}
.job-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.2rem;transition:box-shadow .15s;position:relative}
.job-card:hover{box-shadow:0 1px 8px rgba(0,0,0,.05)}
.job-card .job-top{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
.job-card .job-title{font-weight:600;font-size:.95rem;color:var(--ink)}
.job-card .job-company{font-size:.82rem;color:var(--muted);margin-top:.1rem}
.job-card .job-score{font-size:.78rem;font-weight:700;padding:.15rem .5rem;border-radius:99px;white-space:nowrap}
.score-ready{background:var(--ready-bg);color:var(--ready)}
.score-goal{background:var(--goal-bg);color:var(--goal)}
.job-card .job-why{font-size:.82rem;color:var(--muted);margin-top:.4rem;line-height:1.5}
.job-card .job-link{font-size:.8rem;color:var(--accent);text-decoration:none;font-weight:600}
.job-card .job-link:hover{text-decoration:underline}

/* Skill tree */
.skill-tree{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.4rem;margin:1.2rem 0;font-family:"JetBrains Mono","SF Mono","Cascadia Code",monospace;font-size:.8rem;line-height:1.7;color:var(--ink);white-space:pre;overflow-x:auto}

/* Steps */
.steps{counter-reset:step;list-style:none;padding:0}
.steps li{counter-increment:step;padding:.35rem 0 .35rem 2rem;position:relative;font-size:.88rem;color:var(--ink)}
.steps li::before{content:counter(step);position:absolute;left:0;top:.3rem;width:1.3rem;height:1.3rem;background:var(--accent);color:#fff;border-radius:50%;font-size:.7rem;font-weight:700;display:flex;align-items:center;justify-content:center}

/* Gaps grid */
.gaps-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem}
.gap-col h4{font-size:.76rem;font-weight:600;text-transform:uppercase;letter-spacing:.03em;margin-bottom:.4rem}
.gap-col ul{list-style:none;padding:0}
.gap-col li{font-size:.82rem;padding:.2rem 0;border-bottom:1px dotted var(--border)}
.gap-col li:last-child{border-bottom:none}
.gap-core h4{color:var(--gap-core)}.gap-bridge h4{color:var(--gap-bridge)}.gap-stretch h4{color:var(--gap-stretch)}

/* Section heading */
.section-head{display:flex;align-items:center;gap:.6rem;margin:2rem 0 1rem}
.section-head h2{font-size:1.1rem;font-weight:700;letter-spacing:-0.01em;color:var(--ink)}
.section-head .line{flex:1;height:1px;background:var(--border)}

/* Footer */
.footer{text-align:center;padding:2rem 0 1rem;font-size:.78rem;color:var(--soft);border-top:1px solid var(--border);margin-top:3rem}

@media(max-width:700px){.bento{grid-template-columns:1fr}.gaps-grid{grid-template-columns:1fr}}
"""


def _company_name(name):
    if not name or len(name) < 2: return ""
    import re
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', name): return ""
    return name


def _render_job(job, score_class):
    title = job.get("title", "")
    company = _company_name(job.get("company", ""))
    url = job.get("url", "")
    fit = job.get("fit", 0)
    why = job.get("why", "")
    bridge = job.get("bridge", "")
    eligible = job.get("eligible", True)

    html = '<div class=job-card>\n'
    html += '<div class=job-top>\n'
    html += f'<div><div class=job-title>{title}</div>'
    if company: html += f'<div class=job-company>{company}</div>'
    html += '</div>\n'
    html += f'<div class="job-score {score_class}">{fit}%</div>\n'
    html += '</div>\n'
    if why: html += f'<div class=job-why>{why}</div>\n'
    if bridge: html += f'<div class=job-why style="color:var(--accent)">Bridge: {bridge}</div>\n'
    if not eligible: html += '<div class=job-why style="color:var(--gap-core)">May not be youth-eligible</div>\n'
    if url: html += f'<a class=job-link href="{url}" target=_blank rel=noopener>View position</a>\n'
    html += '</div>\n'
    return html


def generate_report(name, data):
    func = data.get("function", "?")
    sub = data.get("subdomain", "")
    skills = data.get("verified_skills", [])[:8]
    implicit = [s.get("skill","") for s in data.get("implicit_skills", [])]
    ready = data.get("ready_now", [])
    aspirational = data.get("aspirational", [])
    confidence = data.get("confidence", 0)
    skill_tree = data.get("skill_tree", {})
    ideal = data.get("ideal_careers", [])
    needs_review = data.get("needs_review", False)

    func_label = f"{func}/{sub}" if sub else func
    review_class = " needs-review" if needs_review else ""

    html = '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">\n'
    html += f'<title>{name} | SpeakHire</title>\n'
    html += f'<style>{CSS}</style>\n'
    html += f'<div class=container{review_class}>\n'

    # Header
    html += '<div class=header>\n'
    html += f'<h1>{name}</h1>\n'
    html += '<div class=meta>\n'
    html += f'<span>Direction: <strong>{func_label}</strong></span>\n'
    html += f'<span>Confidence: <strong>{confidence}%</strong></span>\n'
    html += f'<span>Lane: <strong>{data.get("lane_used","?")}</strong></span>\n'
    if needs_review: html += '<span class="tag tag-review">Review suggested</span>\n'
    html += '</div>\n'

    # Skills
    if skills:
        html += '<div class=skills-bar>\n'
        for s in skills:
            html += f'<span class=skill>{s}</span>\n'
        for s in implicit[:3]:
            if s not in skills:
                html += f'<span class="skill skill-implicit">{s}</span>\n'
        html += '</div>\n'
    html += '</div>\n'

    # Summary
    if ideal or (skill_tree and skill_tree.get("summary")):
        html += '<div class=summary-card>\n'
        if ideal: html += f'<p><span class=careers>Target careers:</span> {", ".join(ideal)}</p>\n'
        if skill_tree and skill_tree.get("summary"):
            html += f'<p style="margin-top:.4rem">{skill_tree["summary"]}</p>\n'
        html += '</div>\n'

    # Skill tree
    if skill_tree and skill_tree.get("tree"):
        html += '<div class="section-head"><h2>Skill Progression</h2><span class=line></span></div>\n'
        html += f'<pre class=skill-tree>{skill_tree["tree"]}</pre>\n'
        steps = skill_tree.get("steps", [])
        if steps:
            html += '<ol class=steps>\n'
            for s in steps:
                html += f'<li>{s}</li>\n'
            html += '</ol>\n'

    # Jobs bento
    if ready or aspirational:
        html += '<div class="section-head"><h2>Recommendations</h2><span class=line></span></div>\n'
        html += '<div class=bento>\n'

        if ready:
            html += '<div class=bento-card>\n'
            html += '<h3>Available Now</h3>\n'
            html += '<div class=job-list>\n'
            for job in ready:
                html += _render_job(job, "score-ready")
            html += '</div></div>\n'

        if aspirational:
            html += '<div class=bento-card>\n'
            html += '<h3>Career Goals</h3>\n'
            html += '<div class=job-list>\n'
            for job in aspirational:
                html += _render_job(job, "score-goal")
            html += '</div></div>\n'

        html += '</div>\n'

    # Gaps
    core = data.get("core_gaps", [])[:5]
    bridge_gaps = data.get("bridge_gaps", [])[:5]
    stretch = data.get("stretch_gaps", [])[:5]
    if core or bridge_gaps or stretch:
        html += '<div class="section-head"><h2>Skill Gaps</h2><span class=line></span></div>\n'
        html += '<div class=gaps-grid>\n'
        if core:
            html += '<div class="gap-col gap-core"><h4>Core (priority)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in core) + '</ul></div>\n'
        if bridge_gaps:
            html += '<div class="gap-col gap-bridge"><h4>Bridge (developing)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in bridge_gaps) + '</ul></div>\n'
        if stretch:
            html += '<div class="gap-col gap-stretch"><h4>Stretch (future)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in stretch) + '</ul></div>\n'
        html += '</div>\n'

    html += '<div class=footer>SpeakHire Recommender</div>\n'
    html += '</div>\n'
    return html


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(IN_DIR)):
        if not fname.endswith('.json'): continue
        name = fname.replace('.json', '')
        with open(os.path.join(IN_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('error'): continue
        html = generate_report(name, data)
        with open(os.path.join(OUT_DIR, f"{name}.html"), 'w', encoding='utf-8') as f:
            f.write(html)
        count += 1
        print(f"  {name}.html")

    # Index
    ix = '<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">\n'
    ix += f'<title>SpeakHire</title>\n<style>body{{font:15px/1.5 Inter,Segoe UI,system-ui;margin:2rem auto;max-width:800px;padding:0 1rem;background:#f5f4fA;color:#1a1a2e}}h1{{font-size:1.6rem;letter-spacing:-.02em}}table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.04)}}th{{text-align:left;padding:.6rem .8rem;font-size:.76rem;text-transform:uppercase;letter-spacing:.04em;color:#6b6b8a;border-bottom:1px solid #e8e8f0}}td{{padding:.55rem .8rem;font-size:.9rem;border-bottom:1px solid #f0f0f5}}tr:last-child td{{border-bottom:none}}a{{color:#5b5fef;text-decoration:none;font-weight:500}}a:hover{{text-decoration:underline}}.sub{{color:#6b6b8a;font-size:.82rem}}</style>\n'
    ix += '<h1>SpeakHire</h1>\n'
    ix += '<table><thead><tr><th>Student</th><th>Direction</th><th>Conf</th><th>Ready Now</th><th>Career Goal</th></tr></thead><tbody>\n'

    for fname in sorted(os.listdir(IN_DIR)):
        if not fname.endswith('.json'): continue
        name = fname.replace('.json', '')
        with open(os.path.join(IN_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('error'): continue
        func = f"{data.get('function','?')}/{data.get('subdomain','?')}"
        conf = data.get('confidence', 0)
        r = data.get('ready_now', [{}])[0]
        a = data.get('aspirational', [{}])[0]
        ix += f'<tr><td><a href="{name}.html">{name}</a></td><td class=sub>{func}</td><td>{conf}%</td><td>{r.get("title","-")[:40]}</td><td>{a.get("title","-")[:40]}</td></tr>\n'

    ix += '</tbody></table>\n'
    ix += f'<p class=sub style="margin-top:1rem">{count} students</p>\n'
    with open(os.path.join(OUT_DIR, "index.html"), 'w', encoding='utf-8') as f:
        f.write(ix)
    print(f"\n{count} HTML reports + index.html written to {OUT_DIR}")


if __name__ == "__main__":
    main()
