"""Generate HTML reports matching the zip format with skill trees and tiered jobs."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "reports", "tingli_deepseek")
OUT_DIR = IN_DIR

CSS = """
body{font:15px/1.55 system-ui,Segoe UI,Arial;margin:2rem auto;max-width:950px;color:#1a1a1a;padding:0 1rem}
h1{margin:0 0 .2rem} h2{margin:2rem 0 .6rem;font-size:1.15rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
h3{font-size:.95rem;margin:1.5rem 0 .3rem;color:#333}
h4{margin:.3rem 0;font-size:.82rem;text-transform:uppercase;letter-spacing:.03em;color:#666}
.job{border:1px solid #e3e3e3;border-radius:.6rem;padding:1rem;margin:.7rem 0}
.jobhead{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
.rank{color:#999;font-weight:700;margin-right:.4rem}
.title{font-weight:700} .company{color:#555;margin-left:.5rem}
.apply{margin-left:.6rem;color:#1558d6;text-decoration:none;font-weight:600;white-space:nowrap}
.cols{display:flex;gap:2rem;flex-wrap:wrap} .col{flex:1;min-width:200px}
ul{margin:.2rem 0;padding-left:1.1rem} li{margin:.15rem 0}
.sub{color:#888;font-size:.85rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.25rem;font-size:.73rem;font-weight:600;margin-right:.3rem}
.badge-ready{background:#d4edda;color:#155724}
.badge-aspire{background:#cce5ff;color:#004085}
.badge-blocked{background:#f8d7da;color:#721c24}
.skills{display:flex;flex-wrap:wrap;gap:.3rem;margin:.5rem 0}
.skill-tag{background:#f0f0f0;padding:.2rem .5rem;border-radius:.25rem;font-size:.8rem}
.skill-tag-implicit{background:#fff3cd;padding:.2rem .5rem;border-radius:.25rem;font-size:.8rem}
.skill-tree{background:#f8f9fa;padding:1rem 1.2rem;border-radius:.5rem;font-size:.82rem;line-height:1.65;overflow-x:auto;border-left:3px solid #1558d6;margin:1rem 0}
.step-list{counter-reset:step}
.step-list li{counter-increment:step;margin:.5rem 0;list-style:none}
.step-list li::before{content:counter(step);background:#1558d6;color:#fff;border-radius:50%;width:1.4rem;height:1.4rem;display:inline-flex;align-items:center;justify-content:center;font-size:.75rem;margin-right:.5rem;font-weight:700}
.why{color:#2a662a;font-size:.85rem;margin:.3rem 0}
#summary-box{background:#f0f7ff;border:1px solid #cce5ff;border-radius:.6rem;padding:.8rem 1rem;margin:1rem 0}
#summary-box b{color:#004085}
"""


def _company_name(name):
    if not name or len(name) < 2: return ""
    import re
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', name): return ""
    return name


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

    html = f'<!doctype html><meta charset=utf-8><title>{name} — job matches</title>\n'
    html += f'<style>{CSS}</style>\n'
    html += f'<h1>{name}</h1>\n'
    html += f'<p class=sub>Career direction: <b>{func_label}</b> · Confidence: {confidence}%'
    if needs_review:
        html += ' · <span style="color:#856404">⚠ needs review</span>'
    html += '</p>\n'

    # Skills
    if skills:
        html += '<div class=skills>'
        for s in skills:
            html += f'<span class=skill-tag>✓ {s}</span>'
        for s in implicit[:3]:
            if s not in skills:
                html += f'<span class=skill-tag-implicit>~ {s}</span>'
        html += '</div>\n'

    # Summary box
    if ideal:
        html += '<div id=summary-box>'
        html += f'<b>Ideal careers:</b> {", ".join(ideal)}'
        if skill_tree and skill_tree.get("summary"):
            html += f'<br><span class=sub>{skill_tree["summary"]}</span>'
        html += '</div>\n'

    # Skill progression tree
    if skill_tree and skill_tree.get("tree"):
        html += '<h2>Skill Progression</h2>\n'
        html += f'<pre class=skill-tree>{skill_tree["tree"]}</pre>\n'
        steps = skill_tree.get("steps", [])
        if steps:
            html += '<ol class=step-list>'
            for s in steps:
                html += f'<li>{s}</li>'
            html += '</ol>\n'

    # Ready Now
    if ready:
        html += '<h2>✓ Ready Now — jobs you can do today</h2>\n'
        for i, job in enumerate(ready, 1):
            title = job.get("title", "")
            company = _company_name(job.get("company", ""))
            url = job.get("url", "")
            fit = job.get("fit", 0)
            why = job.get("why", "")
            bridge = job.get("bridge", "")
            eligible = job.get("eligible", True)

            html += '<div class=job>\n'
            html += '<div class=jobhead>\n'
            html += f'<div><span class=rank>#{i}</span><span class="badge badge-ready">READY {fit}%</span>'
            html += f'<span class=title>{title}</span>'
            if company: html += f'<span class=company>{company}</span>'
            html += '</div>\n'
            if url:
                html += f'<div><a class=apply href="{url}" target=_blank rel=noopener>Apply →</a></div>\n'
            html += '</div>\n'
            if why:
                html += f'<p class=why>💡 {why}</p>\n'
            if bridge:
                html += f'<p class=sub style="color:#1558d6;"><b>🔄 Bridge:</b> {bridge}</p>\n'
            if not eligible:
                html += '<p class=sub style="color:#c00;"><b>⚠ May not be eligible for youth</b></p>\n'
            html += '</div>\n'

    # Aspirational
    if aspirational:
        html += '<h2>🎯 Work Toward — your ideal career path</h2>\n'
        for i, job in enumerate(aspirational, 1):
            title = job.get("title", "")
            company = _company_name(job.get("company", ""))
            url = job.get("url", "")
            fit = job.get("fit", 0)
            why = job.get("why", "")
            eligible = job.get("eligible", True)

            html += '<div class=job>\n'
            html += '<div class=jobhead>\n'
            html += f'<div><span class=rank>#{i}</span><span class="badge badge-aspire">GOAL {fit}%</span>'
            html += f'<span class=title>{title}</span>'
            if company: html += f'<span class=company>{company}</span>'
            html += '</div>\n'
            if url:
                html += f'<div><a class=apply href="{url}" target=_blank rel=noopener>Apply →</a></div>\n'
            html += '</div>\n'
            if why:
                html += f'<p class=why>💡 {why}</p>\n'
            if not eligible:
                html += '<p class=sub style="color:#c00;"><b>⚠ May not be eligible for youth</b></p>\n'
            html += '</div>\n'

    # Skill Gaps
    core = data.get("core_gaps", [])[:5]
    bridge_gaps = data.get("bridge_gaps", [])[:5]
    stretch = data.get("stretch_gaps", [])[:5]
    if core or bridge_gaps or stretch:
        html += '<h2>📋 Skill Gaps to Close</h2>\n'
        html += '<div class=cols>'
        if core:
            html += '<div class=col><h4>🔴 Core (build now)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in core) + '</ul></div>'
        if bridge_gaps:
            html += '<div class=col><h4>🟡 Bridge (developing)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in bridge_gaps) + '</ul></div>'
        if stretch:
            html += '<div class=col><h4>🟢 Stretch (future)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in stretch) + '</ul></div>'
        html += '</div>\n'

    html += f'<p class=sub style="margin-top:2rem;">Generated by SpeakHire Recommender · {data.get("lane_used","?")} lane</p>\n'
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
    index = '<!doctype html><meta charset=utf-8><title>SpeakHire job matches</title>\n'
    index += f'<style>body{{font:15px/1.5 system-ui;margin:2rem auto;max-width:800px}} table{{border-collapse:collapse;width:100%}} td,th{{padding:.45rem .6rem;border-bottom:1px solid #eee;text-align:left}} a{{color:#1558d6;text-decoration:none}} .sub{{color:#888;font-size:.85rem}}</style>\n'
    index += '<h1>SpeakHire — job matches</h1>\n'
    index += '<table><thead><tr><th>Student</th><th>Function</th><th>Conf</th><th>Top Ready</th><th>Top Goal</th></tr></thead><tbody>\n'

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
        index += f'<tr><td><a href="{name}.html">{name}</a></td><td class=sub>{func}</td><td>{conf}%</td><td>{r.get("title","—")[:40]}</td><td>{a.get("title","—")[:40]}</td></tr>\n'

    index += '</tbody></table>\n'
    index += f'<p class=sub>{count} students</p>\n'
    with open(os.path.join(OUT_DIR, "index.html"), 'w', encoding='utf-8') as f:
        f.write(index)
    print(f"\n{count} HTML reports + index.html written to {OUT_DIR}")


if __name__ == "__main__":
    main()
