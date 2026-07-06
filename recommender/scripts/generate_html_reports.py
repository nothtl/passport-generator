"""Generate HTML reports matching the zip format from our pipeline JSON output."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "reports", "tingli_deepseek")
OUT_DIR = IN_DIR

CSS = """
body{font:15px/1.55 system-ui,Segoe UI,Arial;margin:2rem auto;max-width:900px;color:#1a1a1a;padding:0 1rem}
h1{margin:0 0 .2rem} h2{margin:2rem 0 .6rem;font-size:1.15rem;border-bottom:1px solid #eee;padding-bottom:.3rem}
h4{margin:.2rem 0 .3rem;font-size:.85rem;text-transform:uppercase;letter-spacing:.03em;color:#666}
.job{border:1px solid #e3e3e3;border-radius:.6rem;padding:1rem;margin:.8rem 0}
.jobhead{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem}
.rank{color:#999;font-weight:700;margin-right:.4rem}
.title{font-weight:700} .company{color:#555;margin-left:.5rem}
.apply{margin-left:.6rem;color:#1558d6;text-decoration:none;font-weight:600;white-space:nowrap}
.cols{display:flex;gap:2rem;flex-wrap:wrap} .col{flex:1;min-width:220px}
ul{margin:.2rem 0;padding-left:1.1rem} li{margin:.15rem 0}
.sub{color:#888;font-size:.85rem}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:.25rem;font-size:.75rem;font-weight:600;margin-right:.3rem}
.badge-ready{background:#d4edda;color:#155724}
.badge-aspire{background:#cce5ff;color:#004085}
.skills{display:flex;flex-wrap:wrap;gap:.3rem;margin:.5rem 0}
.skill-tag{background:#f0f0f0;padding:.2rem .5rem;border-radius:.25rem;font-size:.8rem}
"""


def _company_name(name):
    if not name or len(name) < 2:
        return ""
    import re
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', name):
        return ""
    return name


def generate_report(name, data):
    func = data.get("function", "?")
    sub = data.get("subdomain", "")
    skills = data.get("verified_skills", [])[:10]
    ready = data.get("ready_now", [])
    aspirational = data.get("aspirational", [])
    gaps = data.get("core_gaps", [])[:5]
    confidence = data.get("confidence", 0)
    skill_path = data.get("skill_path", "")

    func_label = f"{func}/{sub}" if sub else func

    html = f'<!doctype html><meta charset=utf-8><title>{name} — job matches</title>\n'
    html += f'<style>{CSS}</style>\n'
    html += f'<h1>{name}</h1>\n'
    html += f'<p class=sub>Career direction: <b>{func_label}</b> · Confidence: {confidence}%</p>\n'

    if skill_path:
        html += f'<p class=sub style="color:#333;max-width:700px;"><b>Path:</b> {skill_path}</p>\n'

    # Skill progression tree
    skill_tree = data.get('skill_tree', {})
    if skill_tree and skill_tree.get('tree'):
        html += '<h2>Skill Progression Tree</h2>\n'
        html += f'<pre style="background:#f8f8f8;padding:1rem;border-radius:.5rem;font-size:.8rem;line-height:1.6;overflow-x:auto;">{skill_tree["tree"]}</pre>\n'
        steps = skill_tree.get('steps', [])
        if steps:
            html += '<h4>Action Steps</h4>\n<ol>'
            for s in steps:
                html += f'<li>{s}</li>'
            html += '</ol>\n'

    if skills:
        html += '<div class=skills>'
        for s in skills:
            html += f'<span class=skill-tag>{s}</span>'
        html += '</div>\n'

    # Ready Now
    if ready:
        html += '<h2>Ready Now — jobs you can do today</h2>\n'
        for i, job in enumerate(ready, 1):
            title = job.get("title", "")
            company = _company_name(job.get("company", ""))
            url = job.get("url", "")
            fit = job.get("fit", 0)
            why = job.get("why", "")
            bridge = job.get("bridge", "")
            elligible = job.get("eligible", True)

            html += '<div class=job>\n'
            html += '<div class=jobhead>\n'
            html += f'<div><span class=rank>#{i}</span><span class=badge class=badge-ready>READY {fit}%</span>'
            html += f'<span class=title>{title}</span>'
            if company:
                html += f'<span class=company>{company}</span>'
            html += '</div>\n'
            if url:
                html += f'<div><a class=apply href="{url}" target=_blank rel=noopener>Apply →</a></div>\n'
            html += '</div>\n'

            if why:
                html += f'<p class=sub>{why}</p>\n'
            if bridge:
                html += f'<p class=sub style="color:#1558d6;"><b>Bridge:</b> {bridge}</p>\n'
            if not elligible:
                html += '<p class=sub style="color:#c00;"><b>⚠ May not be eligible for youth</b></p>\n'
            html += '</div>\n'

    # Aspirational
    if aspirational:
        html += '<h2>Work Toward — aspirational jobs</h2>\n'
        for i, job in enumerate(aspirational, 1):
            title = job.get("title", "")
            company = _company_name(job.get("company", ""))
            url = job.get("url", "")
            fit = job.get("fit", 0)
            why = job.get("why", "")
            elligible = job.get("eligible", True)

            html += '<div class=job>\n'
            html += '<div class=jobhead>\n'
            html += f'<div><span class=rank>#{i}</span><span class=badge class=badge-aspire>GOAL {fit}%</span>'
            html += f'<span class=title>{title}</span>'
            if company:
                html += f'<span class=company>{company}</span>'
            html += '</div>\n'
            if url:
                html += f'<div><a class=apply href="{url}" target=_blank rel=noopener>Apply →</a></div>\n'
            html += '</div>\n'
            if why:
                html += f'<p class=sub>{why}</p>\n'
            if not elligible:
                html += '<p class=sub style="color:#c00;"><b>⚠ May not be eligible for youth</b></p>\n'
            html += '</div>\n'

    if gaps:
        html += '<h2>Skill Gaps</h2>\n'
        html += '<div class=cols>'
        core = data.get('core_gaps', [])[:5]
        bridge = data.get('bridge_gaps', [])[:5]
        stretch = data.get('stretch_gaps', [])[:5]
        if core:
            html += '<div class=col><h4>Core Gaps (build now)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in core) + '</ul></div>'
        if bridge:
            html += '<div class=col><h4>Bridge Gaps (developing)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in bridge) + '</ul></div>'
        if stretch:
            html += '<div class=col><h4>Stretch Gaps (future)</h4><ul>' + ''.join(f'<li>{g}</li>' for g in stretch) + '</ul></div>'
        html += '</div>\n'

    html += f'<p class=sub style="margin-top:2rem;">Generated by SpeakHire Recommender · {data.get("lane_used","?")} lane</p>\n'
    return html


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(IN_DIR)):
        if not fname.endswith('.json'):
            continue
        name = fname.replace('.json', '')
        with open(os.path.join(IN_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('error'):
            continue
        html = generate_report(name, data)
        out = os.path.join(OUT_DIR, f"{name}.html")
        with open(out, 'w', encoding='utf-8') as f:
            f.write(html)
        count += 1
        print(f"  {name}.html")

    # Generate index
    index = '<!doctype html><meta charset=utf-8><title>SpeakHire job matches</title>\n'
    index += f'<style>body{{font:15px/1.5 system-ui;margin:2rem auto;max-width:760px}} table{{border-collapse:collapse;width:100%}} td,th{{padding:.45rem .6rem;border-bottom:1px solid #eee;text-align:left}} a{{color:#1558d6;text-decoration:none}} .sub{{color:#888;font-size:.85rem}}</style>\n'
    index += '<h1>SpeakHire — job matches</h1>\n'
    index += '<table><thead><tr><th>Student</th><th>Function</th><th>Top Ready Job</th><th>Top Goal Job</th></tr></thead><tbody>\n'

    for fname in sorted(os.listdir(IN_DIR)):
        if not fname.endswith('.json'):
            continue
        name = fname.replace('.json', '')
        with open(os.path.join(IN_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('error'):
            continue
        func = f"{data.get('function','?')}/{data.get('subdomain','?')}"
        ready = data.get('ready_now', [])
        aspir = data.get('aspirational', [])
        r_top = ready[0].get('title', '—')[:40] if ready else '—'
        a_top = aspir[0].get('title', '—')[:40] if aspir else '—'
        index += f'<tr><td><a href="{name}.html">{name}</a></td><td class=sub>{func}</td><td>{r_top}</td><td>{a_top}</td></tr>\n'

    index += '</tbody></table>\n'
    index += f'<p class=sub>{count} students</p>\n'

    with open(os.path.join(OUT_DIR, "index.html"), 'w', encoding='utf-8') as f:
        f.write(index)

    print(f"\n{count} HTML reports + index.html written to {OUT_DIR}")


if __name__ == "__main__":
    main()
