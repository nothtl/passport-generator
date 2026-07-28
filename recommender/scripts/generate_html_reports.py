"""Generate clean HTML reports — Apple design language, typography-forward, print-ready."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "..", "..", "reports", "tingli_deepseek")
OUT_DIR = IN_DIR

# ── Apple design tokens ──────────────────────────────────────────────

CSS = """
/* ===================================================================
   Apple-style report stylesheet
   =================================================================== */

/* ── Design tokens ─────────────────────────────────────────────── */
:root {
  --bg:            #f5f5f7;
  --surface:       #ffffff;
  --text:          #1d1d1f;
  --text-secondary:#6e6e73;
  --text-tertiary: #aeaeb2;
  --accent:        #0071e3;
  --accent-light:  #e8f2fd;
  --green:         #34c759;
  --green-bg:      #e8f8ed;
  --orange:        #ff9500;
  --orange-bg:     #fff4e5;
  --red:           #ff3b30;
  --red-bg:        #ffeceb;
  --amber:         #ff9f0a;
  --amber-bg:      #fff8e5;
  --border:        #d2d2d7;
  --border-subtle: #e5e5ea;
  --radius:        12px;
  --radius-sm:     8px;
  --radius-lg:     16px;
  --shadow:        0 1px 3px rgba(0,0,0,0.04), 0 0 0 0.5px rgba(0,0,0,0.02);
  --shadow-hover:  0 4px 16px rgba(0,0,0,0.06), 0 0 0 0.5px rgba(0,0,0,0.03);
  --font:          -apple-system, BlinkMacSystemFont, "SF Pro Display",
                   "Helvetica Neue", "Segoe UI", sans-serif;
  --font-mono:     "SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace;
}

/* ── Reset & base ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font: 400 15px/1.6 var(--font);
  color: var(--text);
  background: var(--bg);
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ── Container ─────────────────────────────────────────────────── */
.container {
  max-width: 900px;
  margin: 0 auto;
  padding: 3rem 1.5rem 5rem;
}

/* ── Header ────────────────────────────────────────────────────── */
.report-header {
  padding: 0 0 2rem;
  margin-bottom: 2.5rem;
  border-bottom: 1px solid var(--border-subtle);
}

.report-header h1 {
  font-size: 2rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text);
  margin-bottom: 0.5rem;
}

.report-header .meta-row {
  display: flex;
  gap: 1.5rem;
  flex-wrap: wrap;
  align-items: center;
  font-size: 0.875rem;
  color: var(--text-secondary);
  margin-top: 0.5rem;
}

.report-header .meta-row .stat {
  display: flex;
  align-items: center;
  gap: 0.3rem;
  font-weight: 500;
}

.report-header .meta-row .stat strong {
  color: var(--text);
  font-weight: 600;
}

.report-header .meta-row .stat.conf-high strong { color: var(--green); }
.report-header .meta-row .stat.conf-mid  strong { color: var(--orange); }
.report-header .meta-row .stat.conf-low  strong { color: var(--red); }

/* Review indicator — subtle amber dot */
.review-indicator {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--amber);
}
.review-indicator::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--amber);
}

/* ── Skills bar ────────────────────────────────────────────────── */
.skills-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 1.2rem;
}

.skill-pill {
  display: inline-flex;
  align-items: center;
  padding: 0.3rem 0.7rem;
  border-radius: 99px;
  font-size: 0.78rem;
  font-weight: 500;
  color: var(--text);
  background: var(--surface);
  border: 1px solid var(--border-subtle);
}

.skill-pill.implicit {
  color: var(--text-tertiary);
  border-style: dashed;
}

/* ── Summary card ──────────────────────────────────────────────── */
.summary-card {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  margin: 1.5rem 0 2rem;
}

.summary-card p {
  color: var(--text);
  font-size: 0.92rem;
  line-height: 1.65;
}

.summary-card .careers {
  font-weight: 600;
  color: var(--accent);
}

/* ── Section headings ──────────────────────────────────────────── */
.section-head {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin: 2.5rem 0 1.2rem;
}

.section-head h2 {
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--text);
  white-space: nowrap;
}

.section-head .rule {
  flex: 1;
  height: 1px;
  background: var(--border-subtle);
}

/* ── Progression columns (replaces ASCII tree) ─────────────────── */
.progression {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin: 1.2rem 0;
}

.progression .tier {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
}

.progression .tier .tier-label {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-tertiary);
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-subtle);
}

.progression .tier .tier-item {
  font-size: 0.82rem;
  color: var(--text);
  line-height: 1.4;
  padding: 0.15rem 0;
}

.progression .tier .tier-item:last-child { border-bottom: none; }

/* Color accents per tier */
.progression .tier.tier-current  .tier-label { color: var(--text-secondary); }
.progression .tier.tier-bridge   .tier-label { color: var(--orange); }
.progression .tier.tier-stretch  .tier-label { color: var(--accent); }
.progression .tier.tier-dream    .tier-label { color: var(--green); }

/* ── Steps ──────────────────────────────────────────────────────── */
.steps {
  counter-reset: step;
  list-style: none;
  padding: 0;
  margin-top: 1rem;
}

.steps li {
  counter-increment: step;
  padding: 0.45rem 0 0.45rem 2.2rem;
  position: relative;
  font-size: 0.9rem;
  color: var(--text);
  line-height: 1.55;
}

.steps li::before {
  content: counter(step);
  position: absolute;
  left: 0;
  top: 0.45rem;
  width: 1.4rem;
  height: 1.4rem;
  background: var(--accent);
  color: #fff;
  border-radius: 50%;
  font-size: 0.7rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── Recommendations grid ──────────────────────────────────────── */
.report-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.2rem;
  margin: 1.2rem 0;
}

.stat-card {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 1.4rem;
  transition: box-shadow 0.2s ease;
}

.stat-card:hover {
  box-shadow: var(--shadow-hover);
}

.stat-card h3 {
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-tertiary);
  margin-bottom: 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border-subtle);
}

.stat-card.full { grid-column: 1 / -1; }

/* ── Job cards ──────────────────────────────────────────────────── */
.job-list {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}

.job-card {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  transition: box-shadow 0.15s ease;
}

.job-card:hover {
  box-shadow: var(--shadow);
}

.job-card .job-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  margin-bottom: 0.3rem;
}

.job-card .job-title {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--text);
  line-height: 1.35;
}

.job-card .job-company {
  font-size: 0.82rem;
  color: var(--text-secondary);
  margin-top: 0.15rem;
}

.job-card .job-score {
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.2rem 0.6rem;
  border-radius: 99px;
  white-space: nowrap;
  letter-spacing: 0.01em;
}

.score-ready { background: var(--green-bg); color: var(--green); }
.score-goal  { background: var(--orange-bg); color: var(--orange); }
.score-explore { background: var(--accent-light); color: var(--accent); }

.job-card .job-why {
  font-size: 0.84rem;
  color: var(--text-secondary);
  margin-top: 0.45rem;
  line-height: 1.5;
}

.job-card .job-link {
  display: inline-block;
  font-size: 0.82rem;
  color: var(--accent);
  text-decoration: none;
  font-weight: 500;
  margin-top: 0.5rem;
}

.job-card .job-link:hover {
  text-decoration: underline;
}

.job-card .job-link::after {
  content: " →";
  font-weight: 400;
}

/* ── Skill gaps ────────────────────────────────────────────────── */
.gaps-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 1rem;
  margin: 1.2rem 0;
}

.gap-col {
  background: var(--surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius);
  padding: 1.2rem;
}

.gap-col h4 {
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.8rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-subtle);
}

.gap-col .gap-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}

.gap-col .gap-pill {
  font-size: 0.78rem;
  padding: 0.25rem 0.6rem;
  border-radius: 99px;
  font-weight: 500;
  white-space: nowrap;
}

.gap-core    { border-left: 3px solid var(--red);    }
.gap-core    h4 { color: var(--red);    }
.gap-core    .gap-pill { background: var(--red-bg);    color: var(--red);    }

.gap-bridge  { border-left: 3px solid var(--orange);  }
.gap-bridge  h4 { color: var(--orange); }
.gap-bridge  .gap-pill { background: var(--orange-bg); color: var(--orange); }

.gap-stretch { border-left: 3px solid var(--accent);  }
.gap-stretch h4 { color: var(--accent);  }
.gap-stretch .gap-pill { background: var(--accent-light); color: var(--accent); }

/* ── Footer ────────────────────────────────────────────────────── */
.report-footer {
  text-align: center;
  padding: 2.5rem 0 0;
  margin-top: 3rem;
  font-size: 0.78rem;
  color: var(--text-tertiary);
  border-top: 1px solid var(--border-subtle);
}

/* ── Print styles ──────────────────────────────────────────────── */
@media print {
  body {
    background: #fff;
    font-size: 12px;
    color: #000;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }

  .container { max-width: none; padding: 0.5in; }

  .report-header {
    border-bottom: 0.5pt solid #ccc;
    padding-bottom: 1rem;
    margin-bottom: 1.2rem;
  }

  .section-head .rule { background: #ccc; }

  .stat-card,
  .job-card,
  .summary-card,
  .gap-col,
  .progression .tier {
    box-shadow: none;
    border: 0.5pt solid #ddd;
    break-inside: avoid;
  }

  .job-card .job-link::after { content: ""; }

  .report-grid,
  .gaps-grid,
  .progression { gap: 0.6rem; }

  .report-footer {
    border-top: 0.5pt solid #ccc;
    margin-top: 1.5rem;
    padding-top: 1rem;
  }

  .skill-pill,
  .gap-pill {
    border: 0.5pt solid #ccc;
  }
}

/* ── Responsive ────────────────────────────────────────────────── */
@media (max-width: 700px) {
  .report-grid,
  .gaps-grid,
  .progression {
    grid-template-columns: 1fr;
  }

  .report-header h1 { font-size: 1.6rem; }
  .report-header .meta-row { gap: 0.8rem; }
}
"""


def _company_name(name):
    if not name or len(name) < 2:
        return ""
    import re
    if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', name):
        return ""
    return name


def _conf_class(confidence):
    """Return CSS class for confidence level colouring."""
    if confidence >= 80:
        return "conf-high"
    elif confidence >= 60:
        return "conf-mid"
    else:
        return "conf-low"


def _render_job(job, score_class):
    title = job.get("title", "")
    company = _company_name(job.get("company", ""))
    url = job.get("url", "")
    fit = job.get("fit", 0)
    why = job.get("why", "")
    bridge = job.get("bridge", "")
    eligible = job.get("eligible", True)

    html = '      <div class="job-card">\n'
    html += '        <div class="job-top">\n'
    html += f'          <div><div class="job-title">{title}</div>\n'
    if company:
        html += f'          <div class="job-company">{company}</div>\n'
    html += '          </div>\n'
    html += f'          <div class="job-score {score_class}">{fit}%</div>\n'
    html += '        </div>\n'
    if why:
        html += f'        <div class="job-why">{why}</div>\n'
    if bridge:
        html += f'        <div class="job-why" style="color:var(--accent);font-weight:500;">Bridge: {bridge}</div>\n'
    if not eligible:
        html += f'        <div class="job-why" style="color:var(--red);">May not be youth-eligible</div>\n'
    if url:
        html += f'        <a class="job-link" href="{url}" target="_blank" rel="noopener">View position</a>\n'
    html += '      </div>\n'
    return html


def _render_progression(data):
    """Build a clean four-column skill progression instead of ASCII tree."""
    skills = data.get("verified_skills", [])[:10]
    implicit = [s.get("skill", "") for s in data.get("implicit_skills", [])]
    ideal = data.get("ideal_careers", [])[:5]

    # Use the skill tree to extract bridge and stretch skills if available
    tree_text = ""
    if data.get("skill_tree") and data["skill_tree"].get("tree"):
        tree_text = data["skill_tree"]["tree"]

    # Parse bridge/stretch from tree text
    bridge_skills = []
    stretch_skills = []

    def _is_header(text):
        """Filter out ASCII tree section headers like 'GOAL SKILLS', 'DREAM CAREERS'."""
        upper = text.upper()
        return any(kw in upper for kw in ("SKILLS", "CAREERS", "CURRENT", "DREAM"))

    current_section = None
    for line in tree_text.split("\n"):
        stripped = line.strip()
        if "BRIDGE" in stripped:
            current_section = "bridge"
            continue
        elif "STRETCH" in stripped:
            current_section = "stretch"
            continue
        elif "DREAM" in stripped:
            current_section = "dream"
            continue

        if current_section in ("bridge", "stretch"):
            # Extract skill name from lines like: │   ├── classroom-management
            import re
            m = re.search(r'├──\s+(.+?)$', stripped)
            if not m:
                m = re.search(r'└──\s+(.+?)$', stripped)
            if m:
                skill = m.group(1).strip()
                if _is_header(skill):
                    continue
                if current_section == "bridge":
                    bridge_skills.append(skill)
                else:
                    stretch_skills.append(skill)

    html = '      <div class="progression">\n'

    # Current skills column
    html += '        <div class="tier tier-current">\n'
    html += '          <div class="tier-label">Current Skills</div>\n'
    for s in skills[:6]:
        html += f'          <div class="tier-item">{s}</div>\n'
    for s in implicit[:3]:
        if s not in skills:
            html += f'          <div class="tier-item" style="color:var(--text-tertiary)">{s}</div>\n'
    if not skills:
        html += '          <div class="tier-item" style="color:var(--text-tertiary);font-style:italic">No verified skills</div>\n'
    html += '        </div>\n'

    # Bridge skills column
    html += '        <div class="tier tier-bridge">\n'
    html += '          <div class="tier-label">Learn Next</div>\n'
    for s in bridge_skills[:6]:
        html += f'          <div class="tier-item">{s}</div>\n'
    if not bridge_skills:
        html += '          <div class="tier-item" style="color:var(--text-tertiary);font-style:italic">—</div>\n'
    html += '        </div>\n'

    # Stretch skills column
    html += '        <div class="tier tier-stretch">\n'
    html += '          <div class="tier-label">Future Skills</div>\n'
    for s in stretch_skills[:6]:
        html += f'          <div class="tier-item">{s}</div>\n'
    if not stretch_skills:
        html += '          <div class="tier-item" style="color:var(--text-tertiary);font-style:italic">—</div>\n'
    html += '        </div>\n'

    # Dream careers column
    html += '        <div class="tier tier-dream">\n'
    html += '          <div class="tier-label">Dream Careers</div>\n'
    for c in ideal[:5]:
        html += f'          <div class="tier-item">{c}</div>\n'
    if not ideal:
        html += '          <div class="tier-item" style="color:var(--text-tertiary);font-style:italic">—</div>\n'
    html += '        </div>\n'

    html += '      </div>\n'
    return html


def generate_report(name, data):
    func = data.get("function", "?")
    sub = data.get("subdomain", "")
    skills = data.get("verified_skills", [])[:8]
    implicit = [s.get("skill", "") for s in data.get("implicit_skills", [])]
    ready = data.get("ready_now", [])
    aspirational = data.get("aspirational", [])
    confidence = data.get("confidence", 0)
    skill_tree = data.get("skill_tree", {})
    ideal = data.get("ideal_careers", [])
    needs_review = data.get("needs_review", False)
    lane = data.get("lane_used", "?")

    func_label = f"{func}/{sub}" if sub else func
    conf_class = _conf_class(confidence)

    html = '<!DOCTYPE html>\n'
    html += '<html lang="en">\n'
    html += '<meta charset="utf-8">\n'
    html += '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    html += f'<title>{name} — SpeakHire Career Report</title>\n'
    html += f'<style>{CSS}</style>\n'
    html += '<div class="container">\n'

    # ── Header ──────────────────────────────────────────────────────
    html += '  <header class="report-header">\n'
    html += f'    <h1>{name}</h1>\n'
    html += '    <div class="meta-row">\n'
    html += f'      <span class="stat">Direction: <strong>{func_label}</strong></span>\n'
    html += f'      <span class="stat {conf_class}">Confidence: <strong>{confidence}%</strong></span>\n'
    html += f'      <span class="stat">Lane: <strong>{lane}</strong></span>\n'
    if needs_review:
        html += '      <span class="review-indicator">Review suggested</span>\n'
    html += '    </div>\n'

    # Skills
    if skills:
        html += '    <div class="skills-bar">\n'
        for s in skills:
            html += f'      <span class="skill-pill">{s}</span>\n'
        for s in implicit[:3]:
            if s not in skills:
                html += f'      <span class="skill-pill implicit">{s}</span>\n'
        html += '    </div>\n'

    html += '  </header>\n'

    # ── Summary card ────────────────────────────────────────────────
    if ideal or (skill_tree and skill_tree.get("summary")):
        html += '  <div class="summary-card">\n'
        if ideal:
            html += f'    <p><span class="careers">Target careers:</span> {", ".join(ideal)}</p>\n'
        if skill_tree and skill_tree.get("summary"):
            html += f'    <p style="margin-top:0.5rem">{skill_tree["summary"]}</p>\n'
        html += '  </div>\n'

    # ── Skill progression ──────────────────────────────────────────
    tree_text = ""
    if skill_tree and skill_tree.get("tree"):
        tree_text = skill_tree["tree"]

    if tree_text or skills or ideal:
        html += '  <div class="section-head"><h2>Skill Progression</h2><span class="rule"></span></div>\n'
        html += _render_progression(data)

        # Steps
        steps = skill_tree.get("steps", []) if skill_tree else []
        if steps:
            html += '  <ol class="steps">\n'
            for s in steps:
                html += f'    <li>{s}</li>\n'
            html += '  </ol>\n'

    # ── Recommendations ────────────────────────────────────────────
    if ready or aspirational:
        html += '  <div class="section-head"><h2>Recommendations</h2><span class="rule"></span></div>\n'
        html += '  <div class="report-grid">\n'

        if ready:
            html += '    <div class="stat-card">\n'
            html += '      <h3>Available Now</h3>\n'
            html += '      <div class="job-list">\n'
            for job in ready:
                html += _render_job(job, "score-ready")
            html += '      </div>\n'
            html += '    </div>\n'

        if aspirational:
            html += '    <div class="stat-card">\n'
            html += '      <h3>Career Goals</h3>\n'
            html += '      <div class="job-list">\n'
            for job in aspirational:
                html += _render_job(job, "score-goal")
            html += '      </div>\n'
            html += '    </div>\n'

        html += '  </div>\n'

        # Cross-function explore tier
        explore = data.get("explore_jobs", [])
        if explore:
            html += '  <div class="section-head"><h2>Also Explore</h2><span class="rule"></span></div>\n'
            html += '  <p style="font-size:.84rem;color:var(--text-secondary);margin-bottom:1rem">Jobs from other career fields that match your skills</p>\n'
            html += '  <div class="report-grid">\n'
            html += '    <div class="stat-card full">\n'
            html += '      <h3>Cross-Field Matches</h3>\n'
            html += '      <div class="job-list">\n'
            for job in explore[:6]:
                html += _render_job(job, "score-explore")
            html += '      </div>\n'
            html += '    </div>\n'
            html += '  </div>\n'

    # ── Skill gaps ─────────────────────────────────────────────────
    core = data.get("core_gaps", [])[:6]
    bridge_gaps = data.get("bridge_gaps", [])[:6]
    stretch = data.get("stretch_gaps", [])[:6]

    if core or bridge_gaps or stretch:
        html += '  <div class="section-head"><h2>Skill Gaps</h2><span class="rule"></span></div>\n'
        html += '  <div class="gaps-grid">\n'

        if core:
            html += '    <div class="gap-col gap-core">\n'
            html += '      <h4>Core (priority)</h4>\n'
            html += '      <div class="gap-pills">\n'
            for g in core:
                html += f'        <span class="gap-pill">{g}</span>\n'
            html += '      </div>\n'
            html += '    </div>\n'

        if bridge_gaps:
            html += '    <div class="gap-col gap-bridge">\n'
            html += '      <h4>Bridge (developing)</h4>\n'
            html += '      <div class="gap-pills">\n'
            for g in bridge_gaps:
                html += f'        <span class="gap-pill">{g}</span>\n'
            html += '      </div>\n'
            html += '    </div>\n'

        if stretch:
            html += '    <div class="gap-col gap-stretch">\n'
            html += '      <h4>Stretch (future)</h4>\n'
            html += '      <div class="gap-pills">\n'
            for g in stretch:
                html += f'        <span class="gap-pill">{g}</span>\n'
            html += '      </div>\n'
            html += '    </div>\n'

        html += '  </div>\n'

    # ── Footer ──────────────────────────────────────────────────────
    html += '  <footer class="report-footer">SpeakHire Career Report</footer>\n'
    html += '</div>\n'
    html += '</html>\n'
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
        out_path = os.path.join(OUT_DIR, f"{name}.html")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        count += 1
        print(f"  {name}.html")

    # ── Index page ──────────────────────────────────────────────────
    ix_css = """
    body {
      font: 400 15px/1.6 -apple-system, BlinkMacSystemFont, "SF Pro Display",
           "Helvetica Neue", "Segoe UI", sans-serif;
      margin: 3rem auto;
      max-width: 860px;
      padding: 0 1.5rem;
      background: #f5f5f7;
      color: #1d1d1f;
      -webkit-font-smoothing: antialiased;
    }
    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 1.2rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid #e5e5ea;
    }
    th {
      text-align: left;
      padding: 0.7rem 1rem;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #86868b;
      background: #fafafa;
      border-bottom: 1px solid #e5e5ea;
    }
    td {
      padding: 0.6rem 1rem;
      font-size: 0.9rem;
      border-bottom: 1px solid #f0f0f2;
      vertical-align: top;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #fafafa; }
    a {
      color: #0071e3;
      text-decoration: none;
      font-weight: 500;
    }
    a:hover { text-decoration: underline; }
    .sub {
      color: #86868b;
      font-size: 0.82rem;
    }
    .conf {
      font-weight: 600;
      font-size: 0.85rem;
    }
    .conf-high { color: #34c759; }
    .conf-mid  { color: #ff9500; }
    .conf-low  { color: #ff3b30; }
    @media print { body { background: #fff; } }
    """

    ix = '<!DOCTYPE html>\n<html lang="en">\n'
    ix += '<meta charset="utf-8">\n'
    ix += '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
    ix += '<title>SpeakHire — Career Reports</title>\n'
    ix += f'<style>{ix_css}</style>\n'
    ix += '<h1>SpeakHire Career Reports</h1>\n'
    ix += '<table>\n'
    ix += '<thead><tr><th>Student</th><th>Direction</th><th>Conf</th><th>Ready Now</th><th>Career Goal</th></tr></thead>\n'
    ix += '<tbody>\n'

    for fname in sorted(os.listdir(IN_DIR)):
        if not fname.endswith('.json'):
            continue
        name = fname.replace('.json', '')
        with open(os.path.join(IN_DIR, fname), encoding='utf-8') as f:
            data = json.load(f)
        if data.get('error'):
            continue
        func_label = f"{data.get('function','?')}/{data.get('subdomain','?')}"
        conf = data.get('confidence', 0)
        conf_class = _conf_class(conf)
        r = data.get('ready_now', [{}])[0]
        a = data.get('aspirational', [{}])[0]
        ix += (
            f'<tr>'
            f'<td><a href="{name}.html">{name}</a></td>'
            f'<td class="sub">{func_label}</td>'
            f'<td class="conf {conf_class}">{conf}%</td>'
            f'<td>{r.get("title","-")[:45]}</td>'
            f'<td>{a.get("title","-")[:45]}</td>'
            f'</tr>\n'
        )

    ix += '</tbody>\n</table>\n'
    ix += f'<p style="margin-top:1.2rem;font-size:0.82rem;color:#86868b;">{count} students</p>\n'
    ix += '</html>\n'

    with open(os.path.join(OUT_DIR, "index.html"), 'w', encoding='utf-8') as f:
        f.write(ix)
    print(f"\n{count} HTML reports + index written to {OUT_DIR}")


if __name__ == "__main__":
    main()
