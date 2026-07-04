"""
LinkedIn markdown parser.
Parses the exact format exported by Ousmane's profile and generalises to similar exports.

Format observed:
  # Name (Pronouns)
  **headline**
  Location · connection info
  ---
  ## Experience
  **Title** — Company [· employment type]
  Date · Duration · Location [· On-site/Remote]
  Skills: a, b, c
  ## Education
  **Institution** — years or Degree · dates
  ## Licenses & Certifications
  | Certification | Issuer | Issued |
  |---|---|---|
  | name | issuer | date [· ID: ...] |
  ## Projects
  **Name** — date range
  Description paragraph
  GitHub: url
  Skills: a, b
  ## Skills
  Technical: a, b, c
  Soft Skills: x, y
"""

import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_bold(s: str) -> str:
    return s.strip().strip('*').strip()


def _is_bold_line(line: str) -> bool:
    s = line.strip()
    return s.startswith('**') and s.endswith('**')


def _parse_bold_dash(line: str) -> tuple[str, str] | None:
    """Parse '**Left** — Right' → (left, right), or None if no match."""
    m = re.match(r'^\*\*(.+?)\*\*\s*[—–]\s*(.+)', line.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    header_lines: list[str] = []
    current_key: str | None = None
    current: list[str] = []

    for line in lines:
        m = re.match(r'^##\s+(.+)', line.rstrip())
        if m:
            if current_key is not None:
                sections[current_key] = current
            else:
                header_lines = current
            current_key = m.group(1).strip()
            current = []
        else:
            current.append(line)

    if current_key is not None:
        sections[current_key] = current
    sections['__header__'] = header_lines
    return sections


# ---------------------------------------------------------------------------
# Header block → name / pronouns / headline / location
# ---------------------------------------------------------------------------

def _parse_header(header_lines: list[str]) -> dict:
    name = pronouns = headline = location = None

    for line in header_lines:
        stripped = line.strip()
        if not stripped or stripped == '---':
            continue

        # H1 line
        m = re.match(r'^#\s+(.+)', stripped)
        if m and name is None:
            raw = m.group(1).strip()
            pm = re.search(r'\(([^)]+)\)\s*$', raw)
            if pm:
                pronouns = pm.group(1)
                name = raw[:pm.start()].strip()
            else:
                name = raw
            continue

        # Headline: first bold line after H1
        if name and headline is None and _is_bold_line(stripped):
            headline = _strip_bold(stripped)
            continue

        # Location: first plain (non-bold, non-empty) line after headline
        if headline and location is None and not stripped.startswith('**'):
            # Strip connection noise (· 3rd connection etc.)
            loc = re.sub(r'\s*·\s*\d+(st|nd|rd|th)\s+connection.*', '', stripped).strip()
            if loc:
                location = loc

    return {"name": name, "pronouns": pronouns, "headline": headline, "location": location}


# ---------------------------------------------------------------------------
# Experience
# ---------------------------------------------------------------------------

def _parse_experience(lines: list[str]) -> list[dict]:
    entries: list[dict] = []
    current_header: dict | None = None
    content: list[str] = []

    def _flush():
        if current_header:
            entries.append(_finalize_exp(current_header, content))

    for line in lines:
        stripped = line.strip()
        if stripped == '---':
            _flush()
            current_header = None
            content = []
            continue

        parsed = _parse_bold_dash(stripped)
        if parsed:
            _flush()
            left, right = parsed
            # Company may have ` · Part-time` etc.
            company_parts = re.split(r'\s*·\s*', right, maxsplit=1)
            current_header = {
                'title':           left,
                'company':         company_parts[0].strip(),
            }
            content = []
        elif current_header is not None and stripped:
            content.append(stripped)

    _flush()
    return entries


def _finalize_exp(header: dict, content: list[str]) -> dict:
    duration = location = description = skills = None
    desc_parts: list[str] = []

    for i, line in enumerate(content):
        if line.startswith('Skills:'):
            skills = [s.strip() for s in line[7:].split(',') if s.strip()]
        elif i == 0:
            duration = line
            # Try to extract location from duration line
            parts = [p.strip() for p in line.split(' · ')]
            work_modes = {'on-site', 'remote', 'hybrid'}
            if len(parts) >= 2:
                last = parts[-1].lower()
                second_last = parts[-2].lower() if len(parts) >= 2 else ''
                if last in work_modes:
                    location = parts[-1] if parts[-1].lower() in work_modes else None
                    # Try second-to-last as geographic location
                    if len(parts) >= 3 and not re.match(r'^\d', parts[-2]):
                        location = parts[-2]
                elif not re.match(r'^\d', parts[-1]):
                    location = parts[-1]
        else:
            desc_parts.append(line)

    if desc_parts:
        description = '\n'.join(desc_parts).strip() or None

    return {
        'title':       header['title'],
        'company':     header['company'],
        'duration':    duration,
        'location':    location,
        'description': description,
        'skills':      skills,
    }


# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------

def _parse_education(lines: list[str]) -> list[dict]:
    entries: list[dict] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == '---':
            continue
        parsed = _parse_bold_dash(stripped)
        if parsed:
            institution, rest = parsed
            # rest could be "2022 – 2026" or "Course, ICP · Jun 2025 – Aug 2025"
            rest_parts = re.split(r'\s*·\s*', rest, maxsplit=1)
            if len(rest_parts) == 2:
                degree = rest_parts[0].strip()
                years  = rest_parts[1].strip()
            else:
                # Determine if rest looks like a year range or a degree
                if re.match(r'^\d{4}', rest):
                    degree, years = None, rest
                else:
                    degree, years = rest, None
            entries.append({'institution': institution, 'degree': degree, 'years': years})
        # Ignore non-bold lines (Skills: etc.) in education section

    return entries if entries else None


# ---------------------------------------------------------------------------
# Certifications
# ---------------------------------------------------------------------------

def _parse_certifications(lines: list[str]) -> list[dict]:
    certs: list[dict] = []
    past_separator = False

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith('|') or not stripped.endswith('|'):
            continue
        cells = [c.strip() for c in stripped.split('|')[1:-1]]
        if len(cells) < 3:
            continue
        if re.match(r'^-+$', cells[0]):
            past_separator = True
            continue
        if not past_separator:
            continue   # header row
        name   = cells[0]
        issuer = cells[1]
        issued = re.sub(r'\s*·\s*ID:.*', '', cells[2]).strip()
        if name.lower() in ('certification', 'name', ''):
            continue
        certs.append({'name': name, 'issuer': issuer, 'issued': issued})

    return certs if certs else None


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

def _parse_projects(lines: list[str]) -> list[dict]:
    projects: list[dict] = []
    current_header: dict | None = None
    content: list[str] = []

    def _flush():
        if current_header:
            projects.append(_finalize_project(current_header, content))

    for line in lines:
        stripped = line.strip()
        if stripped == '---':
            _flush()
            current_header = None
            content = []
            continue

        parsed = _parse_bold_dash(stripped)
        if parsed:
            _flush()
            current_header = {'name': parsed[0], 'date_range': parsed[1]}
            content = []
        elif current_header is not None and stripped:
            content.append(stripped)

    _flush()
    return projects if projects else None


def _finalize_project(header: dict, content: list[str]) -> dict:
    desc_parts: list[str] = []
    github_url = skills = None

    for line in content:
        if line.startswith('GitHub:'):
            github_url = line[7:].strip()
        elif line.startswith('Skills:'):
            skills = [s.strip() for s in line[7:].split(',') if s.strip()]
        else:
            desc_parts.append(line)

    return {
        'name':        header['name'],
        'date_range':  header['date_range'],
        'description': '\n'.join(desc_parts).strip() or None,
        'github_url':  github_url,
        'skills':      skills,
    }


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def _parse_skills(lines: list[str]) -> dict | None:
    technical = soft = None
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith('technical:'):
            raw = stripped[stripped.index(':') + 1:]
            technical = [s.strip() for s in raw.split(',') if s.strip()]
        elif re.match(r'soft\s*skills?:', stripped, re.IGNORECASE):
            raw = stripped[stripped.index(':') + 1:]
            soft = [s.strip() for s in raw.split(',') if s.strip()]
    if technical is None and soft is None:
        return None
    return {'technical': technical, 'soft': soft}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_linkedin(md_path: str) -> dict:
    with open(md_path, encoding='utf-8') as f:
        content = f.read()

    lines = content.splitlines()
    sections = _split_sections(lines)
    header   = _parse_header(sections.pop('__header__', []))

    about = None
    if 'About' in sections:
        text = '\n'.join(sections['About']).strip()
        about = text or None

    return {
        **header,
        'about':          about,
        'experience':     _parse_experience(sections.get('Experience', [])),
        'education':      _parse_education(sections.get('Education', [])),
        'certifications': _parse_certifications(sections.get('Licenses & Certifications', [])),
        'projects':       _parse_projects(sections.get('Projects', [])),
        'skills':         _parse_skills(sections.get('Skills', [])),
    }
