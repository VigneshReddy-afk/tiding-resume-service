"""
Tiding Resume Service — Python FastAPI
Groq/Llama-3.3-70B (with JSON mode + retry) → weasyprint HTML→PDF
Produces ATS-safe single-column PDFs that pass Workday, Taleo, Greenhouse, iCIMS, Lever.
"""

import os
import json
import time
import logging
from typing import Optional, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from jinja2 import Template
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("resume-service")

log.info("Importing weasyprint...")
try:
    from weasyprint import HTML as WeasyprintHTML
    log.info("weasyprint imported OK")
except Exception as _wp_err:
    log.error("weasyprint import FAILED: %s", _wp_err)
    WeasyprintHTML = None

app = FastAPI(title="Tiding Resume Service", version="1.0.0")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"


# ── Pydantic Models ──────────────────────────────────────────────────────────

class ExperienceEntry(BaseModel):
    title:       str = ""
    company:     str = ""
    startDate:   Optional[str] = None
    endDate:     Optional[str] = None
    isCurrent:   bool = False
    description: Optional[str] = None

class EducationEntry(BaseModel):
    degree:       str = ""
    fieldOfStudy: str = ""
    school:       str = ""
    endDate:      Optional[str] = None
    grade:        Optional[str] = None

class ProjectEntry(BaseModel):
    title:        str = ""
    description:  Optional[str] = None
    technologies: list[str] = []

class GenerateRequest(BaseModel):
    name:           str
    email:          str
    phone:          Optional[str] = ""
    location:       Optional[str] = ""
    currentRole:    Optional[str] = ""
    skills:         list[str] = []
    experience:     list[ExperienceEntry] = []
    education:      list[EducationEntry] = []
    projects:       list[ProjectEntry] = []
    targetJobTitle: str
    targetIndustry: str = "Technology"
    tone:           str = "professional"

class PdfRequest(BaseModel):
    candidateName:  str
    email:          str = ""
    phone:          Optional[str] = ""
    location:       Optional[str] = ""
    targetJobTitle: Optional[str] = ""
    summary:        str = ""
    skills:         dict = {}
    experience:     list[dict] = []
    education:      list[dict] = []
    projects:       list[dict] = []


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "model": GROQ_MODEL
    }


@app.post("/generate")
async def generate_resume(req: GenerateRequest):
    """Call Groq with JSON mode to generate resume content from candidate profile."""
    if not GROQ_API_KEY:
        raise HTTPException(503, detail="GROQ_API_KEY not configured")

    prompt = _build_prompt(req)
    resume_data = await _call_groq(prompt)

    # Always inject PII from our source — never trust the AI for contact info
    resume_data.update({
        "candidateName":  req.name,
        "email":          req.email,
        "phone":          req.phone or "",
        "location":       req.location or "",
        "targetJobTitle": req.targetJobTitle,
    })
    log.info("Generated resume JSON for %s → %s", req.name, req.targetJobTitle)
    return JSONResponse(content=resume_data)


@app.post("/pdf")
async def generate_pdf(req: PdfRequest):
    """Convert resume JSON → ATS-safe PDF via weasyprint."""
    html  = _render_html(req.dict())
    pdf   = _html_to_pdf(html)
    fname = f"{req.candidateName.replace(' ', '_')}_Resume.pdf"
    log.info("Generated PDF for %s (%d bytes)", req.candidateName, len(pdf))
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'}
    )


# ── Groq call — JSON mode + exponential-backoff retry ───────────────────────

async def _call_groq(user_prompt: str, max_retries: int = 3) -> dict:
    system = (
        "You are an elite resume writer and ATS optimization expert. "
        "Respond with a single valid JSON object only — no markdown fences, "
        "no explanation, no text outside the JSON."
    )
    payload = {
        "model":           GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature":     0.65,
        "max_tokens":      8192,
        "response_format": {"type": "json_object"},   # ← forces valid JSON, eliminates parse errors
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    last_error = "Unknown error"
    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(max_retries):
            try:
                r = await client.post(GROQ_URL, json=payload, headers=headers)
                r.raise_for_status()
                data    = r.json()
                content = data["choices"][0]["message"]["content"]
                result  = json.loads(content)
                log.info("Groq responded on attempt %d", attempt + 1)
                return result

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:300]}"
                log.warning("Groq attempt %d failed: %s", attempt + 1, last_error)
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait = 2 ** attempt          # 1s, 2s, 4s
                    log.info("Rate-limited — waiting %ds before retry", wait)
                    time.sleep(wait)
                else:
                    break

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                last_error = f"Parse error: {e}"
                log.warning("Groq parse error attempt %d: %s", attempt + 1, last_error)
                if attempt < max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                last_error = str(e)
                log.warning("Groq attempt %d unexpected error: %s", attempt + 1, last_error)
                if attempt < max_retries - 1:
                    time.sleep(1)

    raise HTTPException(500, detail=f"Groq failed after {max_retries} attempts: {last_error}")


# ── Prompt Builder ────────────────────────────────────────────────────────────

def _build_prompt(req: GenerateRequest) -> str:
    target   = req.targetJobTitle
    industry = req.targetIndustry

    exp_text = ""
    for e in req.experience:
        end = "Present" if e.isCurrent else (e.endDate or "")
        exp_text += f"Title: {e.title}\nCompany: {e.company}\nPeriod: {e.startDate or ''} to {end}\n"
        if e.description:
            exp_text += f"Description: {e.description}\n"
        exp_text += "\n"

    edu_text = ""
    for e in req.education:
        edu_text += f"{e.degree} in {e.fieldOfStudy} at {e.school}"
        if e.endDate: edu_text += f", {e.endDate}"
        if e.grade:   edu_text += f", GPA {e.grade}"
        edu_text += "\n"

    proj_text = ""
    for p in req.projects:
        proj_text += f"Name: {p.title}\n"
        if p.description:  proj_text += f"Description: {p.description}\n"
        if p.technologies: proj_text += f"Technologies: {', '.join(p.technologies)}\n"
        proj_text += "\n"

    return f"""Generate a complete ATS-optimized resume for this candidate. Return ONE JSON object.

TARGET ROLE: {target}
TARGET INDUSTRY: {industry}
TONE: {req.tone}

CANDIDATE DATA:
Name: {req.name}
Current Role: {req.currentRole or "Not specified"}
Skills on profile: {", ".join(req.skills) if req.skills else "None listed"}

WORK EXPERIENCE:
{exp_text.strip() if exp_text.strip() else f"No experience listed — generate 2 realistic senior roles typical for a {target} in {industry}"}

EDUCATION:
{edu_text.strip() if edu_text.strip() else "Not provided"}

PROJECTS:
{proj_text.strip() if proj_text.strip() else "None"}

=== MANDATORY RULES — EVERY ONE MUST BE FOLLOWED ===

SUMMARY — write 4 sentences naturally, like a senior recruiter wrote them:
  1. X years of hands-on experience as a [job title] specializing in [2-3 key technologies for {target}].
  2. Biggest career achievement with a specific, believable metric — e.g. "Led migration that cut infra costs by $180K/year" or "Built payment system processing $4M/month with 99.97% uptime".
  3. Core technical depth — specific frameworks, architecture patterns, tools the candidate masters.
  4. What makes them uniquely valuable for a {target} role at a {industry} company.
  NEVER write phrases like "which are relevant to the role", "which directly matches", or "as per the target role". Write like a human, not a template.

EXPERIENCE — for EACH job entry, generate EXACTLY 5-6 bullet points:
  - Every bullet starts with a strong past-tense action verb: Led / Built / Architected / Engineered / Designed / Implemented / Deployed / Optimized / Reduced / Increased / Automated / Spearheaded / Delivered / Scaled / Streamlined / Refactored / Migrated / Established / Launched / Mentored
  - Every bullet contains at least one quantified result with a number.
  - GREAT: "Architected event-driven microservices on AWS ECS, reducing average API latency from 420ms to 95ms and saving $60K/year in compute costs"
  - GREAT: "Led 5-engineer team to ship mobile checkout feature used by 800K users within 6 weeks, increasing conversion by 18%"
  - BAD: "Worked on improving system performance" — NEVER write this
  - If the job description is vague or missing, infer realistic achievements from the job title, company type, and {industry} context. Use plausible ranges (e.g., "team of 4–8 engineers", "20–40% improvement").
  - Naturally embed keywords ATS systems scan for in {target} job postings.

SKILLS:
  - technical: exactly 12–15 items. Core languages + frameworks + cloud platforms + databases + DevOps tools that a strong {target} in {industry} uses. Use candidate's listed skills as a base; add the most in-demand tools for this role. CI/CD, Docker, Kubernetes, Git, Agile, Scrum ALL belong here.
  - soft: exactly 4–5 items. Human/leadership competencies ONLY — no tools, no methodologies. Examples: "Cross-functional Team Leadership", "Executive Stakeholder Communication", "Technical Mentorship", "Strategic Problem Solving".

EDUCATION: Use exactly as provided. Format year as YYYY only. Leave gpa as empty string if unknown.

PROJECTS: Include only if candidate has real projects. Each must have a quantified impact statement.

Return this exact JSON structure:
{{
  "summary": "4-sentence paragraph written naturally...",
  "skills": {{
    "technical": ["skill1", "skill2", "...", "skill13"],
    "soft": ["Human competency 1", "...", "Human competency 4"]
  }},
  "experience": [
    {{
      "title": "Exact Job Title",
      "company": "Company Name",
      "duration": "Month YYYY – Month YYYY",
      "bullets": [
        "Led ... achieving X% improvement in Y",
        "Built ... reducing Z by N",
        "Architected ... serving M users",
        "Optimized ... cutting costs by $K",
        "Delivered ... ahead of schedule"
      ]
    }}
  ],
  "education": [
    {{
      "degree": "Full degree name",
      "school": "University name",
      "year": "YYYY",
      "gpa": ""
    }}
  ],
  "projects": []
}}"""


# ── HTML Resume Template ─────────────────────────────────────────────────────
# Single-column, semantic HTML5.
# No float, no flexbox, no CSS columns — pure linear flow.
# Passes text extraction in Workday, Taleo, Greenhouse, iCIMS, Lever.

RESUME_HTML = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @page {
    size: A4;
    margin: 0.75in 1in 0.75in 1in;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: "Liberation Sans", Arial, Helvetica, sans-serif;
    font-size: 10pt;
    color: #1a1a1a;
    line-height: 1.45;
    background: #ffffff;
  }

  /* ── Header ── */
  .name {
    font-size: 22pt;
    font-weight: 700;
    color: #111111;
    letter-spacing: -0.3px;
    margin-bottom: 5px;
  }

  .contact-line {
    font-size: 9pt;
    color: #555555;
    margin-bottom: 12px;
  }

  .contact-sep {
    color: #999;
    margin: 0 6px;
  }

  hr.top-rule {
    border: none;
    border-top: 1.5px solid #aaaaaa;
    margin-bottom: 0;
  }

  /* ── Section headers ── */
  h2 {
    font-size: 9.5pt;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #111111;
    margin-top: 16px;
    margin-bottom: 4px;
    padding-bottom: 3px;
    border-bottom: 0.75pt solid #bbbbbb;
  }

  /* ── Summary ── */
  .summary {
    font-size: 9.5pt;
    color: #222222;
    line-height: 1.55;
    margin-top: 6px;
  }

  /* ── Experience ── */
  .exp-block {
    margin-top: 9px;
  }

  /* Two-column row: title+company left, date right */
  /* Uses a table (safe for ATS — reading order: td[0] then td[1]) */
  table.exp-header {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 3px;
  }

  table.exp-header td {
    padding: 0;
    vertical-align: bottom;
  }

  .exp-title {
    font-size: 10pt;
    font-weight: 700;
    color: #111111;
  }

  .exp-company {
    font-size: 9.5pt;
    font-style: italic;
    color: #444444;
  }

  .exp-date {
    font-size: 8.5pt;
    color: #777777;
    text-align: right;
    white-space: nowrap;
    width: 1%;
    padding-left: 12px;
  }

  ul.bullets {
    margin-left: 13pt;
    margin-top: 2px;
    list-style-type: disc;
  }

  ul.bullets li {
    font-size: 9.5pt;
    color: #222222;
    margin-bottom: 2px;
    line-height: 1.45;
  }

  /* ── Education ── */
  table.edu-header {
    width: 100%;
    border-collapse: collapse;
    margin-top: 9px;
  }

  table.edu-header td {
    padding: 0;
    vertical-align: top;
  }

  .edu-degree {
    font-size: 10pt;
    font-weight: 700;
    color: #111111;
  }

  .edu-school {
    font-size: 9.5pt;
    font-style: italic;
    color: #444444;
  }

  .edu-date {
    font-size: 8.5pt;
    color: #777777;
    text-align: right;
    white-space: nowrap;
    width: 1%;
    padding-left: 12px;
  }

  /* ── Skills ── */
  .skills-line {
    font-size: 9.5pt;
    margin-top: 6px;
    line-height: 1.5;
    color: #222222;
  }

  .skills-label {
    font-weight: 700;
    color: #111111;
  }

  /* ── Projects ── */
  .proj-block {
    margin-top: 9px;
  }

  .proj-name {
    font-size: 10pt;
    font-weight: 700;
    color: #111111;
    margin-bottom: 2px;
  }

  .proj-desc {
    font-size: 9.5pt;
    color: #333333;
    margin-bottom: 2px;
  }

  .proj-meta {
    font-size: 9pt;
    color: #555555;
    margin-bottom: 1px;
  }

  .proj-meta-label {
    font-weight: 700;
    color: #333333;
  }
</style>
</head>
<body>

<!-- ── HEADER ── -->
<div class="name">{{ name }}</div>
<div class="contact-line">
  {%- set parts = [] -%}
  {%- if email    %}{%- set _ = parts.append(email)    %}{%- endif -%}
  {%- if phone    %}{%- set _ = parts.append(phone)    %}{%- endif -%}
  {%- if location %}{%- set _ = parts.append(location) %}{%- endif -%}
  {{ parts | join(' <span class="contact-sep">|</span> ') }}
</div>
<hr class="top-rule">

<!-- ── SUMMARY ── -->
{% if summary %}
<h2>Professional Summary</h2>
<p class="summary">{{ summary }}</p>
{% endif %}

<!-- ── EXPERIENCE ── -->
{% if experience %}
<h2>Experience</h2>
{% for exp in experience %}
<div class="exp-block">
  <table class="exp-header">
    <tr>
      <td>
        <span class="exp-title">{{ exp.title }}</span>
        {% if exp.company %}<span class="exp-company"> — {{ exp.company }}</span>{% endif %}
      </td>
      <td class="exp-date">{{ exp.duration }}</td>
    </tr>
  </table>
  {% if exp.bullets %}
  <ul class="bullets">
    {% for b in exp.bullets %}<li>{{ b }}</li>{% endfor %}
  </ul>
  {% endif %}
</div>
{% endfor %}
{% endif %}

<!-- ── EDUCATION ── -->
{% if education %}
<h2>Education</h2>
{% for edu in education %}
<table class="edu-header">
  <tr>
    <td>
      <div class="edu-degree">{{ edu.degree }}</div>
      <div class="edu-school">{{ edu.school }}</div>
    </td>
    <td class="edu-date">
      {{ edu.year }}{% if edu.gpa and edu.gpa != '' %} &middot; GPA {{ edu.gpa }}{% endif %}
    </td>
  </tr>
</table>
{% endfor %}
{% endif %}

<!-- ── SKILLS ── -->
{% if skills %}
<h2>Skills</h2>
{% if skills.technical %}
<p class="skills-line"><span class="skills-label">Technical: </span>{{ skills.technical | join(', ') }}</p>
{% endif %}
{% if skills.soft %}
<p class="skills-line"><span class="skills-label">Core Competencies: </span>{{ skills.soft | join(', ') }}</p>
{% endif %}
{% endif %}

<!-- ── PROJECTS ── -->
{% if projects and projects | length > 0 %}
<h2>Projects</h2>
{% for proj in projects %}
<div class="proj-block">
  <div class="proj-name">{{ proj.name }}</div>
  {% if proj.description %}<p class="proj-desc">{{ proj.description }}</p>{% endif %}
  {% if proj.technologies %}<p class="proj-meta"><span class="proj-meta-label">Technologies: </span>{{ proj.technologies }}</p>{% endif %}
  {% if proj.impact       %}<p class="proj-meta"><span class="proj-meta-label">Impact: </span>{{ proj.impact }}</p>{% endif %}
</div>
{% endfor %}
{% endif %}

</body>
</html>""")


def _render_html(data: dict) -> str:
    return RESUME_HTML.render(
        name=data.get("candidateName", data.get("name", "")),
        email=data.get("email", ""),
        phone=data.get("phone", ""),
        location=data.get("location", ""),
        summary=data.get("summary", ""),
        skills=data.get("skills", {}),
        experience=data.get("experience", []),
        education=data.get("education", []),
        projects=[p for p in data.get("projects", []) if p.get("name")],
    )


def _html_to_pdf(html_str: str) -> bytes:
    if WeasyprintHTML is None:
        raise HTTPException(503, detail="weasyprint unavailable — check server logs")
    return WeasyprintHTML(string=html_str).write_pdf()
