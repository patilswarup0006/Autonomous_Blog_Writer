from __future__ import annotations

import operator
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Patches image capability using your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words per section (150–350 words).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design", "hybrid", "overview", "deep_dive"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


from typing import Union

class RouterDecision(BaseModel):
    needs_research: Union[bool, str] = Field(..., description="Set to true or false boolean")
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: Union[List[str], str] = Field(default_factory=list)
    max_results_per_query: Union[int, str] = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (optimized lightweight tool calling) ----
class ImageSpec(BaseModel):
    section_title: str = Field(..., description="Title of section where image should be placed, e.g. 'Key Components of Agentic AI'")
    alt: str = Field(..., description="Short alt text for image")
    caption: str = Field(..., description="Caption describing the diagram")
    prompt: str = Field(..., description="Detailed prompt for generating the visual diagram or infographic")
    filename: str = Field(default="diagram.png", description="Filename e.g. architecture_flow.png")
    placeholder: str = Field(default="[[IMAGE_1]]")


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM Configuration (Multi-Tier Fallback Strategy)
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

primary_llm = None
fallback_llms = []

# Primary provider: Groq
if groq_api_key and not groq_api_key.startswith("your_"):
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if "70b" in groq_model.lower():
        groq_model = "llama-3.1-8b-instant"
    try:
        from langchain_groq import ChatGroq
        primary_llm = ChatGroq(
            model=groq_model,
            groq_api_key=groq_api_key,
            max_tokens=2048,
            max_retries=5
        )
    except Exception:
        primary_llm = ChatOpenAI(
            model=groq_model,
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            max_tokens=2048,
            max_retries=5
        )

    if groq_model != "llama-3.1-8b-instant":
        try:
            from langchain_groq import ChatGroq
            fallback_llms.append(ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, max_tokens=2048, max_retries=5))
        except Exception:
            fallback_llms.append(ChatOpenAI(model="llama-3.1-8b-instant", base_url="https://api.groq.com/openai/v1", api_key=groq_api_key, max_tokens=2048, max_retries=5))

# Secondary provider fallback: OpenRouter
if openrouter_api_key and not openrouter_api_key.startswith("your_"):
    openrouter_llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        max_tokens=2048,
        max_retries=5
    )
    if primary_llm is None:
        primary_llm = openrouter_llm
    else:
        fallback_llms.append(openrouter_llm)

if primary_llm is None:
    primary_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2048, max_retries=5)

# Attach automatic fallbacks so rate limits failover silently
if fallback_llms:
    llm = primary_llm.with_fallbacks(fallback_llms)
else:
    llm = primary_llm

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–8 high-signal, scoped search query strings as a JSON array of strings, e.g. ["query 1", "query 2"].
"""

def router_node(state: State) -> dict:
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    # Safely convert boolean or string boolean to Python bool
    needs_res = str(decision.needs_research).lower() in ("true", "1", "yes")

    # Safely parse queries whether returned as list or JSON string
    raw_queries = decision.queries
    queries = []
    if isinstance(raw_queries, str):
        try:
            import json
            parsed = json.loads(raw_queries)
            if isinstance(parsed, list):
                queries = [str(x) for x in parsed]
            else:
                queries = [q.strip(' "[]') for q in raw_queries.split(",") if q.strip()]
        except Exception:
            queries = [q.strip(' "[]') for q in raw_queries.split(",") if q.strip()]
    elif isinstance(raw_queries, list):
        queries = [str(x) for x in raw_queries]

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": needs_res,
        "mode": decision.mode,
        "queries": queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 4) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

def research_node(state: State) -> dict:
    queries = (state.get("queries") or [])[:5]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=4))

    if not raw:
        return {"evidence": []}

    # Python-native zero-token extraction from Tavily results
    dedup = {}
    for r in raw:
        url = r.get("url")
        if url and url not in dedup:
            snippet = (r.get("snippet") or "")[:350]
            dedup[url] = EvidenceItem(
                title=r.get("title") or "Untitled",
                url=url,
                snippet=snippet,
                published_at=r.get("published_at"),
                source=r.get("source"),
            )

    evidence = list(dedup.values())

    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [e for e in evidence if (d := _iso_to_date(e.published_at)) and d >= cutoff]

    return {"evidence": evidence}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a concise, punchy, high-quality outline for a technical blog post.

Requirements:
- 4–6 tasks/sections for a clean, focused post.
- Assign 150–350 target words to each section.
- Each section must have 3–5 actionable bullet points.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    plan = state["plan"]
    evidence_concise = [e.model_dump() for e in state.get("evidence", [])[:5]]
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "blog_title": plan.blog_title,
                "audience": plan.audience,
                "tone": plan.tone,
                "blog_kind": plan.blog_kind,
                "constraints": plan.constraints,
                "evidence": evidence_concise,
            },
        )
        for task in plan.tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url}"
        for e in evidence[:5]
    )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {payload.get('blog_title')}\n"
                    f"Audience: {payload.get('audience')}\n"
                    f"Tone: {payload.get('tone')}\n"
                    f"Blog kind: {payload.get('blog_kind')}\n"
                    f"Topic: {payload['topic']}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence URLs:\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Propose 1 to 2 crisp, high-quality technical visual diagrams/infographics for THIS blog post to visually clarify key concepts or architecture.

Rules:
- Include 1 to 2 distinct image objects for key section titles.
- Visual prompts MUST be extremely detailed: specify clean vector infographic style, dark modern UI theme, clear labels, and bright modern color accents.
"""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    section_titles = [t.title for t in plan.tasks]

    try:
        image_plan = planner.invoke(
            [
                SystemMessage(content=DECIDE_IMAGES_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Blog Kind: {plan.blog_kind}\n"
                        f"Available Section Titles:\n" + "\n".join(f"- {t}" for t in section_titles)
                    )
                ),
            ]
        )
    except Exception as e:
        print(f"[Warning] Image decision failed: {e}. Continuing without images.")
        return {"md_with_placeholders": merged_md, "image_specs": []}

    md_with_placeholders = merged_md
    specs_list = []

    for idx, spec in enumerate(image_plan.images[:2], 1):
        placeholder = f"[[IMAGE_{idx}]]"
        spec.placeholder = placeholder
        if not spec.filename or spec.filename == "diagram.png":
            spec.filename = f"diagram_{idx}.png"

        target_heading = (spec.section_title or "").strip()
        inserted = False
        if target_heading:
            lines = md_with_placeholders.splitlines()
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if not inserted and target_heading.lower() in line.lower():
                    new_lines.append("")
                    new_lines.append(placeholder)
                    inserted = True
            if inserted:
                md_with_placeholders = "\n".join(new_lines)

        if not inserted:
            # Fallback: append near beginning of content
            md_with_placeholders = md_with_placeholders.replace("\n\n", f"\n\n{placeholder}\n\n", 1)

        specs_list.append(spec.model_dump())

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": specs_list,
    }


def _pollinations_generate_image_bytes(prompt: str) -> bytes:
    """
    Free fallback image generator using Pollinations AI (no API key required, 100% free).
    """
    import urllib.parse
    import urllib.request

    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _generate_image_bytes_with_fallback(prompt: str) -> bytes:
    """
    Generates image bytes. Tries Google Gemini first (if key exists).
    If Gemini hits 429 quota limits or fails, automatically falls back to Pollinations AI.
    """
    if os.environ.get("GOOGLE_API_KEY"):
        try:
            return _gemini_generate_image_bytes(prompt)
        except Exception as e:
            print(f"[Image Generation Warning] Gemini API failed: {e}. Falling back to Pollinations AI...")

    return _pollinations_generate_image_bytes(prompt)


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # If no images requested, just write merged markdown
    if not image_specs:
        filename = f"{_safe_slug(plan.blog_title)}.md"
        Path(filename).write_text(md, encoding="utf-8")
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # generate only if needed
        if not out_path.exists():
            try:
                img_bytes = _generate_image_bytes_with_fallback(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    return {"final": md}

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()