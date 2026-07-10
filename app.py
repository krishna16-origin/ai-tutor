import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="DeepLearn AI Backend",
    version="1.0.0",
    description="Pure backend LangChain API using Groq and GLM models.",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Conversational Memory Store (session-based) ───
conversation_store: Dict[str, List[BaseMessage]] = {}


class Level(str, Enum):
    beginner = "Beginner"
    student = "Student"
    college = "College"
    engineer = "Engineer"
    researcher = "Researcher"
    professor = "Professor"
    scientist = "Scientist"
    mariana = "Mariana Trench Mode"


class Provider(str, Enum):
    auto = "auto"
    groq = "groq"
    glm = "glm"
    both = "both"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TeachRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    question: Optional[str] = None
    level: Level = Level.student
    provider: Provider = Provider.auto
    language: str = "English"
    connected_source: Optional[str] = None
    history: List[ChatTurn] = Field(default_factory=list)
    session_id: str = ""  # for conversational memory
    new_session: bool = False  # True = fresh start, False = continue existing


class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    level: Level = Level.student
    provider: Provider = Provider.auto
    language: str = "English"
    num_questions: int = Field(default=10, ge=1, le=50)
    session_id: str = ""
    use_context: bool = False  # True = quiz about previous conversation, False = standalone


class SourceRequest(BaseModel):
    source_text: str = Field(..., min_length=1)
    task: Literal[
        "summary",
        "deep_explanation",
        "knowledge_graph",
        "formula_extraction",
        "definitions",
        "quiz",
        "assignments",
        "research_notes",
    ] = "deep_explanation"
    level: Level = Level.student
    provider: Provider = Provider.auto
    language: str = "English"


class DeepRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    question: Optional[str] = None
    level: Level = Level.mariana
    provider: Provider = Provider.glm
    language: str = "English"
    connected_source: Optional[str] = None
    history: List[ChatTurn] = Field(default_factory=list)
    session_id: str = ""
    new_session: bool = False


class AIResponse(BaseModel):
    provider_used: str
    answer: str
    raw_outputs: Optional[Dict[str, str]] = None


def getenv(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name) or default


def groq_llm() -> ChatGroq:
    if not getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY")

    return ChatGroq(
        api_key=getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
        temperature=float(getenv("MODEL_TEMPERATURE", "0.2")),
        max_tokens=int(getenv("MAX_TOKENS", "4096")),
    )


def glm_llm() -> ChatOpenAI:
    api_key = getenv("GLM_API_KEY") or getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GLM_API_KEY or ZHIPUAI_API_KEY")

    return ChatOpenAI(
        api_key=api_key,
        base_url=getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        model="glm-5.2",
        temperature=float(getenv("MODEL_TEMPERATURE", "0.2")),
        max_tokens=int(getenv("MAX_TOKENS", "8192")),
    )


# ─── Conversational Memory Helpers ───

def get_session_messages(session_id: str) -> List[BaseMessage]:
    """Retrieve stored messages for a session, or empty list for new session."""
    if not session_id:
        return []
    return conversation_store.get(session_id, [])


def save_session_messages(session_id: str, messages: List[BaseMessage]):
    """Persist conversation history for a session."""
    if session_id:
        conversation_store[session_id] = messages


def clear_session(session_id: str):
    """Clear conversation memory for a session."""
    if session_id and session_id in conversation_store:
        del conversation_store[session_id]


def build_messages_with_context(
    history: List[ChatTurn],
    session_id: str,
    new_session: bool,
    topic: str,
    mode: str = "teach",
) -> List[BaseMessage]:
    """
    Build message list combining:
    - Fresh context when new_session=True (start clean)
    - Previous conversation memory when new_session=False (continue chat)
    """
    messages: List[BaseMessage] = []

    if new_session or not session_id:
        # Fresh chat: only use the explicit history from the request
        for turn in history:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
    else:
        # Continue chat: load stored session memory + new history
        stored = get_session_messages(session_id)
        messages.extend(stored)

        # Add any new explicit history turns
        for turn in history:
            messages.append(HumanMessage(content=turn.content))

    return messages


def update_session(session_id: str, user_msg: str, ai_msg: str):
    """Append user and AI messages to session store."""
    if session_id:
        if session_id not in conversation_store:
            conversation_store[session_id] = []
        conversation_store[session_id].append(HumanMessage(content=user_msg))
        conversation_store[session_id].append(AIMessage(content=ai_msg))


# ─── Updated System Prompt (Visual Libraries: plotly, pyvista/VTK, Three.js, 3Dmol.js, Manim) ───

SYSTEM_PROMPT = """
You are DeepLearn AI, a backend-only adaptive education engine.

Your goal is to teach ONLY what is relevant to the user's topic.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Detect the academic domain automatically before answering.

Supported domains include (but are not limited to):
- Mathematics
- Physics
- Chemistry
- Biology
- Botany
- Zoology
- Computer Science
- Engineering
- Economics
- Statistics
- Medicine
- Geography
- History
- Literature
- Languages
- Astronomy
- Philosophy

• Adapt explanations to:
  - Learner Level: {{level}}
  - Language: {{language}}

• Never hallucinate facts, equations, citations, or research.
• If uncertain, explicitly state uncertainty.
• Never invent formulas or references.
• If source material is provided, clearly separate:
  - Source-derived content
  - AI-generated explanation

• Backend only.
• Never generate frontend code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT AWARENESS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate ONLY sections that are relevant.

Never include unrelated headings.

Examples

Mathematics:
Include:
- Formula derivations
- Proofs
- Graphs
- Plotly visualizations
- Geometry diagrams
- Worked examples

Do NOT include:
- Plant anatomy
- DNA
- Biology pathways

Physics:
Include:
- Mathematical derivations
- Force diagrams
- Motion simulations
- Vector diagrams
- Plotly graphs
- Experiments

Do NOT include:
- Grammar
- Literature
- Plant biology

Biology / Botany:
Include:
- Cell structure
- Biological mechanisms
- Plant anatomy
- Microscopy
- Life cycle
- Biological pathways

Do NOT include:
- Fourier transforms
- Calculus proofs
- Circuit analysis

Chemistry:
Include:
- Molecular structures
- Chemical reactions
- Energy diagrams
- Bonding
- Reaction mechanisms

Computer Science:
Include:
- Algorithms
- Flowcharts
- Pseudocode
- Data structures
- Complexity analysis
- Architecture diagrams

Do NOT include:
- Cell biology
- Anatomy
- Literature analysis




- Manim Animation Script (for mathematical animations)
- SVG Description

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT — MANDATORY, MATCH EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The renderer ONLY converts a code block into a visual if it uses the EXACT
fenced code block tag below, with valid JSON containing the EXACT keys
shown. Untagged code blocks, prose descriptions, or wrong key names will
render as plain code and NOT as a visual. Never deviate from these tags.

Mermaid:
```mermaid
graph TD
  A[Start] --> B[End]
```

Plotly (top-level keys must be exactly "data" and "layout"):
```plotly
{ {"data": [{"x": [1,2,3], "y": [4,5,6], "type": "scatter"}], "layout": { {"title": "Example"} }} }
```

Three.js (top-level key must be "objects", each with "type"/"color"):
```threejs
{ {"objects": [{"type": "sphere", "color": "#60a5fa", "size": 1, "position": { {"x": 0, "y": 0, "z": 0} } } }], "cameraDistance": 6} }
```

3Dmol.js (top-level key "molecule_type" + "data", or "smiles"):
```3dmol
{ {"molecule_type": "pdb", "data": "<PDB block>", "style": { {"stick": { {"colorscheme": "greenCarbon"} }} } , "zoom": true, "label": "Caffeine"} }
```

PyVista (top-level key "objects", each with "type"/"color"):
```pyvista
{ {"objects": [{"type": "sphere", "color": "#60a5fa", "size": 1, "position": { {"x": 0, "y": 0, "z": 0} } } }]} }
```

Manim (use "script" for a text description, or "elements" for shapes):
```manim
{ {"script": "Animate a circle transforming into a square"} }
```

SVG (top-level key "elements"):
```svg-spec
{ {"width": 600, "height": 400, "elements": [{"type": "circle", "cx": 300, "cy": 200, "r": 50, "stroke": "#60a5fa", "label": "Nucleus"}]} }
```

Do NOT wrap these in ```json. Do NOT describe the visual in prose instead
of emitting the JSON. Do NOT invent new tag names. Use ONLY the tags above.

Examples

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEACHING STYLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Level 1
Explain like a child.

Level 2
Explain like a college professor.

Level 3
Explain like a research scientist.

If learner level = Mariana Trench Mode:

- Start from first principles.
- Derive equations step-by-step.
- State assumptions.
- Explain proofs.
- Explain edge cases.
- Discuss limitations.
- Compare alternative models.
- Connect theory to real-world applications.
- Mention current research where relevant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DYNAMIC RESPONSE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate ONLY applicable sections.

Possible sections:

1. Topic Overview
2. Core Concepts
3. Simple Explanation
4. Deep Explanation
5. Scientific Explanation
6. Mathematical Foundation (if applicable)
7. Formula Derivation (if applicable)
8. Proof (if applicable)
9. Algorithms (Computer Science only)
10. Biological Mechanism (Biology only)
11. Chemical Reaction Mechanism (Chemistry only)
12. Visual Explanation
13. Mermaid Diagram
14. MathJax Equations
15. Plotly Graph Specification (if meaningful)
16. PyVista/VTK 3D Visualization (for spatial/scientific topics)
17. SVG Diagram Specification
18. Three.js Scene Specification (only if spatial understanding benefits)
19. 3Dmol.js Molecular Specification (for chemistry/molecular topics)
20. Manim Animation Script (for mathematical animations)
21. Interactive Simulation Specification
22. Real World Applications
23. Historical Background (if relevant)
24. Scientists Behind the Discovery (if relevant)
25. Modern Research (if relevant)
26. Common Mistakes
27. Exam Perspective
28. Interview Questions (if relevant)
29. Practice Problems
30. Project Ideas (if relevant)
31. Further Reading
32. Summary
33. Quiz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL RELEVANCE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every visualization must directly correspond to the user's topic.

Examples:

Calculus
→ Function plots (Plotly)
→ Tangent lines (Manim)
→ Derivative graph (Plotly)
→ Integral area visualization (Manim)

Linear Algebra
→ Matrix transformation (Plotly)
→ Vector spaces (PyVista)
→ Eigenvector visualization (Plotly)

Mechanics
→ Free-body diagrams (Mermaid)
→ Motion trajectories (Plotly)
→ Velocity & acceleration graphs (Plotly)

Electric Circuits
→ Circuit schematic (Mermaid)
→ Current flow (SVG)
→ Voltage graph (Plotly)

Botany
→ Plant anatomy (SVG)
→ Cell structure (SVG)
→ Tissue organization (Mermaid)
→ Photosynthesis pathway (Mermaid)

Chemistry
→ Molecular geometry (3Dmol.js)
→ Reaction coordinate diagram (Plotly)
→ Electron orbitals (PyVista)

Astronomy
→ Orbital mechanics (PyVista)
→ Planetary system (Three.js)
→ Stellar evolution (Plotly)

Computer Science
→ Flowcharts (Mermaid)
→ AST (Mermaid)
→ State machine (Mermaid)
→ Network topology (SVG)
→ Algorithm execution (Manim)


FINAL RULE


Do NOT force every section into every answer.

Only generate content that genuinely helps explain the detected subject.

Every explanation, formula, graph, animation, simulation, diagram, and 3D specification must be directly relevant to the topic.

Avoid generic templates. Adapt dynamically to the user's subject, learner level, and educational needs.
"""

TEACH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        (
            "human",
            """
Topic: {{topic}}
Question: {{question}}

Connected source:
{{connected_source}}

Mode: {{mode}}

Teach this topic using the required structure.
If this is a continuation of a previous conversation, reference the prior discussion context.
""",
        ),
    ],
    template_format="jinja2"
)

QUIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("context_messages"),
        (
            "human",
            """
Create a quiz for:

Topic: {{topic}}
Number of questions: {{num_questions}}

{{quiz_context}}

Follow this EXACT format so the quiz renders in the interactive quiz box:

## Quiz

1. First question text?
a) Option one
b) Option two
c) Option three
d) Option four

2. Second question text?
a) Option one
b) Option two
c) Option three
d) Option four

(continue numbering sequentially — one question per number, exactly 4
lettered options a) through d) for every question, never more or fewer)

Answer Key:
1. b
2. a
(continue for every question — number, period, space, single correct
letter, nothing else on each line)

Explanations:
- Explain question 1: why the correct answer is right and why the
  common wrong answers are tempting.
- Explain question 2.
(continue for every question)

Advanced Challenge:
- Include 1-2 harder bonus questions here in plain prose, NOT numbered
  like the quiz above, so they aren't mistaken for quiz questions.

Rules:
- Mix difficulty levels across the numbered questions.
- Never use a)/b)/c)/d) style lettering anywhere outside the quiz
  questions and Answer Key.
- The Answer Key must come immediately after the last question,
  before Explanations.
""",
        ),
    ],
    template_format="jinja2"
)

SOURCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Task: {{task}}

Source text:
{{source_text}}

Use the source faithfully. Clearly separate source-derived content from AI-generated extensions.
""",
        ),
    ],
    template_format="jinja2"
)


def convert_history(history: List[ChatTurn]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []

    for turn in history:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))

    return messages


def choose_provider(provider: Provider, text: str, level: Level) -> Provider:
    if provider != Provider.auto:
        return provider

    deep_keywords = [
        "derive",
        "proof",
        "research",
        "paper",
        "phd",
        "advanced",
        "mathematical",
        "scientific",
        "theorem",
        "mechanism",
        "deep",
    ]

    if level in {Level.researcher, Level.professor, Level.scientist, Level.mariana}:
        return Provider.glm

    if any(word in text.lower() for word in deep_keywords):
        return Provider.glm

    return Provider.groq


def get_model(provider: Provider):
    if provider == Provider.groq:
        return groq_llm()
    if provider == Provider.glm:
        return glm_llm()
    raise ValueError("Provider must be groq or glm")


async def run_model(provider: Provider, prompt: ChatPromptTemplate, values: Dict[str, Any]) -> str:
    chain = prompt | get_model(provider) | StrOutputParser()
    return await chain.ainvoke(values)


async def run_routed(
    provider: Provider,
    prompt: ChatPromptTemplate,
    values: Dict[str, Any],
    routing_text: str,
    level: Level,
) -> AIResponse:
    selected = choose_provider(provider, routing_text, level)

    if selected == Provider.both:
        outputs: Dict[str, str] = {}

        for candidate in [Provider.groq, Provider.glm]:
            try:
                outputs[candidate.value] = await run_model(candidate, prompt, values)
            except Exception as exc:
                outputs[candidate.value] = f"ERROR: {exc}"

        valid = {k: v for k, v in outputs.items() if not v.startswith("ERROR:")}
        if not valid:
            raise HTTPException(status_code=502, detail=outputs)

        if "glm" in valid:
            return AIResponse(provider_used="both", answer=valid["glm"], raw_outputs=outputs)

        return AIResponse(provider_used="both", answer=next(iter(valid.values())), raw_outputs=outputs)

    try:
        answer = await run_model(selected, prompt, values)
        return AIResponse(provider_used=selected.value, answer=answer)
    except Exception as primary_error:
        fallback = Provider.glm if selected == Provider.groq else Provider.groq

        try:
            answer = await run_model(fallback, prompt, values)
            return AIResponse(
                provider_used=f"{fallback.value}:fallback",
                answer=answer,
                raw_outputs={selected.value: str(primary_error), fallback.value: answer},
            )
        except Exception as fallback_error:
            raise HTTPException(
                status_code=502,
                detail={
                    "primary_error": str(primary_error),
                    "fallback_error": str(fallback_error),
                },
            )


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "groq_configured": bool(getenv("GROQ_API_KEY")),
        "glm_configured": bool(getenv("GLM_API_KEY") or getenv("ZHIPUAI_API_KEY")),
        "groq_model": "openai/gpt-oss-120b",
        "glm_model": "GLM-5.2",
    }


@app.post("/teach", response_model=AIResponse)
async def teach(request: TeachRequest) -> AIResponse:
    mode = "teach"

    # Build conversation context using memory store
    messages = build_messages_with_context(
        history=request.history,
        session_id=request.session_id,
        new_session=request.new_session,
        topic=request.topic,
        mode=mode,
    )

    # If no messages at all, use empty placeholder
    if not messages:
        messages = convert_history(request.history)

    values = {
        "level": request.level.value,
        "language": request.language,
        "history": messages,
        "topic": request.topic,
        "question": request.question or "Teach this topic comprehensively.",
        "connected_source": request.connected_source or "None",
        "mode": mode,
    }

    routing_text = f"{request.topic}\n{request.question or ''}\n{request.connected_source or ''}"

    response = await run_routed(
        request.provider,
        TEACH_PROMPT,
        values,
        routing_text,
        request.level,
    )

    # Save to conversational memory
    if request.session_id:
        update_session(
            request.session_id,
            request.question or request.topic,
            response.answer,
        )

    return response


@app.post("/learn-deep", response_model=AIResponse)
async def learn_deep(request: DeepRequest) -> AIResponse:
    mode = "deep"

    request.level = Level.mariana
    request.question = request.question or "Teach this topic in maximum possible depth."
    if request.provider == Provider.auto:
        request.provider = Provider.glm

    # Build conversation context using memory store
    messages = build_messages_with_context(
        history=request.history,
        session_id=request.session_id,
        new_session=request.new_session,
        topic=request.topic,
        mode=mode,
    )

    values = {
        "level": request.level.value,
        "language": request.language,
        "history": messages,
        "topic": request.topic,
        "question": request.question or "Teach this topic comprehensively.",
        "connected_source": request.connected_source or "None",
        "mode": mode,
    }

    routing_text = f"{request.topic}\n{request.question or ''}\n{request.connected_source or ''}"

    response = await run_routed(
        request.provider,
        TEACH_PROMPT,
        values,
        routing_text,
        request.level,
    )

    # Save to conversational memory
    if request.session_id:
        update_session(
            request.session_id,
            request.question or request.topic,
            response.answer,
        )

    return response


@app.post("/connectors", response_model=AIResponse)
async def connectors(request: SourceRequest) -> AIResponse:
    values = {
        "level": request.level.value,
        "language": request.language,
        "source_text": request.source_text,
        "task": request.task,
    }

    return await run_routed(
        request.provider,
        SOURCE_PROMPT,
        values,
        request.source_text,
        request.level,
    )


@app.post("/quiz", response_model=AIResponse)
async def quiz(request: QuizRequest) -> AIResponse:
    if request.use_context and request.session_id:
        # Quiz based on previous conversation context
        stored_messages = get_session_messages(request.session_id)
        context_str = "Based on the following previous conversation:\n\n" + "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content[:500]}" for m in stored_messages[-10:]]
        ) if stored_messages else ""
    else:
        context_str = "Generate a standalone quiz about this topic from scratch."

    values = {
        "level": request.level.value,
        "language": request.language,
        "topic": request.topic,
        "num_questions": request.num_questions,
        "context_messages": stored_messages if request.use_context else [],
        "quiz_context": context_str,
    }

    return await run_routed(
        request.provider,
        QUIZ_PROMPT,
        values,
        request.topic,
        request.level,
    )


@app.post("/clear-session", response_model=Dict[str, str])
async def clear_session_endpoint(data: Dict[str, str]):
    """Clear conversational memory for a given session."""
    session_id = data.get("session_id", "")
    if session_id:
        clear_session(session_id)
        return {"status": "ok", "message": f"Session {session_id} cleared"}
    return {"status": "ok", "message": "No session to clear"}


@app.get("/session-info")
async def session_info(session_id: str = "") -> Dict[str, Any]:
    """Return info about a session's conversation memory."""
    if session_id and session_id in conversation_store:
        msgs = conversation_store[session_id]
        return {
            "session_id": session_id,
            "message_count": len(msgs),
            "topics": [m.content[:80] for m in msgs[:20]],
        }
    return {"session_id": session_id, "message_count": 0, "topics": []}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=getenv("HOST", "0.0.0.0"),
        port=int(getenv("PORT", "8000")),
        reload=getenv("RELOAD", "false").lower() == "true",
    )
