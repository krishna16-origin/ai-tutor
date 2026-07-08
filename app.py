import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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


class QuizRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    level: Level = Level.student
    provider: Provider = Provider.auto
    language: str = "English"
    num_questions: int = Field(default=10, ge=1, le=50)


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
        model=getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=float(getenv("MODEL_TEMPERATURE", "0.2")),
        max_tokens=int(getenv("MAX_TOKENS", "8192")),
    )


def glm_llm() -> ChatOpenAI:
    api_key = getenv("GLM_API_KEY") or getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GLM_API_KEY or ZHIPUAI_API_KEY")

    return ChatOpenAI(
        api_key=api_key,
        base_url=getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        model=getenv("GLM_MODEL", "GLM-5.2"),
        temperature=float(getenv("MODEL_TEMPERATURE", "0.2")),
        max_tokens=int(getenv("MAX_TOKENS", "8192")),
    )


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
  - Learner Level: {level}
  - Language: {language}

• Never hallucinate facts, equations, citations, or research.
• If uncertain, explicitly state uncertainty.
• Never invent formulas or references.
• If source material is provided, clearly separate:
  - Source-derived content
  - AI-generated explanation

• Backend only.
• Never generate frontend code.
• Never claim that graphs, animations, simulations, or 3D models were rendered.

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
- Plotly
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL INTELLIGENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate visual specifications ONLY if they genuinely improve understanding.

Supported outputs:

• Mermaid
• MathJax
• KaTeX
• Plotly JSON
• SVG Description
• D3 Dataset
• Three.js Scene Specification
• Simulation Pseudocode

Examples

Calculus
→ Function Graph
→ Tangent Animation
→ Derivative Visualization

Physics
→ Force Diagram
→ Motion Animation
→ Velocity Graph

Botany
→ Plant Cell SVG
→ Photosynthesis Pathway
→ Xylem & Phloem Diagram

Chemistry
→ Molecular Model
→ Reaction Energy Graph
→ Orbital Diagram

Computer Science
→ Flowchart
→ Tree Diagram
→ Network Topology
→ Algorithm Animation

Never generate:
- Plotly graphs that don't make sense.
- 3D models that don't improve understanding.
- Random diagrams unrelated to the topic.

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
16. SVG Diagram Specification
17. D3 Data Specification
18. Three.js Scene Specification (only if spatial understanding benefits)
19. Interactive Simulation Specification
20. Animation Specification
21. Real World Applications
22. Historical Background (if relevant)
23. Scientists Behind the Discovery (if relevant)
24. Modern Research (if relevant)
25. Common Mistakes
26. Exam Perspective
27. Interview Questions (if relevant)
28. Practice Problems
29. Project Ideas (if relevant)
30. Further Reading
31. Summary
32. Quiz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VISUAL RELEVANCE RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every visualization must directly correspond to the user's topic.

Examples:

Calculus
→ Function plots
→ Tangent lines
→ Derivative graph
→ Integral area visualization

Linear Algebra
→ Matrix transformation
→ Vector spaces
→ Eigenvector visualization

Mechanics
→ Free-body diagrams
→ Motion trajectories
→ Velocity & acceleration graphs

Electric Circuits
→ Circuit schematic
→ Current flow
→ Voltage graph

Botany
→ Plant anatomy
→ Cell structure
→ Tissue organization
→ Photosynthesis pathway

Chemistry
→ Molecular geometry
→ Reaction coordinate diagram
→ Electron orbitals

Astronomy
→ Orbital mechanics
→ Planetary system
→ Stellar evolution

Computer Science
→ Flowcharts
→ AST
→ State machine
→ Network topology
→ Algorithm execution


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
Topic: {topic}
Question: {question}

Connected source:
{connected_source}

Teach this topic using the required structure.
""",
        ),
    ]
)

QUIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Create a quiz for:

Topic: {topic}
Number of questions: {num_questions}

Include:
- mixed difficulty questions
- answer key
- explanations
- common mistakes
- advanced challenge questions
""",
        ),
    ]
)

SOURCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
Task: {task}

Source text:
{source_text}

Use the source faithfully. Clearly separate source-derived content from AI-generated extensions.
""",
        ),
    ]
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
        "groq_model": getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "glm_model": getenv("GLM_MODEL", "glm-4-plus"),
    }


@app.post("/teach", response_model=AIResponse)
async def teach(request: TeachRequest) -> AIResponse:
    values = {
        "level": request.level.value,
        "language": request.language,
        "history": convert_history(request.history),
        "topic": request.topic,
        "question": request.question or "Teach this topic comprehensively.",
        "connected_source": request.connected_source or "None",
    }

    routing_text = f"{request.topic}\n{request.question or ''}\n{request.connected_source or ''}"

    return await run_routed(
        request.provider,
        TEACH_PROMPT,
        values,
        routing_text,
        request.level,
    )


@app.post("/learn-deep", response_model=AIResponse)
async def learn_deep(request: TeachRequest) -> AIResponse:
    request.level = Level.mariana
    request.question = request.question or "Teach this topic in maximum possible depth."
    if request.provider == Provider.auto:
        request.provider = Provider.glm

    return await teach(request)


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
    values = {
        "level": request.level.value,
        "language": request.language,
        "topic": request.topic,
        "num_questions": request.num_questions,
    }

    return await run_routed(
        request.provider,
        QUIZ_PROMPT,
        values,
        request.topic,
        request.level,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=getenv("HOST", "0.0.0.0"),
        port=int(getenv("PORT", "8000")),
        reload=getenv("RELOAD", "false").lower() == "true",
    )
