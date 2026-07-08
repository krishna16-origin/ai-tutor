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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()


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
        model=getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
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
        model=getenv("GLM_MODEL", "glm-4-plus"),
        temperature=float(getenv("MODEL_TEMPERATURE", "0.2")),
        max_tokens=int(getenv("MAX_TOKENS", "8192")),
    )


SYSTEM_PROMPT = """
You are DeepLearn AI, a backend-only AI education engine.

Teach any educational topic with clarity, accuracy, depth, and adaptive pedagogy.

Rules:
- Adapt to learner level: {level}
- Reply in: {language}
- Never hallucinate facts, equations, citations, or research claims.
- If uncertain, clearly say what is uncertain.
- If source text is provided, separate "Source-derived content" from "AI-generated explanation".
- No frontend code.
- For visuals, return backend-friendly specs: Mermaid, MathJax, Plotly JSON, D3 data,
  SVG description, Three.js scene spec, or simulation pseudocode.
- Do not claim that graphs, animations, or 3D models were rendered.

Teaching style:
1. Explain like a child.
2. Explain like a college professor.
3. Explain like a research scientist.

Answer structure:
1. Topic Overview
2. Simple Explanation
3. Deep Explanation
4. Scientific Explanation
5. Mathematical Foundation
6. Visual Explanation
7. Interactive Simulation Spec
8. Animation Spec
9. Graph Spec
10. 3D Model Spec
11. Real World Examples
12. Historical Timeline
13. Scientists Behind Discovery
14. Modern Research
15. Future Scope
16. Common Mistakes
17. Exam Perspective
18. Interview Questions
19. Research Questions
20. Summary
21. Quiz
22. Assignments
23. Projects
24. Further Reading

If level is Mariana Trench Mode, explain from fundamentals, derive equations,
show assumptions, proofs, limits, edge cases, history, applications, and research depth.
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


app = FastAPI(
    title="DeepLearn AI Backend",
    version="1.0.0",
    description="Pure backend LangChain API using Groq and GLM models.",
)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "groq_configured": bool(getenv("GROQ_API_KEY")),
        "glm_configured": bool(getenv("GLM_API_KEY") or getenv("ZHIPUAI_API_KEY")),
        "groq_model": getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "glm_model": getenv("GLM_MODEL", "GLM-5.2"),
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
        "backend:app",
        host=getenv("HOST", "0.0.0.0"),
        port=int(getenv("PORT", "8000")),
        reload=getenv("RELOAD", "false").lower() == "true",
    )
