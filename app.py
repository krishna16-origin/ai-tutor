
import asyncio
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Literal, Optional

from langchain_community.document_loaders import ArxivLoader, PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


class TeachingLevel(str, Enum):
    BEGINNER = "beginner"
    STUDENT = "student"
    COLLEGE = "college"
    RESEARCHER = "researcher"
    PROFESSOR = "professor"
    SCIENTIST = "scientist"
    AUTO = "auto"


class ModelPolicy(str, Enum):
    FAST = "fast"
    DEEP = "deep"
    HYBRID = "hybrid"


class DeepLearnAnswer(BaseModel):
    overview: str
    beginner_explanation: str
    deep_explanation: str
    scientific_explanation: str
    mathematical_foundation: str
    visual_explanation: str
    real_world_examples: list[str]
    applications: list[str]
    common_mistakes: list[str]
    advanced_concepts: list[str]
    research_insights: list[str]
    quiz: list[str]
    summary: str
    visualization_assets: dict[str, str] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class DeepLearnConfig:
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "glm-5.2"))
    gemini_fast_model: str = field(default_factory=lambda: os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash"))
    embedding_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))
    temperature: float = 0.2
    max_retries: int = 2
    chunk_size: int = 1400
    chunk_overlap: int = 180
    retrieval_k: int = 7
    learn_deep_max_rounds: int = 3


DEEPLearn_SYSTEM_PROMPT = """
You are DeepLearn AI, an elite education engine.

Your objective is not to merely answer; your objective is to build complete understanding.

Adapt automatically to the learner level:
beginner, student, college, researcher, professor, scientist.

When Learn Deep is enabled:
- Explore principles, history, intuition, derivations, proofs, edge cases, misconceptions,
  applications, engineering uses, research uses, limitations, open problems, and cross-domain links.
- Show educational reasoning, derivations, and proof steps clearly.
- Do not reveal private hidden chain-of-thought. Provide concise, inspectable reasoning instead.
- Do not hallucinate. If evidence is uncertain, say so.
- Use cited source context when provided.
- Prefer conceptual understanding before memorization.
- Generate useful visualization assets when they improve learning.

Every answer must follow exactly these sections:
1. Overview
2. Beginner Explanation
3. Deep Explanation
4. Scientific Explanation
5. Mathematical Foundation
6. Visual Explanation
7. Real-world Examples
8. Applications
9. Common Mistakes
10. Advanced Concepts
11. Research Insights
12. Quiz
13. Summary
"""


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Classify the educational request.

Return only one label:
FAST - direct coding, short explanation, simple factual teaching
DEEP - advanced reasoning, research depth, math derivation, medicine, physics, proof, complex science
HYBRID - needs fast first draft plus deeper Gemini synthesis
""",
        ),
        ("human", "{question}"),
    ]
)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEEPLearn_SYSTEM_PROMPT),
        (
            "human",
            """
Learner level: {level}
Learn Deep enabled: {learn_deep}

Question:
{question}

Source context:
{context}

Citation hints:
{citations}

Produce a complete educational answer.
""",
        ),
    ]
)


SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEEPLearn_SYSTEM_PROMPT),
        (
            "human",
            """
Synthesize the strongest final educational answer from both model drafts.

Learner level: {level}
Learn Deep enabled: {learn_deep}

Question:
{question}

Groq fast draft:
{groq_draft}

Gemini deep draft:
{gemini_draft}

Source context:
{context}

Citation hints:
{citations}

Return the final answer in the required 13-section structure.
""",
        ),
    ]
)


DEPTH_AUDIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict professor auditing whether an educational answer is deep enough.
Return exactly one of:
ENOUGH
DEEPEN
""",
        ),
        (
            "human",
            """
Question:
{question}

Answer:
{answer}

Should this be expanded with deeper derivations, caveats, edge cases, or research detail?
""",
        ),
    ]
)


DEEPEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", DEEPLearn_SYSTEM_PROMPT),
        (
            "human",
            """
Improve and deepen this answer without changing the required 13-section structure.

Question:
{question}

Current answer:
{answer}

Add missing foundations, derivations, hidden assumptions, edge cases, misconceptions,
applications, research insights, and better visualization assets where useful.
""",
        ),
    ]
)


class DeepLearnAI:
    def __init__(self, config: Optional[DeepLearnConfig] = None) -> None:
        self.config = config or DeepLearnConfig()
        self._validate_env()

        self.groq = ChatGroq(
            model=self.config.groq_model,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )

        self.gemini = ChatGoogleGenerativeAI(
            model=self.config.gemini_model,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )

        self.gemini_fast = ChatGoogleGenerativeAI(
            model=self.config.gemini_fast_model,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.config.embedding_model)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vectorstore: Optional[InMemoryVectorStore] = None
        self.documents: list[Document] = []

    def _validate_env(self) -> None:
        missing = []

        if not os.getenv("GROQ_API_KEY"):
            missing.append("GROQ_API_KEY")

        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")

        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    async def add_websites(self, urls: Iterable[str]) -> None:
        docs: list[Document] = []

        for url in urls:
            loaded = WebBaseLoader(url).load()
            for doc in loaded:
                doc.metadata["source_type"] = "website"
                doc.metadata["source"] = doc.metadata.get("source", url)
            docs.extend(loaded)

        await self._index_documents(docs)

    async def add_pdfs(self, paths: Iterable[str]) -> None:
        docs: list[Document] = []

        for path in paths:
            loaded = PyPDFLoader(path).load()
            for doc in loaded:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["source"] = doc.metadata.get("source", path)
            docs.extend(loaded)

        await self._index_documents(docs)

    async def add_arxiv_papers(self, queries: Iterable[str], max_docs: int = 3) -> None:
        docs: list[Document] = []

        for query in queries:
            loaded = ArxivLoader(query=query, load_max_docs=max_docs).load()
            for doc in loaded:
                doc.metadata["source_type"] = "arxiv"
            docs.extend(loaded)

        await self._index_documents(docs)

    async def _index_documents(self, docs: list[Document]) -> None:
        if not docs:
            return

        chunks = self.splitter.split_documents(docs)
        self.documents.extend(chunks)

        if self.vectorstore is None:
            self.vectorstore = InMemoryVectorStore.from_documents(chunks, self.embeddings)
        else:
            await self.vectorstore.aadd_documents(chunks)

    async def teach(
        self,
        question: str,
        *,
        level: TeachingLevel = TeachingLevel.AUTO,
        learn_deep: bool = True,
        policy: ModelPolicy | Literal["auto"] = "auto",
    ) -> str:
        context_docs = await self._retrieve(question)
        context = self._format_context(context_docs)
        citations = self._format_citations(context_docs)

        chosen_policy = await self._choose_policy(question, policy)

        if chosen_policy == ModelPolicy.FAST:
            answer = await self._answer_with_groq(question, level, learn_deep, context, citations)
        elif chosen_policy == ModelPolicy.DEEP:
            answer = await self._answer_with_gemini(question, level, learn_deep, context, citations)
        else:
            answer = await self._answer_hybrid(question, level, learn_deep, context, citations)

        if learn_deep:
            answer = await self._deepen_until_sufficient(question, answer)

        return answer

    async def teach_structured(
        self,
        question: str,
        *,
        level: TeachingLevel = TeachingLevel.AUTO,
        learn_deep: bool = True,
        policy: ModelPolicy | Literal["auto"] = "auto",
    ) -> DeepLearnAnswer:
        raw = await self.teach(
            question,
            level=level,
            learn_deep=learn_deep,
            policy=policy,
        )

        structured_model = self.gemini.with_structured_output(DeepLearnAnswer)

        return await structured_model.ainvoke(
            [
                SystemMessage(content="Convert the answer into the requested structured schema. Preserve meaning."),
                HumanMessage(content=raw),
            ]
        )

    async def _choose_policy(
        self,
        question: str,
        policy: ModelPolicy | Literal["auto"],
    ) -> ModelPolicy:
        if policy != "auto":
            return ModelPolicy(policy)

        chain = ROUTER_PROMPT | self.gemini_fast | StrOutputParser()
        label = (await chain.ainvoke({"question": question})).strip().upper()

        if "HYBRID" in label:
            return ModelPolicy.HYBRID
        if "DEEP" in label:
            return ModelPolicy.DEEP
        return ModelPolicy.FAST

    async def _answer_with_groq(
        self,
        question: str,
        level: TeachingLevel,
        learn_deep: bool,
        context: str,
        citations: str,
    ) -> str:
        chain = ANSWER_PROMPT | self.groq | StrOutputParser()
        return await chain.ainvoke(
            {
                "question": question,
                "level": level.value,
                "learn_deep": learn_deep,
                "context": context,
                "citations": citations,
            }
        )

    async def _answer_with_gemini(
        self,
        question: str,
        level: TeachingLevel,
        learn_deep: bool,
        context: str,
        citations: str,
    ) -> str:
        chain = ANSWER_PROMPT | self.gemini | StrOutputParser()
        return await chain.ainvoke(
            {
                "question": question,
                "level": level.value,
                "learn_deep": learn_deep,
                "context": context,
                "citations": citations,
            }
        )

    async def _answer_hybrid(
        self,
        question: str,
        level: TeachingLevel,
        learn_deep: bool,
        context: str,
        citations: str,
    ) -> str:
        groq_task = self._answer_with_groq(question, level, learn_deep, context, citations)
        gemini_task = self._answer_with_gemini(question, level, learn_deep, context, citations)

        groq_draft, gemini_draft = await asyncio.gather(groq_task, gemini_task)

        chain = SYNTHESIS_PROMPT | self.gemini | StrOutputParser()

        return await chain.ainvoke(
            {
                "question": question,
                "level": level.value,
                "learn_deep": learn_deep,
                "groq_draft": groq_draft,
                "gemini_draft": gemini_draft,
                "context": context,
                "citations": citations,
            }
        )

    async def _deepen_until_sufficient(self, question: str, answer: str) -> str:
        audit_chain = DEPTH_AUDIT_PROMPT | self.gemini_fast | StrOutputParser()
        deepen_chain = DEEPEN_PROMPT | self.gemini | StrOutputParser()

        current = answer

        for _ in range(self.config.learn_deep_max_rounds):
            verdict = (
                await audit_chain.ainvoke(
                    {
                        "question": question,
                        "answer": current,
                    }
                )
            ).strip().upper()

            if verdict == "ENOUGH":
                break

            current = await deepen_chain.ainvoke(
                {
                    "question": question,
                    "answer": current,
                }
            )

        return current

    async def _retrieve(self, question: str) -> list[Document]:
        if self.vectorstore is None:
            return []

        return await self.vectorstore.asimilarity_search(
            question,
            k=self.config.retrieval_k,
        )

    def _format_context(self, docs: list[Document]) -> str:
        if not docs:
            return "No external source context provided."

        blocks = []

        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("entry_id") or "unknown"
            page = doc.metadata.get("page")
            page_part = f", page {page}" if page is not None else ""

            blocks.append(
                f"[Source {index}: {source}{page_part}]\n"
                f"{doc.page_content[:3500]}"
            )

        return "\n\n".join(blocks)

    def _format_citations(self, docs: list[Document]) -> str:
        if not docs:
            return "No citations."

        seen: set[str] = set()
        citations: list[str] = []

        for doc in docs:
            source = doc.metadata.get("source") or doc.metadata.get("entry_id") or "unknown"
            title = doc.metadata.get("title")
            page = doc.metadata.get("page")

            key = f"{title}|{source}|{page}"
            if key in seen:
                continue

            seen.add(key)

            if title and page is not None:
                citations.append(f"- {title}: {source}, page {page}")
            elif title:
                citations.append(f"- {title}: {source}")
            elif page is not None:
                citations.append(f"- {source}, page {page}")
            else:
                citations.append(f"- {source}")

        return "\n".join(citations)


def ask_bool(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"

    while True:
        value = input(f"{prompt} [{suffix}]: ").strip().lower()

        if not value:
            return default
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False

        print("Enter yes or no.")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    normalized = {choice.lower(): choice for choice in choices}

    while True:
        value = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip().lower()

        if not value:
            return default
        if value in normalized:
            return normalized[value]

        print(f"Choose one of: {', '.join(choices)}")


def ask_many(prompt: str) -> list[str]:
    raw = input(prompt).strip()

    if not raw:
        return []

    return [item.strip() for item in raw.split(",") if item.strip()]


async def interactive_main() -> None:
    engine = DeepLearnAI()

    print("DeepLearn AI backend")
    print("Type 'exit' to quit.")
    print()

    pdf_paths = ask_many("PDF paths to index, comma-separated, or empty: ")
    if pdf_paths:
        await engine.add_pdfs(pdf_paths)

    websites = ask_many("Website URLs to index, comma-separated, or empty: ")
    if websites:
        await engine.add_websites(websites)

    arxiv_queries = ask_many("Arxiv queries to index, comma-separated, or empty: ")
    if arxiv_queries:
        await engine.add_arxiv_papers(arxiv_queries)

    while True:
        question = input("\nAsk a topic/question: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            break

        if not question:
            continue

        level = TeachingLevel(
            ask_choice(
                "Teaching level",
                [item.value for item in TeachingLevel],
                TeachingLevel.AUTO.value,
            )
        )

        policy_input = ask_choice(
            "Model policy",
            ["auto", *[item.value for item in ModelPolicy]],
            "auto",
        )

        learn_deep = ask_bool("Enable Learn Deep", default=True)

        answer = await engine.teach(
            question,
            level=level,
            learn_deep=learn_deep,
            policy=policy_input,
        )

        print("\n" + "=" * 100)
        print(answer)
        print("=" * 100)


if __name__ == "__main__":
    asyncio.run(interactive_main())