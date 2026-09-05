# pip install -U langgraph langchain-groq pydantic python-dotenv langsmith

import operator
import os
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# =========================
# Setup
# =========================

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "your project name"



llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================
# Structured Output
# =========================

class EvaluationSchema(BaseModel):
    feedback: str = Field(
        description="Feedback about the essay"
    )

    score: int = Field(
        description="Score out of 10",
        ge=0,
        le=10
    )


structured_llm = llm.with_structured_output(
    EvaluationSchema
)


# =========================
# Essay
# =========================

essay = """
Technology has changed the way students learn and communicate.
Today, students can use computers, smartphones and online platforms
to access educational material from anywhere.

Online courses are especially useful for students who cannot attend
traditional classrooms. They can watch lectures, read books and
practice different skills using the internet.

However, technology also has some disadvantages. Some students may
spend too much time on social media instead of studying. Students
without reliable internet access may also face difficulties.

Therefore, technology should be used carefully. Teachers, parents
and students should work together to make sure that technology
improves education instead of becoming a distraction.
"""


# =========================
# LangGraph State
# =========================

class EssayState(TypedDict, total=False):

    essay: str

    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str

    # Reducer
    scores: Annotated[List[int], operator.add]

    overall_feedback: str
    average_score: float


# =========================
# Language Evaluation
# =========================

def evaluate_language(state: EssayState):

    prompt = f"""
    Evaluate the language quality of this essay.

    Give feedback and a score out of 10.

    Essay:
    {state["essay"]}
    """

    result = structured_llm.invoke(prompt)

    return {
        "language_feedback": result.feedback,
        "scores": [result.score]
    }


# =========================
# Analysis Evaluation
# =========================

def evaluate_analysis(state: EssayState):

    prompt = f"""
    Evaluate the depth and quality of analysis in this essay.

    Give feedback and a score out of 10.

    Essay:
    {state["essay"]}
    """

    result = structured_llm.invoke(prompt)

    return {
        "analysis_feedback": result.feedback,
        "scores": [result.score]
    }


# =========================
# Clarity Evaluation
# =========================

def evaluate_clarity(state: EssayState):

    prompt = f"""
    Evaluate the clarity of thought in this essay.

    Give feedback and a score out of 10.

    Essay:
    {state["essay"]}
    """

    result = structured_llm.invoke(prompt)

    return {
        "clarity_feedback": result.feedback,
        "scores": [result.score]
    }


# =========================
# Final Evaluation
# =========================

def final_evaluation(state: EssayState):

    prompt = f"""
    Based on the following feedback, write a short overall
    evaluation of the essay.

    Language feedback:
    {state["language_feedback"]}

    Analysis feedback:
    {state["analysis_feedback"]}

    Clarity feedback:
    {state["clarity_feedback"]}
    """

    overall = llm.invoke(prompt).content

    scores = state["scores"]

    average = sum(scores) / len(scores)

    return {
        "overall_feedback": overall,
        "average_score": average
    }


# =========================
# Build Graph
# =========================

graph = StateGraph(EssayState)

graph.add_node("language", evaluate_language)
graph.add_node("analysis", evaluate_analysis)
graph.add_node("clarity", evaluate_clarity)
graph.add_node("final", final_evaluation)


# =========================
# Edges
# =========================

graph.add_edge(START, "language")
graph.add_edge(START, "analysis")
graph.add_edge(START, "clarity")

graph.add_edge("language", "final")
graph.add_edge("analysis", "final")
graph.add_edge("clarity", "final")

graph.add_edge("final", END)


# =========================
# Compile
# =========================

workflow = graph.compile()


# =========================
# Run
# =========================

result = workflow.invoke({
    "essay": essay,
})


# =========================
# Results
# =========================

print("\n===== ESSAY EVALUATION =====")

print("\nLanguage Feedback:")
print(result["language_feedback"])

print("\nAnalysis Feedback:")
print(result["analysis_feedback"])

print("\nClarity Feedback:")
print(result["clarity_feedback"])

print("\nOverall Feedback:")
print(result["overall_feedback"])

print("\nIndividual Scores:")
print(result["scores"])

print("\nAverage Score:")
print(result["average_score"])