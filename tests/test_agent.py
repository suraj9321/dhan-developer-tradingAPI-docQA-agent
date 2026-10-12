"""Agent control flow with a scripted LLM and retriever (no network, no models)."""

import json

from docsqa.agent import AGENT_V1, AGENT_V2, AgentPipeline
from docsqa.ingest.chunk import Chunk
from docsqa.llm import LLMResult
from docsqa.rag import REFUSAL
from docsqa.retrieve import Hit


def _chunk(i, text="Cancel with DELETE /orders/{order-id}."):
    return Chunk(f"s{i:04d}-00", f"s{i:04d}", "Indian Trading & Data API", ["Cancel Order"], "DELETE", "u", f"c{i}", text, 10)


class FakeRetriever:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.queries = []

    def retrieve(self, query, k=None):
        self.queries.append(query)
        return [Hit(c, 1.0) for c in self.rounds.pop(0)]


class ScriptedLLM:
    """Returns canned replies keyed by which agent prompt is being sent."""

    def __init__(self, grade, verify, answer="Use DELETE /orders/{order-id} [1].", revised="Use DELETE [1]."):
        self.grade, self.verify, self.answer, self.revised = list(grade), list(verify), answer, revised
        self.calls = []

    def chat(self, messages, model, max_tokens=1024, **_):
        last = messages[-1]["content"]
        if "Do these excerpts contain" in last:
            kind, text = "grade", json.dumps(self.grade.pop(0))
        elif "Write one new search query" in last:
            kind, text = "rewrite", "cancel order endpoint"
        elif "Check whether an answer is supported" in messages[-1]["content"]:
            kind, text = "verify", json.dumps(self.verify.pop(0))
        elif "Your previous answer contained claims" in last:
            kind, text = "revise", self.revised
        else:
            kind, text = "answer", self.answer
        self.calls.append(kind)
        return LLMResult(text=text, model=model)


def _agent(llm, retriever, cfg=AGENT_V1):
    return AgentPipeline(retriever, llm=llm, model="fake", config=cfg)


def test_happy_path_answers_once():
    llm = ScriptedLLM(grade=[{"sufficient": True}], verify=[{"supported": True}])
    a = _agent(llm, FakeRetriever([[_chunk(1)]])).ask("How do I cancel an order?")
    assert llm.calls == ["grade", "answer", "verify"]
    assert not a.refused and a.citations[0].chunk_id == "s0001-00"


def test_weak_evidence_rewrites_once_and_merges_context():
    llm = ScriptedLLM(grade=[{"sufficient": False, "missing": "endpoint"}, {"sufficient": False}], verify=[{"supported": True}])
    r = FakeRetriever([[_chunk(1)], [_chunk(1), _chunk(2)]])
    a = _agent(llm, r).ask("cancel?")
    assert llm.calls == ["grade", "rewrite", "grade", "answer", "verify"]  # capped at one retry
    assert r.queries == ["cancel?", "cancel order endpoint"]
    assert a.retrieved == ["s0001-00", "s0002-00"]  # merged, de-duplicated


def test_unsupported_answer_is_revised_then_abstains():
    llm = ScriptedLLM(grade=[{"sufficient": True}], verify=[{"supported": False, "unsupported_claims": ["x"]}] * 2)
    a = _agent(llm, FakeRetriever([[_chunk(1)]])).ask("q")
    assert llm.calls == ["grade", "answer", "verify", "revise", "verify"]
    assert a.refused and a.answer == REFUSAL and a.trace[-1] == "abstain"


def test_revision_that_passes_is_kept():
    llm = ScriptedLLM(grade=[{"sufficient": True}], verify=[{"supported": False, "unsupported_claims": ["x"]}, {"supported": True}])
    a = _agent(llm, FakeRetriever([[_chunk(1)]])).ask("q")
    assert not a.refused and a.answer == "Use DELETE [1]."


def test_refusal_skips_verification():
    llm = ScriptedLLM(grade=[{"sufficient": True}], verify=[], answer=REFUSAL)
    a = _agent(llm, FakeRetriever([[_chunk(1)]])).ask("q")
    assert llm.calls == ["grade", "answer"] and a.refused


def test_v2_keeps_revised_answer_when_evidence_was_strong():
    llm = ScriptedLLM(grade=[{"sufficient": True}], verify=[{"supported": False, "unsupported_claims": ["x"]}] * 2)
    a = _agent(llm, FakeRetriever([[_chunk(1)]]), AGENT_V2).ask("q")
    assert llm.calls == ["grade", "answer", "verify", "revise", "verify"]
    assert not a.refused and a.answer == "Use DELETE [1]."


def test_v2_still_abstains_when_evidence_was_weak():
    llm = ScriptedLLM(grade=[{"sufficient": False}, {"sufficient": False}], verify=[{"supported": False, "unsupported_claims": ["x"]}] * 2)
    a = _agent(llm, FakeRetriever([[_chunk(1)], [_chunk(2)]]), AGENT_V2).ask("q")
    assert a.refused
