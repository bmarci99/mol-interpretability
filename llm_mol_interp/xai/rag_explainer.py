# llm_mol_interp/xai/rag_explainer.py
from __future__ import annotations
import yaml, re, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# NEW (version-agnostic):
from transformers.generation.logits_process import (
    LogitsProcessorList, NoRepeatNGramLogitsProcessor
)
try:
    # new-ish name
    from transformers.generation.logits_process import BadWordsLogitsProcessor
except Exception:
    # older/other versions
    from transformers.generation.logits_process import NoBadWordsLogitsProcessor as BadWordsLogitsProcessor

DEFAULT_MODEL = "google/flan-t5-base"  # small enough, more robust than -small

# Optional: lightweight aliases to keep detector names and KB keys aligned
ALIASES = {
    "aromatic nitro (charged)": "aromatic nitro",
    "aromatic nitro": "aromatic nitro",
    "michael acceptor": "michael acceptor (enone)",
    "michael acceptor (enone)": "michael acceptor (enone)",
    "primary aromatic amine": "primary aromatic amine",
    "nitrosamine": "nitrosamine",
    "epoxide": "epoxide",
    "polycyclic aromatic": "polycyclic aromatic",  # PAH
}

def _canon(name: str) -> str:
    return ALIASES.get((name or "").strip().lower(), (name or "").strip().lower())

def _uniq(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

@dataclass
class KBEntry:
    facts: List[str]
    caveats: List[str]
    refs: List[Dict[str, str]]  # [{'id': 'toxalerts_nitro', 'cite': '...'}, ...]

class GroundedExplainer:
    """
    Retrieval-lite (KB lookup) + LLM (optional) with **hard guardrails**:
      - Mentions ONLY detected concepts (decoding-time blocking).
      - Emits inline citations `[cite:<id>]` if refs exist.
      - Deterministic decoding (beam) for reproducibility.
      - Template fallback if model drifts or is offline.

    Public API (backward compatible):
      - generate(sub_name, pred_dict) -> str
    New:
      - generate_multi(sub_names, pred_dict, max_sentences=2, return_json=False)
    """
    def __init__(self, kb_path: str, model_name: Optional[str] = DEFAULT_MODEL, strict: bool = True):
        with open(kb_path, "r") as f:
            raw = yaml.safe_load(f) or {}
        self.kb: Dict[str, KBEntry] = {}
        for k, v in raw.items():
            facts = v.get("facts") or []
            caveats = v.get("caveats") or []
            refs = v.get("refs") or []
            self.kb[_canon(k)] = KBEntry(facts=facts, caveats=caveats, refs=refs)

        self.strict = strict
        self.tok = self.model = None
        if model_name:
            try:
                self.tok = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ).eval()
            except Exception:
                self.tok = self.model = None  # template fallback

        # Build vocab of concept names for guardrails
        self._concept_vocab = sorted(list(self.kb.keys()))

    # ------------- public API -------------
    def generate(self, sub_name: Optional[str], pred_dict: Dict[str, float]) -> str:
        concept = _canon(sub_name or "")
        if concept not in self.kb:
            return self._no_kb_message(pred_dict)

        entry = self.kb[concept]
        pred_summary = self._pred_summary(pred_dict)

        if not self.model or not entry.facts:
            return self._template_one(concept, pred_summary, entry)

        prompt = self._prompt_one(concept, pred_summary, entry)
        badwords = self._badwords_except(allowed=[concept])  # block ALL other concept names
        text = self._generate_text(prompt, badwords=badwords)

        if self._needs_fallback(text, allowed=[concept], has_refs=bool(entry.refs)):
            return self._template_one(concept, pred_summary, entry)
        return text

    def generate_multi(
        self,
        sub_names: Iterable[str],
        pred_dict: Dict[str, float],
        max_sentences: int = 2,
        return_json: bool = False
    ) -> str | Dict[str, Any]:
        # Canonicalize & limit concepts (2 is a good budget)
        concepts = _uniq([_canon(s) for s in sub_names if _canon(s) in self.kb])[:2]
        pred_summary = self._pred_summary(pred_dict)

        if not concepts:
            msg = self._no_kb_message(pred_dict)
            return {"text": msg, "citations": [], "concepts": []} if return_json else msg

        entries = [self.kb[c] for c in concepts]
        if not self.model or any(len(e.facts) == 0 for e in entries):
            text, cites = self._template_multi(concepts, pred_summary, entries, max_sentences=max_sentences)
            return {"text": text, "citations": cites, "concepts": concepts} if return_json else text

        prompt = self._prompt_multi(concepts, pred_summary, entries, max_sentences=max_sentences)
        badwords = self._badwords_except(allowed=concepts)
        text = self._generate_text(prompt, badwords=badwords)

        # Guardrails: if it mentions extra KB concepts or lacks citations where refs exist, fallback
        has_refs = any(bool(e.refs) for e in entries)
        if self._needs_fallback(text, allowed=concepts, has_refs=has_refs):
            text, cites = self._template_multi(concepts, pred_summary, entries, max_sentences=max_sentences)
            return {"text": text, "citations": cites, "concepts": concepts} if return_json else text

        return {"text": text, "citations": self._extract_cites(text), "concepts": concepts} if return_json else text

    # ------------- internals -------------

    def _pred_summary(self, pred_dict: Dict[str, float], k: int = 3) -> str:
        preds = sorted(pred_dict.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return ", ".join([f"{name} p={prob:.2f}" for name, prob in preds]) if preds else "no tasks"

    def _no_kb_message(self, pred_dict: Dict[str, float]) -> str:
        return f"Predicted profile: {self._pred_summary(pred_dict)}. No matched structural alert in KB; no grounded rationale available."

    def _prompt_one(self, concept: str, pred_summary: str, entry: KBEntry) -> str:
        refs_block = ""
        if entry.refs:
            refs_block = "\nReferences:\n" + "\n".join([f"- [{r['id']}] {r.get('cite','')}" for r in entry.refs[:2]])
        caveat = f"\nCaveat:\n- {entry.caveats[0]}" if entry.caveats else ""
        return (
            "You are a medicinal chemistry assistant. "
            "Write a concise, mechanism-focused rationale (1–2 sentences) for the predicted profile, "
            "USING ONLY the provided facts. Mention ONLY the detected concept listed below. "
            "Include inline citations like [cite:<id>] if references are provided.\n\n"
            f"Predicted profile: {pred_summary}\n"
            f"Detected concept: {concept}\n"
            "Facts:\n- " + "\n- ".join(entry.facts[:2]) + caveat + "\n"
            + refs_block + "\n"
            "Rationale:"
        )

    def _prompt_multi(self, concepts: List[str], pred_summary: str, entries: List[KBEntry], max_sentences: int) -> str:
        # Concatenate up to 2 concept fact blocks; instruct 1–2 sentences total
        blocks = []
        for c, e in zip(concepts, entries):
            refs_block = ""
            if e.refs:
                refs_block = "\n  References:\n  " + "\n  ".join([f"- [{r['id']}] {r.get('cite','')}" for r in e.refs[:1]])
            caveat = f"\n  Caveat: {e.caveats[0]}" if e.caveats else ""
            blocks.append(f"- {c}:\n  Facts:\n  - " + "\n  - ".join(e.facts[:1]) + caveat + refs_block)
        instr = (
            "You are a medicinal chemistry assistant. "
            f"Write a concise, mechanism-focused rationale in <= {max_sentences} sentences that mentions ONLY the detected concepts below. "
            "Use inline citations like [cite:<id>]. If references are not provided, omit citations.\n"
        )
        return instr + f"\nPredicted profile: {pred_summary}\nDetected concepts:\n" + "\n".join(blocks) + "\nRationale:"

    def _template_one(self, concept: str, pred_summary: str, entry: KBEntry) -> str:
        parts = []
        if entry.facts:
            parts.append(f"{concept} is consistent with the predicted profile ({pred_summary}). {entry.facts[0]}")
            if len(entry.facts) > 1:
                parts.append(entry.facts[1])
        else:
            parts.append(f"Detected {concept}, consistent with the predicted profile ({pred_summary}).")
        if entry.caveats:
            parts.append(f"Caveat: {entry.caveats[0]}")
        text = " ".join(parts)
        if entry.refs:
            text = self._append_first_citation(text, entry.refs[0]["id"])
        return text

    def _template_multi(self, concepts: List[str], pred_summary: str, entries: List[KBEntry], max_sentences: int = 2) -> tuple[str, List[str]]:
        # Compose up to 2 sentences total
        sents, cites = [], []
        for c, e in zip(concepts[:2], entries[:2]):
            base = e.facts[0] if e.facts else f"{c} may contribute to the predicted profile."
            sent = f"{c} is consistent with the predicted profile ({pred_summary}). {base}"
            if e.caveats:
                sent += f" Caveat: {e.caveats[0]}"
            if e.refs:
                sent = self._append_first_citation(sent, e.refs[0]["id"])
                cites.append(e.refs[0]["id"])
            sents.append(sent)
            if len(sents) >= max_sentences:
                break
        return " ".join(sents), cites

    def _append_first_citation(self, text: str, ref_id: str) -> str:
        return text.rstrip(".") + f" [cite:{ref_id}]."

    def _extract_cites(self, text: str) -> List[str]:
        return re.findall(r"\[cite:([^\]]+)\]", text)

    # ---------- generation & guardrails ----------
    def _badwords_except(self, allowed: List[str]) -> List[List[int]]:
        """
        Build 'bad words' token ID sequences for ALL concept names except the allowed ones.
        This prevents the decoder from naming off‑concept mechanisms.
        """
        if not self.model or not self.tok:
            return []
        allowed = {a.lower() for a in allowed}
        forbidden = [c for c in self._concept_vocab if c.lower() not in allowed]
        token_lists = []
        for phrase in forbidden:
            ids = self.tok(phrase, add_special_tokens=False).input_ids
            if ids:
                token_lists.append(ids)
        return token_lists

    def _generate_text(self, prompt: str, badwords: List[List[int]]) -> str:
        dev = next(self.model.parameters()).device
        lp = LogitsProcessorList()
        if badwords:
            eos_id = getattr(self.tok, "eos_token_id", None)
            lp.append(BadWordsLogitsProcessor(bad_words_ids=badwords, eos_token_id=eos_id))
        lp.append(NoRepeatNGramLogitsProcessor(3))
        enc = self.tok(prompt, return_tensors="pt").to(dev)
        out = self.model.generate(
            **enc,
            max_new_tokens=96,
            num_beams=4,
            do_sample=False,
            length_penalty=0.0,
            logits_processor=lp
        )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def _mentions_other_concepts(self, text: str, allowed: List[str]) -> bool:
        t = " " + text.lower() + " "
        allowed = {a.lower() for a in allowed}
        for c in self._concept_vocab:
            if c.lower() in allowed:
                continue
            if f" {c.lower()} " in t:
                return True
        return False

    def _needs_fallback(self, text: str, allowed: List[str], has_refs: bool) -> bool:
        if not text or len(text.split()) < 5:
            return True
        if self._mentions_other_concepts(text, allowed):
            return True
        if self.strict and has_refs and "[cite:" not in text:
            return True
        return False


"""
# llm_mol_interp/xai/rag_explainer.py
import yaml, torch
from importlib.resources import files
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL = "google/flan-t5-small"

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_molinst_lora(base="meta-llama/Llama-2-7b-hf", adapter="zjunlp/llama-molinst-molecule-7b"):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="auto")
    mdl = PeftModel.from_pretrained(mdl, adapter).eval()
    return tok, mdl

def generate_explanation(tok, mdl, substructures, top_assays, max_new_tokens=96):
    subs = ", ".join(substructures) if substructures else "none detected"
    tops = ", ".join(f"{k} p={v:.2f}" for k,v in top_assays)
    prompt = (
      "You are a careful medicinal-chemistry assistant. "
      "Given detected substructures and assay probabilities, "
      "write 1–2 precise sentences explaining plausible mechanisms or risks. "
      "Be specific to the listed signals; avoid speculation.\n"
      f"Detected substructures: {subs}\nAssay probabilities: {tops}\nExplanation:"
    )
    ids = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True).split("Explanation:")[-1].strip()



class GroundedExplainer:
    def __init__(self, kb_path=None, model_name=DEFAULT_MODEL):
        kb_path = kb_path or files("llm_mol_interp.xai").joinpath("kb.yaml")
        with open(kb_path, "r") as f:
            self.kb = yaml.safe_load(f)
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception:
            self.tok = self.model = None  # offline fallback

    def generate(self, sub_name, pred_dict):
        facts = self.kb.get(sub_name, {}).get("facts", [])
        caveats = self.kb.get(sub_name, {}).get("caveats", [])
        if not facts or self.model is None:  # safe fallback
            return f"Detected {sub_name}." + (f" {facts[0]}" if facts else " No curated facts available.")
        tasks = ", ".join([f"{k} p={v:.2f}" for k,v in pred_dict.items()])
        prompt = (
            "Write a 1–2 sentence toxicology rationale USING ONLY the facts below.\n"
            f"Prediction summary: {tasks}\nDetected substructure: {sub_name}\n"
            "Facts:\n- " + "\n- ".join(facts[:2]) + "\n"
            "Caveats (optional):\n- " + ("\n- ".join(caveats[:1]) if caveats else "") + "\n"
            "Rationale:"
        )
        dev = next(self.model.parameters()).device
        out = self.model.generate(**self.tok(prompt, return_tensors="pt").to(dev), max_new_tokens=60, do_sample=False)
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        return text
"""