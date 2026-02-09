"""
L4: Specification Compliance Checking (Enhanced)

Two-phase verification:
  Phase 1 (Extract): 6 category-specific LLM calls -> focused checklists -> merge & deduplicate
  Phase 2 (Verify):  LLM reads checklist + code + L1-L3 context -> checks each item -> verdict

Design principles:
  - Category-specific extraction prompts for higher recall
  - Each category is independently analyzed, then results are merged
  - Verification receives L1-L3 execution context for cross-layer insight
  - Results map to standard Diagnostic schema
"""

import json
import re
import logging
from collections import Counter
from typing import List, Dict, Any, Callable, Optional

from .verification import Diagnostic

logger = logging.getLogger(__name__)


# ============================================================
# Phase 1: Category-Specific Specification Extraction
# ============================================================

CATEGORY_PROMPTS = {
    "VARIABLE_TYPE": {
        "prompt": """You are an optimization modeling expert. Read this problem and answer ONE question:

What types should the decision variables be?
- CONTINUOUS (fractional values allowed, e.g., 2.5 units)
- INTEGER (must be whole numbers, e.g., 3 units)
- BINARY (0 or 1 decisions)

Look for clues like: "number of items", "how many", "units",
"assign", "select", "choose" (often binary), or any context
where fractional values would be physically meaningless
(e.g., "number of trucks", "workers to hire", "packages to ship").

Problem:
---
{problem_desc}
---

Output (strict JSON array):
```json
[
  {{
    "id": 1,
    "category": "VARIABLE_TYPE",
    "requirement": "quoted text from problem that indicates variable type",
    "checkable_criterion": "specific thing to check in code, e.g., 'vtype=GRB.INTEGER'"
  }}
]
```
If all variables should be continuous (default), output an empty array: []
Output ONLY the JSON array.""",
    },

    "VARIABLE_SCOPE": {
        "prompt": """You are an optimization modeling expert. Read this problem and answer ONE question:

What is the SCOPE of the decision variables? Specifically:
- Between which sets of entities are variables defined?
- Should variables cover ALL pairs/combinations, or only a subset?
- What are the index sets (time periods, locations, products, routes)?

Look for clues like: "any X can Y to any Z" (= all pairs), "from surplus to deficit"
(= subset), "over T periods" (= time-indexed), "each product at each location"
(= product x location).

Problem:
---
{problem_desc}
---

Output (strict JSON array):
```json
[
  {{
    "id": 1,
    "category": "VARIABLE_SCOPE",
    "requirement": "quoted text indicating variable scope",
    "checkable_criterion": "what to check, e.g., 'variables defined for ALL region pairs, not just surplus->deficit'"
  }}
]
```
If scope is straightforward/unambiguous, output an empty array: []
Output ONLY the JSON array.""",
    },

    "CONSTRAINT": {
        "prompt": """You are an optimization modeling expert. Read this problem and list ALL constraints.

Include:
- Explicitly stated constraints ("capacity cannot exceed 500")
- Implied constraints (non-negativity, budget limits)
- Balance/conservation constraints ("supply = demand", "inflow = outflow")
- Bound constraints ("at most 3 shifts per worker")
- Linking constraints between different variable groups

Problem:
---
{problem_desc}
---

Output (strict JSON array):
```json
[
  {{
    "id": 1,
    "category": "CONSTRAINT",
    "requirement": "quoted text or description of constraint",
    "checkable_criterion": "what to verify in code, e.g., 'm.addConstr(supply[i] >= demand[i])'"
  }}
]
```
Output ONLY the JSON array.""",
    },

    "OBJECTIVE": {
        "prompt": """You are an optimization modeling expert. Read this problem and answer:

1. Is this a MINIMIZATION or MAXIMIZATION problem?
2. What exactly is being optimized? (cost, profit, distance, time, etc.)
3. What terms should appear in the objective function?
4. Are there any terms that should NOT be in the objective?

Problem:
---
{problem_desc}
---

Output (strict JSON array):
```json
[
  {{
    "id": 1,
    "category": "OBJECTIVE",
    "requirement": "what the objective should be",
    "checkable_criterion": "specific check, e.g., 'objective should be MINIMIZE total cost, including transport + penalty'"
  }}
]
```
Output ONLY the JSON array.""",
    },

    "LOGICAL_CONDITION": {
        "prompt": """You are an optimization modeling expert. Read this problem and identify any:

- If-then conditions ("if production > 0, then setup cost applies")
- Either-or constraints ("either machine A or machine B, not both")
- Conditional activation ("warehouse is only used if opened")
- Piecewise relationships
- Big-M formulations needed

Problem:
---
{problem_desc}
---

Output (strict JSON array):
```json
[
  {{
    "id": 1,
    "category": "LOGICAL_CONDITION",
    "requirement": "description of logical condition",
    "checkable_criterion": "what to check in code"
  }}
]
```
If no logical conditions exist, output an empty array: []
Output ONLY the JSON array.""",
    },

    "DATA_MAPPING": {
        "prompt": """You are an optimization modeling expert. Read this problem and check for tricky parameter interpretations:

- "Return of 0.7 per dollar invested" — does this mean profit=0.7 (total=1.7x) or total=0.7x?
- "Cost is 50 cents" — is this 0.5 dollars or 50 cents?
- "Rate of 5%%" — is the multiplier 0.05 or 1.05?
- Any unit conversions needed?
- Any parameters that combine multiple components?

Problem:
---
{problem_desc}
---

Output (strict JSON array):
```json
[
  {{
    "id": 1,
    "category": "DATA_MAPPING",
    "requirement": "the tricky parameter and its correct interpretation",
    "checkable_criterion": "what the code coefficient should be"
  }}
]
```
If all parameters are straightforward, output an empty array: []
Output ONLY the JSON array.""",
    },
}

# Keep old prompt as backup (commented out for rollback if needed)
# SPEC_EXTRACTION_PROMPT = """..."""


def extract_specification(
    problem_desc: str,
    llm_fn: Callable,
) -> List[Dict]:
    """
    Phase 1: Extract specification checklist from problem description.

    Uses 6 category-specific prompts for focused extraction.
    Each category is independently analyzed, then results are merged and deduplicated.

    Args:
        problem_desc: The original problem description text
        llm_fn: Function to call LLM. Signature: (system_prompt, user_prompt) -> str

    Returns:
        List of specification items, each a dict with keys:
        {id, category, requirement, checkable_criterion}
    """
    all_specs = []
    global_id = 1

    for category, config in CATEGORY_PROMPTS.items():
        prompt = config["prompt"].format(problem_desc=problem_desc)

        try:
            response = llm_fn(
                f"You are an optimization expert focused on {category}. Output only valid JSON.",
                prompt,
            )
        except Exception as e:
            logger.warning(f"L4 Phase 1: {category} extraction failed: {e}")
            continue

        specs = _parse_json_response(response)

        # Validate and normalize each item
        for spec in specs:
            if not all(k in spec for k in ["category", "requirement", "checkable_criterion"]):
                continue
            spec["id"] = global_id
            spec["category"] = category  # Force correct category
            global_id += 1
            all_specs.append(spec)

        logger.debug(f"L4 Phase 1: {category} -> {len(specs)} items")

    # Deduplicate
    all_specs = _deduplicate_specs(all_specs)

    if not all_specs:
        logger.warning("L4 Phase 1: No specifications extracted across all categories")
        return []

    # Per-category stats
    cat_counts = Counter(s["category"] for s in all_specs)
    logger.info(
        f"L4 Phase 1: Extracted {len(all_specs)} specs across categories: "
        f"{', '.join(f'{cat}={cnt}' for cat, cnt in cat_counts.items())}"
    )

    return all_specs


def _deduplicate_specs(specs: List[Dict]) -> List[Dict]:
    """
    Deduplicate specs based on checkable_criterion text.

    Simple exact-match deduplication (case-insensitive, stripped).
    """
    seen = set()
    unique = []
    for spec in specs:
        criterion = spec.get("checkable_criterion", "").strip().lower()
        if criterion not in seen:
            seen.add(criterion)
            unique.append(spec)

    # Renumber IDs
    for i, spec in enumerate(unique, 1):
        spec["id"] = i

    return unique


# ============================================================
# Phase 2: Specification Verification (with L1-L3 context)
# ============================================================

SPEC_VERIFICATION_PROMPT = """You are a code reviewer verifying optimization code against requirements.

For EACH specification item, do THREE things:
1. STATE what the requirement demands (from the spec)
2. FIND what the code actually does (quote exact code lines)
3. COMPARE: does the code match the requirement?

This three-step process is critical. Do not skip step 2.

Example of a FAIL:
  Spec: "variables defined for ALL pairs of regions (i,j)"
  Requirement demands: decision variables for every combination of regions
  Code actually does: `for i in surplus: for j in deficit:` (line 23)
  Compare: Code only creates variables between surplus and deficit regions,
  NOT all pairs. FAIL.

Example of a PASS:
  Spec: "minimize total transportation cost"
  Requirement demands: objective = minimize, includes transport costs
  Code actually does: `m.setObjective(total_cost, GRB.MINIMIZE)` (line 45)
  Compare: Matches. PASS.

Specification checklist:
{checklist_json}

Code to verify:
```python
{code}
```

{execution_context_section}

For each item, output (strict JSON array):
```json
[
  {{
    "spec_id": 1,
    "requirement_demands": "what the spec requires",
    "code_actually_does": "exact quote from code with line reference",
    "verdict": "PASS or FAIL or UNCERTAIN",
    "explanation": "why they match or don't match"
  }}
]
```

CRITICAL: For VARIABLE_SCOPE specs, carefully check the loop ranges
and index sets. If the spec says "all pairs" but the code iterates
over a subset, that is a FAIL.

Output ONLY the JSON array."""


def verify_specification(
    specs: List[Dict],
    code: str,
    llm_fn: Callable,
    l1_diagnostics: Optional[List] = None,
    l2_diagnostics: Optional[List] = None,
    l3_diagnostics: Optional[List] = None,
    z_base: Optional[float] = None,
) -> List[Dict]:
    """
    Phase 2: Verify each specification item against the code.

    Now accepts L1-L3 diagnostic results as additional context.

    Args:
        specs: Specification checklist from Phase 1
        code: LLM-generated optimization code
        llm_fn: Function to call LLM
        l1_diagnostics: Diagnostic objects from L1 (execution)
        l2_diagnostics: Diagnostic objects from L2 (direction analysis)
        l3_diagnostics: Diagnostic objects from L3 (CPT)
        z_base: Current objective value

    Returns:
        List of verification results, each a dict with keys:
        {spec_id, verdict, code_evidence, explanation, category, requirement}
    """
    if not specs:
        return []

    checklist_json = json.dumps(specs, indent=2, ensure_ascii=False)

    # Build execution context section
    execution_context_section = _build_execution_context(
        l1_diagnostics, l2_diagnostics, l3_diagnostics, z_base
    )

    prompt = SPEC_VERIFICATION_PROMPT.format(
        checklist_json=checklist_json,
        code=code,
        execution_context_section=execution_context_section,
    )

    response = llm_fn(
        "You are a precise code reviewer. Output only valid JSON.",
        prompt,
    )

    results = _parse_json_response(response)

    if not results:
        logger.warning("L4 Phase 2: No verification results parsed")
        return []

    # Enrich results with spec info
    spec_map = {s["id"]: s for s in specs}
    enriched = []
    for r in results:
        spec_id = r.get("spec_id")
        if spec_id in spec_map:
            r["category"] = spec_map[spec_id].get("category", "UNKNOWN")
            r["requirement"] = spec_map[spec_id].get("requirement", "")
            r["checkable_criterion"] = spec_map[spec_id].get("checkable_criterion", "")
        enriched.append(r)

    logger.info(f"L4 Phase 2: Verified {len(enriched)} specification items")
    return enriched


def _build_execution_context(
    l1_diagnostics: Optional[List] = None,
    l2_diagnostics: Optional[List] = None,
    l3_diagnostics: Optional[List] = None,
    z_base: Optional[float] = None,
) -> str:
    """
    Build execution context text for injection into Phase 2 prompt.

    If no context is available (all params None), returns empty string.
    Evidence is truncated to 200 chars to control token usage.
    """
    sections = []

    if z_base is not None:
        sections.append(f"Solver objective value: {z_base}")

    if l1_diagnostics:
        l1_texts = []
        for d in l1_diagnostics:
            if hasattr(d, "evidence"):
                l1_texts.append(f"  - [{d.issue_type}] {d.evidence[:200]}")
        if l1_texts:
            sections.append("L1 (Execution) findings:\n" + "\n".join(l1_texts))

    if l2_diagnostics:
        l2_texts = []
        for d in l2_diagnostics:
            if hasattr(d, "evidence"):
                l2_texts.append(f"  - [{d.issue_type}] {d.evidence[:200]}")
        if l2_texts:
            sections.append("L2 (Direction Analysis) findings:\n" + "\n".join(l2_texts))

    if l3_diagnostics:
        l3_texts = []
        for d in l3_diagnostics:
            if hasattr(d, "evidence"):
                l3_texts.append(f"  - [{d.issue_type}] {d.evidence[:200]}")
        if l3_texts:
            sections.append("L3 (Constraint Presence) findings:\n" + "\n".join(l3_texts))

    if not sections:
        return ""

    return (
        "Execution Context from earlier verification layers:\n"
        "---\n"
        + "\n\n".join(sections)
        + "\n---\n"
        "Use this context to inform your verification. For example:\n"
        "- If a parameter shows NO EFFECT in L2, the constraint may be ineffective\n"
        "- If L3 reports a MISSING constraint, verify the corresponding spec carefully\n"
        "- If objective value seems unreasonable, check formulation details"
    )


# ============================================================
# Phase 3: Convert results to Diagnostics
# ============================================================

def results_to_diagnostics(results: List[Dict]) -> List[Diagnostic]:
    """
    Convert verification results to Diagnostic objects.

    Only FAIL results become ERROR diagnostics (triggers_repair=True).
    UNCERTAIN results become WARNING diagnostics (triggers_repair=False).
    PASS results are not converted.
    """
    diagnostics = []

    for r in results:
        verdict = r.get("verdict", "").upper()

        if verdict == "FAIL":
            # Support both old "code_evidence" and new "code_actually_does" fields
            code_ev = r.get("code_actually_does") or r.get("code_evidence", "N/A")
            diagnostics.append(Diagnostic(
                layer="L4",
                issue_type="SPEC_VIOLATION",
                severity="ERROR",
                target_name=f"{r.get('category', 'UNKNOWN')}: {r.get('requirement', '')[:80]}",
                evidence=(
                    f"Specification requirement NOT met.\n"
                    f"Category: {r.get('category', 'N/A')}\n"
                    f"Requirement: {r.get('requirement', 'N/A')}\n"
                    f"Expected: {r.get('checkable_criterion', 'N/A')}\n"
                    f"Code actually does: {code_ev}\n"
                    f"Explanation: {r.get('explanation', 'N/A')}"
                ),
                triggers_repair=True,
            ))

        elif verdict == "UNCERTAIN":
            code_ev = r.get("code_actually_does") or r.get("code_evidence", "N/A")
            diagnostics.append(Diagnostic(
                layer="L4",
                issue_type="SPEC_UNCERTAIN",
                severity="WARNING",
                target_name=f"{r.get('category', 'UNKNOWN')}: {r.get('requirement', '')[:80]}",
                evidence=(
                    f"Specification requirement UNCERTAIN.\n"
                    f"Category: {r.get('category', 'N/A')}\n"
                    f"Requirement: {r.get('requirement', 'N/A')}\n"
                    f"Code actually does: {code_ev}\n"
                    f"Explanation: {r.get('explanation', 'N/A')}"
                ),
                triggers_repair=False,
            ))
        # PASS items: no diagnostic generated

    return diagnostics


# ============================================================
# Main entry point
# ============================================================

def run_l4(
    code: str,
    problem_desc: str,
    llm_fn: Callable,
    z_base: Optional[float] = None,
    l1_diagnostics: Optional[List] = None,
    l2_diagnostics: Optional[List] = None,
    l3_diagnostics: Optional[List] = None,
    max_specs: int = 25,
) -> Dict[str, Any]:
    """
    L4: Specification Compliance Checking (Enhanced)

    Phase 1: Category-specific extraction (6 focused LLM calls)
    Phase 2: Verification with L1-L3 execution context

    Args:
        code: LLM-generated optimization code
        problem_desc: Original problem description
        llm_fn: LLM calling function (system_prompt, user_prompt) -> str
        z_base: Current objective value (for context in diagnostics)
        l1_diagnostics: Diagnostic objects from L1 verification
        l2_diagnostics: Diagnostic objects from L2 verification
        l3_diagnostics: Diagnostic objects from L3 verification
        max_specs: Maximum number of specification items (to control cost)

    Returns:
        {
            'specs': List[Dict],          # Extracted specifications
            'results': List[Dict],        # Verification results
            'diagnostics': List[Diagnostic],  # Actionable diagnostics
            'summary': {
                'total': int,
                'pass': int,
                'fail': int,
                'uncertain': int
            }
        }
    """
    # Phase 1: Extract specifications (6 category-specific calls)
    specs = extract_specification(problem_desc, llm_fn)

    if not specs:
        return {
            "specs": [],
            "results": [],
            "diagnostics": [],
            "summary": {"total": 0, "pass": 0, "fail": 0, "uncertain": 0},
        }

    # Truncate if too many specs (control LLM cost)
    if len(specs) > max_specs:
        specs = specs[:max_specs]

    # Phase 2: Verify with execution context
    results = verify_specification(
        specs, code, llm_fn,
        l1_diagnostics=l1_diagnostics,
        l2_diagnostics=l2_diagnostics,
        l3_diagnostics=l3_diagnostics,
        z_base=z_base,
    )

    # Phase 3: Convert to diagnostics
    diagnostics = results_to_diagnostics(results)

    # Summary statistics
    verdicts = [r.get("verdict", "").upper() for r in results]
    summary = {
        "total": len(results),
        "pass": verdicts.count("PASS"),
        "fail": verdicts.count("FAIL"),
        "uncertain": verdicts.count("UNCERTAIN"),
    }

    logger.info(
        f"L4 Summary: {summary['total']} specs checked - "
        f"PASS={summary['pass']}, FAIL={summary['fail']}, "
        f"UNCERTAIN={summary['uncertain']}"
    )

    return {
        "specs": specs,
        "results": results,
        "diagnostics": diagnostics,
        "summary": summary,
    }


# ============================================================
# Utilities
# ============================================================

def _parse_json_response(response: str) -> list:
    """
    Parse JSON from LLM response, handling common formatting issues.

    LLM may wrap JSON in ```json ... ``` blocks or include preamble text.
    """
    # Try direct parse first
    try:
        result = json.loads(response)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try finding first [ ... ] block
    bracket_match = re.search(r"\[.*\]", response, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []
