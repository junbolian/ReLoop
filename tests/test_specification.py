"""Unit tests for L4 Specification Compliance Checking (Enhanced)."""

import json
import pytest
from reloop.specification import (
    _parse_json_response,
    _deduplicate_specs,
    _build_execution_context,
    extract_specification,
    verify_specification,
    results_to_diagnostics,
    run_l4,
    CATEGORY_PROMPTS,
)
from reloop.verification import Diagnostic


# ---- _parse_json_response ----

class TestParseJsonResponse:
    """Tests for the JSON parsing utility."""

    def test_direct_json_array(self):
        response = '[{"id": 1, "category": "VARIABLE_TYPE"}]'
        result = _parse_json_response(response)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_json_in_code_block(self):
        response = 'Here is the result:\n```json\n[{"id": 1, "verdict": "PASS"}]\n```\n'
        result = _parse_json_response(response)
        assert len(result) == 1
        assert result[0]["verdict"] == "PASS"

    def test_json_in_plain_code_block(self):
        response = '```\n[{"id": 1}]\n```'
        result = _parse_json_response(response)
        assert len(result) == 1

    def test_json_with_preamble_text(self):
        response = 'The specification checklist is:\n[{"id": 1, "category": "CONSTRAINT"}]'
        result = _parse_json_response(response)
        assert len(result) == 1

    def test_empty_array(self):
        result = _parse_json_response("[]")
        assert result == []

    def test_invalid_json(self):
        result = _parse_json_response("this is not json at all")
        assert result == []

    def test_json_object_not_array(self):
        result = _parse_json_response('{"key": "value"}')
        assert result == []

    def test_unicode_content(self):
        response = '[{"id": 1, "requirement": "integer variables required"}]'
        result = _parse_json_response(response)
        assert len(result) == 1


# ---- _deduplicate_specs ----

class TestDeduplicateSpecs:
    """Tests for deduplication logic."""

    def test_no_duplicates(self):
        specs = [
            {"id": 1, "category": "A", "checkable_criterion": "check_a"},
            {"id": 2, "category": "B", "checkable_criterion": "check_b"},
        ]
        result = _deduplicate_specs(specs)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_exact_duplicate_removed(self):
        specs = [
            {"id": 1, "category": "A", "checkable_criterion": "vtype=GRB.INTEGER"},
            {"id": 2, "category": "B", "checkable_criterion": "vtype=GRB.INTEGER"},
        ]
        result = _deduplicate_specs(specs)
        assert len(result) == 1
        assert result[0]["id"] == 1  # Renumbered

    def test_case_insensitive(self):
        specs = [
            {"id": 1, "category": "A", "checkable_criterion": "Check This"},
            {"id": 2, "category": "B", "checkable_criterion": "check this"},
        ]
        result = _deduplicate_specs(specs)
        assert len(result) == 1

    def test_whitespace_stripped(self):
        specs = [
            {"id": 1, "category": "A", "checkable_criterion": "  check  "},
            {"id": 2, "category": "B", "checkable_criterion": "check"},
        ]
        result = _deduplicate_specs(specs)
        assert len(result) == 1

    def test_renumbering(self):
        specs = [
            {"id": 10, "category": "A", "checkable_criterion": "a"},
            {"id": 20, "category": "B", "checkable_criterion": "a"},  # dup
            {"id": 30, "category": "C", "checkable_criterion": "c"},
        ]
        result = _deduplicate_specs(specs)
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_empty_list(self):
        assert _deduplicate_specs([]) == []


# ---- _build_execution_context ----

class TestBuildExecutionContext:
    """Tests for execution context builder."""

    def test_no_context(self):
        result = _build_execution_context()
        assert result == ""

    def test_z_base_only(self):
        result = _build_execution_context(z_base=53.9)
        assert "53.9" in result
        assert "Execution Context" in result

    def test_l1_diagnostics(self):
        d = Diagnostic(
            layer="L1", issue_type="DUALITY_GAP", severity="INFO",
            target_name="duality", evidence="Gap = 0.01%", triggers_repair=False
        )
        result = _build_execution_context(l1_diagnostics=[d])
        assert "L1 (Execution)" in result
        assert "DUALITY_GAP" in result

    def test_l2_diagnostics(self):
        d = Diagnostic(
            layer="L2", issue_type="DIRECTION_VIOLATION", severity="ERROR",
            target_name="param_a", evidence="Expected increase", triggers_repair=True
        )
        result = _build_execution_context(l2_diagnostics=[d])
        assert "L2 (Direction Analysis)" in result

    def test_l3_diagnostics(self):
        d = Diagnostic(
            layer="L3", issue_type="MISSING_CONSTRAINT", severity="WARNING",
            target_name="protein", evidence="0.0% change", triggers_repair=True
        )
        result = _build_execution_context(l3_diagnostics=[d])
        assert "L3 (Constraint Presence)" in result

    def test_all_combined(self):
        d1 = Diagnostic(layer="L1", issue_type="X", severity="INFO",
                        target_name="t", evidence="e1", triggers_repair=False)
        d2 = Diagnostic(layer="L2", issue_type="Y", severity="ERROR",
                        target_name="t", evidence="e2", triggers_repair=True)
        d3 = Diagnostic(layer="L3", issue_type="Z", severity="WARNING",
                        target_name="t", evidence="e3", triggers_repair=True)
        result = _build_execution_context(
            l1_diagnostics=[d1], l2_diagnostics=[d2],
            l3_diagnostics=[d3], z_base=100.0
        )
        assert "L1" in result
        assert "L2" in result
        assert "L3" in result
        assert "100.0" in result

    def test_evidence_truncation(self):
        long_evidence = "x" * 500
        d = Diagnostic(layer="L1", issue_type="TEST", severity="INFO",
                        target_name="t", evidence=long_evidence, triggers_repair=False)
        result = _build_execution_context(l1_diagnostics=[d])
        # Evidence should be truncated to 200 chars — check the evidence portion, not total string
        assert "x" * 201 not in result
        assert "x" * 100 in result  # truncated but still present


# ---- results_to_diagnostics ----

class TestResultsToDiagnostics:
    """Tests for converting verification results to Diagnostic objects."""

    def test_fail_becomes_error_diagnostic(self):
        results = [{
            "spec_id": 1, "verdict": "FAIL",
            "code_evidence": "line 12: m.addVar(lb=0)",
            "explanation": "Variable is continuous, should be integer",
            "category": "VARIABLE_TYPE",
            "requirement": "food servings should be whole numbers",
            "checkable_criterion": "vtype=GRB.INTEGER",
        }]
        diags = results_to_diagnostics(results)
        assert len(diags) == 1
        assert diags[0].layer == "L4"
        assert diags[0].issue_type == "SPEC_VIOLATION"
        assert diags[0].severity == "ERROR"
        assert diags[0].triggers_repair is True
        assert "VARIABLE_TYPE" in diags[0].target_name

    def test_uncertain_becomes_warning_diagnostic(self):
        results = [{
            "spec_id": 2, "verdict": "UNCERTAIN",
            "code_evidence": "No relevant code found",
            "explanation": "Cannot determine",
            "category": "LOGICAL_CONDITION",
            "requirement": "if demand > supply then penalize",
        }]
        diags = results_to_diagnostics(results)
        assert len(diags) == 1
        assert diags[0].severity == "WARNING"
        assert diags[0].triggers_repair is False

    def test_pass_produces_no_diagnostic(self):
        results = [{
            "spec_id": 3, "verdict": "PASS",
            "code_evidence": "line 15: m.addVar(vtype=GRB.INTEGER)",
            "explanation": "Correctly integer",
            "category": "VARIABLE_TYPE",
            "requirement": "integer variables",
        }]
        diags = results_to_diagnostics(results)
        assert len(diags) == 0

    def test_mixed_verdicts(self):
        results = [
            {"spec_id": 1, "verdict": "PASS", "category": "A", "requirement": "x"},
            {"spec_id": 2, "verdict": "FAIL", "category": "B", "requirement": "y",
             "code_evidence": "x", "explanation": "wrong", "checkable_criterion": "z"},
            {"spec_id": 3, "verdict": "UNCERTAIN", "category": "C", "requirement": "z",
             "code_evidence": "x", "explanation": "unclear"},
            {"spec_id": 4, "verdict": "PASS", "category": "D", "requirement": "w"},
        ]
        diags = results_to_diagnostics(results)
        assert len(diags) == 2
        assert diags[0].severity == "ERROR"
        assert diags[1].severity == "WARNING"

    def test_empty_results(self):
        assert results_to_diagnostics([]) == []

    def test_missing_verdict_key(self):
        results = [{"spec_id": 1, "category": "A"}]
        diags = results_to_diagnostics(results)
        assert len(diags) == 0


# ---- extract_specification (mock LLM, 6-category) ----

class TestExtractSpecification:
    """Tests for Phase 1: category-specific extraction."""

    def _make_category_llm_fn(self, category_responses: dict):
        """Create mock LLM that returns different responses per category.

        category_responses: dict mapping category name -> JSON response string.
        Categories not in the dict return '[]'.
        """
        def llm_fn(system_prompt, user_prompt):
            for cat, resp in category_responses.items():
                if cat in system_prompt:
                    return resp
            return "[]"
        return llm_fn

    def test_single_category_extraction(self):
        """Should extract specs from one category."""
        responses = {
            "VARIABLE_TYPE": json.dumps([{
                "id": 1, "category": "VARIABLE_TYPE",
                "requirement": "integer servings",
                "checkable_criterion": "vtype=GRB.INTEGER",
            }]),
        }
        specs = extract_specification("diet problem", self._make_category_llm_fn(responses))
        assert len(specs) >= 1
        vt_specs = [s for s in specs if s["category"] == "VARIABLE_TYPE"]
        assert len(vt_specs) == 1
        assert vt_specs[0]["requirement"] == "integer servings"

    def test_multi_category_extraction(self):
        """Should merge specs from multiple categories."""
        responses = {
            "VARIABLE_TYPE": json.dumps([{
                "id": 1, "category": "VARIABLE_TYPE",
                "requirement": "integer vars",
                "checkable_criterion": "vtype=GRB.INTEGER",
            }]),
            "CONSTRAINT": json.dumps([{
                "id": 1, "category": "CONSTRAINT",
                "requirement": "protein >= 83",
                "checkable_criterion": "protein constraint",
            }]),
            "OBJECTIVE": json.dumps([{
                "id": 1, "category": "OBJECTIVE",
                "requirement": "minimize cost",
                "checkable_criterion": "GRB.MINIMIZE",
            }]),
        }
        specs = extract_specification("problem", self._make_category_llm_fn(responses))
        categories = {s["category"] for s in specs}
        assert "VARIABLE_TYPE" in categories
        assert "CONSTRAINT" in categories
        assert "OBJECTIVE" in categories

    def test_empty_categories_produce_no_specs(self):
        """Categories returning [] should not contribute specs."""
        responses = {
            "LOGICAL_CONDITION": "[]",
            "DATA_MAPPING": "[]",
        }
        specs = extract_specification("simple LP", self._make_category_llm_fn(responses))
        lc_specs = [s for s in specs if s["category"] == "LOGICAL_CONDITION"]
        dm_specs = [s for s in specs if s["category"] == "DATA_MAPPING"]
        assert len(lc_specs) == 0
        assert len(dm_specs) == 0

    def test_all_categories_fail(self):
        """If all categories return invalid, should return empty list."""
        def bad_llm(system_prompt, user_prompt):
            return "I cannot help"
        specs = extract_specification("problem", bad_llm)
        assert specs == []

    def test_category_forced_correct(self):
        """Category should be overridden to match the prompt category."""
        responses = {
            "CONSTRAINT": json.dumps([{
                "id": 1, "category": "WRONG_CATEGORY",
                "requirement": "budget limit",
                "checkable_criterion": "budget <= 1000",
            }]),
        }
        specs = extract_specification("problem", self._make_category_llm_fn(responses))
        c_specs = [s for s in specs if s["category"] == "CONSTRAINT"]
        assert len(c_specs) == 1

    def test_invalid_items_filtered(self):
        """Items missing required keys should be filtered out."""
        responses = {
            "CONSTRAINT": json.dumps([
                {"id": 1, "category": "CONSTRAINT"},  # missing keys
                {"id": 2, "category": "CONSTRAINT",
                 "requirement": "x", "checkable_criterion": "y"},
            ]),
        }
        specs = extract_specification("problem", self._make_category_llm_fn(responses))
        c_specs = [s for s in specs if s["category"] == "CONSTRAINT"]
        assert len(c_specs) == 1

    def test_deduplication_across_categories(self):
        """Duplicate checkable_criterion across categories should be deduped."""
        responses = {
            "VARIABLE_TYPE": json.dumps([{
                "id": 1, "category": "VARIABLE_TYPE",
                "requirement": "integer",
                "checkable_criterion": "vtype=GRB.INTEGER",
            }]),
            "CONSTRAINT": json.dumps([{
                "id": 1, "category": "CONSTRAINT",
                "requirement": "also integer",
                "checkable_criterion": "vtype=GRB.INTEGER",
            }]),
        }
        specs = extract_specification("problem", self._make_category_llm_fn(responses))
        # Should be deduped to 1
        int_specs = [s for s in specs if "integer" in s["checkable_criterion"].lower()]
        assert len(int_specs) == 1

    def test_ids_are_sequential(self):
        """After merging and deduplication, IDs should be sequential from 1."""
        responses = {
            "VARIABLE_TYPE": json.dumps([{
                "id": 99, "category": "VARIABLE_TYPE",
                "requirement": "a", "checkable_criterion": "a",
            }]),
            "OBJECTIVE": json.dumps([{
                "id": 88, "category": "OBJECTIVE",
                "requirement": "b", "checkable_criterion": "b",
            }]),
        }
        specs = extract_specification("problem", self._make_category_llm_fn(responses))
        ids = [s["id"] for s in specs]
        assert ids == list(range(1, len(specs) + 1))


# ---- verify_specification (mock LLM) ----

class TestVerifySpecification:
    """Tests for Phase 2: specification verification."""

    def _make_llm_fn(self, response: str):
        def llm_fn(system_prompt, user_prompt):
            return response
        return llm_fn

    def test_basic_verification(self):
        specs = [
            {"id": 1, "category": "VARIABLE_TYPE", "requirement": "integer vars",
             "checkable_criterion": "vtype=GRB.INTEGER"},
        ]
        mock_response = json.dumps([{
            "spec_id": 1, "verdict": "FAIL",
            "code_evidence": "line 12: m.addVar(lb=0)",
            "explanation": "Continuous variable, not integer",
        }])
        results = verify_specification(
            specs, "code here", self._make_llm_fn(mock_response)
        )
        assert len(results) == 1
        assert results[0]["verdict"] == "FAIL"
        assert results[0]["category"] == "VARIABLE_TYPE"

    def test_empty_specs(self):
        results = verify_specification([], "code", self._make_llm_fn("[]"))
        assert results == []

    def test_unknown_spec_id_still_included(self):
        specs = [{"id": 1, "category": "A", "requirement": "x", "checkable_criterion": "y"}]
        mock_response = json.dumps([
            {"spec_id": 1, "verdict": "PASS", "code_evidence": "x", "explanation": "ok"},
            {"spec_id": 99, "verdict": "FAIL", "code_evidence": "y", "explanation": "bad"},
        ])
        results = verify_specification(
            specs, "code", self._make_llm_fn(mock_response)
        )
        assert len(results) == 2

    def test_with_execution_context(self):
        """Execution context should be included in prompt."""
        specs = [{"id": 1, "category": "A", "requirement": "x", "checkable_criterion": "y"}]
        mock_response = json.dumps([{
            "spec_id": 1, "verdict": "PASS",
            "code_evidence": "x", "explanation": "ok",
        }])

        captured_prompts = []
        def capturing_llm(system_prompt, user_prompt):
            captured_prompts.append(user_prompt)
            return mock_response

        d = Diagnostic(layer="L1", issue_type="TEST", severity="INFO",
                       target_name="t", evidence="test evidence", triggers_repair=False)

        results = verify_specification(
            specs, "code", capturing_llm,
            l1_diagnostics=[d], z_base=42.0
        )
        assert len(results) == 1
        # Check that execution context was injected
        assert "42.0" in captured_prompts[0]
        assert "Execution Context" in captured_prompts[0]

    def test_without_execution_context(self):
        """No context should work fine (backward compat)."""
        specs = [{"id": 1, "category": "A", "requirement": "x", "checkable_criterion": "y"}]
        mock_response = json.dumps([{
            "spec_id": 1, "verdict": "PASS",
            "code_evidence": "x", "explanation": "ok",
        }])
        results = verify_specification(
            specs, "code", self._make_llm_fn(mock_response)
        )
        assert len(results) == 1


# ---- run_l4 (mock LLM, 6+1 calls) ----

class TestRunL4:
    """Tests for the L4 main entry point (enhanced with 6-category extraction)."""

    def _make_llm_fn(self, category_responses: dict, verify_response: str):
        """Create mock LLM for run_l4.

        Phase 1: 6 calls (one per category). Matches category in system_prompt.
        Phase 2: 1 call (verification). Matches "code reviewer" in system_prompt.
        """
        def llm_fn(system_prompt, user_prompt):
            if "code reviewer" in system_prompt.lower():
                return verify_response
            for cat, resp in category_responses.items():
                if cat in system_prompt:
                    return resp
            return "[]"
        return llm_fn

    def test_full_l4_with_failures(self):
        """L4 should detect FAIL items and produce ERROR diagnostics."""
        cat_responses = {
            "VARIABLE_TYPE": json.dumps([{
                "id": 1, "category": "VARIABLE_TYPE",
                "requirement": "integer servings",
                "checkable_criterion": "vtype=GRB.INTEGER",
            }]),
            "CONSTRAINT": json.dumps([{
                "id": 1, "category": "CONSTRAINT",
                "requirement": "protein >= 83",
                "checkable_criterion": "protein constraint",
            }]),
        }
        verify_resp = json.dumps([
            {"spec_id": 1, "verdict": "FAIL",
             "code_evidence": "line 5: m.addVar(lb=0)",
             "explanation": "Continuous, not integer"},
            {"spec_id": 2, "verdict": "PASS",
             "code_evidence": "line 10: m.addConstr(...)",
             "explanation": "Constraint present"},
        ])

        result = run_l4(
            code="some code",
            problem_desc="diet problem",
            llm_fn=self._make_llm_fn(cat_responses, verify_resp),
            z_base=53.9,
        )

        assert result["summary"]["fail"] >= 1
        error_diags = [d for d in result["diagnostics"] if d.severity == "ERROR"]
        assert len(error_diags) >= 1
        assert error_diags[0].triggers_repair is True

    def test_all_pass(self):
        """When all specs PASS, no diagnostics should be generated."""
        cat_responses = {
            "CONSTRAINT": json.dumps([{
                "id": 1, "category": "CONSTRAINT",
                "requirement": "demand met",
                "checkable_criterion": "demand constraint",
            }]),
        }
        verify_resp = json.dumps([
            {"spec_id": 1, "verdict": "PASS",
             "code_evidence": "line 8: m.addConstr(...)",
             "explanation": "Correctly implemented"},
        ])

        result = run_l4(
            code="code",
            problem_desc="problem",
            llm_fn=self._make_llm_fn(cat_responses, verify_resp),
        )

        assert result["summary"]["fail"] == 0
        assert result["diagnostics"] == []

    def test_empty_extraction(self):
        """If all Phase 1 categories return nothing, should return empty."""
        def empty_llm(system_prompt, user_prompt):
            return "[]"

        result = run_l4(
            code="code",
            problem_desc="problem",
            llm_fn=empty_llm,
        )
        assert result["specs"] == []
        assert result["diagnostics"] == []
        assert result["summary"]["total"] == 0

    def test_max_specs_truncation(self):
        """Specs should be truncated to max_specs."""
        # Generate many CONSTRAINT specs
        many_specs = [
            {"id": i, "category": "CONSTRAINT",
             "requirement": f"req_{i}",
             "checkable_criterion": f"check_{i}"}
            for i in range(30)
        ]
        cat_responses = {
            "CONSTRAINT": json.dumps(many_specs),
        }
        verify_resp = json.dumps([
            {"spec_id": i, "verdict": "PASS",
             "code_evidence": "x", "explanation": "ok"}
            for i in range(1, 6)
        ])

        result = run_l4(
            code="code",
            problem_desc="problem",
            llm_fn=self._make_llm_fn(cat_responses, verify_resp),
            max_specs=5,
        )
        assert len(result["specs"]) == 5

    def test_uncertain_produces_warning(self):
        """UNCERTAIN verdicts should produce WARNING diagnostics."""
        cat_responses = {
            "DATA_MAPPING": json.dumps([{
                "id": 1, "category": "DATA_MAPPING",
                "requirement": "rate interpretation",
                "checkable_criterion": "coefficient check",
            }]),
        }
        verify_resp = json.dumps([{
            "spec_id": 1, "verdict": "UNCERTAIN",
            "code_evidence": "unclear",
            "explanation": "Cannot determine coefficient meaning",
        }])

        result = run_l4(
            code="code",
            problem_desc="problem",
            llm_fn=self._make_llm_fn(cat_responses, verify_resp),
        )

        assert result["summary"]["uncertain"] == 1
        assert len(result["diagnostics"]) == 1
        assert result["diagnostics"][0].severity == "WARNING"
        assert result["diagnostics"][0].triggers_repair is False

    def test_l1_l2_l3_diagnostics_passed_through(self):
        """L1-L3 diagnostics should be forwarded to verification."""
        cat_responses = {
            "CONSTRAINT": json.dumps([{
                "id": 1, "category": "CONSTRAINT",
                "requirement": "budget",
                "checkable_criterion": "budget constraint",
            }]),
        }

        captured = []
        def capturing_llm(system_prompt, user_prompt):
            captured.append({"system": system_prompt, "user": user_prompt})
            if "code reviewer" in system_prompt.lower():
                return json.dumps([{
                    "spec_id": 1, "verdict": "PASS",
                    "code_evidence": "x", "explanation": "ok",
                }])
            for cat in CATEGORY_PROMPTS:
                if cat in system_prompt:
                    return cat_responses.get(cat, "[]")
            return "[]"

        d1 = Diagnostic(layer="L1", issue_type="TEST", severity="INFO",
                        target_name="t", evidence="l1 evidence", triggers_repair=False)

        result = run_l4(
            code="code",
            problem_desc="problem",
            llm_fn=capturing_llm,
            l1_diagnostics=[d1],
            z_base=99.0,
        )

        # Find the verification call (last one)
        verify_calls = [c for c in captured if "code reviewer" in c["system"].lower()]
        assert len(verify_calls) == 1
        assert "l1 evidence" in verify_calls[0]["user"]
        assert "99.0" in verify_calls[0]["user"]


# ---- CATEGORY_PROMPTS structure ----

class TestCategoryPrompts:
    """Tests for category prompt structure."""

    def test_all_six_categories_present(self):
        expected = {"VARIABLE_TYPE", "VARIABLE_SCOPE", "CONSTRAINT",
                    "OBJECTIVE", "LOGICAL_CONDITION", "DATA_MAPPING"}
        assert set(CATEGORY_PROMPTS.keys()) == expected

    def test_each_has_prompt_key(self):
        for cat, config in CATEGORY_PROMPTS.items():
            assert "prompt" in config, f"{cat} missing 'prompt' key"

    def test_prompts_have_placeholder(self):
        for cat, config in CATEGORY_PROMPTS.items():
            assert "{problem_desc}" in config["prompt"], \
                f"{cat} prompt missing {{problem_desc}} placeholder"

    def test_prompts_format_without_error(self):
        """All prompts should format without error."""
        for cat, config in CATEGORY_PROMPTS.items():
            formatted = config["prompt"].format(problem_desc="test problem")
            assert "test problem" in formatted
