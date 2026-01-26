# ReLoop: Core Contributions

## Paper Information

**Title:** ReLoop: Reliable LLM-based Optimization Modeling via Sensitivity-Based Behavioral Verification

**Target:** NeurIPS 2025 / AAAI 2026

---

## Three Core Contributions

### Contribution 1: Silent Failure Problem (Problem Definition)

**What:**
We identify and quantify a critical failure mode in LLM-generated optimization code: **Silent Failure** - code that executes successfully and returns OPTIMAL status, but produces incorrect solutions due to constraint semantic errors.

**Why Important:**
- Crashes → Easy to detect (runtime error)
- INFEASIBLE → Easy to detect (solver status)
- **Silent Failure → UNDETECTABLE** without verification

**Key Insight:** Even with detailed mathematical specifications (variable definitions, constraint formulas), LLMs fail to correctly translate constraints into code. This is a **reasoning gap**, not a prompting gap.

**Novelty:** First systematic study of silent failures in LLM-generated OR code.

---

### Contribution 2: 6-Layer Behavioral Verification (Method)

**What:**
A universal verification framework using sensitivity-based testing to detect constraint semantic errors by comparing model behavior against expected outcomes, **without parsing generated code**.

**6-Layer Verification System:**

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Execution Verification                        │
│    └── Does code run without errors?                    │
│                                                         │
│  Layer 2: Feasibility Verification                      │
│    └── OPTIMAL? INFEASIBLE? UNBOUNDED?                  │
│                                                         │
│  Layer 3: Monotonicity Verification (Universal - Core)  │
│    └── Does each parameter affect objective?            │
│    └── "No effect" = constraint likely MISSING          │
│                                                         │
│  Layer 4: Sensitivity Verification (Role-Based)         │
│    └── demand↑ → cost↑? capacity↓ → cost↑?              │
│                                                         │
│  Layer 5: Boundary Verification                         │
│    └── param=0 behavior? param=∞ behavior?              │
│                                                         │
│  Layer 6: Domain Probes (optional, for RetailOpt)       │
│    └── init, holding cost, lost sales, substitution     │
└─────────────────────────────────────────────────────────┘
```

**Design Principles:**
1. **Universal (No Domain Knowledge)** - Layer 3 works for ANY optimization problem
2. **Parameter Perturbation** - Sensitivity analysis via ±20% perturbation
3. **Observable Outcomes** - Judge by objective/status only, no code parsing
4. **No Ground Truth Required** - Tests relative behavior, not absolute correctness

**Key Insight (Layer 3 - Monotonicity):**
> If a parameter appears in a constraint, perturbing it should affect the objective.
> "No effect" indicates the constraint is likely MISSING from the model.

**Smart Parameter Filtering:**
- Skip zero-value parameters (e.g., `lead_time=0`, `return_rate=0`)
- Skip Big-M values (e.g., `labor_cap=99999`) that won't bind
- Only test parameters that should reasonably affect the objective

**Supported Datasets:**

| Dataset | Layers 1-5 | Layer 6 |
|---------|------------|---------|
| RetailOpt-190 | ✅ | ✅ RetailProbes |
| MAMO | ✅ | ❌ N/A |
| NL4OPT | ✅ | ❌ N/A |
| Any LP/MILP | ✅ | Extensible |

**Novelty:** First universal behavioral verification for LLM-generated optimization code via sensitivity analysis (not LLM self-checking).

---

### Contribution 3: RetailOpt-190 Benchmark (Dataset)

**What:**
A vertically-focused benchmark for retail supply chain optimization:
- **38 structural archetypes** × **5 numerical variants** = **190 instances**
- **8 mechanism families**
- **Universal solver** provides ground truth
- **Semantic probes** enable automated verification
- **No code template** - tests actual translation ability

**Novelty:** First OR benchmark with integrated semantic verification and no code scaffolding.

---

## Current Limitations (Honest Assessment)

| Limitation | Description | Impact |
|------------|-------------|--------|
| **Manual Probe Design** | 8 probes are hand-crafted | Does not scale to new domains |
| **No Coverage Guarantee** | Unknown if probes detect all errors | May miss some failure modes |
| **Domain-Specific** | Probes designed for retail OR | Generalization unclear |
| **Empirical Validation** | No formal proof of correctness | Theoretical gap |

---

## Future Research Directions

### Direction 1: Automatic Probe Generation

**Research Question:** Given a constraint formula, can we automatically generate test data that distinguishes correct from incorrect implementations?

**Approach:**
```
Input: Constraint C(x) ≤ b
Output: Test data d such that:
  - Correct implementation: satisfies C
  - Incorrect implementation (missing C): violates C or has wrong objective
  
Method: SMT solving / symbolic execution / LLM-in-the-loop
```

**Potential Contribution:** Generalizable probe generation framework

---

### Direction 2: Semantic Coverage Analysis

**Research Question:** How do we measure and guarantee that probes cover all critical constraint types?

**Formalization:**
```
Definition (Semantic Coverage):
  For model M = (Variables V, Constraints C, Objective O)
  Coverage(Probes P) = |{c ∈ C : ∃p ∈ P that tests c}| / |C|

Challenge: Constraints interact - testing A may implicitly test B
```

**Approach:** Borrow from mutation testing in software engineering

---

### Direction 3: Formal Verification Framework

**Research Question:** Can we prove probe soundness (no false negatives)?

**Definitions:**
```
Definition (Silent Failure):
  Model M has silent failure on data d ⟺ 
    M.solve(d).status = OPTIMAL ∧ 
    M.solve(d).obj ≠ ground_truth(d)

Definition (Probe Soundness):
  Probe P is sound for constraint C ⟺ 
    ∀ implementation M' missing C: P(M') = FAIL

Definition (Probe Set Completeness):
  Probe set {P₁...Pₖ} is complete ⟺ 
    ∀ constraint Cᵢ ∈ M, ∃ Pⱼ that is sound for Cᵢ
```

**Open Problem:**
> Given optimization model specification S, construct minimal sound and complete probe set for single-constraint errors.

---

### Direction 4: Cross-Domain Generalization

**Research Question:** Do probe design principles transfer to other OR domains?

**Candidate Domains:**
- Vehicle Routing (VRP)
- Scheduling
- Facility Location
- Portfolio Optimization

**Hypothesis:** Probe design principles (mechanism isolation, boundary conditions, behavioral testing) are domain-agnostic.

---

## Experimental Design

### Baseline Models

| Model | Type | Why Selected |
|-------|------|--------------|
| **GPT-4o** | Closed-source | SOTA general LLM |
| **Claude Opus 4.5** | Closed-source | SOTA general LLM |
| **Qwen3-Max** | Closed-source | Strong reasoning + code |
| SIRL-7B | Open-source | Training-time RL method |
| ORLM-8B | Open-source | Training-time SFT method |

### Evaluation Modes

| Mode | Pipeline | Verification | Repair |
|------|----------|--------------|--------|
| Baseline | Single LLM call | Post-hoc only | No |
| ReLoop | 3-step generation | 6-layer verification | Yes (up to 5x) |

### Metrics

| Metric | Definition |
|--------|------------|
| Execution Rate | % of scripts that run without error |
| Layers Passed | Number of verification layers passed (0-6) |
| Objective Accuracy | % within 1% of ground truth |
| Silent Failure Rate | Execution OK but objective wrong |

---

## Paper Structure (Proposed)

```
1. Introduction
   - Motivation: LLMs for optimization modeling
   - Problem: Silent failures are prevalent
   - Solution: Semantic probe verification

2. Related Work
   - LLMs for code generation
   - LLMs for optimization (ORLM, OptiMus)
   - Program testing and verification

3. Silent Failure Problem [Contribution 1]
   - Definition and taxonomy
   - Prevalence study on RetailOpt-190
   - Why traditional testing fails

4. Semantic Probe Framework [Contribution 2]
   - Design principles
   - 8 probe specifications
   - Integration with repair loop
   - Limitations and coverage analysis

5. RetailOpt-190 Benchmark [Contribution 3]
   - Design rationale (no code template)
   - 38 archetypes × 5 variants
   - Universal solver

6. Experiments
   - RQ1: Silent failure rates across LLMs
   - RQ2: Probe detection effectiveness
   - RQ3: Repair improvement with probes
   - RQ4: Generalization to other benchmarks

7. Analysis & Discussion
   - Error patterns by constraint type
   - Ablation: number of probes
   - Limitations and future work

8. Conclusion
```

---

## Key Claims (To Be Validated by Experiments)

| Claim | Required Evidence |
|-------|-------------------|
| Silent failures are prevalent | Table: Failure rates on RetailOpt-190 |
| 6-layer verification detects most silent failures | Precision/recall analysis |
| Sensitivity analysis (L3) is universally effective | Results on MAMO, NL4OPT without domain probes |
| Diagnosis-guided repair improves accuracy | Comparison with/without diagnosis |
| Framework generalizes across models | Results on GPT-4o, Claude, SIRL, ORLM |

---

## Reviewer Anticipated Questions

**Q: Is this just prompt engineering?**
A: No. The core contribution is the verification framework (probes via code execution), not the prompts. Probes work on any LLM output without modification.

**Q: Why not use unit tests?**
A: Unit tests require knowing implementation details. Probes test behavioral correctness of any code claiming to solve the same problem.

**Q: Why not let LLM self-verify?**
A: LLMs are unreliable at detecting their own errors. Probes use actual code execution for ground-truth verification.

**Q: Do probes guarantee correctness?**
A: No. Probes provide necessary but not sufficient conditions. A script passing all probes may still have errors not covered by probes.

---

## One-Sentence Summary

1. **Problem:** LLM-generated optimization code often runs successfully but produces wrong answers (silent failure).

2. **Method:** Semantic probes detect constraint errors through boundary testing via code execution, not code parsing or LLM prompting.

3. **Benchmark:** RetailOpt-190 provides 190 instances with ground truth, semantic probes, and no code scaffolding.