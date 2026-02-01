# ==============================================================================
# STEP 1: PROBLEM UNDERSTANDING (Universal)
# ==============================================================================

STEP1_PROMPT = """
[STEP 1: PROBLEM UNDERSTANDING]

You are an expert operations research analyst. Analyze this optimization problem and extract its key components DIRECTLY from the natural language description.

## Problem Description
{business_narrative}

## Task
Extract the ESSENTIAL problem structure. Focus on what is NECESSARY for a correct model.
Do NOT over-engineer - include only constraints that are explicitly required.

Output a JSON object with:

{
  "problem_type": "LP/IP/MILP/NLP",

  "objective": {
    "sense": "minimize" or "maximize",
    "description": "what we are optimizing in plain language",
    "components": ["list every cost/revenue/penalty term mentioned"]
  },

  "decisions": [
    {
      "name": "descriptive name of the decision",
      "symbol": "suggested variable name",
      "type": "continuous/binary/integer",
      "indexed_by": ["list of index sets this variable depends on (infer sets from text)"],
      "meaning": "what this decision variable represents"
    }
  ],

  "constraints": [
    {
      "name": "constraint name",
      "type": "equality/inequality/bound",
      "category": "balance/capacity/demand/flow/logical/linking",
      "description": "plain language description of what this constraint enforces"
    }
  ],

  "key_relationships": [
    "describe key dynamic or linking relationships in the problem"
  ],

  "special_considerations": [
    "edge cases, boundary handling, optional features inferred from text"
  ]
}

Output ONLY valid JSON. No other text.
"""

# ==============================================================================
# STEP 2: MATHEMATICAL SPECIFICATION (Universal)
# ==============================================================================

STEP2_PROMPT = """
[STEP 2: MATHEMATICAL SPECIFICATION]

Based on the problem understanding, create a formal mathematical specification.

## Problem Understanding (from Step 1)
{step1_output}

## Task
Create a MINIMAL but correct mathematical formulation.
Focus on ESSENTIAL variables and constraints - do NOT add unnecessary complexity.
A simpler model that works is better than a complex model.

## CRITICAL: Objective Function Completeness
List every cost/revenue/penalty term mentioned in the narrative in the objective.

## CRITICAL: Copy Original Equations EXACTLY
If the Original Problem Equations section below contains specific formulas, you MUST:
1. Copy them EXACTLY into the constraints section
2. Do NOT add, remove, or modify any terms
3. Preserve the EXACT index patterns - DO NOT shift indices!
4. Do NOT add extra terms to equations

Output a JSON object with:

{
  "sets": {
    "symbol": "definition and meaning (e.g., T = set of time periods)"
  },

  "parameters": {
    "symbol[indices]": "meaning (state if inferred from text)"
  },

  "variables": {
    "symbol[indices]": {
      "type": "continuous/binary/integer",
      "bounds": ">=0, [0,1], etc.",
      "meaning": "what this variable represents"
    }
  },

  "objective": {
    "sense": "minimize/maximize",
    "expression": "mathematical expression in words",
    "terms": [
      {"name": "term_name", "formula": "mathematical formula"}
    ]
  },

  "constraints": [
    {
      "name": "constraint_name",
      "formula": "mathematical formula using symbols defined above",
      "forall": "index ranges where this constraint applies",
      "explanation": "what this constraint ensures"
    }
  ],

  "boundary_conditions": [
    {
      "condition": "when this applies (e.g., first period, last period)",
      "formula": "mathematical formula or special handling",
      "explanation": "why this boundary condition is needed"
    }
  ]
}

Be mathematically precise. Use standard optimization notation.
Output ONLY valid JSON.
"""

# ==============================================================================
# STEP 3: CODE GENERATION (Universal)
# ==============================================================================

STEP3_PROMPT = """
[STEP 3: CODE GENERATION]

Generate executable GurobiPy code from the mathematical specification. If schema/data is absent, infer sets/parameters from the narrative and create minimal placeholders so the code runs.

## CRITICAL: EQUATION COPYING RULES (READ FIRST!)
- No extra terms; copy equations verbatim.
- No index shifting; keep indices exactly as written.
- Separate constraints; do not merge multiple relationships.
- Use all declared indices for each variable.

## Mathematical Specification (from Step 2)
{step2_output}

## Data Access Patterns and Problem Context (may be empty)
{data_access}

## Implementation Notes
- Ensure all variables and constraints are implemented based on the problem's natural language description.
- If no data is provided, define minimal placeholders for sets and parameters based on the problem description (e.g., time periods, budget, etc.).
- Include every cost/revenue/penalty term mentioned in the narrative or spec in the objective.
- Preserve boundary conditions and index expressions exactly.
- Solver params: OutputFlag=0, Threads=1, Seed=0, TimeLimit=120.

## Code Skeleton
```python
import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model()
m.Params.OutputFlag = 0
m.Params.Threads = 1
m.Params.Seed = 0
m.Params.TimeLimit = 120

# Decision variables
# Define decision variables based on the extracted specification
# ...

# Objective function
# Define the objective function based on the specification
# ...

# Constraints
# Add constraints based on the problem's description
# ...

# Optimize the model
m.optimize()

# Print the results
print(f"status: {m.Status}")
if m.Status == 2:
    print(f"objective: {m.ObjVal}")

## Task
Generate complete, executable code. Keep the model SIMPLE and efficient.
Avoid overcomplicating with unnecessary variables or constraints.

Output ONLY Python code. No markdown, no explanations.
"""


# ==============================================================================
# REPAIR PROMPT (Universal)
# ==============================================================================

REPAIR_PROMPT = """
[CODE REPAIR]

The generated code has verification failures. Fix the issues based on the diagnosis.

## Original Code
```python
{code}
```

## Verification Report
{verification_report}

## Specific Failures and Diagnoses
{diagnoses}

## Common Fix Patterns

1. TIMEOUT (CODE_9) or Slow Solving:
   - SIMPLIFY the model - reduce unnecessary variables/constraints
   - Add time limit: m.Params.TimeLimit = 120
   - Use fewer age cohorts if shelf_life is large
   - Aggregate similar constraints where possible

2. INFEASIBLE Model:
   - Add slack/penalty variables for constraints that may be too tight
   - Check if demand can exceed supply without a slack variable

3. No Effect on Objective (Monotonicity Failure):
   - The parameter is not being used in a constraint
   - Check if the constraint was actually added to the model
   - Verify the parameter appears in the constraint formula

4. Wrong Direction (Sensitivity Failure):
   - Check constraint inequality direction (≤ vs ≥)
   - Verify parameter appears on correct side of constraint

5. Objective Near Zero or Unexpected Value:
   - Check initialization/boundary conditions at first period
   - Verify all cost/revenue terms are included in objective
   - Check for "free" resources from missing constraints

6. Boundary Test Failure:
   - Check handling of zero values
   - Verify extreme value handling doesn't cause errors

7. KeyError or IndexError:
   - Network edges must be tuples: [tuple(e) for e in edges]
   - Check if indices match variable dimensions
   - Variable index ranges must match how they are accessed in constraints
   - For varying ranges (e.g., shelf_life differs by product), use dict comprehension:
     {{(p,l,t,r): m.addVar(...) for p in products ... for r in range(1, shelf_life[p]+1)}}

8. Negative Objective in Minimization:
   - Check if deficit terms can become negative (e.g., demand - sales when sales > demand)
   - Use explicit slack variables with lb=0 for deficits:
     unmet = m.addVars(..., lb=0); m.addConstr(sales + unmet == demand)

## Task
Fix the code to pass verification. Make minimal, targeted changes focused on the diagnosed issues.

Output ONLY the corrected Python code. No explanations.
"""


# ==============================================================================
# BASELINE PROMPT (Direct generation without structured steps)
# ==============================================================================

BASELINE_PROMPT = """
[OPTIMIZATION PROBLEM]

## Problem Description
{business_narrative}

## Task
Write a complete GurobiPy optimization model that:
1. Reads parameters from the `data` dict
2. Creates appropriate decision variables
3. Sets the objective function
4. Adds all necessary constraints
5. Solves and prints results

## Output Format
- Import: import gurobipy as gp; from gurobipy import GRB
- Set Gurobi params: OutputFlag=0, Threads=1, Seed=0
- Print at end:
  print(f"status: {{m.Status}}")
  if m.Status == 2:
      print(f"objective: {{m.ObjVal}}")

Output ONLY executable Python code. No markdown, no explanations.
"""

# ==============================================================================
# PROMPT GENERATOR CLASS
# ==============================================================================

class PromptGenerator:
    """Generate appropriate prompts based on dataset type and step"""

    @staticmethod
    def baseline(narrative: str, schema: str = "", dataset_type: str = "prompt_only") -> str:
        """Generate baseline prompt (direct generation without steps)"""
        return BASELINE_PROMPT.format(
            business_narrative=narrative,
            data_schema=""  # schema no longer used
        )

    @staticmethod
    def step1(narrative: str, schema: str = "") -> str:
        """Generate Step 1 prompt (Problem Understanding)"""
        return STEP1_PROMPT.format(business_narrative=narrative)

    @staticmethod
    def step2(step1_output: str, schema: str = "", original_problem: str = None) -> str:
        """Generate Step 2 prompt (Mathematical Specification)"""
        equations_section = ""
        if original_problem:
            equations_section = f"\n\n## Original Problem Equations (MUST COPY EXACTLY)\n{original_problem}"
        return STEP2_PROMPT.format(
            step1_output=step1_output + equations_section
        )

    @staticmethod
    def step3(step2_output: str, data_access: str = "") -> str:
        """Generate Step 3 prompt (Code Generation)"""
        return STEP3_PROMPT.format(
            step2_output=step2_output,
            data_access=data_access
        )

    @staticmethod
    def repair(code: str, report: str, diagnoses: str) -> str:
        """Generate repair prompt"""
        return REPAIR_PROMPT.format(
            code=code,
            verification_report=report,
            diagnoses=diagnoses
        )



# Placeholders removed: schema-free pipeline only uses core prompts above.
