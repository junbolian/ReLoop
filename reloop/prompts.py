"""
ReLoop Prompt Templates

Universal prompts for the ReLoop pipeline:
- Step 1: Problem Understanding
- Step 2: Mathematical Specification
- Step 3: Code Generation
- Repair: Diagnosis-guided repair

Design Principle:
- These prompts are UNIVERSAL GUIDES, not domain-specific templates
- The actual business description comes from scenario-specific files (scenarios/prompts/)
- Step prompts provide structure and format guidance for ANY optimization problem
"""

# ==============================================================================
# STEP 1: PROBLEM UNDERSTANDING (Universal)
# ==============================================================================

STEP1_PROMPT = """
[STEP 1: PROBLEM UNDERSTANDING]

You are an expert operations research analyst. Analyze this optimization problem and extract its key components.

## Problem Description
{business_narrative}

## Data Schema
{data_schema}

## Task
Extract the ESSENTIAL problem structure. Focus on what is NECESSARY for a correct model.
Do NOT over-engineer - include only constraints that are explicitly required.

CRITICAL: Check the Data Schema's "costs" section carefully.
ALL cost parameters in the schema MUST appear in the objective function.

Output a JSON object with:

{{
  "problem_type": "LP/IP/MILP/NLP",

  "objective": {{
    "sense": "minimize" or "maximize",
    "description": "what we are optimizing in plain language",
    "components": ["MUST list EVERY cost/revenue/penalty term from data['costs']"]
  }},

  "decisions": [
    {{
      "name": "descriptive name of the decision",
      "symbol": "suggested variable name",
      "type": "continuous/binary/integer",
      "indexed_by": ["list of index sets this variable depends on"],
      "meaning": "what this decision variable represents"
    }}
  ],

  "constraints": [
    {{
      "name": "constraint name",
      "type": "equality/inequality/bound",
      "category": "balance/capacity/demand/flow/logical/linking",
      "description": "plain language description of what this constraint enforces"
    }}
  ],

  "key_relationships": [
    "describe key dynamic or linking relationships in the problem"
  ],

  "special_considerations": [
    "any special handling needed (time boundaries, edge cases, optional features, etc.)"
  ]
}}

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

## Data Schema
{data_schema}

## Task
Create a MINIMAL but correct mathematical formulation.
Focus on ESSENTIAL variables and constraints - do NOT add unnecessary complexity.
A simpler model that works is better than a complex model.

## CRITICAL: Objective Function Completeness
Check the Data Schema for ALL cost parameters (usually in data['costs']).
The objective function MUST include ALL cost terms from the data schema.
Do NOT omit any cost terms - each cost in the schema affects the optimal solution.

## CRITICAL: Copy Original Equations EXACTLY
If the Original Problem Equations section below contains specific formulas, you MUST:
1. Copy them EXACTLY into the constraints section
2. Do NOT add, remove, or modify any terms
3. Preserve the EXACT index patterns - DO NOT shift indices!
   - If original says x[t+1] = f(x[t]), keep t+1 on LHS, do NOT change to x[t] = f(x[t-1])
   - If original says y[i,j+1], keep j+1, do NOT change to y[i,j]
4. Do NOT add extra terms to equations (if original says A = B, don't write A = B - C)
5. Do NOT "rewrite" equations by uniformly shifting all time indices

Output a JSON object with:

{{
  "sets": {{
    "symbol": "definition and meaning (e.g., T = set of time periods)"
  }},

  "parameters": {{
    "symbol[indices]": "meaning and how to access from data"
  }},

  "variables": {{
    "symbol[indices]": {{
      "type": "continuous/binary/integer",
      "bounds": ">=0, [0,1], etc.",
      "meaning": "what this variable represents"
    }}
  }},

  "objective": {{
    "sense": "minimize/maximize",
    "expression": "mathematical expression in words",
    "terms": [
      {{"name": "term_name", "formula": "mathematical formula"}}
    ]
  }},

  "constraints": [
    {{
      "name": "constraint_name",
      "formula": "mathematical formula using symbols defined above",
      "forall": "index ranges where this constraint applies",
      "explanation": "what this constraint ensures"
    }}
  ],

  "boundary_conditions": [
    {{
      "condition": "when this applies (e.g., first period, last period)",
      "formula": "mathematical formula or special handling",
      "explanation": "why this boundary condition is needed"
    }}
  ]
}}

Be mathematically precise. Use standard optimization notation.
Output ONLY valid JSON.
"""


# ==============================================================================
# STEP 3: CODE GENERATION (Universal)
# ==============================================================================

STEP3_PROMPT = """
[STEP 3: CODE GENERATION]

Generate executable GurobiPy code from the mathematical specification.

## CRITICAL: EQUATION COPYING RULES (READ FIRST!)

The problem context below contains EXPLICIT EQUATIONS. You MUST implement them EXACTLY.
FAILURE TO COPY EQUATIONS EXACTLY IS THE #1 SOURCE OF ERRORS.

### RULE 1: NO EXTRA TERMS
If the problem says:
    x[t] = f(a, b)

You MUST write:
    x[t] == f(a, b)

DO NOT write:
    x[t] == f(a, b) - c      # WRONG: added -c that's not in original
    x[t] == f(a, b) + d - e  # WRONG: added extra terms

WHY: Each equation describes ONE relationship. Other relationships are SEPARATE constraints.

### RULE 2: NO INDEX SHIFTING
If the problem says:
    x[t+1] = g(x[t], u[t])

You MUST write:
    x[t+1] == g(x[t], u[t])

DO NOT write:
    x[t] == g(x[t-1], u[t-1])  # WRONG: shifted all indices by -1

These look mathematically equivalent but have DIFFERENT computational loop structures.

### RULE 3: SEPARATE CONSTRAINTS
When the problem defines multiple relationships, implement each as a SEPARATE constraint.
DO NOT combine equations. If the problem says:
    - Equation A: x = f(y)
    - Equation B: z = g(x, w)

Write TWO constraints, not one merged equation.

## Mathematical Specification (from Step 2)
{step2_output}

## Data Access Patterns and Problem Context
{data_access}

## Additional Error Prevention

COMMON ERRORS TO AVOID:

1. WRONG INDEX SUBSCRIPTS:
   - If equation says "x[i,j+1]", do NOT use "x[i,j]"
   - Copy indices EXACTLY as written

2. ADDING EXTRA TERMS:
   - If equation says "A = B", do NOT write "A = B - C"
   - Copy the RHS EXACTLY as written in the original

3. INDEX SHIFTING:
   - WRONG: Transforming "x[t+1] = f(x[t])" into "x[t] = f(x[t-1])"
   - ALWAYS preserve the EXACT index expressions

## CRITICAL: Objective Function Completeness Check

BEFORE writing the objective function:
1. Examine the Original Problem Context and data to identify ALL cost/revenue terms
2. Check if there is a 'costs' section in the data - include ALL listed cost parameters
3. The objective function MUST include ALL relevant terms from the data schema
4. If the Mathematical Specification from Step 2 is MISSING any cost terms that exist in the data,
   you MUST ADD them to the objective function

## CRITICAL: State Variable Semantics

When implementing state equations (inventory, stock, balance, etc.):
1. Be consistent about timing: start-of-period vs end-of-period
2. If x[t] is start-of-period state, then end-of-period = x[t] - consumption[t]
3. Costs on state variables typically apply to END-of-period values
4. Follow the problem context for specific variable definitions

## Gurobi API Patterns (Universal)

When implementing the mathematical model in Gurobi, follow these API patterns:

1. VARIABLES WITH VARYING INDEX RANGES:
   If different items have different index ranges, use dict comprehension:
   ```python
   # WRONG: Creates same range for all items
   x = m.addVars(items, times, max(range_per_item.values()), ...)

   # CORRECT: Creates item-specific ranges
   x = {{(i, t, k): m.addVar(...)
        for i in items for t in times
        for k in range(1, range_per_item[i] + 1)}}
   ```

2. SLACK VARIABLES FOR DEFICITS:
   For quantities that represent deficits (shortage, unmet requirement, etc.), use explicit slack variables:
   ```python
   # WRONG: Can become negative if supply > requirement
   deficit[i,t] = requirement[i,t] - supply[i,t]  # May be negative!

   # CORRECT: Use explicit non-negative slack variable
   slack = m.addVars(..., lb=0, name="slack")  # Always >= 0
   m.addConstr(supply[i,t] + slack[i,t] == requirement[i,t])
   ```

3. CONSTRAINT BOUNDS FOR SLACK:
   When adding balance constraints with slack, ensure the slack variable captures the deficit:
   ```python
   m.addConstr(x[i,t] + slack[i,t] == target[i,t])  # slack >= 0 by lb
   ```

## Implementation Notes

1. DATA ACCESS:
   - The variable `data` is pre-loaded as a Python dict
   - Use data.get() for optional/nested fields to avoid KeyError
   - Lists in data are 0-indexed (period t in model uses index [t-1] in data)

2. INDEXING ALIGNMENT:
   - Model indices may be 1-based (periods 1..T) while data arrays are 0-based
   - CRITICAL: In constraint equations, preserve the exact index relationships from the problem
   - If equation has x[i,t+1,k] on LHS and x[i,t,k+1] on RHS, implement this EXACTLY

3. BOUNDARY CONDITIONS:
   - Handle first period initialization carefully (no "previous" period)
   - Handle last period boundaries (no "next" period references)
   - Check for edge cases in constraints

4. SLACK/PENALTY VARIABLES:
   - Include slack variables where constraints may be too tight
   - Unmet demand, overflow, etc. should have slack to avoid infeasibility

5. ROBUSTNESS:
   - Check if optional features exist before implementing
   - Use data.get(key, default) for optional data fields

6. NETWORK EDGES:
   - sub_edges and trans_edges are lists of lists [[a,b], ...]
   - MUST convert to tuples: sub_edges = [tuple(e) for e in data.get('network', {{}}).get('sub_edges', [])]

7. SOLVER SETTINGS:
   - Set time limit to avoid timeout: m.Params.TimeLimit = 120
   - Keep model simple - avoid unnecessary complexity

## Code Structure
```python
import gurobipy as gp
from gurobipy import GRB

# Extract data from pre-loaded 'data' dict
# ...

# Create model
m = gp.Model()
m.Params.OutputFlag = 0
m.Params.Threads = 1
m.Params.Seed = 0
m.Params.TimeLimit = 120  # Avoid timeout

# Decision variables (keep simple)
# ...

# Objective function
# ...

# Constraints (with proper indexing and boundary handling)
# ...

# Solve
m.optimize()

# Output
print(f"status: {{m.Status}}")
if m.Status == 2:
    print(f"objective: {{m.ObjVal}}")
```

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

## Data Schema
The variable `data` is a pre-loaded Python dict with this structure:
{data_schema}

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
# SPECIALIZED BASELINE PROMPTS FOR DIFFERENT DATASETS
# ==============================================================================

MAMO_BASELINE_PROMPT = """
[OPTIMIZATION PROBLEM - MAMO]

## Problem Description
{problem_description}

## Problem Type
{problem_type}

## Variables
{variables}

## Objective
{objective}

## Constraints
{constraints}

## Task
Write GurobiPy code to solve this optimization problem.

Requirements:
- Import gurobipy
- Create model with appropriate variables
- Set objective function
- Add all constraints
- Solve and print results

Output ONLY Python code.
"""

NL4OPT_BASELINE_PROMPT = """
[NL4OPT OPTIMIZATION PROBLEM]

## Problem Statement
{problem_statement}

## Task
Formulate and solve this optimization problem using GurobiPy.

Steps:
1. Identify decision variables from the problem
2. Determine the objective function
3. Extract all constraints
4. Write executable code

Requirements:
- Use GurobiPy (import gurobipy as gp)
- Handle both LP and MILP problems
- Print solver status and optimal objective value

Output ONLY executable Python code.
"""


# ==============================================================================
# PROMPT GENERATOR CLASS
# ==============================================================================

class PromptGenerator:
    """Generate appropriate prompts based on dataset type and step"""

    @staticmethod
    def baseline(narrative: str, schema: str, dataset_type: str = "generic") -> str:
        """Generate baseline prompt (direct generation without steps)"""
        if dataset_type == "mamo":
            return MAMO_BASELINE_PROMPT.format(
                problem_description=narrative,
                problem_type="",
                variables="",
                objective="",
                constraints=""
            )
        elif dataset_type == "nl4opt":
            return NL4OPT_BASELINE_PROMPT.format(problem_statement=narrative)
        else:
            return BASELINE_PROMPT.format(
                business_narrative=narrative,
                data_schema=schema
            )

    @staticmethod
    def step1(narrative: str, schema: str) -> str:
        """Generate Step 1 prompt (Problem Understanding)"""
        return STEP1_PROMPT.format(
            business_narrative=narrative,
            data_schema=schema
        )

    @staticmethod
    def step2(step1_output: str, schema: str, original_problem: str = None) -> str:
        """Generate Step 2 prompt (Mathematical Specification)"""
        # Include original problem equations if provided
        equations_section = ""
        if original_problem:
            equations_section = f"\n\n## Original Problem Equations (MUST COPY EXACTLY)\n{original_problem}"
        return STEP2_PROMPT.format(
            step1_output=step1_output,
            data_schema=schema + equations_section
        )

    @staticmethod
    def step3(step2_output: str, data_access: str) -> str:
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


# ==============================================================================
# DATA SCHEMA TEMPLATES (Reference only - actual schemas come from scenario files)
# ==============================================================================

RETAIL_SCHEMA_TEMPLATE = """
{{
  "name": str,                          # scenario identifier
  "periods": int,                       # number of time periods
  "products": [str, ...],               # list of product IDs
  "locations": [str, ...],              # list of location IDs

  "shelf_life": {{p: int}},             # shelf life per product
  "lead_time": {{p: int}},              # order lead time

  "demand_curve": {{p: [float, ...]}},  # demand per product per period (0-indexed)
  "demand_share": {{l: float}},         # demand fraction per location

  "production_cap": {{p: [float, ...]}},# max production per period (0-indexed)
  "cold_capacity": {{l: float}},        # storage capacity per location
  "cold_usage": {{p: float}},           # storage units per unit product

  "costs": {{
    "purchasing": {{p: float}},         # cost per unit ordered
    "inventory": {{p: float}},          # holding cost per unit per period
    "waste": {{p: float}},              # cost per unit expired
    "lost_sales": {{p: float}},         # penalty per unit unmet demand
    "fixed_order": float,               # fixed cost per order
    "transshipment": float              # cost per unit transshipped
  }},

  "constraints": {{
    "moq": float,                       # minimum order quantity
    "pack_size": int,                   # order multiple
    "budget_per_period": float|null,    # max purchasing cost per period
    "waste_limit_pct": float|null       # max waste as fraction of demand
  }},

  "network": {{
    "sub_edges": [[p_from, p_to], ...], # substitution edges
    "trans_edges": [[l_from, l_to], ...]# transshipment edges
  }}
}}

DATA ACCESS:
- demand[p,l,t] = data['demand_curve'][p][t-1] * data['demand_share'][l]
- production_cap[p,t] = data['production_cap'][p][t-1]
- Network: data.get('network', {{}}).get('sub_edges', [])
- IMPORTANT: sub_edges and trans_edges are lists of lists [[a,b], ...].
  Convert to tuples for Gurobi indexing: [tuple(e) for e in edges]
"""

MAMO_SCHEMA_TEMPLATE = """
Standard LP/MILP format with variables, objective, and constraints defined in the problem description.
"""

NL4OPT_SCHEMA_TEMPLATE = """
Natural language problem description. Extract:
- Decision variables from context
- Objective function from "minimize" or "maximize" statements
- Constraints from conditional statements
"""
