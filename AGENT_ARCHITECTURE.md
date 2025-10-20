# Competitive Programming Agent - Architecture

## Overview
A generalized agent that solves algorithm problems using the same methodology as competitive programmers in ACM contests.

## Core Innovation
**Instead of:** Direct code generation (leads to stubs)
**We use:** Multi-step reasoning with explicit phases

## Architecture

### Phase 1: Problem Analysis 🔍
**Goal:** Understand the problem deeply before writing any code

**Prompt asks:**
- What type of problem? (Observer, Graph, DP, etc.)
- What are inputs/outputs for each method?
- What constraints and edge cases exist?
- What patterns can we exploit?
- What data structures are appropriate?

**Output:** Structured analysis with key insights

### Phase 2: Algorithm Design 📐
**Goal:** Plan the solution in detail with pseudo-code

**Prompt asks:**
- Overall approach in plain English
- What attributes does each class need?
- Detailed pseudo-code for each method
- Example walkthrough showing data flow
- Complexity analysis

**Output:** Complete algorithm design with pseudo-code

### Phase 3: Implementation 💻
**Goal:** Translate design into working code

**Prompt receives:**
- Algorithm design from Phase 2
- Code skeleton

**Prompt enforces:**
- Store ALL parameters (no `self.value = None`)
- Implement ALL methods (no `pass` statements)
- Follow the pseudo-code exactly
- Use identified data structures
- Handle edge cases

**Output:** Complete, working main.py

### Phase 4: Testing & Refinement 🧪
**Goal:** Debug and fix issues systematically

**Process:**
1. Run tests with pytest
2. If tests fail, analyze failure
3. Debug with TEST_AND_DEBUG_PROMPT
4. Implement fix
5. Repeat up to 3 iterations

**Output:** Improved implementation

## Key Features

### 1. Generalized Approach
- Works for ANY problem type
- No hard-coded solutions
- Pattern recognition through analysis

### 2. Explicit Reasoning
- Forces LLM to think before coding
- Each phase builds on previous
- Traceable decision-making

### 3. Anti-Stub Enforcement
- Shows negative examples (❌ what NOT to do)
- Shows positive examples (✅ what TO do)
- Repeats requirements at each phase

### 4. Iterative Refinement
- Tests early and often
- Learns from failures
- Systematic debugging

## Example Flow

**Problem:** Reactive spreadsheet cells

```
Phase 1 (Analysis):
→ "This is Observer pattern + dependency graph"
→ "Need to track dependents for each cell"
→ "Changes must propagate in order"

Phase 2 (Design):
→ "InputCell: stores value + list of dependents"
→ "ComputeCell: stores inputs + function + callbacks"
→ "Pseudo-code: when value changes, notify all dependents"

Phase 3 (Implementation):
→ class InputCell:
      def __init__(self, initial_value):
          self.value = initial_value  # ✅ Stored!
          self.dependents = []        # ✅ Initialized!

Phase 4 (Testing):
→ Run tests → Some fail → Analyze → Fix propagation logic → Retest → Pass!
```

## Comparison to Old Agent

### Old Agent (Single-Shot)
```
Problem → LLM → Code (often stubs)
```
- Jumps straight to code
- No reasoning visible
- Often generates incomplete implementations

### New Agent (Multi-Step)
```
Problem → Analysis → Design → Implementation → Testing → Working Code
```
- Thinks through problem first
- Plans before coding
- Generates complete implementations
- Debugs systematically

## Benefits

1. **Higher Success Rate**: Forces complete implementations
2. **Better Code Quality**: Follows algorithmic thinking
3. **Debuggable**: Can see where reasoning failed
4. **Generalizable**: Works for any problem domain
5. **Competitive**: Mimics expert problem-solving

## Usage

```python
# Input
input_dict = {
    "problem_statement": "Implement a reactive spreadsheet..."
}

# Agent processes through 4 phases
patch = agent_main(input_dict)

# Output: Git diff with complete solution
```

## Next Steps

1. ✅ Test on benchmark problems
2. ✅ Measure improvement vs single-shot
3. Tune prompts based on failures
4. Add more sophisticated debugging
5. Optimize for speed/token usage
