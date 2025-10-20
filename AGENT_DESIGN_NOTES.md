# Agent Design Notes

## Our Approach vs Reference Implementation

### Reference Agent (Complex Multi-Agent Framework)
**Size**: 3000+ lines  
**Approach**: Tool-based with COT (Chain of Thought)  
**Strengths**:
- Robust error handling with detailed error types
- Multiple model routing (GLM, Kimi, DeepSeek, Qwen)
- Extensive retry logic
- FIX vs CREATE task discrimination
- Rich logging and debugging

**Weaknesses**:
- Over-engineered for simple problems
- Hard to understand and maintain
- Many moving parts (tools, COT, state management)
- Requires deep infrastructure

---

### Our Agent (Focused Competitive Programming)
**Size**: ~400 lines  
**Approach**: 4-phase structured reasoning  
**Philosophy**: Simple, clear, purpose-built

```
Phase 1: ANALYSIS     → Understand problem deeply
Phase 2: DESIGN       → Plan algorithm & data structures
Phase 3: IMPLEMENT    → Generate complete code (NO STUBS!)
Phase 4: SELF-TEST    → Run tests & debug (3 iterations)
```

**What We Learned (But Didn't Copy)**:
1. ✅ **Better git initialization** - Proper repo setup with commits
2. ✅ **Retry logic** - Handle LLM failures gracefully
3. ✅ **Standard library enforcement** - Validate imports
4. ✅ **Structured logging** - Clear progress indicators

**What We Kept Different**:
1. ❌ **No tool framework** - Direct function calls are clearer
2. ❌ **Single model** - Qwen is sufficient, simpler
3. ❌ **No COT tracking** - 4 phases are enough
4. ❌ **No task types** - Focus on competitive programming only

---

## Key Improvements Made

### 1. Robust Git Initialization
```python
def init_git(repo_dir: str):
    """Initialize git repository with proper config and initial commit."""
    # Proper setup with user config
    # Initial commit for clean diffs
    # Error handling
```

### 2. LLM Call Retry Logic
```python
def call_llm(messages: list, max_retries: int = 3) -> str:
    """Call LLM via sandbox proxy with retry logic."""
    # 3 attempts with exponential backoff
    # Timeout handling
    # Response validation
```

### 3. Import Validation
```python
def validate_imports(code: str) -> tuple[bool, list[str]]:
    """Check if code only uses standard library imports."""
    # Forbidden: numpy, pandas, requests, etc.
    # Allowed: collections, itertools, math, etc.
```

### 4. Better Code Extraction
```python
def write_solution(repo_dir: str, code: str):
    """Write solution to main.py - extracts Python code from LLM response."""
    # Regex-based extraction of ```python blocks
    # Fallback to keyword detection
    # Syntax validation before writing
```

---

## Results

### Comparison Table

| Feature | Reference | Our Agent |
|---------|-----------|-----------|
| Lines of Code | 3000+ | ~400 |
| Models Used | 4 (GLM, Kimi, DeepSeek, Qwen) | 1 (Qwen) |
| Approach | Tool-based COT | 4-phase reasoning |
| Task Types | FIX + CREATE | Competitive programming |
| Retry Logic | ✅ Complex | ✅ Simple |
| Git Setup | ✅ Advanced | ✅ Basic |
| Import Validation | ❌ None | ✅ Yes |
| Self-Testing | ✅ Via tools | ✅ Built-in |
| Maintainability | 😰 Hard | 😊 Easy |
| Debuggability | 😰 Complex | 😊 Clear |

---

## Testing Strategy

### Our Self-Testing Loop
```python
for iteration in range(3):
    test_output, passed = run_tests(repo_dir)
    
    if passed:
        ✅ Done!
        break
    
    if iteration < 2:
        🔧 Debug with LLM
        💾 Write fixed code
    else:
        ⚠️ Return best attempt
```

### Why This Works
1. **Simple**: One clear loop
2. **Fast**: Max 3 iterations
3. **Effective**: Most bugs caught in first debug
4. **Predictable**: Always returns something

---

## Conclusion

**What Makes Our Agent Better for This Use Case:**
- ✅ **Focused**: Built for competitive programming specifically
- ✅ **Simple**: Easy to understand and modify
- ✅ **Effective**: 4-phase approach ensures quality
- ✅ **Robust**: Has key reliability features (retry, validation)
- ✅ **Maintainable**: ~400 lines anyone can read

**When You'd Want the Reference Approach:**
- Complex SWE tasks requiring multiple tools
- Need to FIX existing codebases (not just CREATE)
- Want multi-model routing for different problem types
- Have infrastructure to support complex tool frameworks

**Our Choice: Keep It Simple, Keep It Focused** 🎯
