# Phase 2: UnifiedReporter Implementation
# Phase 2: UnifiedReporter Agent Implementation

## Real-World Insights from LCZero Optimization Project

Based on optimization of 28 LCZero kernels on Intel BMG B60 GPU:

### Key Metrics to Report

**Performance Metrics:**
- Baseline GFLOPS (before optimization)
- Optimized GFLOPS (after each round)
- Improvement percentage (show both positive and negative)
- Memory bandwidth utilization
- Occupancy percentage

**Example:**
```
Kernel: batch_norm
Baseline: 70 GFLOPS
Round 1 (FP16): 85 GFLOPS (+21%)
Round 2 (Memory): 92 GFLOPS (+31%)
Round 3 (SLM): 109 GFLOPS (+56%) ← Final
Peak memory bandwidth: 45%
Occupancy: 75%
```

### Critical Findings to Report

**1. Most Kernels Already Optimal**
- 26/28 kernels (93%) showed <10% improvement potential
- Don't report "optimization applied" when gain is negligible
- Focus on WHY optimization didn't help

**2. Memory Coalescing Issues**
- Report when access patterns break coalescing
- Show before/after memory transaction counts
- Example: "Fixed strided access, reduced transactions from 16 to 1 per warp"

**3. SLM Caching Results**
- Only report when parameters are small (<100KB) and reused
- Show SLM usage: "Used 20KB SLM for 5 parameters × 1024 channels"
- Report bank conflict check: "No conflicts (padded 16×17 array)"

**4. Grid Configuration Impact**
- Report grid dimension changes:
  - Before: 2D grid (batch, channels) → 33 GFLOPS
  - After: 1D flattened grid → 1094 GFLOPS
- This is often more important than algorithm changes

### Report Structure (Based on 5-Round Process)

```
## Executive Summary
- Total kernels: 28
- Improved: 2/28 (7%)
- Already optimal: 26/28 (93%)
- Best improvement: +179% (softmax)
- Worst regression: -10% (expand_planes)

## Per-Kernel Results

### batch_norm [IMPROVED +56%]
**Baseline:** 70 GFLOPS
**Optimized:** 109 GFLOPS
**Technique:** SLM parameter caching
**Why it worked:** Small parameters (20KB) reused per element
**Code change:** Cached 5 parameters to SLM, barrier per work-group

### softmax [IMPROVED +179%]
**Baseline:** 26 GFLOPS  
**Optimized:** 73 GFLOPS
**Technique:** Cooperative reduction with SLM
**Why it worked:** Three-phase reduction using sub-group shuffle
**Code change:** Max→exp→normalize with SLM for max/sum

### add_vectors [ALREADY OPTIMAL]
**Baseline:** 83 GFLOPS
**Optimization attempts:** Rounds 1-3
**Result:** No improvement (<5% variance)
**Conclusion:** Memory bandwidth bound, already coalesced

### expand_planes [REGRESSION -10%]
**Baseline:** 55 GFLOPS
**Optimized:** 50 GFLOPS
**Technique:** "Improved" indexing
**Why it failed:** Broke memory coalescing
**Lesson:** Don't fix what isn't broken

## Optimization Techniques Applied

### Successful Patterns
1. **SLM Parameter Caching** (batch_norm, softmax)
   - Gain: +56% to +179%
   - Requirement: Small parameters (<100KB), high reuse
   
2. **1D Grid Flattening** (layer_norm)
   - Gain: 33→1094 GFLOPS (33×!)
   - Requirement: 2D/3D kernels with poor occupancy

3. **Cooperative Reduction** (softmax)
   - Gain: +179%
   - Technique: Sub-group shuffle + SLM for cross-sub-group

### Failed Patterns
1. **Unnecessary SLM** (element-wise kernels)
   - Result: 20-50% slower
   - Why: Adds barriers to memory-bound kernels

2. **Strided Access "Optimization"** (expand_planes)
   - Result: -10%
   - Why: Broke coalesced memory access

3. **Wrong Work-Group Size** (layer_norm early attempt)
   - Result: 1094→33 GFLOPS
   - Why: 2D grid limited parallelism

## Key Takeaways

1. **Profile before optimizing** - 93% of kernels already optimal
2. **Memory coalescing is sacred** - Never break existing patterns
3. **Grid config > algorithm** - 33× improvement from flattening
4. **SLM only for parameters** - 256KB is large, use it wisely
5. **Test incrementally** - Each optimization in isolation

## Recommendations for Future Projects

- Budget 70% time for profiling, 30% for optimization
- Focus on kernels with <50% memory bandwidth utilization
- Skip element-wise kernels unless profiling shows issues
- Always compare against baseline, not theoretical peak

## Task Description

Create a UnifiedReporter Agent that generates comprehensive reports in multiple formats (JSON, HTML, Markdown).

## Requirements

### 1. Create UnifiedReporter Class

```python
class UnifiedReporter:
    def __init__(self, tracer: UnifiedTracer):
        self.tracer = tracer
        self.base_dir = Path(__file__).parent.parent
    
    async def generate_reports(self, session_data: dict) -> dict:
        '''Generate all report formats'''
        pass
    
    def generate_json_report(self, data: dict) -> str:
        '''Generate JSON format report'''
        pass
    
    def generate_html_report(self, data: dict) -> str:
        '''Generate HTML format report with styling'''
        pass
    
    def generate_markdown_report(self, data: dict) -> str:
        '''Generate Markdown format report'''
        pass
```

### 2. Report Content

All reports must include:

#### Header Section
- Session ID
- Kernel ID
- Timestamp
- Overall Status

#### Phase Summary
- Phase 1: Analysis (duration, results)
- Phase 2: Conversion (duration, output file)
- Phase 3: Validation (compilation status, fixes applied)
- Phase 4: Accuracy (pass rate, error metrics)
- Phase 5: Reporting (this phase)

#### Performance Metrics
- Total duration
- Phase durations breakdown
- Compilation time
- Accuracy test time

#### Trace Metrics
- Total steps
- Tool calls count
- Errors encountered
- Fixes applied

### 3. JSON Report Format

```json
{
  "session_id": "winograd_20260304_120000",
  "kernel_id": "winograd_input_transform",
  "timestamp": "2026-03-04T12:05:00",
  "overall_status": "success",
  "phases": {
    "analysis": {
      "status": "completed",
      "duration_seconds": 45.2,
      "lines_analyzed": 217,
      "complexity_level": 3
    },
    "conversion": {
      "status": "completed",
      "duration_seconds": 120.5,
      "output_file": "kernel_dataset/sycl/..."
    },
    "validation": {
      "status": "completed",
      "duration_seconds": 16.5,
      "compilation_success": true,
      "fixes_applied": 2
    },
    "accuracy": {
      "status": "completed",
      "duration_seconds": 180.3,
      "pass_rate": 1.0,
      "max_abs_diff": 0.0,
      "max_rel_diff": 0.0
    }
  },
  "performance": {
    "total_duration_seconds": 362.5,
    "compilation_time_seconds": 16.5,
    "accuracy_test_time_seconds": 180.3
  },
  "trace_summary": {
    "total_steps": 156,
    "total_tool_calls": 23,
    "errors_encountered": 2,
    "fixes_applied": 2
  }
}
```

### 4. HTML Report Format

Create a clean, professional HTML report with:
- Header with status badge (green for success, red for fail)
- Table of phases with status indicators
- Performance charts (optional, can be simple tables)
- Trace metrics cards
- Download links for other formats

Use inline CSS for simplicity. Avoid external dependencies.

### 5. Markdown Report Format

Create a human-readable Markdown report:
- Executive summary at top
- Section for each phase
- Code blocks for metrics
- Tables for comparison
- Suitable for GitHub/GitLab

### 6. Integration with Phase 5

Replace the existing simple reporting in run_phase5_reporting:

```python
async def run_phase5_reporting(self, analysis, build, accuracy):
    '''Phase 5: Reporting'''
    self.state["current_phase"] = Phase.REPORTING
    
    # Collect all data
    session_data = {
        "session_id": self.session_id,
        "kernel_id": self.kernel_id,
        "timestamp": datetime.now().isoformat(),
        "phases_completed": [p.value for p in self.state["phases_completed"]],
        "fixes_applied": self.state["fixes_applied"],
        "analysis": analysis,
        "compilation": build,
        "accuracy": accuracy,
        "trace_metrics": self.tracer.metrics
    }
    
    # Generate reports
    reports = await self.reporter.generate_reports(session_data)
    
    print(f"   ✅ Reports generated:")
    print(f"      JSON: {reports['json']}")
    print(f"      HTML: {reports['html']}")
    print(f"      Markdown: {reports['markdown']}")
    
    self.state["phases_completed"].append(Phase.REPORTING)
    return reports
```

### 7. File Output Structure

```
results/
└── {session_id}/
    ├── final_report.json
    ├── final_report.html
    └── final_report.md
```

### 8. HTML Template Requirements

Use this structure:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Conversion Report - {kernel_id}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 8px; }
        .status-success { color: green; }
        .status-fail { color: red; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .metric-card { display: inline-block; margin: 10px; padding: 15px; 
                       background: #e3f2fd; border-radius: 8px; }
    </style>
</head>
<body>
    <!-- Content here -->
</body>
</html>
```

### 9. Error Handling

All methods must handle errors gracefully:

```python
def generate_html_report(self, data: dict) -> str:
    try:
        # Generate report
        html = self._build_html(data)
        return html
    except Exception as e:
        self.tracer.log("UnifiedReporter", "html_error", {"error": str(e)})
        # Return simple error message as HTML
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"
```

### 10. Testing Requirements

Create test_phase2.py that tests:
- JSON report generation
- HTML report generation
- Markdown report generation
- Error handling
- File output

## Code Standards

### String Handling
- Use """ for docstrings
- Use format() for complex strings
- Extract variables before formatting

### Example
```python
# Good
header = """
# Conversion Report

**Kernel**: {kernel_id}
**Status**: {status}
""".format(kernel_id=data['kernel_id'], status=data['status'])

# Bad (nested f-strings)
header = f"**Kernel**: {data['kernel_id']}"
```

### Encoding
- All comments in English
- No special Unicode characters
- UTF-8 encoding

## Deliverables

1. UnifiedReporter class in unified_converter.py
2. Updated run_phase5_reporting() method
3. test_phase2.py with unit tests
4. Example reports in results/test_session/

## Success Criteria

- [ ] JSON report is valid and complete
- [ ] HTML report displays correctly in browser
- [ ] Markdown report renders properly on GitHub
- [ ] All three formats include required sections
- [ ] Error handling works correctly
- [ ] Unit tests pass (100%)
- [ ] No syntax errors
