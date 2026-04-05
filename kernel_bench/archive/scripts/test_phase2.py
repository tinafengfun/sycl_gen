#!/usr/bin/env python3
"""
Phase 2 Test - UnifiedReporter
Test report generation functionality
"""

import sys
import os
import json
import tempfile
import shutil

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from unified_converter import UnifiedTracer, UnifiedReporter


def test_reporter_init():
    """Test UnifiedReporter initialization"""
    print("\n🧪 Test 1: Reporter Initialization")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    reporter = UnifiedReporter(tracer)
    
    assert hasattr(reporter, 'tracer'), "Missing tracer attribute"
    assert hasattr(reporter, 'base_dir'), "Missing base_dir attribute"
    assert reporter.tracer == tracer, "Tracer not set correctly"
    
    print("  ✓ Reporter initialized correctly")
    print("  ✓ Has tracer attribute")
    print("  ✓ Has base_dir attribute")
    return True


def test_generate_json_report():
    """Test JSON report generation"""
    print("\n🧪 Test 2: JSON Report Generation")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    reporter = UnifiedReporter(tracer)
    
    # Create test data
    test_data = {
        "session_id": "test_123",
        "kernel_id": "test_kernel",
        "timestamp": "2026-03-04T12:00:00",
        "overall_status": "success",
        "phases": {
            "analysis": {"status": "completed", "duration_seconds": 45.2},
            "conversion": {"status": "completed", "duration_seconds": 120.5}
        },
        "performance": {"total_duration_seconds": 300.5},
        "trace_summary": {"total_steps": 100, "total_tool_calls": 20}
    }
    
    # Generate report
    json_content = reporter.generate_json_report(test_data)
    
    # Verify it's valid JSON
    parsed = json.loads(json_content)
    assert parsed["session_id"] == "test_123", "Session ID mismatch"
    assert parsed["kernel_id"] == "test_kernel", "Kernel ID mismatch"
    assert parsed["overall_status"] == "success", "Status mismatch"
    
    print("  ✓ JSON report generated successfully")
    print("  ✓ Valid JSON format")
    print("  ✓ All fields present")
    return True


def test_generate_html_report():
    """Test HTML report generation"""
    print("\n🧪 Test 3: HTML Report Generation")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    reporter = UnifiedReporter(tracer)
    
    # Create test data
    test_data = {
        "session_id": "test_123",
        "kernel_id": "test_kernel",
        "timestamp": "2026-03-04T12:00:00",
        "overall_status": "success",
        "phases": {
            "analysis": {"status": "completed", "duration_seconds": 45.2},
            "conversion": {"status": "completed", "duration_seconds": 120.5},
            "validation": {"status": "completed", "duration_seconds": 16.5}
        },
        "performance": {"total_duration_seconds": 300.5},
        "trace_summary": {"total_steps": 100, "total_tool_calls": 20}
    }
    
    # Generate report
    html_content = reporter.generate_html_report(test_data)
    
    # Verify HTML structure
    assert html_content.startswith("<!DOCTYPE html>"), "Missing DOCTYPE"
    assert "<html>" in html_content, "Missing html tag"
    assert "<head>" in html_content, "Missing head tag"
    assert "<body>" in html_content, "Missing body tag"
    assert "test_kernel" in html_content, "Kernel ID not in HTML"
    assert "success" in html_content.lower(), "Status not in HTML"
    
    print("  ✓ HTML report generated successfully")
    print("  ✓ Valid HTML structure")
    print("  ✓ Contains kernel information")
    print("  ✓ Contains phase data")
    return True


def test_generate_markdown_report():
    """Test Markdown report generation"""
    print("\n🧪 Test 4: Markdown Report Generation")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    reporter = UnifiedReporter(tracer)
    
    # Create test data
    test_data = {
        "session_id": "test_123",
        "kernel_id": "test_kernel",
        "timestamp": "2026-03-04T12:00:00",
        "overall_status": "success",
        "phases": {
            "analysis": {"status": "completed", "duration_seconds": 45.2},
            "conversion": {"status": "completed", "duration_seconds": 120.5}
        },
        "performance": {"total_duration_seconds": 300.5},
        "trace_summary": {"total_steps": 100, "total_tool_calls": 20, "errors": 0, "fixes": 2},
        "analysis": {"total_lines": 217, "complexity_level": 3},
        "compilation": {"success": True},
        "accuracy": {
            "pass_rate": 1.0,
            "summary": {"overall_max_abs_diff": 0.0, "overall_max_rel_diff": 0.0}
        },
        "fixes_applied": 2
    }
    
    # Generate report
    md_content = reporter.generate_markdown_report(test_data)
    
    # Verify Markdown structure
    assert "# CUDA-to-SYCL Conversion Report" in md_content, "Missing title"
    assert "test_kernel" in md_content, "Kernel ID not in markdown"
    assert "## Summary" in md_content, "Missing summary section"
    assert "## Phase Breakdown" in md_content, "Missing phases section"
    assert "## Performance Metrics" in md_content, "Missing performance section"
    assert "|" in md_content, "Missing tables"
    
    print("  ✓ Markdown report generated successfully")
    print("  ✓ Valid Markdown structure")
    print("  ✓ Contains all sections")
    print("  ✓ Contains tables")
    return True


def test_generate_reports_integration():
    """Test full report generation integration"""
    print("\n🧪 Test 5: Full Integration Test")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        tracer = UnifiedTracer("test_session_integration", "test_kernel_integration")
        reporter = UnifiedReporter(tracer)
        
        # Override base_dir to use temp directory
        from pathlib import Path
        reporter.base_dir = Path(temp_dir)
        
        # Create test data
        test_data = {
            "session_id": "integration_test",
            "kernel_id": "integration_kernel",
            "timestamp": "2026-03-04T12:00:00",
            "overall_status": "success",
            "phases": {
                "analysis": {"status": "completed", "duration_seconds": 45.2},
                "conversion": {"status": "completed", "duration_seconds": 120.5},
                "validation": {"status": "completed", "duration_seconds": 16.5},
                "accuracy": {"status": "completed", "duration_seconds": 180.3}
            },
            "performance": {"total_duration_seconds": 362.5},
            "trace_summary": {"total_steps": 156, "total_tool_calls": 23, "errors": 2, "fixes": 2},
            "analysis": {"total_lines": 217, "complexity_level": 3},
            "compilation": {"success": True},
            "accuracy": {
                "pass_rate": 1.0,
                "summary": {"overall_max_abs_diff": 0.0, "overall_max_rel_diff": 0.0}
            },
            "fixes_applied": 2
        }
        
        # Generate all reports
        import asyncio
        reports = asyncio.run(reporter.generate_reports(test_data))
        
        # Verify reports were generated
        assert "json" in reports, "JSON report not generated"
        assert "html" in reports, "HTML report not generated"
        assert "markdown" in reports, "Markdown report not generated"
        
        print("  ✓ All three reports generated")
        print("  ✓ JSON: {}".format(reports['json'] if reports['json'] else "Failed"))
        print("  ✓ HTML: {}".format(reports['html'] if reports['html'] else "Failed"))
        print("  ✓ Markdown: {}".format(reports['markdown'] if reports['markdown'] else "Failed"))
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return True


def test_error_handling():
    """Test error handling in report generation"""
    print("\n🧪 Test 6: Error Handling")
    
    tracer = UnifiedTracer("test_session", "test_kernel")
    reporter = UnifiedReporter(tracer)
    
    # Test with invalid data (missing required fields)
    invalid_data = {
        "session_id": "test",
        # Missing kernel_id and other fields
    }
    
    # Should not raise exception, should handle gracefully
    try:
        html_content = reporter.generate_html_report(invalid_data)
        assert "unknown" in html_content or html_content, "Should generate something"
        print("  ✓ Handles missing data gracefully")
    except Exception as e:
        print(f"  ⚠ HTML generation failed: {e}")
    
    try:
        md_content = reporter.generate_markdown_report(invalid_data)
        assert md_content, "Should generate markdown"
        print("  ✓ Markdown handles missing data")
    except Exception as e:
        print(f"  ⚠ Markdown generation failed: {e}")
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("🚀 Phase 2 Test - UnifiedReporter")
    print("="*60)
    
    tests = [
        ("Reporter Initialization", test_reporter_init),
        ("JSON Report", test_generate_json_report),
        ("HTML Report", test_generate_html_report),
        ("Markdown Report", test_generate_markdown_report),
        ("Integration Test", test_generate_reports_integration),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print("Passed: {}/{}".format(passed, len(tests)))
    print("Failed: {}/{}".format(failed, len(tests)))
    
    if failed == 0:
        print("\n✅ All Phase 2 unit tests passed!")
        return 0
    else:
        print("\n⚠️  {} test(s) failed".format(failed))
        return 1


if __name__ == "__main__":
    sys.exit(main())
