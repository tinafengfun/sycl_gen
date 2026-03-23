#!/usr/bin/env python3
"""
Cleanup script to remove placeholder files and update project status
Run this to clean up Phase 1 and prepare for honest reassessment
"""

import os
from pathlib import Path

def analyze_files():
    """Analyze all kernel files and report status"""
    base_path = Path("01_kernels")
    
    print("=== Phase 1 File Analysis ===\n")
    
    complete = []
    placeholders = []
    
    for kernel_dir in base_path.iterdir():
        if kernel_dir.is_dir():
            generated_dir = kernel_dir / "generated"
            if generated_dir.exists():
                for cpp_file in generated_dir.glob("*.cpp"):
                    line_count = len(cpp_file.read_text().splitlines())
                    if line_count > 10:
                        complete.append((cpp_file, line_count))
                    else:
                        placeholders.append((cpp_file, line_count))
    
    print(f"Complete files: {len(complete)}")
    for f, lines in sorted(complete):
        print(f"  ✅ {f}: {lines} lines")
    
    print(f"\nPlaceholder files: {len(placeholders)}")
    for f, lines in sorted(placeholders):
        print(f"  ⚠️  {f}: {lines} lines")
    
    print(f"\nTotal: {len(complete) + len(placeholders)} files")
    print(f"Completion rate: {len(complete)/(len(complete) + len(placeholders))*100:.1f}%")
    
    return complete, placeholders

def cleanup_placeholders():
    """Remove placeholder files (optional)"""
    _, placeholders = analyze_files()
    
    if not placeholders:
        print("\nNo placeholder files to clean up.")
        return
    
    print(f"\n⚠️  Found {len(placeholders)} placeholder files")
    print("Options:")
    print("1. Keep placeholders (mark as TODO)")
    print("2. Delete placeholders (clean state)")
    print("3. Do nothing")
    
    # For now, just mark them
    print("\nMarking placeholder files with TODO comments...")
    for f, _ in placeholders:
        content = f.read_text()
        if not content.startswith("// TODO:"):
            new_content = f"""// TODO: Implement this kernel version
// Status: Placeholder - needs actual implementation
// See phase1_honest_assessment.md for details
// Original content:
{content}
"""
            f.write_text(new_content)
            print(f"  Marked: {f}")

def main():
    print("Phase 1 Cleanup and Assessment Tool\n")
    
    complete, placeholders = analyze_files()
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    if len(complete) >= 12:
        print(f"\n✅ You have {len(complete)} complete kernels")
        print("   This is sufficient for MVP testing")
        print("\nSuggested next steps:")
        print("1. Test the complete kernels (add_vectors, batch_norm)")
        print("2. Generate performance report")
        print("3. Document what's needed for remaining kernels")
        print("\nOption: Proceed to Phase 2 with 12 kernels")
    
    if len(placeholders) > 0:
        print(f"\n⚠️  {len(placeholders)} placeholder files detected")
        print("   These represent incomplete work")
        print("\nOption: Complete remaining kernels first")
    
    print("\n" + "="*60)
    
    # Ask for action
    print("\nActions:")
    print("1. Mark placeholders with TODO")
    print("2. Exit without changes")
    
    # Auto-mark for now
    cleanup_placeholders()
    
    print("\n✅ Assessment complete!")
    print("See: 04_results/reports/phase1_honest_assessment.md")

if __name__ == "__main__":
    main()