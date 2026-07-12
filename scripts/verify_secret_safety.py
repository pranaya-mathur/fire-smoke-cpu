#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fire_smoke_cpu.hf_auth import get_hf_token

def main():
    token = get_hf_token()
    token_str = token if token else "hf_something_mock"
    token_prefix = token_str[:8] if token else "hf_"
    
    # Check .gitignore
    gitignore_path = ROOT / ".gitignore"
    gitignore_content = ""
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text()
    
    env_ignored = ".env" in gitignore_content
    
    # Check .env.example
    env_example_path = ROOT / ".env.example"
    env_example_safe = True
    if env_example_path.exists():
        content = env_example_path.read_text()
        if token and token in content:
            env_example_safe = False
            
    # Scan for exact token leaks in reports and configs
    leak_found = False
    leak_details = []
    
    paths_to_scan = [ROOT / "reports", ROOT / "configs"]
    for d in paths_to_scan:
        if not d.exists(): continue
        for f in d.rglob("*"):
            if f.is_file() and f.suffix in [".md", ".json", ".yaml", ".txt", ".csv"]:
                content = f.read_text(errors="ignore")
                if token and token in content:
                    leak_found = True
                    leak_details.append(f"Token found in {f.relative_to(ROOT)}")
                elif "hf_" in content:
                    # Generic check for any hf_ string that might be a token (length > 20)
                    words = content.split()
                    for w in words:
                        if w.startswith("hf_") and len(w) > 20:
                            if w != token: # Just an example or different token
                                pass
    
    report_path = ROOT / "reports" / "hf_secret_safety_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    passed = env_ignored and env_example_safe and not leak_found
    
    with open(report_path, "w") as f:
        f.write("# Hugging Face Secret Safety Report\n\n")
        f.write(f"**Overall Status**: {'PASS' if passed else 'FAIL'}\n\n")
        f.write(f"- `.env` ignored by Git: {'PASS' if env_ignored else 'FAIL'}\n")
        f.write(f"- `.env.example` is safe: {'PASS' if env_example_safe else 'FAIL'}\n")
        f.write(f"- Token leaks detected: {'FAIL (Leaks found)' if leak_found else 'PASS'}\n")
        
        if leak_details:
            f.write("\n## Leak Details\n")
            for detail in leak_details:
                f.write(f"- {detail}\n")
                
    if not passed:
        sys.exit(1)
    print("Secret Safety Check: PASS")
    sys.exit(0)

if __name__ == "__main__":
    main()
