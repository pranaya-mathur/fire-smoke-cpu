#!/usr/bin/env python3
import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fire_smoke_cpu.hf_auth import get_hf_token, masked_token_status

def verify_auth():
    print("Verifying Hugging Face Authentication...")
    status = masked_token_status()
    
    report = {
        "token_detected": status["exists"],
        "token_preview": status["preview"],
        "auth_status": "FAIL",
        "authenticated_user": None,
        "reason": None
    }
    
    print(f"HF_TOKEN detected: {'yes' if status['exists'] else 'no'}")
    if status["exists"]:
        print(f"Token preview: {status['preview']}")
    
    if not status["exists"]:
        print("Authentication: FAIL")
        print("Reason: HF_TOKEN environment variable is not set")
        report["reason"] = "HF_TOKEN environment variable is not set"
        write_report(report)
        sys.exit(1)
        
    token = get_hf_token()
    
    try:
        cmd = ["curl", "-s", "-H", f"Authorization: Bearer {token}", "https://huggingface.co/api/whoami-v2"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            raise Exception("Curl command failed")
            
        data = json.loads(result.stdout)
        
        if "error" in data:
            reason = "Invalid or expired token"
            print("Authentication: FAIL")
            print(f"Reason: {reason}")
            report["reason"] = reason
            write_report(report)
            sys.exit(1)
            
        user = data.get("name", "unknown")
        print("Authentication: PASS")
        print(f"Authenticated user: {user}")
        report["auth_status"] = "PASS"
        report["authenticated_user"] = user
        write_report(report)
        sys.exit(0)
        
    except Exception as e:
        reason = f"Network or unexpected error: {str(e)}"
        print("Authentication: FAIL")
        print(f"Reason: {reason}")
        report["reason"] = reason
        write_report(report)
        sys.exit(1)

def write_report(report):
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = report_dir / "hf_auth_status.json"
    md_path = report_dir / "hf_auth_status.md"
    
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
        
    md = [
        "# Hugging Face Authentication Status\n",
        f"- **Token Detected:** {'Yes' if report['token_detected'] else 'No'}",
        f"- **Token Preview:** {report['token_preview'] or 'N/A'}",
        f"- **Status:** {report['auth_status']}"
    ]
    if report["authenticated_user"]:
        md.append(f"- **Authenticated User:** {report['authenticated_user']}")
    if report["reason"]:
        md.append(f"- **Reason:** {report['reason']}")
        
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

if __name__ == "__main__":
    verify_auth()
