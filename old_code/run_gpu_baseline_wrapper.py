"""
Wrapper script to run GPU baseline test and capture output
"""
import subprocess
import sys
from pathlib import Path

# Set working directory
project_dir = Path(__file__).parent
output_file = project_dir / "gpu_baseline_output.txt"

# Run the baseline test
venv_python = project_dir / ".venv" / "Scripts" / "python.exe"
test_script = project_dir / "run_baseline_test.py"

print(f"Running GPU baseline test...")
print(f"Python: {venv_python}")
print(f"Script: {test_script}")
print(f"Output will be saved to: {output_file}")
print("-" * 80)

try:
    with open(output_file, 'w', encoding='utf-8') as f:
        process = subprocess.run(
            [str(venv_python), str(test_script)],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # Write both stdout and stderr
        f.write("=== STDOUT ===\n")
        f.write(process.stdout)
        f.write("\n\n=== STDERR ===\n")
        f.write(process.stderr)
        f.write(f"\n\n=== EXIT CODE: {process.returncode} ===\n")

        # Also print to console
        print(process.stdout)
        if process.stderr:
            print("STDERR:", file=sys.stderr)
            print(process.stderr, file=sys.stderr)

        print(f"\n\nExit code: {process.returncode}")
        print(f"Full output saved to: {output_file}")

except Exception as e:
    print(f"ERROR: {e}")
    with open(output_file, 'w') as f:
        f.write(f"ERROR: {e}\n")
    sys.exit(1)
