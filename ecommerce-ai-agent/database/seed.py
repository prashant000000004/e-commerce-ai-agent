"""
database/seed.py
Convenience wrapper to seed the database.
Calls scripts/generate_sample_data.py logic directly.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    script = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "generate_sample_data.py")
    subprocess.run([sys.executable, script], check=True)
