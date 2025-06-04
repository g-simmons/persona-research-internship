from pathlib import Path
import subprocess
import re
from typing import Generator, Tuple

SCRIPT_DIR = Path(__file__).parent
CODE_DIR = SCRIPT_DIR / "../llm_ontology"

def get_git_blame(file: Path, lineno: int) -> str:
    result = subprocess.run(["git", "blame", "-w", "-L", f"{lineno},{lineno}", file], capture_output=True, text=True)
    return result.stdout

def _check_for_regex(file: Path, regex: str) -> Generator[Tuple[int, str], None, None]:
    with open(file, "r") as f:
        lines = f.readlines()
    for lineno, line in enumerate(lines):
        if re.search(regex, line):
            yield lineno, line

def _show_instances(file: Path, regex: str, template: str) -> None:
    for lineno, line in _check_for_regex(file, regex):
        print(template.format(lineno=lineno, file=file, line=line))
        print("\t" + line)
        print("\t" + get_git_blame(file, lineno))

def lint_file(file: Path):
    _show_instances(file, r"print", "Print statement found in {file} on line {lineno}")
    _show_instances(file, r"import code", "Debugger statement found in {file} on line {lineno}")
    _show_instances(file, r"plt.", "Matplotlib plt. statement found in {file} on line {lineno}")

def lint_code(code_dir: Path = CODE_DIR):
    for file in code_dir.glob("*.py"):
        lint_file(file)

if __name__ == "__main__":
    lint_code()