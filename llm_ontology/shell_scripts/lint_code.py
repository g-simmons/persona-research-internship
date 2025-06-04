from pathlib import Path
import subprocess
import re
from typing import Generator, Tuple, Literal, Optional
import os
import json
import logging

# Set up logger with file and stdout handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
script_dir = Path(__file__).parent
log_file = script_dir / 'lint_code.log'
file_handler = logging.FileHandler(log_file)
stdout_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

SCRIPT_DIR = Path(__file__).parent
CODE_DIR = SCRIPT_DIR / "../llm_ontology"

def get_git_blame(file: Path, lineno: int) -> Tuple[str, str, str]:
    """
    Get git blame information for a specific line using porcelain format.
    
    Args:
        file: Path to the file
        lineno: Line number (1-indexed)
        
    Returns:
        Tuple of (author, date, commit)
        
    Raises:
        subprocess.CalledProcessError: If git blame command fails
        ValueError: If the output format is unexpected
    """
    try:
        # Use porcelain format for more structured output
        result = subprocess.run([
            "git", "blame", 
            "--porcelain",  # Machine-readable format
            "-L", f"{lineno},{lineno}", 
            str(file)
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        if not lines:
            raise ValueError(f"No git blame output for {file}:{lineno}")
        
        # Parse porcelain format
        commit = ""
        author = ""
        date = ""
        
        for line in lines:
            if line.startswith('author '):
                author = line[7:]  # Remove 'author ' prefix
            elif line.startswith('author-time '):
                # Convert Unix timestamp to readable format
                timestamp = int(line[12:])
                import datetime
                date = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            elif not line.startswith(('\t', 'author-', 'committer-', 'summary ', 'boundary', 'filename ')):
                # First line contains commit hash
                if not commit:
                    commit = line.split()[0]
        
        if not all([author, date, commit]):
            logger.warning(f"Incomplete git blame data for {file}:{lineno}")
            
        return author, date, commit
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git blame failed for {file}:{lineno}: {e.stderr}")
        return "Unknown", "Unknown", "Unknown"
    except Exception as e:
        logger.error(f"Error parsing git blame for {file}:{lineno}: {e}")
        return "Unknown", "Unknown", "Unknown"

def get_git_blame_json(file: Path, lineno: int) -> Optional[dict]:
    """
    Alternative implementation that returns structured data as a dictionary.
    
    Args:
        file: Path to the file
        lineno: Line number (1-indexed)
        
    Returns:
        Dictionary with blame information or None if failed
    """
    try:
        result = subprocess.run([
            "git", "blame", 
            "--porcelain",
            "-L", f"{lineno},{lineno}", 
            str(file)
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        blame_info = {}
        
        for line in lines:
            if line.startswith('author '):
                blame_info['author'] = line[7:]
            elif line.startswith('author-mail '):
                blame_info['author_email'] = line[12:].strip('<>')
            elif line.startswith('author-time '):
                blame_info['author_timestamp'] = int(line[12:])
                import datetime
                blame_info['author_date'] = datetime.datetime.fromtimestamp(
                    int(line[12:])
                ).strftime('%Y-%m-%d %H:%M:%S')
            elif line.startswith('committer '):
                blame_info['committer'] = line[10:]
            elif line.startswith('committer-time '):
                blame_info['committer_timestamp'] = int(line[15:])
            elif line.startswith('summary '):
                blame_info['summary'] = line[8:]
            elif not line.startswith(('\t', 'author-', 'committer-', 'boundary', 'filename ')):
                if 'commit' not in blame_info:
                    parts = line.split()
                    if parts:
                        blame_info['commit'] = parts[0]
                        blame_info['source_line'] = int(parts[1]) if len(parts) > 1 else lineno
                        blame_info['result_line'] = int(parts[2]) if len(parts) > 2 else lineno
        
        return blame_info if blame_info else None
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git blame failed for {file}:{lineno}: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Error parsing git blame for {file}:{lineno}: {e}")
        return None

def _check_for_regex(file: Path, regex: str) -> Generator[Tuple[int, str], None, None]:
    with open(file, "r") as f:
        lines = f.readlines()
    for lineno, line in enumerate(lines):
        if re.search(regex, line):
            yield lineno, line

def _show_instances(file: Path, regex: str, template: str, format: Literal["text", "json"] = "text") -> None:
    for lineno, line in _check_for_regex(file, regex):
        if format == "text":
            print(template.format(lineno=lineno, file=file, line=line))
            print("\t" + line)
            blame_info = get_git_blame_json(file, lineno + 1)  # git uses 1-indexed lines
            if blame_info:
                print(f"\t{blame_info['author']} {blame_info['author_date']} {blame_info['commit']}")
            else:
                print("\tUnable to get git blame information")
        elif format == "json":
            blame_info = get_git_blame_json(file, lineno + 1)  # git uses 1-indexed lines
            output = {
                "lineno": lineno + 1,  # Convert to 1-indexed for consistency
                "file": str(file),
                "line": line.strip(),
            }
            if blame_info:
                output.update(blame_info)
            print(json.dumps(output))

def lint_file(file: Path, format: Literal["text", "json"] = "text"):
    _show_instances(file, r"print", "Print statement found in {file} on line {lineno}", format)
    _show_instances(file, r"import code", "Debugger statement found in {file} on line {lineno}", format)
    _show_instances(file, r"plt.", "Matplotlib plt. statement found in {file} on line {lineno}", format)
    # check for hardcoded paths using a regex that matches Unix or Windows-style paths
    hardcoded_path_regex = r"(?:[\"'])(?:[A-Za-z]:)?(?:/|\\)[^\"']+[\"']"
    _show_instances(file, hardcoded_path_regex, "Hardcoded path found in {file} on line {lineno}", format)

def lint_code(code_dir: Path = CODE_DIR):
    for file in code_dir.glob("*.py"):
        lint_file(file, format="json")
        # os.system(f"mypy --disallow-untyped-defs {file}")

def main() -> None:
    """Main function to run the linting process."""
    lint_code()

if __name__ == "__main__":
    main()