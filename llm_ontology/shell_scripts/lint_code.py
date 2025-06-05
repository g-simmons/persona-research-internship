from pathlib import Path
import subprocess
import re
from typing import Generator, Tuple, Literal, Optional, List
import os
import json
import logging
from pydantic import BaseModel


class CodeContribution(BaseModel):
    author: str
    lines: int
class LintingResult(BaseModel):
    lineno: int
    file: Path
    line: str
    type: Literal["hardcoded_path", "print", "debugger", "plt."]
    author: str
    author_date: str
    commit: str


# Set up logger with file and stdout handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
script_dir = Path(__file__).parent
log_file = script_dir / "lint_code.log"
file_handler = logging.FileHandler(log_file)
stdout_handler = logging.StreamHandler()

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        result = subprocess.run(
            [
                "git",
                "blame",
                "--porcelain",  # Machine-readable format
                "-L",
                f"{lineno},{lineno}",
                str(file),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().split("\n")
        if not lines:
            raise ValueError(f"No git blame output for {file}:{lineno}")

        # Parse porcelain format
        commit = ""
        author = ""
        date = ""

        for line in lines:
            if line.startswith("author "):
                author = line[7:]  # Remove 'author ' prefix
            elif line.startswith("author-time "):
                # Convert Unix timestamp to readable format
                timestamp = int(line[12:])
                import datetime

                date = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            elif not line.startswith(
                ("\t", "author-", "committer-", "summary ", "boundary", "filename ")
            ):
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
        result = subprocess.run(
            ["git", "blame", "--porcelain", "-L", f"{lineno},{lineno}", str(file)],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().split("\n")
        blame_info = {}

        for line in lines:
            if line.startswith("author "):
                blame_info["author"] = line[7:]
            elif line.startswith("author-mail "):
                blame_info["author_email"] = line[12:].strip("<>")
            elif line.startswith("author-time "):
                blame_info["author_timestamp"] = int(line[12:])
                import datetime

                blame_info["author_date"] = datetime.datetime.fromtimestamp(
                    int(line[12:])
                ).strftime("%Y-%m-%d %H:%M:%S")
            elif line.startswith("committer "):
                blame_info["committer"] = line[10:]
            elif line.startswith("committer-time "):
                blame_info["committer_timestamp"] = int(line[15:])
            elif line.startswith("summary "):
                blame_info["summary"] = line[8:]
            elif not line.startswith(
                ("\t", "author-", "committer-", "boundary", "filename ")
            ):
                if "commit" not in blame_info:
                    parts = line.split()
                    if parts:
                        blame_info["commit"] = parts[0]
                        blame_info["source_line"] = (
                            int(parts[1]) if len(parts) > 1 else lineno
                        )
                        blame_info["result_line"] = (
                            int(parts[2]) if len(parts) > 2 else lineno
                        )

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


def _show_instances(
    file: Path,
    regex: str,
    template: str,
    type: Literal["hardcoded_path", "print", "debugger", "plt."],
    format: Literal["json"] = "json",
) -> Generator[LintingResult, None, None]:
    for lineno, line in _check_for_regex(file, regex):
        blame_info = get_git_blame_json(file, lineno + 1)  # git uses 1-indexed lines
        output = {
            "lineno": lineno + 1,  # Convert to 1-indexed for consistency
            "file": str(file),
            "line": line.strip(),
            "type": type,
        }
        if blame_info:
            output.update(blame_info)
        yield LintingResult(**output)


def lint_file(file: Path, format: Literal["json"] = "json"):
    results = []
    results.extend(_show_instances(
        file,
        r"print",
        "Print statement found in {file} on line {lineno}",
        type="print",
        format=format,
    ))
    results.extend(_show_instances(
        file,
        r"import code",
        "Debugger statement found in {file} on line {lineno}",
        type="debugger",
        format=format,
    ))
    results.extend(_show_instances(
        file,
        r"plt.",
        "Matplotlib plt. statement found in {file} on line {lineno}",
        type="plt.",
        format=format,
    ))
    # check for hardcoded paths using a regex that matches Unix or Windows-style paths
    hardcoded_path_regex = r"(?:[\"'])(?:[A-Za-z]:)?(?:/|\\)[^\"']+[\"']"
    results.extend(_show_instances(
        file,
        hardcoded_path_regex,
        "Hardcoded path found in {file} on line {lineno}",
        type="hardcoded_path",
        format=format,
    ))
    return results


def lint_code(code_dir: Path = CODE_DIR) -> List[LintingResult]:
    results = []
    for file in code_dir.glob("*.py"):
        results.extend(lint_file(file, format="json"))
    
        # os.system(f"mypy --disallow-untyped-defs {file}")
    return results

def _get_total_code_contributions() -> List[CodeContribution]:
    """
    Get the total lines of code contributed by each author using git log --pretty and --numstat.
    This counts added lines per author across all commits in the repo for .py files in CODE_DIR.
    """
    from collections import defaultdict

    contributions = defaultdict(int)
    code_dir = CODE_DIR.resolve()

    # Use git log with --numstat to get per-author line additions for .py files
    try:
        # Get all .py files relative to the repo root
        py_files = [str(f.relative_to(code_dir.parent)) for f in code_dir.glob("*.py")]
        if not py_files:
            logger.warning("No Python files found for code contribution analysis.")
            return []

        # Build the git log command
        cmd = [
            "git", "log", "--pretty=format:%an", "--numstat", "--", *py_files
        ]
        result = subprocess.run(
            cmd,
            cwd=code_dir.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        current_author = None
        for line in result.stdout.splitlines():
            if line.strip() == "":
                continue
            # If the line is an author name (from --pretty=format:%an)
            if re.match(r"^[^\d\s]+\s*[^\d\s]*$", line.strip()):
                current_author = line.strip()
            # If the line is a numstat line: added\tremoved\tfile
            elif re.match(r"^\d+\s+\d+\s+.+\.py$", line):
                if current_author is not None:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            added = int(parts[0])
                        except ValueError:
                            added = 0  # Handle binary files or parse errors
                        contributions[current_author] += added
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to run git log for code contributions: {e}")
        return []

    return [CodeContribution(author=author, lines=lines) for author, lines in contributions.items()]

def main() -> None:
    """Main function to run the linting process."""
    linting_results = lint_code()
    total_code_contributions = _get_total_code_contributions()

    with open("linter_data.jsonl", "w") as f:
        for result in linting_results:
            f.write(result.model_dump_json() + "\n")

    with open("total_code_contributions.jsonl", "w") as f:
        for contribution in total_code_contributions:
            f.write(contribution.model_dump_json() + "\n")

    import pandas as pd
    linter_data = pd.read_json('../shell_scripts/linter_data.jsonl', lines=True)
    total_contributions_data = pd.read_json('../shell_scripts/total_code_contributions.jsonl', lines=True)

    linter_data = linter_data[linter_data.author != 'Not Committed Yet']
    linter_data = linter_data[linter_data.author != 'Zach Wang']
    grouped = linter_data.groupby(['type', 'author']).agg({'lineno': 'count'}).reset_index()
    grouped['total_lines'] = grouped.author.map(total_contributions_data.set_index('author')['lines'])
    grouped['fraction'] = grouped['lineno'] / grouped['total_lines']

    from itertools import product
    import numpy as np

    unique_authors = grouped.author.unique()
    unique_types = grouped.type.unique()

    for type, author in product(unique_types, unique_authors):
        print(type, author)
        if len(grouped[(grouped.type == type) & (grouped.author == author)]) == 0:
            grouped = pd.concat([grouped, pd.DataFrame({'type': [type], 'author': [author], 'lineno': [0], 'total_lines': [0], 'fraction': [np.nan]})], ignore_index=True)

            grouped.sort_values(by='fraction', ascending=False).sort_values(by='type', ascending=False)
    
    grouped_not_incl_gabe = grouped[grouped.author != 'Gabriel Simmons']

    clown_quotes = {
        'hardcoded_path': 'i guess the code doesn\'t need to be portable...',
        'plt.': 'five figures on one is fine, right?',
        'print': 'everyone needs to know about this'
    }

    technical_notes = {
        'hardcoded_path': 'Hardcoded paths to data and outputs make it difficult for multiple people to work on the same codebase. They also make the code sensitive to the current working directory, which hurts reproducibility.',
        'plt.': 'Matplotlib\'s plt.xyz methods act on the "current figure", a global variable. This is bad practice, and can lead to bugs when multiple figures are in use. Specify a figure and call `fig.xyz` or `ax.xyz` instead.',
        'print': 'Use a logger instead.'
    }

    from datetime import datetime

    def generate_contributions_report(grouped, total_contributions_data, unique_authors) -> str:
        output = ""
        output += "\n## Code Contributions\n"
        output += "| Author | Lines of Code |\n"
        output += "|--------|---------------|\n"
        # Prepare a DataFrame for sorting
        df = total_contributions_data[total_contributions_data['author'].isin(unique_authors)]
        df = df[df['author'] != 'Gabriel Simmons']
        df_sorted = df.sort_values(by='lines', ascending=False)
        for _, row in df_sorted.iterrows():
            output += f"| {row['author']} | {row['lines']} |\n"
        return output

    def generate_antipatterns_report(grouped, unique_types) -> str:
        output = ""
        output += f"\n## Code Anti-Patterns"
        for type in unique_types:
            output += f"\n### {type}"
            output += f"\n_{technical_notes[type]}_"
            output += "\n"
            output += "||||"
            output += "\n|--|--|--|"
            # get the top and bottom author for this type by fraction
            type_group = grouped[grouped.type == type].sort_values(by='fraction', ascending=False)
            top_row = type_group.iloc[0]
            bottom_row = type_group.sort_values(by='fraction', ascending=True).iloc[0]
            top_author = top_row.author
            top_fraction = top_row.fraction
            bottom_author = bottom_row.author
            bottom_fraction = bottom_row.fraction

            # Format author names and fractions
            def format_author(author):
                return f"**{author}**"
            def format_fraction(frac):
                if frac is None or (isinstance(frac, float) and (frac != frac)):
                    return "N/A"
                return f"{frac:.2%}"

            # Lone ranger case
            if top_author == bottom_author:
                output += f"\n| :cowboy_hat_face: **_lone ranger_** | {format_author(top_author)} ({format_fraction(top_fraction)} of total contributions) | |"
            else:
                output += f"\n| :innocent: **_innocent_** | {format_author(bottom_author)} ({format_fraction(bottom_fraction)} of total contributions) | Way to go! |"
                output += f"\n| :clown_face: **_class clown_** | {format_author(top_author)} ({format_fraction(top_fraction)} of total contributions) | {clown_quotes[type]} |"
        return output

    def generate_report(grouped, total_contributions_data, unique_authors, unique_types) -> str:
        output = ""
        date = datetime.now().strftime("%Y-%m-%d")
        title = f"Code Report {date}"
        output += f"\n# {title}"
        output += generate_contributions_report(grouped, total_contributions_data, unique_authors)
        output += generate_antipatterns_report(grouped, unique_types)
        return output

    output = generate_report(grouped_not_incl_gabe, total_contributions_data, unique_authors, unique_types)

    with open('report.md', 'w') as f:
        f.write(output)



if __name__ == "__main__":
    main()
