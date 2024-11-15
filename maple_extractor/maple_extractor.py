import typer
import xml.etree.ElementTree as ET
from pathlib import Path

app = typer.Typer()


def extract_maple_code(xml_content):
    root = ET.fromstring(xml_content)
    code_blocks = root.findall(".//Input/Text-field[@style='Maple Input']")

    return "\n\n".join(
        "".join(block.itertext()).strip().encode("ascii").decode("unicode_escape")
        for block in code_blocks
    )


@app.command()
def extract(
    input_file: Path = typer.Argument(..., help="Path to the Maple workbook XML file"),
    output_file: Path = typer.Option(
        None, help="Path to save the extracted Maple script"
    ),
):
    """
    Extract Maple code from a Maple workbook XML file and save it as a plain Maple script.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            xml_content = f.read()

        maple_code = extract_maple_code(xml_content)

        if not output_file:
            output_file = input_file.with_suffix(".mpl")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(maple_code)

        typer.echo(f"Maple code extracted and saved to {output_file}")

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
