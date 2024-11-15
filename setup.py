from setuptools import setup, find_packages

setup(
    name="maple-code-extractor",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "typer>=0.4.0",
    ],
    entry_points={
        "console_scripts": [
            "maple-extract=maple_extractor:app",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to extract Maple code from Maple workbook XML files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/maple-code-extractor",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
