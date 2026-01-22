# setup.py
from setuptools import find_packages, setup
from pathlib import Path

# Read dependencies from requirements.txt
def read_requirements():
    requirements_path = Path(__file__).parent / "requirements.txt"
    with open(requirements_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="persona-elicitation",
    version="0.1.0",
    description="Persona Elicitation with ICM algorithm",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=read_requirements(),
    python_requires=">=3.9",
    include_package_data=True,
)