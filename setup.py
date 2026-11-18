"""Install NAVSIM and DriveSTA without bundling external research assets."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
requirements = [
    line.split("#", 1)[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
]

setup(
    name="drivesta",
    version="0.1.0",
    description="DriveSTA: learning from scored trajectory alternatives for driving",
    packages=find_packages(include=["navsim", "navsim.*"]),
    python_requires=">=3.9",
    install_requires=[line for line in requirements if line],
    include_package_data=True,
    license="Apache-2.0",
)
