"""Build the project page from its explicit static assets."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "_site")
    output = parser.parse_args().output.resolve()
    # A fresh directory prevents stale files from earlier builds from being deployed.
    output.mkdir(parents=True, exist_ok=False)
    for name in ("index.html", "app.js", "style.css", ".nojekyll"):
        shutil.copy2(SOURCE / name, output / name)
    shutil.copytree(SOURCE / "assets", output / "assets")
    print(f"Built project page at {output}")


if __name__ == "__main__":
    main()
