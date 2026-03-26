import os
import sys
from pathlib import Path


def ensure_local_site_packages():
    root = Path(__file__).resolve().parent

    site_packages_candidates = [
        root / "Dg" / "Lib" / "site-packages",
        root / ".venv" / "Lib" / "site-packages",
        root / "venv" / "Lib" / "site-packages",
    ]
    scripts_candidates = [
        root / "Dg" / "Scripts",
        root / ".venv" / "Scripts",
        root / "venv" / "Scripts",
    ]

    for site_packages in site_packages_candidates:
        if site_packages.exists():
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                sys.path.insert(0, site_packages_str)
            break

    for scripts_dir in scripts_candidates:
        if scripts_dir.exists():
            scripts_dir_str = str(scripts_dir)
            current_path = os.environ.get("PATH", "")
            if scripts_dir_str not in current_path.split(os.pathsep):
                os.environ["PATH"] = scripts_dir_str + os.pathsep + current_path
            break
