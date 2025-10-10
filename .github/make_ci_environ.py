import tomllib
from pathlib import Path
import yaml
import os

def parse_pyproject(path: str, optional_sections_to_skip=None):
    with open(path, "rb") as f:
        data = tomllib.load(f)

    deps = set()

    # project.dependencies (PEP 621)
    for dep in data.get("project", {}).get("dependencies", []):
        if "numpy" in dep:
            # numpy is also listed in build requirements with a higher version number
            # so we skip it here to avoid conflicts.
            continue
        deps.add(dep)

    # optional dependencies (PEP 621)
    if optional_sections_to_skip is None:
        optional_sections_to_skip = []
    for group, group_deps in data.get("project", {}).get("optional-dependencies", {}).items():
        if group in optional_sections_to_skip:
            print("Skipping optional dependency group:", group)
            continue
        deps.update(group_deps)

    deps.discard("geoana[all]")
    deps.discard("geoana[doc,all]")
    deps.discard("geoana[plot,extras,jittable]")

    if "matplotlib" in deps:
        deps.discard("matplotlib")
        deps.add("matplotlib-base")
    return sorted(deps)

def create_env_yaml(deps, name="env", python_version=None, free_threaded=False):
    conda_pkgs = []
    pip_pkgs = []

    for dep in deps:
        # crude split: try to detect conda vs pip-only packages
        if any(dep.startswith(pip_only) for pip_only in ["git+", "http:", "https:", "file:"]):
            pip_pkgs.append(dep)
        else:
            conda_pkgs.append(dep)

    dependencies = conda_pkgs
    if pip_pkgs:
        dependencies.append({"pip": pip_pkgs})

    if python_version:
        if free_threaded:
            dependencies.insert(0, f"python-threading={python_version}")
        dependencies.insert(0, f"python={python_version}")

    return {
        "name": name,
        "channels": ["conda-forge"],
        "dependencies": dependencies,
    }

if __name__ == "__main__":
    pyproject_path = Path("pyproject.toml")

    py_vers = os.environ.get("PYTHON_VERSION", "3.11")
    is_free_threaded = os.environ.get("FREE_THREADED", "false").lower() == "true"
    no_doctest = os.environ.get("NO_DOCTEST", "false").lower() == "true"
    no_numba = os.environ.get("NO_NUMBA", "false").lower() == "true"
    env_name = os.environ.get("ENV_NAME", "geoana_env")

    skips = ["all"]
    if no_numba:
        skips.append("jittable")
    if no_doctest:
        skips.append("doc")

    deps = parse_pyproject(pyproject_path, optional_sections_to_skip=skips)
    env_data = create_env_yaml(deps, name=env_name, python_version=py_vers, free_threaded=is_free_threaded)

    out_name = "environment_ci.yml"
    with open(out_name, "w") as f:
        yaml.safe_dump(env_data, f, sort_keys=False)

    print("✅ Generated environment_ci.yml with", len(deps), "dependencies")