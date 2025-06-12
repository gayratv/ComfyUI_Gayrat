import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Dict, Any

# Central registries that will be progressively filled at runtime
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}


def _import_module_from_path(path: pathlib.Path) -> ModuleType | None:
    """Safely import a Python module located at an arbitrary *path*.

    Returns the imported module or **None** if the import fails for any reason.
    """
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"Cannot create spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module  # Allows intra-module relative imports
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    except Exception as exc:  # broad catch to keep discovery robust
        print(f"[load_custom_nodes] Skipped {path}: {exc}", file=sys.stderr)
        return None


def load_custom_nodes() -> None:
    """Discover ``*.py`` files **only in subdirectories** of this script's folder.

    Python files located directly beside this script are ignored; only files in
    nested subfolders are imported. Any ``NODE_CLASS_MAPPINGS`` or
    ``NODE_DISPLAY_NAME_MAPPINGS`` they expose are merged into the global
    registries defined above.
    """

    root_path = pathlib.Path(__file__).parent.resolve()

    for file_path in root_path.rglob("*.py"):
        # Skip files that live directly in root_path (want subdirs only)
        if file_path.parent == root_path:
            continue

        module = _import_module_from_path(file_path)
        if module is None:
            continue

        # Merge NODE_CLASS_MAPPINGS, if present
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS"))

        # Merge NODE_DISPLAY_NAME_MAPPINGS, if present
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(
                getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
            )


if __name__ == "__main__":
    load_custom_nodes()

    # Developer-friendly output ---------------------------------------------
    print("Loaded NODE_CLASS_MAPPINGS:")
    for k, v in NODE_CLASS_MAPPINGS.items():
        print(f"  {k}: {v}")

    print("\nLoaded NODE_DISPLAY_NAME_MAPPINGS:")
    for k, v in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"  {k}: {v}")


WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
