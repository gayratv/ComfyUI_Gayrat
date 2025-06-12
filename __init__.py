import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Dict, Any

# ---------------------------------------------------------------------------
# GLOBAL REGISTRIES THAT COMFYUI EXPECTS
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS: Dict[str, Any] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# HELPER: DYNAMIC IMPORT WITH PACKAGE SUPPORT
# ---------------------------------------------------------------------------

def _import_module_from_path(path: pathlib.Path, root: pathlib.Path) -> ModuleType | None:
    """Import *path* as a sub‑module of the current package hierarchy.

    This function fabricates the package structure required for **relative
    imports** (``from .foo import bar``) to resolve correctly.  For a file like

        <root>/ImageResize/image_scale.py

    the resulting dotted name becomes ``ImageResize.image_scale``.  All
    intermediate packages (``ImageResize``) are created on‑the‑fly and entered
    into :pydata:`sys.modules` with proper ``__path__`` attributes so that the
    standard import machinery recognises them.
    """

    dotted_name = ".".join(path.relative_to(root).with_suffix("").parts)
    # Example: "ImageResize.image_scale_by_aspect_ratio_v2"

    # -----------------------------------------------------------------------
    # Ensure every parent package exists in sys.modules
    pkg_path = root
    pkg_name = ""
    for part in dotted_name.split(".")[:-1]:
        pkg_name = f"{pkg_name+'.' if pkg_name else ''}{part}"
        if pkg_name not in sys.modules:
            pkg = ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_path / part)]  # mandatory for sub‑imports
            sys.modules[pkg_name] = pkg
        pkg_path = pkg_path / part

    # -----------------------------------------------------------------------
    # Import the target module itself
    try:
        spec = importlib.util.spec_from_file_location(dotted_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[dotted_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module

    except Exception as exc:
        print(f"[Gayrat load_custom_nodes] Skipped {path}: {exc}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# MAIN DISCOVERY FUNCTION
# ---------------------------------------------------------------------------

def load_custom_nodes() -> None:
    """Recursively discover ``*.py`` files in *sub‑directories* and merge their
    node mappings into the global registries above.
    """

    root_path = pathlib.Path(__file__).parent.resolve()

    for file_path in root_path.rglob("*.py"):
        # Ignore Python files located directly in *root_path* (incl. this __init__)
        if file_path.parent == root_path:
            continue

        module = _import_module_from_path(file_path, root_path)
        if module is None:
            continue

        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS"))
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(
                getattr(module, "NODE_DISPLAY_NAME_MAPPINGS")
            )

# ---------------------------------------------------------------------------
# RUN DISCOVERY IMMEDIATELY ON PACKAGE IMPORT
# ---------------------------------------------------------------------------
# print("\033[31m[Gayrat] Scanning sub‑directories for custom nodes …\033[0m")
load_custom_nodes()
print("[Gayrat] Loaded", len(NODE_CLASS_MAPPINGS), "NODE_CLASS_MAPPINGS and",
      len(NODE_DISPLAY_NAME_MAPPINGS), "NODE_DISPLAY_NAME_MAPPINGS")

# ---------------------------------------------------------------------------
# PUBLIC EXPORTS
# ---------------------------------------------------------------------------
WEB_DIRECTORY = "./js"
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
