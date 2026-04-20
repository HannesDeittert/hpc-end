#!/usr/bin/env python3
"""Check simple training tools against the wire registry definitions."""

from __future__ import annotations

import ast
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEVICE_FIELDS = [
    "name",
    "velocity_limit",
    "length",
    "tip_radius",
    "tip_angle",
    "tip_outer_diameter",
    "tip_inner_diameter",
    "straight_outer_diameter",
    "straight_inner_diameter",
    "poisson_ratio",
    "young_modulus_tip",
    "young_modulus_straight",
    "mass_density_tip",
    "mass_density_straight",
    "visu_edges_per_mm",
    "collis_edges_per_mm_tip",
    "collis_edges_per_mm_straight",
    "beams_per_mm_tip",
    "beams_per_mm_straight",
    "color",
]

SOFA_FIELDS = [
    "length",
    "straight_length",
    "spire_diameter",
    "spire_height",
    "poisson_ratio",
    "young_modulus",
    "young_modulus_extremity",
    "radius",
    "radius_extremity",
    "inner_radius",
    "inner_radius_extremity",
    "mass_density",
    "mass_density_extremity",
    "num_edges",
    "num_edges_collis",
    "density_of_beams",
    "key_points",
    "color",
]


@dataclass(frozen=True)
class ToolPair:
    label: str
    custom_file: str
    custom_module: str
    custom_class: str
    registry_module: str
    registry_class: str


TOOL_PAIRS = [
    ToolPair(
        label="amplatz/tight_j",
        custom_file="steve_recommender/bench/custom_tools_amplatz_tight_j_simple.py",
        custom_module="steve_recommender.bench.custom_tools_amplatz_tight_j_simple",
        custom_class="JShapedAmplatzSuperStiffTightJSimple",
        registry_module="data.wire_registry.amplatz_super_stiff.wire_versions.tight_j.tool",
        registry_class="JShaped_AmplatzSuperStiff_TightJ",
    ),
    ToolPair(
        label="amplatz/gentle",
        custom_file="steve_recommender/bench/custom_tools_amplatz_gentle_simple.py",
        custom_module="steve_recommender.bench.custom_tools_amplatz_gentle_simple",
        custom_class="JShapedAmplatzSuperStiffGentleSimple",
        registry_module="data.wire_registry.amplatz_super_stiff.wire_versions.gentle.tool",
        registry_class="JShaped_AmplatzSuperStiff_Gentle",
    ),
    ToolPair(
        label="amplatz/straight",
        custom_file="steve_recommender/bench/custom_tools_amplatz_straight_simple.py",
        custom_module="steve_recommender.bench.custom_tools_amplatz_straight_simple",
        custom_class="JShapedAmplatzSuperStiffStraightSimple",
        registry_module="data.wire_registry.amplatz_super_stiff.wire_versions.straight.tool",
        registry_class="JShaped_AmplatzSuperStiff_Straight",
    ),
    ToolPair(
        label="amplatz/strong_hook",
        custom_file="steve_recommender/bench/custom_tools_amplatz_strong_hook_simple.py",
        custom_module="steve_recommender.bench.custom_tools_amplatz_strong_hook_simple",
        custom_class="JShapedAmplatzSuperStiffStrongHookSimple",
        registry_module="data.wire_registry.amplatz_super_stiff.wire_versions.strong_hook.tool",
        registry_class="JShaped_AmplatzSuperStiff_StrongHook",
    ),
    ToolPair(
        label="steve_default/tight_j",
        custom_file="steve_recommender/bench/custom_tools_steve_tight_j_simple.py",
        custom_module="steve_recommender.bench.custom_tools_steve_tight_j_simple",
        custom_class="JShapedDefaultTightJSimple",
        registry_module="data.wire_registry.steve_default.wire_versions.tight_j.tool",
        registry_class="JShaped_Default_TightJ",
    ),
    ToolPair(
        label="steve_default/gentle",
        custom_file="steve_recommender/bench/custom_tools_steve_gentle_simple.py",
        custom_module="steve_recommender.bench.custom_tools_steve_gentle_simple",
        custom_class="JShapedDefaultGentleSimple",
        registry_module="data.wire_registry.steve_default.wire_versions.gentle.tool",
        registry_class="JShaped_Default_Gentle",
    ),
    ToolPair(
        label="steve_default/straight",
        custom_file="steve_recommender/bench/custom_tools_steve_straight_simple.py",
        custom_module="steve_recommender.bench.custom_tools_steve_straight_simple",
        custom_class="JShapedDefaultStraightSimple",
        registry_module="data.wire_registry.steve_default.wire_versions.straight.tool",
        registry_class="JShaped_Default_Straight",
    ),
    ToolPair(
        label="steve_default/strong_hook",
        custom_file="steve_recommender/bench/custom_tools_steve_strong_hook_simple.py",
        custom_module="steve_recommender.bench.custom_tools_steve_strong_hook_simple",
        custom_class="JShapedDefaultStrongHookSimple",
        registry_module="data.wire_registry.steve_default.wire_versions.strong_hook.tool",
        registry_class="JShaped_Default_StrongHook",
    ),
    ToolPair(
        label="universal_ii/tight_j",
        custom_file="steve_recommender/bench/custom_tools_universalii_tight_j_simple.py",
        custom_module="steve_recommender.bench.custom_tools_universalii_tight_j_simple",
        custom_class="JShapedUniversalIITightJSimple",
        registry_module="data.wire_registry.universal_ii.wire_versions.tight_j.tool",
        registry_class="JShaped_UniversalII_TightJ",
    ),
    ToolPair(
        label="universal_ii/gentle",
        custom_file="steve_recommender/bench/custom_tools_universalii_gentle_simple.py",
        custom_module="steve_recommender.bench.custom_tools_universalii_gentle_simple",
        custom_class="JShapedUniversalIIGentleSimple",
        registry_module="data.wire_registry.universal_ii.wire_versions.gentle.tool",
        registry_class="JShaped_UniversalII_Gentle",
    ),
    ToolPair(
        label="universal_ii/straight",
        custom_file="steve_recommender/bench/custom_tools_universalii_straight_simple.py",
        custom_module="steve_recommender.bench.custom_tools_universalii_straight_simple",
        custom_class="JShapedUniversalIIStraightSimple",
        registry_module="data.wire_registry.universal_ii.wire_versions.straight.tool",
        registry_class="JShaped_UniversalII_Straight",
    ),
    ToolPair(
        label="universal_ii/strong_hook",
        custom_file="steve_recommender/bench/custom_tools_universalii_strong_hook_simple.py",
        custom_module="steve_recommender.bench.custom_tools_universalii_strong_hook_simple",
        custom_class="JShapedUniversalIIStrongHookSimple",
        registry_module="data.wire_registry.universal_ii.wire_versions.strong_hook.tool",
        registry_class="JShaped_UniversalII_StrongHook",
    ),
]


def load_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def ast_name(node: ast.AST) -> str:
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    if isinstance(node, ast.Attribute):
        return f"{ast_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return f"{ast_name(node.func)}(...)"
    if isinstance(node, ast.Subscript):
        return f"{ast_name(node.value)}[...]"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    return node.__class__.__name__


def check_style(pair: ToolPair) -> tuple[bool, list[str]]:
    path = REPO_ROOT / pair.custom_file
    tree = ast.parse(path.read_text(), filename=str(path))

    issues = []
    imports_eve = any(
        isinstance(node, ast.Import) and any(alias.name == "eve" for alias in node.names)
        for node in tree.body
    )
    if not imports_eve:
        issues.append("missing `import eve`")

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1:
        issues.append(f"expected exactly 1 top-level class, found {len(classes)}")
        return False, issues

    base_names = [ast_name(base) for base in classes[0].bases]
    if base_names != ["eve.intervention.device.JShaped"]:
        issues.append(f"unexpected base class: {base_names}")

    return not issues, issues


def compare_fields(lhs, rhs, fields: list[str]) -> list[str]:
    diffs = []
    for field in fields:
        lhs_value = getattr(lhs, field)
        rhs_value = getattr(rhs, field)
        if lhs_value != rhs_value:
            diffs.append(
                f"{field}: custom={lhs_value!r} registry={rhs_value!r}"
            )
    return diffs


def main() -> int:
    overall_ok = True

    print("=== style check ===")
    for pair in TOOL_PAIRS:
        ok, issues = check_style(pair)
        if ok:
            print(f"{pair.custom_file}: OK")
        else:
            overall_ok = False
            print(f"{pair.custom_file}: FAIL")
            for issue in issues:
                print(f"  - {issue}")

    print()
    print("=== registry comparison ===")
    for pair in TOOL_PAIRS:
        custom_cls = load_class(pair.custom_module, pair.custom_class)
        registry_cls = load_class(pair.registry_module, pair.registry_class)

        custom_obj = custom_cls()
        registry_obj = registry_cls()

        device_diffs = compare_fields(custom_obj, registry_obj, DEVICE_FIELDS)
        sofa_diffs = compare_fields(custom_obj.sofa_device, registry_obj.sofa_device, SOFA_FIELDS)

        device_ok = not device_diffs
        sofa_ok = not sofa_diffs
        overall_ok = overall_ok and device_ok and sofa_ok

        print(
            f"{pair.label}: "
            f"device={'MATCH' if device_ok else 'DIFF'} "
            f"sofa={'MATCH' if sofa_ok else 'DIFF'}"
        )

        if device_diffs:
            print("  device diffs:")
            for diff in device_diffs:
                print(f"    - {diff}")

        if sofa_diffs:
            print("  sofa diffs:")
            for diff in sofa_diffs:
                print(f"    - {diff}")

    print()
    print("OVERALL:", "MATCH" if overall_ok else "DIFFERENT")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
