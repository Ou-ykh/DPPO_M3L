"""Generate a lighter MJCF variant for MJX state-based RL experiments.

The output is a separate XML file. The original model is never modified.
Defaults remove rendering-only content while keeping collision dynamics.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def _rewrite_asset_paths(root, xml_path: Path, repo_root: Path) -> int:
    changed = 0
    xml_dir = xml_path.parent
    for asset in root.findall(".//asset/*[@file]"):
        raw = asset.get("file")
        if not raw:
            continue

        normalized = raw.replace("\\", "/")
        lowered = normalized.lower()
        marker = "dppo/"
        marker_index = lowered.find(marker)

        if marker_index >= 0:
            rel = normalized[marker_index + len(marker) :]
            rewritten = repo_root / Path(*rel.split("/"))
        elif ":" in normalized[:4]:
            rewritten = Path(raw)
        elif normalized.startswith("/"):
            continue
        elif Path(normalized).is_absolute():
            rewritten = Path(normalized)
        else:
            rewritten = xml_dir / Path(*normalized.split("/"))

        rewritten_str = str(rewritten)
        if rewritten_str != raw:
            asset.set("file", rewritten_str)
            changed += 1
    return changed


def _parent_map(root):
    return {child: parent for parent in root.iter() for child in parent}


def _remove(root, predicate) -> int:
    parents = _parent_map(root)
    removed = 0
    for elem in list(root.iter()):
        if elem is root:
            continue
        if predicate(elem):
            parent = parents.get(elem)
            if parent is not None:
                parent.remove(elem)
                removed += 1
    return removed


def _remove_default_class(root, class_name: str) -> int:
    removed = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "default" and child.get("class") == class_name:
                parent.remove(child)
                removed += 1
    return removed


def _insert_before_first_body(parent, elem) -> None:
    children = list(parent)
    for index, child in enumerate(children):
        if child.tag == "body":
            parent.insert(index, elem)
            return
    parent.append(elem)


def _make_grip_pad_geom(side: str):
    return ET.Element(
        "geom",
        {
            "name": f"{side}_grip_pad_collision",
            "class": "collision",
            "type": "box",
            "pos": "0 -0.0026 0.01875",
            "size": "0.0115 0.0045 0.01875",
            "mass": "0",
            "friction": "1.5 0.05 0.001",
            "solimp": "0.98 0.995 0.001",
            "solref": "0.006 1",
            "priority": "2",
            "condim": "4",
            "rgba": "0 0 0 0",
        },
    )


def _simplify_pad_geoms(root) -> tuple[int, int]:
    removed = 0
    added = 0

    for side in ("right", "left"):
        body = root.find(f".//body[@name='{side}_pad']")
        if body is None:
            continue

        generated_name = f"{side}_grip_pad_collision"
        for child in list(body):
            if child.tag != "geom":
                continue
            if child.get("class") == "pad":
                body.remove(child)
                removed += 1
            elif child.get("name") == generated_name:
                body.remove(child)

        _insert_before_first_body(body, _make_grip_pad_geom(side))
        added += 1

    return removed, added


def _set_joint_default(root, class_name: str, updates: dict[str, str]) -> int:
    joint = root.find(f".//default[@class='{class_name}']/joint")
    if joint is None:
        return 0

    changed = 0
    for key, value in updates.items():
        if joint.get(key) != value:
            joint.set(key, value)
            changed += 1
    return changed


def _stabilize_gripper_defaults(root) -> int:
    changed = 0
    changed += _set_joint_default(root, "follower", {"damping": "0.02"})
    changed += _set_joint_default(root, "coupler", {"damping": "0.02"})
    changed += _set_joint_default(root, "spring_link", {"damping": "0.005"})
    return changed


def _ensure_option(root):
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    return option


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", nargs="?", default="mjmodel.xml")
    parser.add_argument("-o", "--output", default="mjmodel_mjx_slim.xml")
    parser.add_argument("--keep-cameras", action="store_true")
    parser.add_argument("--keep-lights", action="store_true")
    parser.add_argument("--keep-visual-geoms", action="store_true")
    parser.add_argument(
        "--remove-pad-geoms",
        action="store_true",
        help="Remove class='pad' tactile grid geoms. This changes contact behavior.",
    )
    parser.add_argument(
        "--simplify-pads",
        action="store_true",
        help="Replace tactile pad grids with one stable collision pad per finger.",
    )
    parser.add_argument(
        "--stabilize-gripper",
        action="store_true",
        help="Add modest damping to passive gripper defaults to reduce chatter.",
    )
    parser.add_argument(
        "--tune-options",
        action="store_true",
        help="Set common MJX-friendly solver options for initial experiments.",
    )
    parser.add_argument(
        "--maxhullvert",
        type=int,
        default=None,
        help="Set compiler maxhullvert, useful for MJX-JAX mesh collision tuning.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Root used to rewrite old G:\\DPPO\\... asset paths. Defaults to the repository root.",
    )
    parser.add_argument("--no-normalize-assets", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.xml).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    tree = ET.parse(in_path)
    root = tree.getroot()

    counts = {}
    if not args.no_normalize_assets:
        repo_root = (
            Path(args.repo_root).expanduser().resolve()
            if args.repo_root
            else Path(__file__).resolve().parents[1]
        )
        counts["asset_path"] = _rewrite_asset_paths(root, in_path, repo_root)

    counts["extension"] = _remove(root, lambda e: e.tag == "extension")

    if not args.keep_cameras:
        counts["camera"] = _remove(root, lambda e: e.tag == "camera")
    if not args.keep_lights:
        counts["light"] = _remove(root, lambda e: e.tag == "light")
    if not args.keep_visual_geoms:
        counts["visual_geom"] = _remove(
            root,
            lambda e: e.tag == "geom"
            and (
                e.get("class") == "visual"
                or (e.get("contype") == "0" and e.get("conaffinity") == "0")
            ),
        )
    if args.simplify_pads:
        removed, added = _simplify_pad_geoms(root)
        counts["pad_geom"] = removed
        counts["grip_pad_geom"] = added
        counts["pad_default"] = _remove_default_class(root, "pad")
    elif args.remove_pad_geoms:
        counts["pad_geom"] = _remove(root, lambda e: e.tag == "geom" and e.get("class") == "pad")

    if args.stabilize_gripper:
        counts["gripper_default_attr"] = _stabilize_gripper_defaults(root)

    if args.maxhullvert is not None:
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            root.insert(0, compiler)
        compiler.set("maxhullvert", str(args.maxhullvert))

    if args.tune_options:
        option = _ensure_option(root)
        tuned = copy.copy(option.attrib)
        tuned.update(
            {
                "solver": "Newton",
                "iterations": "2",
                "ls_iterations": "4",
                "jacobian": "dense",
            }
        )
        option.attrib.clear()
        option.attrib.update(tuned)

    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)

    print("input:", in_path)
    print("output:", out_path)
    for key, value in counts.items():
        print(f"removed {key}: {value}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
