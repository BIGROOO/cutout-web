"""Fuse rembg's expanded deformable convolution for ONNX Runtime Web >= 1.30.

Preserves FP32 trained weights and the original 1024 input. Requires onnx==1.19.0.
Only accepts the known rembg BiRefNet export pattern; never silently modifies other graphs.
"""
import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import helper


def convert(source: Path, destination: Path):
    model = onnx.load(source)
    prefixes = [node.name.removesuffix("offset_conv/Conv") for node in model.graph.node
                if node.name.endswith("/atrous_conv/offset_conv/Conv")]
    if len(prefixes) != 20:
        raise ValueError(f"Expected 20 known deformable convolution blocks, found {len(prefixes)}")
    by_name = {node.name: node for node in model.graph.node}
    replacements = {}
    removed = set()
    for prefix in prefixes:
        nodes = [node for node in model.graph.node if node.name.startswith(prefix)]
        offset = by_name[prefix + "offset_conv/Conv"]
        regular = by_name[prefix + "Conv"]
        mask = by_name[prefix + "Mul"]
        tail = by_name[prefix + "Add_3"]
        assert len(nodes) == 120 and len(regular.input) == 2
        assert list(tail.input) == [regular.output[0], prefix + "Constant_48_output_0"]
        retained_names = ["offset_conv/Conv", "modulator_conv/Conv", "Sigmoid", "Constant", "Mul"]
        retained = [by_name[prefix + name] for name in retained_names]
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in offset.attribute}
        assert attributes["group"] == 1
        replacement = helper.make_node(
            "DeformConv", [offset.input[0], regular.input[1], offset.output[0], "", mask.output[0]],
            list(regular.output), name=prefix + "NativeDeformConv",
            group=1, offset_group=1, pads=attributes["pads"],
            strides=attributes["strides"], dilations=attributes["dilations"],
        )
        retained += [replacement, by_name[prefix + "Constant_48"], tail]
        produced = {output for node in nodes for output in node.output}
        group_names = {node.name for node in nodes}
        external_uses = {item for node in model.graph.node if node.name not in group_names
                         for item in node.input if item in produced}
        assert external_uses == set(tail.output)
        replacements[offset.name] = retained
        removed.update(node.name for node in nodes)
    # This pinned export has no Reduce* operators needing the opset-18 axes migration.
    # All Split nodes already supply split sizes; existing float/int operators keep
    # their semantics in opset 19. DeformConv was introduced in opset 19.
    assert len(model.opset_import) == 1 and model.opset_import[0].version == 17
    assert all(not node.op_type.startswith("Reduce") for node in model.graph.node)
    assert all(len(node.input) == 2 for node in model.graph.node if node.op_type == "Split")
    original_count = len(model.graph.node)
    model.opset_import[0].version = 19
    new_nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            new_nodes.extend(replacements[node.name])
        elif node.name not in removed:
            new_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.doc_string = "Cutout browser export: legacy deformable-convolution blocks fused; FP32 weights and 1024 input unchanged."
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(f"{source.name}: {original_count} -> {len(new_nodes)} nodes; {destination.stat().st_size} bytes; SHA256 {digest}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
