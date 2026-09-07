"""Actual scene evidence contact sheets and JSON, with no HTML artifacts."""
from ._common import *


def write_report(root, snapshot_path, geometry_path):
    snapshot, geometry = read_sealed(snapshot_path), read_sealed(geometry_path)
    path = root / "report/receipt.json"
    if path.exists():
        prior = read_sealed(path)
        require(prior["source_semantics"] == artifact(snapshot_path)
                and prior["source_geometry"] == artifact(geometry_path), "Report source drift")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup = {row["instance_id"]: row for row in geometry["objects"]}
    pages, records = [], []
    objects = snapshot["objects"]
    for page_index, first in enumerate(range(0, len(objects), 6)):
        page_objects = objects[first:first+6]
        canvas = Image.new("RGB", (1060, 210*len(page_objects)+45), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), "Fixed scene " + snapshot["scene_id"] + " | real scene evidence; unknowns retained", fill="black")
        for index, obj in enumerate(page_objects):
            geo, y = lookup[obj["instance_id"]], 45+index*210
            semantic = obj["semantic"]
            text = [obj["instance_id"], semantic["object_class"] + " / " + semantic["mask_state"],
                "identity=" + str(obj["decision"]["query_usable"]) + "  sit=" + str(geo["query_usable_for_sit"]),
                "generic planes=" + str(len(geo.get("generic_face_candidates", [])))]
            for line, label in enumerate(text):
                draw.text((12, y+line*19), label, fill="black")
            for view_index, image_record in enumerate(obj["geometry_source"].get("classifier_images", [])[:2]):
                image_path = verified(image_record)
                with Image.open(image_path) as source:
                    tile = source.convert("RGB")
                    tile.thumbnail((320, 198))
                    canvas.paste(tile, (410+view_index*325, y))
            records.append({"instance_id": obj["instance_id"], "semantic": semantic,
                "decision": obj["decision"], "query_usable_for_sit": geo["query_usable_for_sit"],
                "generic_face_count": len(geo.get("generic_face_candidates", []))})
        target = path.parent / f"scene_inventory_{page_index:03d}.png"
        require(not target.exists(), "Report image already exists")
        canvas.save(target)
        pages.append(artifact(target))
    write_once(path, {"schema": "hsi.scene_inventory_visualization.v1",
        "source_semantics": artifact(snapshot_path), "source_geometry": artifact(geometry_path),
        "pages": pages, "objects": records, "new_ai_images_generated": False,
        "html_generated": False})
    return path
