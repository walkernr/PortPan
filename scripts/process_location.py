"""End-to-end processing script for generating master stitched panoramas, previews, DZI pyramids, and landmark crops for a location."""
import sys
import gc
import cv2
import numpy as np
from pathlib import Path
from portpan.geometry.transform_store import TransformStore
from portpan.segmentation.polar_segmenter import PolarChromaSegmenter
from portpan.core.audit import VisualAuditor

def process_location(loc: str) -> None:
    print(f"==================================================")
    print(f"STARTING PIPELINE EXECUTION FOR: {loc}")
    print(f"==================================================")
    
    loc_path = Path(f"data/locations/{loc}")
    output_dir = Path(f"output/{loc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tf = TransformStore().load(loc_path)
    if tf is None:
        raise ValueError(f"No transforms found for {loc}")
        
    base_path = loc_path / "base" / tf.base_image_name
    print(f"Loading master canvas: {base_path}...")
    base_canvas = cv2.imread(str(base_path))
    if base_canvas is None:
        raise ValueError(f"Could not load base image from {base_path}")
        
    h_b, w_b = base_canvas.shape[:2]
    print(f"Base canvas loaded: {w_b}x{h_b}")
    
    auditor = VisualAuditor(output_dir=output_dir)
    segmenter = PolarChromaSegmenter()
    
    # 1. Segment apertures and export isolated portholes if not already present
    iso_dir = output_dir / "isolated_portholes"
    iso_dir.mkdir(parents=True, exist_ok=True)
    images_dir = loc_path / "images"
    
    apertures_map = {}
    print("Extracting apertures and isolated portholes...")
    for name, res in tf.transforms.items():
        if res.status == "FAILED":
            continue
            
        img_p = images_dir / name
        if not img_p.exists():
            continue
            
        stem = img_p.stem
        iso_p = iso_dir / f"{loc}_{stem}_porthole.png"
        
        target_img = cv2.imread(str(img_p))
        if target_img is None:
            continue
            
        h_t, w_t = target_img.shape[:2]
        try:
            ap = segmenter.segment(target_img)
        except Exception as err:
            print(f"Warning: segmenter fallback for {name}: {err}")
            from portpan.core.models import CircleAperture
            ap = CircleAperture(
                center_x=w_t / 2.0,
                center_y=h_t / 2.0,
                radius=min(w_t, h_t) * 0.45,
                image_width=w_t,
                image_height=h_t,
            )
            
        apertures_map[name] = ap
        if not iso_p.exists():
            auditor.export_isolated_porthole(target_img, ap, f"{loc}_{stem}")
            print(f"  Exported isolated porthole: {iso_p.name}")
        else:
            print(f"  Existing isolated porthole found: {iso_p.name}")
            
        del target_img
        gc.collect()

    class LazyImageMap(dict):
        def __init__(self, img_dir):
            super().__init__()
            self.img_dir = img_dir
        def __contains__(self, k):
            return (self.img_dir / k).exists()
        def __getitem__(self, k):
            return cv2.imread(str(self.img_dir / k))
        def get(self, k, default=None):
            p = self.img_dir / k
            if p.exists():
                return cv2.imread(str(p))
            return default

    images_map = LazyImageMap(images_dir)
    
    print("\nGenerating static composite preview and detail crop...")
    auditor.draw_static_composite_preview(
        base_canvas=base_canvas,
        results=tf.transforms,
        images_map=images_map,
        apertures_map=apertures_map,
        location_name=loc,
    )
    
    print("\nGenerating master gigapixel panorama and DZI pyramid...")
    auditor.draw_master_gigapixel_panorama(
        base_canvas=base_canvas,
        results=tf.transforms,
        images_map=images_map,
        apertures_map=apertures_map,
        location_name=loc,
    )
    
    # 2. Extract inspection crops from the master panorama
    master_path = output_dir / f"{loc}_master_stitched_panorama.jpg"
    if master_path.exists():
        print("Extracting native-resolution inspection crops...")
        crops_dir = output_dir / "landmark_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        master = cv2.imread(str(master_path))
        if master is not None:
            h_m, w_m = master.shape[:2]
            
            # Identify top 2 highest-confidence/highest-saliency portholes to crop
            centers = []
            for name, res in tf.transforms.items():
                if res.status == "SUCCESS" and res.bounding_box_in_base:
                    pts = np.array(res.bounding_box_in_base) * 2.5
                    cx = int(round(float(np.mean(pts[:, 0]))))
                    cy = int(round(float(np.mean(pts[:, 1]))))
                    centers.append((name, cx, cy, res.inlier_count))
                    
            centers.sort(key=lambda c: c[3], reverse=True)
            for idx, (name, cx, cy, inl) in enumerate(centers[:3]):
                w_box, h_box = 800, 800
                x1 = max(0, cx - w_box // 2)
                x2 = min(w_m, cx + w_box // 2)
                y1 = max(0, cy - h_box // 2)
                y2 = min(h_m, cy + h_box // 2)
                crop = master[y1:y2, x1:x2]
                crop_name = f"{loc}_landmark_crop_{idx + 1}_{Path(name).stem}.jpg"
                cv2.imwrite(str(crops_dir / crop_name), crop)
                print(f"  Saved inspection crop: {crop_name}")
                
            del master
            gc.collect()

    print(f"\n==================================================")
    print(f"SUCCESSFULLY COMPLETED {loc}!")
    print(f"==================================================\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        locations = sys.argv[1:]
    else:
        locations = ["kolmer_gulch", "windermere_point"]
        
    for loc in locations:
        process_location(loc)
