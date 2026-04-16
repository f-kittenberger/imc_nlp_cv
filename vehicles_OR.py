import os
import json
from pycocotools.coco import COCO

# 1. Setup Paths (Adjust these to your local paths)
annDir = 'annotations'
instFile = f'{annDir}/instances_val2017.json'
captFile = f'{annDir}/captions_val2017.json'

# Initialize APIs
coco = COCO(instFile)
coco_caps = COCO(captFile)

target_cats = ['car', 'bus', 'truck']
category_stats = {}
all_unique_ids = set()

print("\n--- Category Breakdown ---")

# 2. Get counts per category
for cat_name in target_cats:
    # Get ID for this specific category
    catId = coco.getCatIds(catNms=[cat_name])
    imgIds = coco.getImgIds(catIds=catId)
    
    # Store stats
    category_stats[cat_name] = len(imgIds)
    all_unique_ids.update(imgIds)
    
    print(f"{cat_name.capitalize()}: {len(imgIds)} images")

# 3. Final Summary
print("-" * 30)
print(f"Total Sum (with overlaps): {sum(category_stats.values())}")
print(f"Total Unique Images:      {len(all_unique_ids)}")
print("-" * 30)

# 4. Generate the Subset Metadata (Descriptions)
subset_metadata = []

for img_id in all_unique_ids:
    img_info = coco.loadImgs(img_id)[0]
    
    # Get all captions for this image
    annIds = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(annIds)
    descriptions = [ann['caption'] for ann in anns]
    
    subset_metadata.append({
        "file_name": img_info['file_name'],
        "coco_url": img_info['coco_url'],
        "descriptions": descriptions
    })

# Save the metadata to a JSON file
with open('vehicle_subset_metadata.json', 'w') as f:
    json.dump(subset_metadata, f, indent=4)

print(f"\nSaved metadata for {len(subset_metadata)} unique images to 'vehicle_subset_metadata.json'")