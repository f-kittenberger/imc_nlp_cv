import os
import requests
import json
from pycocotools.coco import COCO

# Setup Paths
dataDir = 'annotations'
instFile = f'{dataDir}/instances_val2017.json'
captFile = f'{dataDir}/captions_val2017.json'
saveDir = 'vehicle_subset_OR'
os.makedirs(saveDir, exist_ok=True)

# Initialize APIs
coco_inst = COCO(instFile)
coco_caps = COCO(captFile)

# Get specific categories
target_cats = ['car', 'bus', 'truck']
unique_img_ids = set()

print("\n--- Category Breakdown ---")
for cat_name in target_cats:
    catId = coco_inst.getCatIds(catNms=[cat_name])
    imgIds = coco_inst.getImgIds(catIds=catId)
    print(f"{cat_name.capitalize()}: {len(imgIds)} images found")
    unique_img_ids.update(imgIds) # Adds IDs to set, automatically ignoring duplicates

final_img_list = list(unique_img_ids)
print(f"\nTotal unique images (OR query): {len(final_img_list)}")

# Dictionary to store captions
subset_metadata = {}

# Process EVERY image in the subset
for img_id in final_img_list:
    # Use .get() or dict access to bypass Pylance TypedDict warnings
    img_info = coco_inst.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    url = img_info.get('coco_url')
    
    # Get Captions
    annIds = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(annIds)
    captions = [dict(ann).get('caption') for ann in anns if 'caption' in dict(ann)]  # Safely access 'caption' key
    
    # Store in metadata dictionary
    subset_metadata[file_name] = captions
    
    # Download image (only if not already downloaded)
    target_path = os.path.join(saveDir, file_name)
    if not os.path.exists(target_path):
        try:
            img_data = requests.get(url, timeout=10).content
            with open(target_path, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

# Save all captions to a single file
with open('vehicle_captions_OR.json', 'w') as f:
    json.dump(subset_metadata, f, indent=4)

print(f"Successfully extracted {len(subset_metadata)} images and their captions.")