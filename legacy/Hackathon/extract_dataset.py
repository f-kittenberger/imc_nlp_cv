import os
import requests
import json
from pycocotools.coco import COCO

# Setup Paths
dataDir = 'annotations'
instFile = f'{dataDir}/instances_train2017.json'
captFile = f'{dataDir}/captions_train2017.json'
saveDir = 'vehicle_subset'
os.makedirs(saveDir, exist_ok=True)

# Initialize APIs
coco_inst = COCO(instFile)
coco_caps = COCO(captFile)

# Get specific categories
target_cats = ['car', 'bus', 'truck']
catIds = coco_inst.getCatIds(catNms=target_cats)
imgIds = coco_inst.getImgIds(catIds=catIds)

print(f"Total matching images found: {len(imgIds)}")

# Dictionary to store captions
subset_metadata = {}

# Process EVERY image in the subset
for img_id in imgIds:
    # Use .get() or dict access to bypass Pylance TypedDict warnings
    img_info = coco_inst.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    url = img_info['coco_url']
    
    # Get Captions
    annIds = coco_caps.getAnnIds(imgIds=img_id)
    anns = coco_caps.loadAnns(annIds)
    captions = [ann['caption'] for ann in anns]
    
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
with open('vehicle_captions.json', 'w') as f:
    json.dump(subset_metadata, f, indent=4)

print(f"Successfully extracted {len(subset_metadata)} images and their captions.")