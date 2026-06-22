import json
from pycocotools.coco import COCO

# 1. Load both annotation files
inst_coco = COCO('annotations/instances_train2017.json')
capt_coco = COCO('annotations/captions_train2017.json')

# 2. Get your specific vehicle categories
catIds = inst_coco.getCatIds(catNms=['bus', 'car', 'truck'])
imgIds = inst_coco.getImgIds(catIds=catIds)

subset_data = []

print(f"Processing {len(imgIds)} images...")

for img_id in imgIds:
    # Get image metadata (filename, etc.)
    img_info = inst_coco.loadImgs(img_id)[0]
    
    # Get all captions associated with this image ID
    annIds = capt_coco.getAnnIds(imgIds=img_id)
    anns = capt_coco.loadAnns(annIds)
    
    # Clean the captions into a simple list of strings
    descriptions = [ann['caption'] for ann in anns]
    
    # Store the result
    subset_data.append({
        "file_name": img_info['file_name'],
        "image_path": f"vehicle_subset/{img_info['file_name']}",  # add path prefix for your subset
        "id": img_id,
        "descriptions": descriptions
    })

# 3. Save your custom subset description file
with open('vehicle_subset_descriptions.json', 'w') as f:
    json.dump(subset_data, f, indent=4)

print("Done! You now have 'vehicle_subset_descriptions.json'.")