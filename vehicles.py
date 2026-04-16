from pycocotools.coco import COCO

# Initialize COCO api
annFile = 'annotations/instances_train2017.json'
coco = COCO(annFile)

# 1. Use catNms for specific subcategories
# Note: 'vehicle' is the supercategory for all of these
target_categories = ['car', 'bus', 'truck']
catIds = coco.getCatIds(catNms=target_categories)

# 2. Get all image IDs containing ANY of these categories
imgIds = coco.getImgIds(catIds=catIds)

# 3. Load image metadata
images = coco.loadImgs(imgIds)

print(f"Target Categories: {target_categories}")
print(f"Category IDs found: {catIds}")
print(f"Total images containing at least one of these: {len(imgIds)}")