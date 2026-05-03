import os, sys
import glob
import cv2
import PIL.Image as Image
import numpy as np
import h5py
from tqdm import tqdm
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter 

# Path to TRANCOS dataset with file structure:
# - images/
# - results/
# - code/
# - image_sets/
dataset_path = "/mnt/d/common/datasets/TRANCOS_v3"
test_set = "image_sets/test.txt"
train_val_set = "image_sets/trainval.txt"
density_map_set = "density_gt"

# Get path sets
test_set_path = os.path.join(dataset_path, test_set)
train_val_set_path = os.path.join(dataset_path, train_val_set)  
path_sets = [test_set_path, train_val_set_path]

# Get image list
img_paths = []
for path_set in path_sets:
    if ".txt" in path_set:
        with open(path_set, "r") as f:
            count = 0
            for line in f:
                line = line.strip()   # remove newline / whitespace
                img_path = os.path.join(dataset_path, "images", line)
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                    count += 1
                else:
                    print(f">>> Warning: {img_path} does not exist.")
    if "test" in path_set:
        print(f">>> Test set: {count} images found.")
    elif "trainval" in path_set:
        print(f">>> Train/Val set: {count} images found.")
print(f">>> Total images found: {len(img_paths)}")

#==========================================
# Generate density map
#==========================================
# The adaptive sigma is calculated based on the average distance to the 3 nearest neighbors, 
# multiplied by a factor (e.g., 0.3).
def adaptive_sigma(points):
    tree = KDTree(points)
    distances, _ = tree.query(points, k=4)
    sigma = distances[:, 1:4].mean(axis=1) * 0.3
    return sigma

# Gauusian filter density map generation
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array((np.nonzero(gt)[1], np.nonzero(gt)[0]))
    pts = pts.T
    leafsize = 2048
    # build kdtree
    tree = KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.08
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    return density

os.makedirs(os.path.join(dataset_path, density_map_set), exist_ok=True)
print(">>> Generating density maps and saving to .h5 files...")
for i in tqdm(range(len(img_paths))):
    img_path = img_paths[i]
    gt_path = img_path.replace('.jpg', '.txt')
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    # Get GT points
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            gt_point_x, gt_point_y = map(int, line.split())
            if gt_point_y < img.shape[0] and gt_point_x < img.shape[1]:
                k[gt_point_y, gt_point_x] = 1
    # Generate density map
    k = gaussian_filter_density(k)

    # Save density map as .h5 file
    with h5py.File(img_path.replace('.jpg','.h5').replace('images', density_map_set), 'w') as hf:
        hf['density'] = k

# Test with 1st image
ori_img = Image.open(img_paths[0])
gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images', density_map_set),'r')
groundtruth = np.asarray(gt_file['density'])
np.sum(groundtruth)# don't mind this slight variation

plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
plt.imshow(ori_img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(groundtruth,cmap=CM.jet)
plt.title("Density Map")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(ori_img)
plt.imshow(groundtruth,cmap=CM.jet, alpha=0.6)
plt.title("Overlay")
plt.axis("off")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
