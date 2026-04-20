from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

DATAROOT   = 'D:/nuscenes_project/data/nuscenes'
OUTPUT_DIR = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

nusc         = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
my_scene     = nusc.scene[0]
sample_token = my_scene['first_sample_token']
sample       = nusc.get('sample', sample_token)

cameras = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle(f"Scene: {my_scene['name']} — All 6 Cameras", fontsize=14)

for idx, cam in enumerate(cameras):
    row = idx // 3
    col = idx  % 3

    cam_token = sample['data'][cam]
    cam_data  = nusc.get('sample_data', cam_token)
    img_path  = os.path.join(DATAROOT, cam_data['filename'])

    img = mpimg.imread(img_path)
    axes[row][col].imshow(img)
    axes[row][col].set_title(cam, fontsize=10)
    axes[row][col].axis('off')

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'cameras_scene0.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()