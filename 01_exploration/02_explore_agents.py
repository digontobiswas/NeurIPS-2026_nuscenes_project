from nuscenes.nuscenes import NuScenes
from collections import Counter

DATAROOT = 'D:/nuscenes_project/data/nuscenes'

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

my_scene = nusc.scene[0]
print(f"Scene      : {my_scene['name']}")
print(f"Description: {my_scene['description']}")
print(f"Frames     : {my_scene['nbr_samples']}")
print()

sample_token   = my_scene['first_sample_token']
frame_count    = 0
all_categories = []

print(f"{'Frame':<8} {'Timestamp':<22} {'Agents':<10} {'Categories'}")
print('-' * 70)

while sample_token:
    sample      = nusc.get('sample', sample_token)
    annotations = sample['anns']

    cats = []
    for ann_token in annotations:
        ann = nusc.get('sample_annotation', ann_token)
        cats.append(ann['category_name'].split('.')[1])
        all_categories.append(ann['category_name'])

    cat_summary = ', '.join(sorted(set(cats)))
    print(f"{frame_count:<8} {sample['timestamp']:<22} {len(annotations):<10} {cat_summary}")

    frame_count  += 1
    sample_token  = sample['next']

print()
print('AGENT CATEGORY BREAKDOWN:')
print('-' * 40)
counts = Counter(all_categories)
for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<40} {count:>5} instances")