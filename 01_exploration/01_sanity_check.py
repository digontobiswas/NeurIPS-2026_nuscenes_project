from nuscenes.nuscenes import NuScenes

DATAROOT = 'D:/nuscenes_project/data/nuscenes'
VERSION  = 'v1.0-mini'

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

print('=' * 50)
print('DATASET SUMMARY')
print('=' * 50)
print(f'Version       : {VERSION}')
print(f'Scenes        : {len(nusc.scene)}')
print(f'Samples       : {len(nusc.sample)}')
print(f'Annotations   : {len(nusc.sample_annotation)}')
print(f'Categories    : {len(nusc.category)}')
print(f'Instances     : {len(nusc.instance)}')
print(f'Sensors       : {len(nusc.sensor)}')
print('=' * 50)

print('\nALL SCENES:')
print(f"{'#':<5} {'Name':<15} {'Frames':<10} {'Description'}")
print('-' * 70)
for i, scene in enumerate(nusc.scene):
    print(f"{i:<5} {scene['name']:<15} {scene['nbr_samples']:<10} {scene['description'][:40]}")

print('\nALL CATEGORIES:')
print(f"{'#':<5} {'Category Name'}")
print('-' * 40)
for i, cat in enumerate(nusc.category):
    print(f"{i:<5} {cat['name']}")