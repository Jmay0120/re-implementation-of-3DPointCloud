import numpy as np
import os.path as osp

from ..src.evaluation_metrics import minimum_mathing_distance, jsd_between_point_cloud_sets, coverage

from ..src.in_out import snc_category_to_synth_id, load_all_point_clouds_under_folder

top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.
class_name = input('Give me the class name (e.g. "chair"): ').lower()
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

n_ref = 100 # size of ref_pcs.
n_sam = 150 # size of sample_pcs.
all_ids = np.arange(all_pc_data.num_examples)
ref_ids = np.random.choice(all_ids, n_ref, replace=False)
sam_ids = np.random.choice(all_ids, n_sam, replace=False)
ref_pcs = all_pc_data.point_clouds[ref_ids]
sample_pcs = all_pc_data.point_clouds[sam_ids]

ae_loss = 'chamfer'  # Which distance to use for the matchings.

if ae_loss == 'emd':
    use_EMD = True
else:
    use_EMD = False  # Will use Chamfer instead.

batch_size = 100  # Find appropriate number that fits in GPU.
normalize = True  # Matched distances are divided by the number of
# points of thepoint-clouds.

mmd, matched_dists = minimum_mathing_distance(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD)

cov, matched_ids = coverage(sample_pcs, ref_pcs, batch_size, normalize=normalize, use_EMD=use_EMD)

jsd = jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28)

print(mmd, cov, jsd)

print(coverage.__doc__)
print(minimum_mathing_distance.__doc__)
print(jsd_between_point_cloud_sets.__doc__)