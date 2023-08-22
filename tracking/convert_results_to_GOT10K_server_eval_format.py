import os
import shutil
import glob

import numpy as np

src_dir = os.path.join('/home/goutam/VisualTracking/research_code_for_github/MVT/output/test/tracking_results/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing', 'got10k')
dst_dir = os.path.join('/home/goutam/VisualTracking/research_code_for_github/MVT/output/test/tracking_results/mobilevit_track/mobilevit_256_128x1_got10k_ep100_cosine_annealing', 'got10k_eval_server')

got10k_test_seqs = list(range(1, 181))

for seq_id in got10k_test_seqs:

    if os.path.exists(os.path.join(dst_dir, 'GOT-10k_Test_' + str(seq_id).zfill(6))) is False:
        os.makedirs(os.path.join(dst_dir, 'GOT-10k_Test_' + str(seq_id).zfill(6)))

    for seq_result in glob.glob(os.path.join(src_dir, "*" + str(seq_id).zfill(6) + "*")):
        print(seq_result)
        if "time" in seq_result:
            shutil.copy(seq_result, os.path.join(dst_dir, 'GOT-10k_Test_' + str(seq_id).zfill(6),
                                                 seq_result.split('/')[-1]))
        else:
            try:
                seq_bboxes = np.loadtxt(seq_result)
            except:
                seq_bboxes = np.loadtxt(seq_result, delimiter=',')
            np.savetxt(os.path.join(dst_dir, 'GOT-10k_Test_' + str(seq_id).zfill(6),
                                    seq_result.split('/')[-1].split('.')[0] + '_' + str(1).zfill(3) + '.txt'),
                       seq_bboxes, delimiter=',', fmt='%i')
