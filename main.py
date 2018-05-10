import os
import os.path as osp
from pathlib import Path

import use_model as um

home_dir = Path.home()
work_dir = osp.join(home_dir, 'work', 'work-greenscreen')
print("work_dir: '{}'".format(work_dir))
if not osp.exists(work_dir):
    os.makedirs(work_dir)
out_file = osp.join(work_dir, 'data-img100.csv')

model_file = osp.join(work_dir, "model-img100.h5")

# ctd.create(out_file)
# tm.train(out_file, model_file)
um.use(model_file)
