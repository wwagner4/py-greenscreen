from gs import create_training_data as ctd
from gs import train_model as tm
from gs import use_model as um
from gs import config as cf
import time
import sys
import os.path as osp

if len(sys.argv) != 1:
    print("One arguments required. 'id'. Possible values 'img100', 'img500', ...")

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
root_dir = osp.dirname(sys.argv[0])
_id = sys.argv[1]

print("Arguments")
print("   id        : {}".format(_id))
print("   rootdir   : {}".format(root_dir))
print("   timestamp : {}".format(timestamp))

cfg = cf.conf(_id, root_dir)

ctd.create(cfg)
tm.train(cfg)
um.use(cfg, timestamp)
