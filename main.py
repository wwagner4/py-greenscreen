from gs import create_training_data as ctd
from gs import train_model as tm
from gs import use_model as um
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

ctd.create(_id, root_dir)
tm.train(_id)
um.use(_id, timestamp, root_dir)
