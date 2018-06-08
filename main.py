from gs import create_training_data as ctd
from gs import train_model as tm
from gs import train_model_opt as tmo
from gs import use_model as um
from gs import config as cf
from gs import plot as pl
import time
import sys
import os.path as osp


def fullRun():
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


def create():
    root_dir = osp.dirname(sys.argv[0])
    work_dir = "C:/ta30/entw1/work/work-greenscreen"
    _id = 'img500'

    print("Arguments")
    print("   id        : {}".format(_id))
    print("   rootdir   : {}".format(root_dir))
    print("   workdir   : {}".format(work_dir))
    cfg = cf.conf(_id, root_dir)
    ctd.create(work_dir, cfg)
    print("FINISHED")


def opt():
    tmo._run()


def plot():
    pl._tryout_multi()
    pl._tryout()
    
# plot()
# opt()
create()
