from gs import create_training_data as ctd
from gs import train_model as tm
from gs import use_model as um
import time

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
_id = "img100"

ctd.create(_id)
tm.train(_id)
um.use(_id, timestamp)
