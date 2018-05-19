from gs import create_training_data as ctd
from gs import train_model as tm
from gs import use_model as um

_id = "img100"

ctd.create(_id)
tm.train(_id)
um.use(_id)
