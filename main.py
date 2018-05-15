from gs import common as co
from gs import create_training_data as ctd
from gs import train_model as tm, use_model as um

out_file = co.work_file('data-img100.csv')
model_file = co.work_file("model-img100.h5")

_id = "img500"

ctd.create(_id)
tm.train(out_file, model_file)
um.use(model_file)
