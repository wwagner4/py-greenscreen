import common as co
import create_training_data as ctd
import train_model as tm
import use_model as um

out_file = co.work_file('data-img100.csv')
model_file = co.work_file("model-img100.h5")

ctd.create(out_file)
tm.train(out_file, model_file)
um.use(model_file)
