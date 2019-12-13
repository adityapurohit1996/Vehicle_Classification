import numpy as np
import csv 
import tensorflow as tf
from keras.models import load_model, Model
import pdb

model = tf.keras.models.load_model('simple_nn.h5')
# model.summary()
# csv_name = 'test_labels_13_bounds_orig.csv'
csv_name = 'train_labels_12_bounds_orig.csv'
b_first_line = True
data = np.array([[]])
with open(csv_name) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        if b_first_line:
            data = np.array([line[1], line[2],line[3],line[4],line[5]])
            # pdb.set_trace()
            b_first_line = False
        else:
            data = np.vstack((data, [line[1], line[2],line[3],line[4],line[5]]))
one_hots = model.predict(data)
print(one_hots.shape)
# pdb.set_trace()
labels = [np.argmax(one_hot) for one_hot in one_hots]
print(labels)

output_csv_name = 'output_' + csv_name
cnt = 0
with open(output_csv_name, 'w') as f:
    writer = csv.writer(f, delimiter=',', lineterminator='\n')
    with open(csv_name) as csvfile:
        reader = csv.reader(csvfile)
        writer.writerow(['guid/image', 'label'])
        next(reader)
        for line in reader:    
            writer.writerow([line[1], line[2],line[3],line[4],line[5], labels[cnt]])
            cnt += 1
print('Wrote report file `{}`'.format(output_csv_name))