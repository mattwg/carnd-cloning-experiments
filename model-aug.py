from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam 
from keras.models import model_from_json
from itertools import zip_longest
import cv2
import numpy as np
import csv, argparse
import os, errno

def file_len(fname):
    i = -1
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def remove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise
            

def clean_driving_log(logfile, img_dir):
    with open(logfile,'r') as r, open('clean_log.csv','w') as w:
        reader = csv.reader(r, delimiter=',')
        writer = csv.writer(w, delimiter=',')
        for row in reader:
            imgc_file = img_dir + row[0].strip()
            imgl_file = img_dir + row[1].strip()
            imgr_file = img_dir + row[2].strip()
            if (os.path.isfile(imgc_file) & os.path.isfile(imgl_file) & os.path.isfile(imgr_file)):
                writer.writerow(row)
                
                
def split_driving_log(f, train_percent, seed=1973):
    ft = open('train_log.csv','w')
    fv = open('valid_log.csv','w')
    np.random.seed(seed)
    with open(f, 'r') as f:
        for line in f:
            if (np.random.random() <= train_percent):
                ft.write(line)
            else:
                fv.write(line)
    ft.close()
    fv.close()

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def load_image(f):
    img = cv2.imread(f,-1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img 
    
def extract_csv(log):
    f = []
    y = []
    with open(log,'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            f.append(row[0].strip())
            y.append(np.float32(row[3]))
            f.append(row[1].strip())
            y.append(np.float32(row[3])+0.2)
            f.append(row[2].strip())
            y.append(np.float32(row[3])-0.2)
    return f, y

def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def preprocess_image(img):
    img = img[60:140,:,:]
    img = cv2.resize(img,(64, 64))
    return(img)
    
def generate_image(img, y):
    
    angle = y

    X_OFFSET_RANGE = 10
    Y_OFFSET_RANGE = 10
    X_OFFSET_ANGLE = 0.2
    
    img = preprocess_image(img)
    
    bright_factor = 0.2 + (3.8 * np.random.uniform())
    img = gamma_correction(img, bright_factor)

    if (np.random.uniform() > 1.0):
        img = np.fliplr(img)
        angle = -1.0 * angle

    x_translation = (X_OFFSET_RANGE * np.random.uniform()) - (X_OFFSET_RANGE / 2)
    y_translation = (Y_OFFSET_RANGE * np.random.uniform()) - (Y_OFFSET_RANGE / 2)

    angle = angle + ((x_translation / X_OFFSET_RANGE) * 2) * X_OFFSET_ANGLE
    t = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    img = cv2.warpAffine(img, t, (img.shape[1], img.shape[0]))

    return (img, angle)

            
def generator_random_zero(n, img_dir, logfile):
    f, y = extract_csv(logfile)
    l = len(f)
    while True:
        xs = []
        ys = []
        for _ in range(n):
            i = np.random.randint(low=0,high=l)
            if ( (np.float32(y[i]) >= -0.01 
                    and np.float32(y[i]) <= 0.01 ) 
                    and np.random.uniform() > 0.1 ):
                while (np.float32(y[i]) >= -0.01 
                       and np.float32(y[i]) <= 0.01):
                    i = np.random.randint(low=0,high=l)
            elif ( (np.float32(y[i]) >= -0.21 
                    and np.float32(y[i]) <= -0.19 ) 
                    and np.random.uniform() > 0.1 ):
                while (np.float32(y[i]) >= -0.21 
                       and np.float32(y[i]) <= -0.19):
                    i = np.random.randint(low=0,high=l)
            elif ( (np.float32(y[i]) >= 0.19 
                    and np.float32(y[i]) <= 0.21 ) 
                    and np.random.uniform() > 0.1 ):
                while (np.float32(y[i]) >= 0.19 
                       and np.float32(y[i]) <= 0.21):
                    i = np.random.randint(low=0,high=l)
            img, angle = generate_image(load_image(img_dir+f[i]),np.float32(y[i]))
            xs.append(img)
            ys.append(y[i])
        yield (np.asarray(xs), np.asarray(ys))

def generator_all_batch(n, img_dir, logfile):
    f, y = extract_csv(logfile)
    l = len(f)
    while True:
        xs = []
        ys = []
        for i in range(0,l,n):
            xs.append(preprocess_image(load_image(img_dir+f[i])))
            ys.append(np.float32(y[i]))
            yield (np.asarray(xs), np.asarray(ys))

        
def get_model():
    ch, row, col = 3, 64, 64  

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=( row, col, ch),
            output_shape=( row, col, ch)))
    model.add(Convolution2D(32, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))   
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dense(64))
    model.add(ELU())
    model.add(Dense(16))
    model.add(ELU())
    model.add(Dense(1))
    return model


parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--model', type=str, help='Optional path to model to continue training')
args = parser.parse_args()

image_dir = 'data/images_run01/'

if args.model:
    print('Loading model for continued training!')
    with open(args.model, 'r') as f:
        json = f.read()
        model = model_from_json(json)
else:
    print('Training new model!')
    model = get_model()
    remove('clean_log.csv')
    remove('train_log.csv')
    remove('valid_log.csv')
    clean_driving_log('data/driving_log_3.csv', image_dir)
    split_driving_log('clean_log.csv',0.5)

n_train = file_len('train_log.csv') * 3 
n_valid = file_len('valid_log.csv') * 3

print(n_train, n_valid)

early_stopping_patience = 10
epochs_since_better = 0
n_epochs = 20
continue_training = True
consider_stopping = False
n_epoch = 0

batch_size = 1
n_batches = n_train / batch_size

model.compile(optimizer=Adam(lr=0.00001), loss="mse")

if args.model:
    model.load_weights('model.h5')
    best_mse = model.evaluate_generator(
        generator_all(image_dir,'valid_log.csv'),
        val_samples = 1)
else:
    best_mse = float("inf")
    
while n_epoch < n_epochs and continue_training:
    
    model.fit_generator(
        generator_random_zero(batch_size, image_dir,'train_log.csv'),
        samples_per_epoch=n_batches,
        nb_epoch=1, verbose=0)

    mse = model.evaluate_generator(
        generator_all_batch(2,image_dir,'valid_log.csv'),
        val_samples = 1)
    
    print(mse)
    
    # Early stopping?
    if mse > best_mse:    
        if consider_stopping:
            epochs_since_better += 1
        else:
            consider_stopping = True
            epochs_since_better = 1
    else:
        print('Improved accuracy of {} at epoch {}'.format(mse,n_epoch))
        best_mse = mse
        json = model.to_json()
        with open('model.json', 'w') as f:
            f.write(json)  
        model.save_weights('model.h5')
        consider_stopping = False

    if epochs_since_better > early_stopping_patience:
        print('Stopping no improvement for {} epochs'.format(early_stopping_patience))
        continue_training = False

    
    n_epoch += 1
