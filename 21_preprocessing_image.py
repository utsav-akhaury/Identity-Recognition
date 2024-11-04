from pathlib import Path
import numpy as np
from skimage.transform import resize
import dlib

datadir = Path('datadir')
filename = datadir / 'audVisIdn.npz'



################ use this class to make your models,
################ see `trialFred` above for an example.
class dataHolder():
    def __init__(self):
        self.prepareData(f=filename)
        
    def prepareData(self, f):
        f = np.load(filename)
        self.trainimg = f['imgFrames_train'] / 255.
        self.valimg = f['imgFrames_val'] / 255.
        self.testimg = f['imgFrames_test'] / 255.
        
  

# Load the detector
detector = dlib.get_frontal_face_detector()
FINALSIZE = 160

g = dataHolder()

def extractFace(index, dataset='train'):
    if dataset == 'train':
        img = g.trainimg[index,:,:,:]
    if dataset == 'test':
        img = g.testimg[index,:,:,:]
    if dataset == 'val':
        img = g.valimg[index,:,:,:]
        
    faces = detector((255*img).astype(np.uint8))
    
    for face in faces[:1]:
        x1 = face.left() # left point
        y1 = face.top() # top point
        x2 = face.right() # right point
        y2 = face.bottom() # bottom point
        
        lenx = (x2-x1)
        leny = (y2-y1)
        diff = lenx - leny
        # make the crop a square:
        if diff > 0:
            x1 += diff // 2 
            x2 -= diff // 2
        if diff < 0 :
            y1 += diff // 2
            y2 -= diff // 2
    if len(faces) == 0:
        # well....then no crop.
        imgr = img
    else:
        # found a face? draw a square.
        imgr = img[y1:y2, x1:x2]
        
    try:
        imgr = resize(imgr, (FINALSIZE, FINALSIZE))
    except:
        imgr = resize(img, (FINALSIZE, FINALSIZE))
        
    mean, std = np.nanmean(imgr), np.nanstd(imgr)
    imgr =  (imgr - mean) / std

    return imgr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    alls = []
    labs = []
    for i in range(g.trainimg.shape[0]):
        try:
            img = extractFace(i, 'train')
            alls.append(img)
            labs.append(g.trainlabelsnorm[i])
        except Exception as e:
            print(e)
            plt.imshow(g.trainimg[i])
            plt.waitforbuttonpress()
            
        print(i)
    np.save(datadir / f'train_faces_only_{FINALSIZE}.npy', alls)
    np.save(datadir / f'train_lab_faces_only_{FINALSIZE}.npy', labs)
    
    alls = []
    labs = []
    for i in range(g.testimg.shape[0]):
        try:
            img = extractFace(i, 'test')
            alls.append(img)
            labs.append(g.testlabelsnorm[i])
        except:
            pass
        print('test', i)
    np.save(datadir / f'test_faces_only_{FINALSIZE}.npy', alls)
    np.save(datadir / f'test_lab_faces_only_{FINALSIZE}.npy', labs)

    alls = []
    labs = []
    for i in range(g.valimg.shape[0]):
        try:
            img = extractFace(i, 'val')
            alls.append(img)
            labs.append(g.vallabelsnorm[i])
        except:
            pass
        print('val', i)
    np.save(datadir / f'val_faces_only_{FINALSIZE}.npy', alls)
    np.save(datadir / f'val_lab_faces_only_{FINALSIZE}.npy', labs)