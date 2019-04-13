# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob


class dataProcess(object):
    def __init__(self, out_rows, out_cols, train_path="content/segnet/CamVid/train", train_label="content/segnet/CamVid/trainannot",
                 val_path="content/segnet/CamVid/val", val_label="content/segnet/CamVid/valannot",
                 test_path="content/segnet/CamVid/test", test_label='content/segnet/CamVid/testannot', npy_path="content/unet-rgb/npydata", img_type="png"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.train_path = train_path
        self.train_label = train_label
        self.img_type = img_type
        self.val_path = val_path
        self.val_label = val_label
        self.test_path = test_path
        self.test_label = test_label
        self.npy_path = npy_path

    def label2class(self, label):
        x = np.zeros([self.out_rows, self.out_cols, 19])
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                x[i, j, int(label[i][j])] = 1  # 属于第m类，第三维m处值为1
        return x

    def create_train_data(self):
        i = 0
        print('Creating training images...')
        imgs0 = sorted(glob.glob(self.train_path+"/*."+self.img_type))
        imgs1 = sorted(glob.glob(self.test_path+"/*."+self.img_type))
        imgs = imgs0 + imgs1
        labels0 = sorted(glob.glob(self.train_label+"/*."+self.img_type))
        labels1 = sorted(glob.glob(self.test_label + "/*." + self.img_type))
        labels = labels0 + labels1
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, 19), dtype=np.uint8)
        print(len(imgs), len(labels))

        for x in range(len(imgs)):
            imgpath = imgs[x]
            labelpath = labels[x]
            img = load_img(imgpath, grayscale=False)

            label = load_img(labelpath)
            print('image '+labelpath+' loaded')
            img = img_to_array(img)
            a = self.decode_segmap(img_to_array(label))
            label = self.label2class(a)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        print('loading done')
        np.save('/content/unet-rgb/npydata/camvid_train.npy', imgdatas)
        np.save('content/unet-rgb/npydata/camvid_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('Creating test images...')
        imgs = glob.glob(self.val_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False)
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1

        txtname = './results/camvid.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('loading done')
        np.save('content/unet-rgb/npydata/camvid_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('load train images...')
        imgs_train = np.load(self.npy_path + "/camvid_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/camvid_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        imgs_mask_train /= 255
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/camvid_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        return imgs_test


    def decode_segmap(self, temp, plot=False):
        Sky = [128, 128, 128]
        Building=[128,0,0]
        Pole = [192, 192, 128]
        Road = [128, 64, 128]
        Pavement = [60, 40, 222]
        Tree = [128, 128, 0]
        SignSymbol = [192, 128, 128]
        Fence = [64, 64, 128]
        Car=[64,0,128]
        Pedestrian = [64, 64, 0]
        Bicyclist = [0, 128, 192]
        Unlabelled=[0,0,0]
        Animal=[64,128,64]
        Archway=[192,0,128]
        Bridge = [0,128,64]
        CartLuggagePram=[64,0,192]
        Child=[192,1128,64]
        Column_Pole=[ 192 ,192,   128]

        label_colours = np.array(
            [
                Sky,
                Building,
                Pole,
                Road,
                Pavement,
                Tree,
                SignSymbol,
                Fence,
                Car,
                Pedestrian,
                Bicyclist,
                Unlabelled,
                Bridge,
                Animal,
                Archway,
                Bridge,
                CartLuggagePram,
                Child,
                Column_Pole,

            ]
        )
        a,b,c=np.shape(temp)
        categorized = np.zeros((a,b))
        for i in range (np.shape(temp)[0]):
            for j in range (np.shape(temp)[1]):
                for m in range (19):
                    if (temp[i][j]==label_colours[m]).all():
                        categorized[i][j]=m


        return categorized

if __name__ == "__main__":
    mydata = dataProcess(720 ,960)
    mydata.create_train_data()
    mydata.create_test_data()
