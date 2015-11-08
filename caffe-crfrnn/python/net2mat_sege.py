# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on
top of the Caffe deep learning library.

Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Par$

Supervisor:
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""
def main(argv):
    caffe_root = '../'
    import sys
    sys.path.insert(0, caffe_root + 'python')
    import exifutil 
    import os
    import cPickle
    import logging
    import numpy as np
    import pandas as pd
    from PIL import Image
    #import Image
    import cStringIO as StringIO
    import caffe
    import matplotlib.pyplot as plt
    import scipy.io as sio
    MODELNAME = argv[1]
    if MODELNAME == 'fcn8':
      MODEL = 'fcn-8s-pascal'
    elif MODELNAME == 'fcn32':
      MODEL = 'fcn-32s-pascal'

    #MODEL_FILE = 'TVG_CRFRNN_COCO_VOC.prototxt'
    #PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
    MODEL_FILE = '../models/'+MODEL+'/'+'deploy.prototxt'
    PRETRAINED = '../models/'+MODEL+'/'+MODEL + '.caffemodel'
    IMAGE_FILE = '../../../caff/examples/images/'+ argv[2]+'.jpg'
    MATFILE = '../../../caff/python/_temp/' + MODELNAME + '_' +argv[2]+'.mat'
    MATFILE_PARA = '../../../caff/python/_temp/' + 'paras_'+MODELNAME+'_'+argv[1]+'.mat'
    outputImage = '../../../caff/python/_temp/' + MODELNAME + '_'+argv[1]+'.png'
    #caffe.set_mode_gpu()
    net = caffe.Segmenter(MODEL_FILE, PRETRAINED)

   # input_image = 255 * caffe.io.load_image(IMAGE_FILE)
    input_image = 255 * exifutil.open_oriented_im(IMAGE_FILE)

    #width = input_image.shape[0]
    #height = input_image.shape[1]
    #maxDim = max(width,height)

    #image = PILImage.fromarray(np.uint8(input_image))
    #image = np.array(image)

    pallete = [0,0,0,128,0,0,0,128,0,128,128,0,0,0,128,128,0,128,0,128,128,128,128,128,64,0,0,192,0,0,64,128,0,192,128,0,64,0,128,192,0,128,64,128,128,192,128,128,0,64,0,128,64,0,0,192,0,128,192,0,0,64,128,128,64,128,0,192,128,128,192,128,64,64,0,192,64,0,64,192,0,192,192,0]

    # Mean values in BGR format
    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

    # Rearrange channels to form BGR
    im = input_image[:,:,::-1]
    # Subtract mean
    im = im - reshaped_mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = 500 - cur_h
    pad_w = 500 - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

    # Get predictions
    segmentation = net.predict([im])
    #segmentation2 = segmentation[0:cur_h, 0:cur_w]
    output_im = Image.fromarray(segmentation)
    output_im.putpalette(pallete)

    # plt.imshow(output_im)
    #plt.savefig(outputImage)
    output_im.save(outputImage)

    collectDic = {}
    for keys,values in net.blobs.items():
	print 'collecting...'
        collectDic[keys] = values.data

    collectDic['segmentationResult'] = segmentation

    sio.savemat('MATFILE',collectDic)

    collectDicPara = {}
    for keys,values in net.params.items():
        para = values
        for i in range(0,len(para)):
	    print 'collecting...'
	    layerName = keys + str(i)
            collectDicPara[layerName] = para[i].data

   # sio.savemat('MATFILE_PARA',collectDicPara)
    print('done')

if __name__ == '__main__':
  import sys
  main(sys.argv)



