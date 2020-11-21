import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks



from http.server import BaseHTTPRequestHandler, HTTPServer
import time




from PIL import Image
from PIL import ImageDraw
import os
#import dlib.cuda
import numpy
import base64
import json
import time
#from compress_pickle import dump, load
#print(dlib.DLIB_USE_CUDA)



import io

import PIL.ImageDraw

_G, _D, Gs = pretrained_networks.load_networks('stylegan2-ffhq-config-f.pkl')
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
Gs_kwargs.truncation_psi = 1.0
rnd = np.random.RandomState()
z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
g = {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}
z = np.ones((1,512))
tflib.set_vars(g) # [height, width]
images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
#plt.figure(figsize=(15, 15))
#plt.rcParams["image.interpolation"] = "lanczos"
img = PIL.Image.fromarray(images[0], 'RGB').resize((256,256),PIL.Image.LANCZOS)    







hostName = ""
serverPort = 8083



class MyServer(BaseHTTPRequestHandler):
    
    def do_GET(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()
            p = self.path

            
            z = np.ones((1,512))

            for i in range(0,512):
                if p[max(len(p)-i-1,0)] == '0':
                    z[0][i]=-1

            tflib.set_vars(g) # [height, width]
            images = Gs.run(z, None, **Gs_kwargs)
            img = PIL.Image.fromarray(images[0], 'RGB')

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            self.wfile.write(img_byte_arr)
    
if __name__ == "__main__":        
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")