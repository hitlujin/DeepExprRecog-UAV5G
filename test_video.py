import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
import json
# addr = 'http://355m892v19.51vip.biz:31614' # test domain:port


addr = 'http://127.0.0.1:5000'
test_url = addr + '/api/demo' 
content_type = 'image/jpeg'
headers = {'content-type': content_type}
cap = cv2.VideoCapture("./testdata/face_video.mp4")

while True:
    ret,img = cap.read()
    _, img_encode = cv2.imencode('.jpg',img) # compless image
    start = default_timer()
    response = requests.post(test_url,data=img_encode.tobytes(),headers=headers) # send request
    end = default_timer()
    # print('-- %s: %.6f 秒' % ("endtime", end - start))
    # print(response.text)
    result = json.loads(response.text)
    bbox = result['data']['tags'][0]['bbox']
    # print(response.text.data.tags[0])
    cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1,4)
    cv2.imshow('test',img)
    cv2.waitKey()