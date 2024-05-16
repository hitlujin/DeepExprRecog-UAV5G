import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer

# addr = 'http://355m892v19.51vip.biz:31614' # test domain:port
addr = 'http://127.0.0.1:5000'
test_url = addr + '/api/facefeatureextract?videoId=1' 
video_url = addr+'/api/videofeatureextract'
# test_url = addr + '/api/regitface' 
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('e1.jpg')
_, img_encode = cv2.imencode('.jpg',img) # compless image
start = default_timer()
response = requests.post(test_url,data=img_encode.tobytes(),headers=headers) # send request
end = default_timer()
print('-- %s: %.6f 秒' % ("endtime", end - start))
print(response.text)


# img = cv2.imread('tiandi.jpg')
# _, img_encode = cv2.imencode('.jpg',img) # compless image
# start = default_timer()
# params = {"name":"赖工"}
# response = requests.post(test_url,data=img_encode.tobytes(),params=params,headers=headers) # send request
# end = default_timer()
# print('-- %s: %.6f 秒' % ("endtime", end - start))
# print(response.text)


# # img = cv2.imread('tiandi.jpg')
# # _, img_encode = cv2.imencode('.jpg',img) # compless image
# start = default_timer()
# # params = {"name":"赖工"}
# response = requests.post(video_url) # send request
# end = default_timer()
# print('-- %s: %.6f 秒' % ("endtime", end - start))
# print(response.text)