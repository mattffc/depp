#test talk to our local server and send it an image

import requests
import cv2
import os


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\mattf\Downloads\ml-bp-project-17caf281d604'


url = "https://helloworld-nxcbafywma-oa.a.run.app"#'http://localhost:80/test2'
files = {'media': open('depp1.jpeg', 'rb')}
'''
requests.post(url, files=files)
'''

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
'''
img = open('depp1.jpeg', 'rb').read()
response = requests.post(url, data=img, headers=headers)
'''
ontent_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('depp1.jpeg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(url, data=img_encoded.tostring(), headers=headers)
print(response)