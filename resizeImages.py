#depp learning resize images to squares, get in same format as the cactus ones

#take in directory of images

#loop over each image

# resize the image via strech to desired size

# save image to new folder called resized images
import os
from PIL import Image
import matplotlib.pyplot as plt
pathToImages = r"C:\Users\mattf\Downloads\man _ Google Search"
pathToSave = r"C:\Users\mattf\OneDrive\Documents\deppLearning\johnnyDepp1"
i = 0
for filename in os.listdir(pathToImages):
    print(filename)
    print(str(i),"out of ",str(len(os.listdir(pathToImages))))
    i += 1
    im = Image.open(os.path.join(pathToImages,filename))
    
    new_image = im.resize((100, 100))
    
    new_image.save(os.path.join(pathToImages,filename))