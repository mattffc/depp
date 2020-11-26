#relabel images
import os
pathToImages = r"C:\Users\mattf\OneDrive\Documents\deppLearning\notJohnny"
i = 753
for filename in os.listdir(pathToImages):
    print(filename)
    os.rename(os.path.join(pathToImages,filename), os.path.join(pathToImages,"img_"+str(i)+".jpeg"))
    i += 1
    