import os
import imageio
path='imge/'
filenames=[]
for i in range(100,600):
        file=path+str(i)+'.jpg'
        if os.access(file, os.F_OK):
            filenames.append(file)
images=[]
for filename in filenames:
        images.append(imageio.imread(filename))
imageio.mimsave('paris.gif',images,duration=0.06)