import os, os.path, sys
from glob import glob
cPath = os.getcwd()
time_gap = 1.0/8.0
print("time gap: " + str(time_gap))
print("loading: "+ sys.argv[1])
relative_path = sys.argv[1]


with open(relative_path + "/times.txt","w") as wfile:
    image_num = len([name for name in os.listdir(relative_path+'/image_0') if name.endswith(".png")])
    print(image_num)
    times = 0.0
    for i in range(image_num):
        num = "%06d"%i
        wfile.write(str(times)+"\t"+num+"\n")
        times += time_gap
