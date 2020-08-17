import os, os.path, sys
cPath = os.getcwd()
time_gap = 1.0/8.0
print("time gap: " + str(time_gap))
print("loading: "+ sys.argv[1])
relative_path = sys.argv[1]

checklist = os.listdir(relative_path)
if any("image_left" in s for s in checklist):
    os.rename(relative_path +"/image_left",relative_path +"/image_0")
if any("image_right" in s for s in checklist):
    os.rename(relative_path +"/image_right",relative_path +"/image_1")

with open(relative_path + "/pose_left.txt","r") as rfile:
    with open(relative_path + "/pose_left_times.txt","w") as wfile:
        gttimes = 0.0
        for line in rfile:
            wfile.write( str(gttimes)+' '+line)
            gttimes += time_gap

with open(relative_path + "/times.txt", 'w') as file:
    image_num = len([name for name in os.listdir(relative_path+'/image_0') if name.endswith(".png")])
    print(image_num)
    times = 0.0
    for i in range(image_num):

        num = "%06d"%i
        old_path_name1= relative_path + "/image_0/"+num+"_left.png"
        new_path_name1= relative_path + "/image_0/"+num+".png"
        old_path_name2= relative_path + "/image_1/"+num+"_right.png"
        new_path_name2= relative_path + "/image_1/"+num+".png"
        try:
            os.rename(old_path_name1,new_path_name1)
        except:
            pass
        try:
            os.rename(old_path_name2,new_path_name2)
        except:
            pass
        file.write(str(times)+"\t"+num+"\n")
        times+=time_gap
    print(times)
