import os
import random

# train_project_num = 9000
# valid_project_num = 200
# test_project_num = 1022

# projects = os.listdir("/data2/liufang/datasets_javam/icse_java")


projects = open("/data2/liufang/datasets_javam/repos.txt",'r').readlines()
n = len(projects)
print('total projects: {}'.format(n))
random.shuffle(projects)

wf = open("/data2/liufang/datasets_javam/train_projects.txt", "w")
for x in projects[:9000]:
    wf.write(x)
wf.close()

wf = open("/data2/liufang/datasets_javam/eval_projects.txt", "w")
for x in projects[9000:9200]:
    wf.write(x)
wf.close()

wf = open("/data2/liufang/datasets_javam/test_projects.txt", "w")
for x in projects[9200:]:
    wf.write(x)
wf.close()

train_projects = open("/data2/liufang/datasets_javam/train_projects.txt", "r").readlines()
wf = open("/data2/liufang/datasets_javam/small_train_projects.txt", "w")
for x in train_projects[:4000]:
    wf.write(x)
wf.close()
