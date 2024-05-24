import subprocess
import os

run_file = "../main.py"

#--save_name
#--ttrank

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cmd_list = []
seed_list = ["1111", "2222", "3333"] #
models = ["resnet18", "mobilenet_v2", "wresnet50"] #"densenet121", "resnext50", 
data_list = ["cifar10", "cifar100"] #
lr_list = ["0.0001"]

for data in data_list:
    for lr in lr_list:
        for model in models:
            for s in seed_list:
                for r in range(10):
                    rr = str(r)

                    fr = str(4)
                    cmd_list.append(["python", run_file, "-dn", data, "-flr", lr, "-fo", "1", "--model", model, "--seed", s, "-rr", rr, "-fr", fr, "-rl", "0"])

                    fr = str(3)
                    cmd_list.append(["python", run_file, "-dn", data, "-flr", lr, "-fo", "1", "--model", model, "--seed", s, "-rr", rr, "-fr", fr, "-rl", "0"])

for i, cmd in enumerate(cmd_list):
    p = subprocess.Popen(cmd)
    p.wait()
print('completed!')