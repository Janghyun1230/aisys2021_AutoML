# aisys2021_AutoML
AutoML System team project page (AI system 2021 class).

The main objective of this project is to build an **automatic RL search algorithm** for network scaling. 

### Requirements 
These codes are tested with
```
PyTorch == 1.7.0
torchvision == 0.8.0
CUDA == 10.2
```


### To implement NAS scaling algorithm
```
python main.py -p [platform] -v [device] --latency [latency bound] --mem [memory bound] --id [process number]
```
**Arguments**  
- platforms: raspberrypi, jetson, desktop   
- device: cpu, cuda (only for jetson, desktop)
- latency: required latency bound in second (s)
- mem: required memory bound in MB 

If running multi-process for the certain experimental setting (platform, device, latency, mem bounds), each process should have different id.

### To test a certain model
```
python env.py -p [platform] -v [device] -w [width] -d [depth] -r [resolution] -e [training epoch]
```
This command will print model and return evaluation results including validation accuracy, latency, and memory.

### To test a search algorithgm on Toy Env
```
python test.py 
```
