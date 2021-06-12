# aisys2021_AutoML
AutoML System team project page (AI system 2021 class)

### To implement NAS scaling algorithm
```
python main.py -p [platform] -v [device]
```
platforms: raspberrypi, jetson, desktop   
device: cpu, cuda (only for jetson, destop)

### To test a certain model
```
python env.py -p [platform] -v [device] -w [width] -d [depth] -r [resolution]
```

### To test a search algorithgm on Toy Env
```
python test.py 
```