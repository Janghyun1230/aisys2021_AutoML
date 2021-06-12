import yaml
from parse import compile
from os import listdir
from os.path import isfile, join, abspath
import pickle
import statistics

eff_block_0 = compile(
"""({:d}, {:d}, MBConvBlock(
  (_expand_conv): Conv2dStaticSamePadding(
    {:d}, {:d}, kernel_size={}, stride={}, bias=False
    (static_padding): Identity()
  )
  (_bn0): BatchNorm2d({:d}, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_depthwise_conv): Conv2dStaticSamePadding(
    {:d}, {:d}, kernel_size={}, stride={}, groups={}, bias=False
    (static_padding): ZeroPad2d(padding={}, value=0.0)
  )
  (_bn1): BatchNorm2d({:d}, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_se_reduce): Conv2dStaticSamePadding(
    {:d}, {:d}, kernel_size={}, stride={}
    (static_padding): Identity()
  )
  (_se_expand): Conv2dStaticSamePadding(
    {:d}, {:d}, kernel_size={}, stride={}
    (static_padding): Identity()
  )
  (_project_conv): Conv2dStaticSamePadding(
    {:d}, {:d}, kernel_size={}, stride={}, bias=False
    (static_padding): Identity()
  )
  (_bn2): BatchNorm2d({:d}, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_swish): MemoryEfficientSwish()
))""")

eff_block_1 = compile(
"""({}, {}, MBConvBlock(
  (_depthwise_conv): Conv2dStaticSamePadding(
    {}, {}, kernel_size={}, stride={}, groups={}, bias=False
    (static_padding): ZeroPad2d(padding={}, value=0.0)
  )
  (_bn1): BatchNorm2d({}, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_se_reduce): Conv2dStaticSamePadding(
    {}, {}, kernel_size={}, stride={}
    (static_padding): Identity()
  )
  (_se_expand): Conv2dStaticSamePadding(
    {}, {}, kernel_size={}, stride={}
    (static_padding): Identity()
  )
  (_project_conv): Conv2dStaticSamePadding(
    {}, {}, kernel_size={}, stride={}, bias=False
    (static_padding): Identity()
  )
  (_bn2): BatchNorm2d({}, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True)
  (_swish): MemoryEfficientSwish()
))""")

eff_avg = compile(
"""({}, {}, AdaptiveAvgPool2d(output_size={}))""")

eff_norm = compile(
"""({}, {}, BatchNorm2d({}, eps=0.001, momentum=0.010000000000000009, affine=True, track_running_stats=True))""")

eff_drop = compile(
"""({}, {}, Dropout(p={}, inplace=False))""")

eff_id = compile("""({}, {}, Identity())""")

eff_line = compile("""({}, {}, Linear(in_features={}, out_features={}, bias=True))""")

eff_swish = compile("""({}, {}, MemoryEfficientSwish())""")

eff_zero = compile("""({}, {}, ZeroPad2d(padding={}, value=0.0))""")

eff_blocks = {"EFF_BLOCK_0":eff_block_0, "EFF_BLOCK_1":eff_block_1, "EFF_AVG":eff_avg,
        "EFF_NORM":eff_norm, "EFF_DROP":eff_drop, "EFF_ID":eff_id, "EFF_LINE":eff_line,
        "EFF_SWISH":eff_swish, "EFF_ZERO":eff_zero}

pre_block_0 = compile("""({a0:d}, {a1:d}, PreActBlock(
  (bn1): BatchNorm2d({a2:d}, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv1): Conv2d({a3:d}, {a4:d}, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d({a5:d}, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d({a6:d}, {a7:d}, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
))""")

pre_block_1 =\
compile("""({b0:d}, {b1:d}, PreActBlock(
  (bn1): BatchNorm2d({b2:d}, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv1): Conv2d({b3:d}, {b4:d}, kernel_size=(3, 3), stride=({b5:d}, {b6:d}), padding=(1, 1), bias=False)
  (bn2): BatchNorm2d({b7:d}, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d({b8:d}, {b9:d}, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (shortcut): Sequential(
    (0): Conv2d({b10:d}, {b11:d}, kernel_size=(1, 1), stride=({b12:d}, {b13:d}), bias=False)
  )
))""")

pre_block_2 = compile("""({c0:d}, {c1:d}, Linear(in_features={c2:d}, out_features=100, bias=True))""")

pre_block_3 = compile("""({d0:d}, {d1:d}, Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False))""")

pre_blocks = [
        (pre_block_0,"block_0",['a0','a1','a2'], False),
        (pre_block_1,"block_1",['b0','b1','b2','b4','b5'], False),
        (pre_block_2,"block_2",['c0','c1','c2'], True),
        (pre_block_3,"block_3",['d0','d1'], True)
       ]

def parse(key):
    for parser in pre_blocks:
        result = parser[0].parse(key)
        if result == None:
            continue
        else:
            return [ result[param] for param in parser[2] ], parser[1]
    print("No Matching Key Found. Key: {}".format(key))
    return [0], "block_non"
    
def get_median(data):
    if type(data).__name__ == "list":
        med = statistics.median(data)
    elif type(data).__name__ == "dict":
        values = [data[k]["value"] for k in data]
        med = statistics.median(values)
    return med

def get_threshold(data, mult = 3):
    if type(data).__name__ == "list":
        med = statistics.median(data)
        minimum = min(data)
        thr = med + (med - minimum) * mult
    elif type(data).__name__ == "dict":
        values = [data[k]["value"] for k in data]
        med = statistics.median(values)
        minimum = min(values)
        thr = med + (med - minimum) * mult
    return thr

def remove_outlier(data, med, mult=3):
    if type(data).__name__ == "list":
        keys = [x[0] for x in data]
        values = [x[1] for x in data]
        minimum = min(values)
        thr = med + (med - minimum) * mult
        filtered = [x for x in data if x[1] < thr]
        return filtered

    elif type(data).__name__ == "dict":
        minimum = min(values)
        thr = med + (med - minimum) * mult
        filtered = {x:data[x] for x in data if data[x]["value"] > thr}
        return filtered


def main(r_path, w_path):
    
    filelist = [f for f in listdir(r_path) if isfile(join(r_path, f)) and ("image_" in f)]
    r_filelist = [join(r_path,f) for f in filelist]
    w_filelist = [join(w_path,f) for f in filelist]

    final_yaml= {filename:dict() for filename in w_filelist}

    for parser, name, params, rm_outlier in pre_blocks:
        
        data = list()
        latency = list()
        yaml_list = dict()
        
        print("Parsing {} in {} ...".format(name,r_path))
        for r_filename in r_filelist:
            f = open(r_filename)
            c = yaml.load(f, Loader=yaml.FullLoader)
            yaml_list[r_filename] = c
            for el in c:
                    parsed = parser.parse(el)
                    if parsed != None:
                        latency.append(c[el]["value"])
                    else:
                        continue
            f.close() #after finishing el

        th = get_threshold(latency,2.5)
        print("Threshold: {}".format(th))
        latency = list()

        for r_filename, w_filename in zip(r_filelist, w_filelist):
            c = yaml_list[r_filename]
            for el in c:
                parsed = parser.parse(el)
                if (parsed != None) and ((not rm_outlier) or (c[el]["value"] < th)):
                    tup = parsed.named
                    tup = [tup[param] for param in params]
                    data.append(tup)
                    latency.append(c[el]["value"])
                    final_yaml[w_filename][el] = c[el]
                else:
                    continue
        
        #if rm_outlier == True:
        #    med = get_median(latency)
        #    zipped = list(zip(data,latency))
        #    zipped = remove_outlier(zipped, med)
        #    data = [x[0] for x in zipped]
        #    latency = [x[1] for x in zipped]
        #    #print("Ori:{}, Filtered:{}".format(len(latency), len(filtered)))
        
        data_filename = "{}/{}_data.pickle".format(w_path,name)
        latency_filename = "{}/{}_latency.pickle".format(w_path,name)
        d = open(data_filename, 'wb')
        l = open(latency_filename, 'wb')
        pickle.dump(data,d)
        pickle.dump(latency,l)
        print("Writing Pickle On {} ...\n".format(latency_filename))
        d.close()
        l.close()
    
    for filename in final_yaml:
        f = open(filename,'w')
        yaml.dump(final_yaml[filename],f)

if __name__ == "__main__":
    paths = list()
    paths.append("latency_data/desktop/preactresnet18/cpu/")
    paths.append("latency_data/desktop/preactresnet18/cuda/")
    paths.append("latency_data/jetson/preactresnet18/cpu/")
    paths.append("latency_data/jetson/preactresnet18/cuda/")
    paths.append("latency_data/raspberrypi/preactresnet18/cpu/")
    read_paths = [join(path,"origin") for path in paths]
    write_paths = paths

    for r_path, w_path in zip(read_paths,write_paths):
        main(r_path, w_path)

