import yaml
from parse import compile
from os import listdir
from os.path import isfile, join, abspath
import pickle

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

pre_blocks = [(pre_block_0,"block_0",['a0','a1','a2']),
        (pre_block_1,"block_1",['b0','b1','b2','b4','b5']),
        (pre_block_2,"block_2",['c0','c1','c2']),
        (pre_block_3,"block_3",['d0','d1'])]

def parse(key):
    for parser in pre_blocks:
        result = parser[0].parse(key)
        if result == None:
            continue
        else:
            return [ result[param] for param in parser[2] ], parser[1]
    print("No Matching Key Found")
    return [0], "block_non"

def main():
    paths = list()
    paths.append("latency_data/desktop/preactresnet18/cpu")
    paths.append("latency_data/desktop/preactresnet18/cuda")
    paths.append("latency_data/jetson/preactresnet18/cpu")
    paths.append("latency_data/jetson/preactresnet18/cuda")
    paths.append("latency_data/raspberrypi/preactresnet18/cpu")
    
    for path in paths:
        filelist = [join(path,f) for f in listdir(path) if isfile(join(path, f)) and ("image" in f)]
        for parser, name, params in pre_blocks:
            data = list()
            latency = list()
            print("Parsing {} in {} ...".format(name,path))
            for filename in filelist:
                f = open(filename)
                c = yaml.load(f, Loader=yaml.FullLoader)
                for el in c:
                        parsed = parser.parse(el)
                        if parsed != None:
                            tup = parsed.named
                            tup = [tup[param] for param in params]
                            #tup_ = list(tup_)
                            data.append(tup)
                            latency.append(c[el]["value"])
                        else:
                            continue
                f.close() #after finishing el
            data_filename = "{}/{}_data.pickle".format(path,name)
            latency_filename = "{}/{}_latency.pickle".format(path,name)
            d = open(data_filename, 'wb')
            l = open(latency_filename, 'wb')
            print(len(data)," ",len(latency))
            pickle.dump(data,d)
            pickle.dump(latency,l)
            print("Writing Pickle On {} ...".format(latency_filename))
            d.close()
            l.close()

if __name__ == "__main__":
    main()

