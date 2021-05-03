import model
from utils import round_filters

import json
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import numpy as np
from model import MBConvBlock
from model import EfficientNet
import matplotlib.pyplot as plt
from collections import OrderedDict
import os.path
import torch_summary

######################### Hyper-parameter #########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Input ##
input_resolution = 64 # for data preprocessing
bs = 8 # batch size

## Optimizer ##
wd = 1e-4 # weight decay
moment = 0.9 # momentum
lr_rate = 0.1

## Training ##
epoch = 1


######################### Functions #########################
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def accuracy(preds,target): # top-5
    N = target.shape[0]
    acc = 0
    for i, pred in enumerate(preds):
      for idx in torch.topk(pred, k=5).indices.squeeze(0).tolist():
        if idx == target[i]:
          acc += 1
          break
    return acc / N

def train(model, optimizer, dataloader, device, loss_fn, eval_fn, epoch, scheduler=None):
    model.train()
    avg_loss = 0
    avg_accuracy = 0
    if device == "cuda":
      model.cuda()

    for step, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)

        preds = logits.softmax(dim=1)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        avg_accuracy += eval_fn(preds, targets)
        if scheduler is not None:
            scheduler.step()
    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy

def test(model, dataloader, device, loss_fn, eval_fn):
    model.eval()
    avg_loss = 0
    avg_accuracy = 0

    if device == "cuda":
      model.cuda()

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.softmax(dim=1)
            loss = loss_fn(preds, targets)
            avg_loss += loss.item()
            avg_accuracy += eval_fn(preds, targets)

    avg_loss /= len(dataloader)
    avg_accuracy /= len(dataloader)
    return avg_loss, avg_accuracy

def PerfEval(inp_arg):
  """
  Make model according to the input argument and return the accuracy and memory of the model.

  ** Currently, the resolution does not affect the model. **

  Args
  inp_arg: tuple(width_scale, depth_scale, resolution) e.g., (1.0, 1.0, 224)

  Return
  accuracy, memory of the model
  """
  w, d, r = inp_arg
  acc, mem = 0, 0
  path = "./" + str(w) + "_" + str(d) + "_" + str(r)
  if os.path.exists(path):
    print("Load existing model: " + path)
    model = torch.load(path)
  else:
    print("Create new model")
    #model = EfficientNet.from_name('efficientnet-b0', width_coefficient = w, depth_coefficient = d)
    model = EfficientNet.from_pretrained('efficientnet-b0', width_coefficient = w, depth_coefficient = d)

  final_filters = 1280 # for efficientnet-b0
  new_filters = round_filters(final_filters, model._global_params) # for custom model
  model._fc = torch.nn.Linear(new_filters, 100, bias=True)

  optimizer = torch.optim.SGD(model.parameters(), lr= lr_rate, momentum= moment, weight_decay=wd)
  #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(2.4*len(small_trainloader)), gamma=0.97)
  scheduler = None
  loss_fn = torch.nn.CrossEntropyLoss().to(device)
  eval_fn = accuracy

  best_test_acc = 0
  estimated_mem = 0

  report, _ = torch_summary.summary_string(model, (3, input_resolution, input_resolution), bs, device = torch.device(device))
  estimated_mem = float(report.split('\n')[-3].split(' ')[-1]) # (MB)

  for i in range(epoch):
      train_loss, train_accuracy = train(model, optimizer, small_trainloader,
                                              device, loss_fn, eval_fn, i, scheduler)
      test_loss, test_accuracy = test(model, small_testloader,
                                          device, loss_fn, eval_fn)
      if test_accuracy > best_test_acc:
        print("Save best model")
        torch.save(model, path)
        best_test_acc = test_accuracy
      if scheduler is not None:
        print(f'''========   epoch {i:>3} (lr: {scheduler.get_lr()[0]:.5f})  ========
                  train loss = {train_loss:.5f} | train acc = {train_accuracy:.2%} |
                  test loss = {test_loss:.5f} | test acc = {test_accuracy:.2%}''')
      else:
        print(f'''========   epoch {i:>3}   ========
                  train loss = {train_loss:.5f} | train acc = {train_accuracy:.2%} |
                  test loss = {test_loss:.5f} | test acc = {test_accuracy:.2%}''')
  print('=== Success ===')

  return best_test_acc, estimated_mem



######################### Data preprocessing #########################

# image size 32x32 => 64x64
tfms = transforms.Compose([transforms.Resize(input_resolution), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download = True, transform = tfms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True, num_workers = 0)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download = True, transform = tfms)
testloader = torch.utils.data.DataLoader(testset, batch_size = bs, shuffle = True, num_workers = 0)

indices = torch.arange(16) # 5만개중 10분의 1인 5000개만 사용
small_trainset = torch.utils.data.Subset(trainset, indices)
small_trainloader = torch.utils.data.DataLoader(small_trainset, batch_size = bs, shuffle = True, num_workers = 0)
dataiter = iter(small_trainloader) # 테스트용

indices = torch.arange(8) # 1만개중 10분의 1인 1000개만 사용
small_testset = torch.utils.data.Subset(testset, indices)
small_testloader = torch.utils.data.DataLoader(small_testset, batch_size = bs, shuffle = True, num_workers = 0)

classes = [
'apple','aquarium_fish',
'baby','bear','beaver','bed','bee','beetle','bicycle','bottle','bowl','boy','bridge','bus','butterfly',
'camel','can','castle','caterpillar','cattle','chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup',
'dinosaur','dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','computer_keyboard','lamp',
'lawn_mower','leopard','lion','lizard','lobster',
'man','maple_tree','motorcycle','mountain','mouse','mushroom',
'oak_tree','orange','orchid','otter',
'palm_tree','pear','pickup_truck','pine_tree','plain','plate','poppy','porcupine','possum',
'rabbit','raccoon','ray','road','rocket','rose',
'sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider','squirrel','streetcar','sunflower','sweet_pepper',
'table','tank','telephone','television','tiger','tractor','train','trout','tulip','turtle',
'wardrobe','whale','willow_tree','wolf','woman',
'worm',
]

######################### Data plot #########################

"""
images, labels = dataiter.next()
print("")
print("=============== TEST PLOT ===============")
print(' '.join('%5s' % classes[labels[j]] for j in range(8))) # batch size = 16
print("=========================================")
print("")
imshow(torchvision.utils.make_grid(images))
"""

if __name__ == "__main__":
    inp_argument = (1.0, 1.0, 224) # width scale factor, depth scale factor, resolution
    acc, mem = PerfEval(inp_argument)

    print(str(acc) + "%", str(mem) + "(MB)")
