import torch
from torchvision import transforms
import math
import pickle
import numpy as np
import copy

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

def make_dataloader(file_name, batch_size = 16, device = "cuda"):
    """
    Args.
    file_name (string): e.g., ./latency_data/block_0
    """
    data_path = file_name + "_data.pickle"
    latency_path = file_name + "_latency.pickle"

    f = open(data_path, "rb")
    g = open(latency_path, "rb")

    x = pickle.load(f) # (data_num, 2) => each row is (image resolution, width)
    x = torch.FloatTensor(x)
    x = x.to(device)

    y = pickle.load(g) # (data_num) => each row is latency
    y = torch.tensor(y).to(device)
    
    f.close()
    g.close()

    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x = (x - x_min) / (x_max - x_min)

    dataset = CustomDataset(x, y)

    total_len = len(dataset)

    train_len = int(total_len * 0.6)
    valid_len = int(total_len * 0.2)
    test_len = total_len - train_len - valid_len

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)

    return train_loader, valid_loader, test_loader, train_len, valid_len, test_len, x_min, x_max

def main(file_name, save_path):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    train_loader, valid_loader, test_loader, train_len, valid_len, test_len, x_min, x_max = make_dataloader(file_name = file_name, batch_size = 16, device = device)

    input_size = train_loader.dataset[0][0].shape[0]

    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )
    # 4 layer input_size, 128 => 128,64 => 64,16 => 16, 1
    
    model.min = x_min
    model.max = x_max

    model.to(device)
    best_model = None
    best_err = -1

    learning_rate = 1e-2
    weight_decay = 0.1
    epochs = 300

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150, 200], gamma=0.1) # lr = gamma * lr

    if best_model is not None:
      model = best_model

    for epoch in range(epochs):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()


            y_pred = np.squeeze(model(x), axis = 1)
            loss = loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()
        model.eval()

        Avg_train_err = 0
        Avg_valid_err = 0

        total_err = 0
        for x, y in train_loader:
            y_pred = np.squeeze(model(x), axis = 1)
            err = sum((y_pred - y) / y *100)
            total_err = total_err + abs(err)
        Avg_train_err = (total_err / train_len)

        total_err = 0
        for x, y in valid_loader:
            y_pred = np.squeeze(model(x), axis = 1)
            err = sum((y_pred - y) / y *100)
            total_err = total_err + abs(err)
        Avg_valid_err = (total_err / valid_len)

        if best_err == -1 or best_err >= Avg_valid_err:
            best_model = copy.deepcopy(model)
            best_err = Avg_valid_err

        #print("train_err: {}%, valid_err: {}%".format(Avg_train_err, Avg_valid_err))
        exp_lr_scheduler.step()
        best_model.eval()

    ## Evaluation on testset
    #print("========== Test ==========")
    total_err = 0
    for x, y in test_loader:
        y_pred = np.squeeze(best_model(x), axis = 1)
        err = sum((y_pred - y) / y)*100
        total_err = total_err + abs(err)
    print("Average err: {}%".format(total_err / test_len))
    print("Saving {}".format(save_path))
    print("")
    torch.save(best_model,save_path)

if __name__ == "__main__":
   main("./latency_data/desktop/preactresnet18/cpu/block_0", "./trained_model/desktop_cpu/block_0.pt")
   main("./latency_data/desktop/preactresnet18/cpu/block_1", "./trained_model/desktop_cpu/block_1.pt")
   main("./latency_data/desktop/preactresnet18/cpu/block_2", "./trained_model/desktop_cpu/block_2.pt")
   main("./latency_data/desktop/preactresnet18/cpu/block_3", "./trained_model/desktop_cpu/block_3.pt")
   main("./latency_data/desktop/preactresnet18/cuda/block_0", "./trained_model/desktop_gpu/block_0.pt")
   main("./latency_data/desktop/preactresnet18/cuda/block_1", "./trained_model/desktop_gpu/block_1.pt")
   main("./latency_data/desktop/preactresnet18/cuda/block_2", "./trained_model/desktop_gpu/block_2.pt")
   main("./latency_data/desktop/preactresnet18/cuda/block_3", "./trained_model/desktop_gpu/block_3.pt")
   main("./latency_data/jetson/preactresnet18/cpu/block_0", "./trained_model/jetson_cpu/block_0.pt")
   main("./latency_data/jetson/preactresnet18/cpu/block_1", "./trained_model/jetson_cpu/block_1.pt")
   main("./latency_data/jetson/preactresnet18/cpu/block_2", "./trained_model/jetson_cpu/block_2.pt")
   main("./latency_data/jetson/preactresnet18/cpu/block_3", "./trained_model/jetson_cpu/block_3.pt")
   main("./latency_data/jetson/preactresnet18/cuda/block_0", "./trained_model/jetson_gpu/block_0.pt")
   main("./latency_data/jetson/preactresnet18/cuda/block_1", "./trained_model/jetson_gpu/block_1.pt")
   main("./latency_data/jetson/preactresnet18/cuda/block_2", "./trained_model/jetson_gpu/block_2.pt")
   main("./latency_data/jetson/preactresnet18/cuda/block_3", "./trained_model/jetson_gpu/block_3.pt")
   main("./latency_data/raspberrypi/preactresnet18/cpu/block_0", "./trained_model/raspberrypi_cpu/block_0.pt")
   main("./latency_data/raspberrypi/preactresnet18/cpu/block_1", "./trained_model/raspberrypi_cpu/block_1.pt")
   main("./latency_data/raspberrypi/preactresnet18/cpu/block_2", "./trained_model/raspberrypi_cpu/block_2.pt")
   main("./latency_data/raspberrypi/preactresnet18/cpu/block_3", "./trained_model/raspberrypi_cpu/block_3.pt")
