import env
import models
model_name = "preactresnet18"

device = "cuda"
platform = "desktop"

a = env.NasEnv(device=device, platform=platform)

s= 8
for w in range(10, 31, 1):
    for d in [10, 20]:
        model = models.__dict__[model_name](100, False, 1, w/10, d/10).to('cuda')
        a.eval_latency(model, s, True)

for s in [32, 64, 128, 224]:
    for w in range(1, 9, 1):
        for d in [10, 20]:
            model = models.__dict__[model_name](100, False, 1, w/10, d/10).to('cuda')
            a.eval_latency(model, s, True)
