import env
import models
model_name = "preactresnet18"
model = models.__dict__[model_name](100, False, 1, 30 / 10, 1.0).to('cpu')

a = env.NasEnv(device="cpu", platform="jetson")
#print(a.latency_model)
#print(a.latency_model.models)
for size in range(4, 500, 4):
    a.eval_latency(model, size)
