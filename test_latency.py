import env
import models
model_name = "preactresnet18"

a = env.NasEnv(device="cuda", platform="desktop")
#print(a.latency_model)
#print(a.latency_model.models)
#size = 36
#for size in range(4, 225, 4):
#    for width in range(10, 51, 1):
#        model = models.__dict__[model_name](100, False, 1, width/10, 1.0).to('cuda')
#        a.eval_latency(model, size, True)
size = 32
model = models.__dict__[model_name](100, False, 1, 10/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 11/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 12/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 13/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 14/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 15/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 16/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 17/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 18/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 20/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 21/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 22/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
model = models.__dict__[model_name](100, False, 1, 23/10, 10/10).to('cuda')
a.eval_latency(model, size, True)
