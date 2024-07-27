import torch
from torchvision import datasets, transforms
from torchvision.transforms import v2
def find_parent(module, name: str):
    """Recursively apply getattr and returns parent of module"""
    if name == '':
        raise ValueError('Cannot Found')
    for sub_name in name.split('.')[: -1]:
        if hasattr(module, sub_name):
            module = getattr(module, sub_name)
        else:
            raise ValueError('submodule name not exist')
    return module

class ActivationHook():
    """
    Forward_hook used to get the output of the intermediate layer. 
    """

    def __init__(self, module):
        super(ActivationHook, self).__init__()
        self.inputs, self.outputs = None, None
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
      #  print("input shape", input[0].shape)
        self.inputs = input[0]  # arg tuple
        self.outputs = output

    def remove(self):
        self.handle.remove()


@torch.no_grad()
def evaluate_classifier(dataset, model, batch_size=64, workers=4, print_freq=50):
    device = next(model.parameters()).device
    model.to(device).eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    total, correct = 0, 0
    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        pred = model(images)
        correct += int((pred.argmax(dim=1)==target).sum())
        total += images.shape[0]

        if i % print_freq == 0:
            print(f"Test {i}/{len(data_loader)}: {correct/total*100:.2f}")

    print(f"Test: {correct/total*100:.2f}")
    accuracy = correct/total
    return accuracy

@torch.no_grad()
def evaluate_train(data_loader, model, batch_size=64, workers=4, print_freq=50):
    device = next(model.parameters()).device
    model.to(device).eval()
   # data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    total, correct = 0, 0
    for i, (images, target) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        pred = model(images)
        if type(pred) is tuple:
            pred = pred[0]
        correct += int((pred.argmax(dim=1)==target).sum())
        total += images.shape[0]

        if i % print_freq == 0:
            print(f"Test {i}/{len(data_loader)}: {correct/total*100:.2f}")

    print(f"Test: {correct/total*100:.2f}")
    accuracy = correct/total
    return accuracy
class LinearTempDecay:
    def __init__(self, iter_max, rel_start_decay, start_t, end_t):
        self.t_max = iter_max
        self.start_decay = rel_start_decay * iter_max
        self.start_b = start_t
        self.end_b = end_t

    def __call__(self, cur_iter):
        if cur_iter < self.start_decay:
            return self.start_b
        else:
            rel_t = (cur_iter-self.start_decay) / (self.t_max-self.start_decay)
            return self.end_b + (self.start_b-self.end_b)*max(0.0, 1 - rel_t)

