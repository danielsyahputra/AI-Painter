import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils.dataset import get_unloader
from .settings import get_device

DEVICE = get_device()
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class ContentLoss(nn.Module):
    def __init__(self, target) -> None:
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, content_input):
        self.loss = F.mse_loss(content_input, self.target)
        return content_input

    def callback(self):
        return self.loss.backward(retain_graph=True)
    
def gram_matrix(style_input):
    batch, color, height, width = style_input.size()
    features = style_input.view(batch * color, height * width)
    gram = torch.mm(features, torch.t(features))
    return gram.div(batch * color * height * width)

class StyleLoss(nn.Module):
    def __init__(self, target) -> None:
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, style_input):
        gram = gram_matrix(style_input)
        self.loss = F.mse_loss(gram, self.target)
        return style_input
    
    def callback(self):
        return self.loss.backward(retain_graph=True)

class Normalization(nn.Module):
    def __init__(self, mean, std) -> None:
        super().__init__()
        normalization_mean = torch.tensor(mean).to(DEVICE)
        normalization_std = torch.tensor(std).to(DEVICE)
        self.mean = normalization_mean.detach().clone().view(-1, 1, 1)
        self.std = normalization_std.detach().clone().view(-1, 1, 1)

    def forward(self, img):
        return (img.to(DEVICE) - self.mean) / self.std

def get_style_model_and_losses(base_model, style, content, 
                            mean=MEAN, std=STD,
                            content_layers=['conv_4'],
                            style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    normalization = Normalization(mean, std).to(DEVICE)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    iteration = 0
    for layer in base_model.children():
        if isinstance(layer, nn.Conv2d):
            iteration += 1
            name = f"conv_{iteration}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{iteration}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{iteration}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{iteration}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")
        
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{iteration}", content_loss)
            content_losses.append(content_loss)
        
        if name in style_layers:
            target = model(style).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{iteration}", style_loss)
            style_losses.append(style_loss)
    
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, content, style, input_img, 
        num_steps=300, style_weight=100000, content_weight=1):
    
    print("Building the Style Transfer Model...")
    model, style_losses, content_losses = get_style_model_and_losses(base_model=cnn, content=content, style=style)

    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")
    step = [0]
    while step[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score

            loss.backward()

            step[0] += 1
            if step[0] % 50 == 0:
                print(f"Run {step}:")
                print(f"Style Loss: {style_score.item():.4f} | Content Loss: {content_score.item():.4f}")
                print()
            return style_score + content_score

        optimizer.step(closure)
    
    with torch.no_grad():
        input_img.clamp_(0, 1)
    
    input_img = get_unloader(input_img.squeeze())
    return input_img