import pathlib
from types import MethodType
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.models as models


def to_dense(x, delimiter=','):
  s = ["%.6f" % val for val in x]
  return delimiter.join(s)


def forward(self, x):
  x = self.conv1(x)
  x = self.bn1(x)
  x = self.relu(x)
  x = self.maxpool(x)

  x = self.layer1(x)
  x = self.layer2(x)
  x = self.layer3(x)
  x = self.layer4(x)

  x = self.avgpool(x)
  x = x.view(x.size(0), -1)

  return x.data.cpu().numpy()
  # return [to_dense(f) for f in x.data.cpu().numpy()]


transform = transforms.Compose([
    transforms.Resize((224, 224), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def init_resnet50_model(cuda=False):
  model = models.resnet50(pretrained=True)
  model.forward = MethodType(forward, model)
  model.eval()
  if cuda:
    model = model.cuda()
  return model


model = init_resnet50_model(torch.cuda.is_available())

if __name__ == '__main__':
  cwd = pathlib.Path(__file__).parent.resolve()
  lenna = cwd.joinpath('lenna.png')
  img = Image.open(lenna)
  f = model(torch.stack([transform(img)]))
  print(f.size, f)
