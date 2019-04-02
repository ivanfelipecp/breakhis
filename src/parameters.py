from models import get_model
from torchsummary import summary
models = ["fractalnet","densenet161", "inceptionv3","squeezenet", "traditionalcnn"]
for i in models:
    model, img_size = get_model(1, None, i)
    model = model.cuda()
    print(i)
    summary(model , (3, img_size, img_size))