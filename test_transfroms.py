# from torchvision import transforms
# from PIL import Image
import random

# img = Image.open("dataset/s02/1.pgm")
# print(img.getpixel((10, 10)))
# # img.show()

# img2 = transforms.CenterCrop((100, 100))(img)   # 中心裁剪
# # img2.show()

# for i in range(10):
#     img3 = transforms.RandomAffine(0, (0.2, 0.2), fillcolor=38)(img)
#     # img3 = transforms.RandomRotation(10)(img3)
#     # img3 = transforms.Resize((112*2, 92*2))(img3)
#     # img3 = transforms.RandomCrop((112, 92))(img3)
#     # img3 = transforms.ColorJitter(0.5, 0.3, 10)(img3)
#     img3.show()
for i in range(10):
    a = bool(random.getrandbits(1)) 
    print(a)