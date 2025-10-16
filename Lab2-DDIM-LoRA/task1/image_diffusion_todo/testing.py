from PIL import Image
img = Image.open('./data/afhq/eval/flickr_cat_000008.jpg')
print(img.size, img.mode)  # 應該是 (256, 256) 和 'RGB'