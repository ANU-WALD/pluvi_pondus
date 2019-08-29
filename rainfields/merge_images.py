from PIL import Image

def merge_images(f1, f2, f3, f4):
    image1 = Image.open(f1)
    image2 = Image.open(f2)
    image3 = Image.open(f3)
    image4 = Image.open(f4)

    (width, height) = image1.size

    result_width = 2*width
    result_height = 2*height

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width, 0))
    result.paste(im=image3, box=(0, height))
    result.paste(im=image4, box=(width, height))

    return result

for i in range(780):
    print(i)
    res = merge_images("h8_b8_{:03d}.png".format(i), "crr_{:03d}.png".format(i), "forecasted_{:03d}.png".format(i), "rainfields_{:03d}.png".format(i))
    res.save("res_{:03d}.png".format(i),"PNG")

