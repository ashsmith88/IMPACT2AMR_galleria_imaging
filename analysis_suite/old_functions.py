"""
Old and non used functions

"""
def create_heatmap(img):

    import os
    import matplotlib.pyplot as plt
    import skimage.io as skio

    print(os.path.isfile(img))
    img = skio.imread(img)

    cmap = plt.get_cmap('jet')

    rgba_img = cmap(img)

    plt.figure()
    plt.imshow(rgba_img)
    plt.show()
