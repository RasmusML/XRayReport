import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import textwrap

from utils import crop_and_scale


def plot_image(image):
    plt.imshow(image)
    plt.axis('off')


def plot_images_2d(images, title=None, path=None):
    n_rows = images.shape[0]
    n_cols = images.shape[1]
    
    fig, axs = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(n_cols,n_rows))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    if title:
        fig.suptitle(title)
    
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axs[r][c]

            ax.imshow(images[r][c])
            ax.set_yticks([])
            ax.set_xticks([])

    if path:
        plt.savefig(path)
    else:
        plt.show()


def plot_images(images, title=None, path=None):
    n_cols = images.shape[0]
    
    fig, axs = plt.subplots(ncols=n_cols, figsize=(n_cols,1))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    if title:
        fig.subplots_adjust(top=0.75)
        fig.suptitle(title)
        
    for c in range(n_cols):
        ax = axs[c]

        ax.imshow(images[c])
        ax.set_yticks([])
        ax.set_xticks([])

    if path:
        plt.savefig(path)
    else:
        plt.show()

def text_to_image(text, title, fontsize=22, wrap_width=45, textarea_width=512):
    # Measure the height of the text area
    textarea_height = 0
    title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    textarea_height += title_font.getbbox(title)[3]

    text_wrap = textwrap.wrap(text, width=wrap_width)

    body_font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    for line in text_wrap:
        textarea_height += body_font.getbbox(line)[3]

    img = np.zeros([textarea_height, textarea_width, 3], dtype=np.uint8)
    img.fill(255)
    img = Image.fromarray(img)

    # Draw the text
    draw = ImageDraw.Draw(img)
    margin = offset = 0

    title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", fontsize)
    draw.text((margin, offset), title, font=title_font, fill="#000000")
    offset += title_font.getbbox(title)[3]

    body_font = ImageFont.truetype("DejaVuSans.ttf", fontsize)
    for line in text_wrap:
        draw.text((margin, offset), line, font=body_font, fill="#000000")
        offset += body_font.getbbox(line)[3]
    
    return img


def merge_textimage_and_image(text_image, image):
    yBuffer = 10
    width = text_image.width
    height = image.height + text_image.height + yBuffer

    combined_image = Image.new('RGB', (width, height), (250, 250, 250))
    combined_image.paste(text_image, (0, 0))
    combined_image.paste(image,(0, text_image.height + yBuffer))

    return np.asarray(combined_image)

def prepare_text_and_image(textbody, title, image):
    text_image = text_to_image(textbody, title)
    scaled_image = crop_and_scale(image, (text_image.width, text_image.width))
    image = Image.fromarray(scaled_image)
    combined_image = merge_textimage_and_image(text_image, image)
    return combined_image


def plot_text_and_image(textbody, title, image):
    combined_image = prepare_text_and_image(textbody, title, image)

    plt.imshow(combined_image)
    plt.axis('off')


def plot_multiple_text_and_images(textbodies, shared_title, images):
    n_cols = images.shape[0]
    
    scalar = 10
    fig, axs = plt.subplots(ncols=n_cols, figsize=(scalar*n_cols,scalar))
    fig.subplots_adjust(wspace=0, hspace=0)

    for c in range(n_cols):
        ax = axs[c]

        image = prepare_text_and_image(textbodies[c], shared_title, images[c])

        ax.imshow(image)
        ax.set_yticks([])
        ax.set_xticks([])
