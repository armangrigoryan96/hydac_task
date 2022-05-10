import argparse
from aligner import ImageAligner
import os
parser = argparse.ArgumentParser(description='Aligning the images according to the text')
parser.add_argument('--input_path', type=str,  help='Input path', default = 'Input images')
parser.add_argument('--output_path', type=str,  help='Output path', default = 'Output images')

args = parser.parse_args()

if __name__ == '__main__':
    img_aligner = ImageAligner(input_path=args.input_path, output_path=args.output_path)

    #process all the images in the input path and align them accordingly
    for img_filename in os.listdir(args.input_path):
        if img_filename.split('.')[-1] in ['jpg', 'jpeg', 'png']:
            img_aligner.align(img_filename)



