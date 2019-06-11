import glob
import os

import matplotlib.image as img
import numpy as np
import skvideo.io
from SITI import siti
from utils import util

sl = util.check_system()['sl']


def main():
    for video in glob.glob(f'..{sl}yuv-full{sl}*.mp4'):
        siti.siti(filename=video, debug=True)

    siti.multi_plot(f'..{sl}yuv-full{sl}*.csv', output_folder='multiplot')
    # siti.single_plot(f'..{sl}yuv-full{sl}*.csv', output_folder='singleplot')


def salva_diff(filename: str, output_folder: str = None, size: str = '0x0', pix_fmt: str = 'yuv420p', form: str = None, num_frames: int = 0, inputdict: dict = None):
    if output_folder is None:
        output_folder = os.path.splitext(os.path.basename(filename))[0]
    if form is not None and inputdict is None:
        [width, height] = list(map(int, size.split('x')))
        inputdict = {"-s": f"{width}x{height}", "-pix_fmt": pix_fmt, '-f': form}

    os.makedirs(output_folder, exist_ok=True)
    previous_frame = None
    print(f"Video {filename}")

    for counter, frame in enumerate(skvideo.io.vreader(fname=filename, inputdict=inputdict, as_grey=True, num_frames=num_frames), 1):
        print(f"frame {counter:04d}", end='\r')

        height = frame.shape[1]
        width = frame.shape[2]
        frame = frame.reshape((height, width)).astype('float32')

        if previous_frame is None:
            previous_frame = frame
            continue

        img.imsave(f'{output_folder}\\diff_{counter}.jpg', np.abs(frame - previous_frame).astype('uint8'), cmap='plasma')
        # Image.fromarray( , mode='L').resize((600, int(600 * height / width))).save(f'{output_folder}\\diff_{counter}.jpg')

        previous_frame = frame


if __name__ == "__main__":
    main()
