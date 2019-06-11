#!/usr/bin/env python3

import argparse
import collections
import glob
import os
import platform
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skvideo.io
from scipy import ndimage

if platform.system() == 'Windows':
    sl = '\\'
else:
    sl = '/'

ax = None
fig = None


def siti(filename: str, csv_output: str = None, size: str = '0x0', pix_fmt: str = 'yuv420p', form: str = None,
         num_frames: int = 0, inputdict: dict = None, debug: bool=False) -> bool:
    """
    Python script to calculate SI/TI according recommendation ITU-T P.910.
    :param filename: Filename of video.
    :param csv_output: CSV name to salve.
    :param size: Video dimension in "MxN" pixels.
    :param pix_fmt: Force pixel format to ffmpeg style.
    :param form: Force video format to ffmpeg style.
    :param num_frames: Number of frames to process.
    :param inputdict: A dict to ffmpeg backend.
    :return: None
    """
    print(f'Calculating SI/TI for {filename}.')
    frame_counter = 0
    [width, height] = list(map(int, size.split('x')))
    name = os.path.splitext(os.path.basename(filename))[0]

    if form is not None and inputdict is None:
        inputdict = {"-s": f"{width}x{height}", "-pix_fmt": pix_fmt, '-f': form}

    measures = {'si': Si(), 'ti': Ti()}

    video = skvideo.io.vreader(fname=filename, inputdict=inputdict, as_grey=True, num_frames=num_frames)

    for frame in video:
        # todo: Supondo que é colorido... se for p&b vai dar problema. ou não?
        width = frame.shape[1]
        height = frame.shape[2]

        frame_counter += 1
        print(f"\nframe {frame_counter} of video {filename}")
        frame = frame.reshape((width, height)).astype('float32')

        for measure in measures:
            value = measures[measure].calc(frame, destiny=name, debug=debug)
            print(f"{measure} -> {value}")
    if not debug:
        if csv_output is None:
            csv_output = name + '.csv'

        df = pd.DataFrame()
        df["frame"] = range(1, measures['si'].frame_counter + 1)
        df['si'] = measures['si'].values
        df['ti'] = measures['ti'].values
        df.to_csv(csv_output)

    subprocess.run(f'ffmpeg -r 30 -i {name}\\siti_%d.jpg -c:v libx264 -vf fps=30 -pix_fmt yuv420p {name}\\out.mp4', shell=True, encoding='utf-8')

    return True


class Features:
    def __init__(self):
        self.values = []
        self.frame_counter = 0

    def calc(self, frame):
        raise NotImplementedError("not implemented")


class Si(Features):
    def calc(self, frame, destiny='sobel', debug=False, si=0):
        self.frame_counter += 1
        sobx = ndimage.sobel(frame, axis=0)
        soby = ndimage.sobel(frame, axis=1)
        sob = np.hypot(sobx, soby)
        sob_std = sob.std()
        si = sob_std
        self.values.append(si)

        if debug:
            plt.close()
            os.makedirs(f'{destiny}', exist_ok=True)
            global fig, ax

            fig, ax = plt.subplots(2, 2, figsize=(12, 6))

            ax[0][0].imshow(frame, cmap='gray')
            ax[0][0].set_xlabel('Frame luma')
            ax[0][0].get_xaxis().set_ticks([])
            ax[0][0].get_yaxis().set_ticks([])

            ax[1][0].imshow(sob, cmap='gray')
            ax[1][0].set_xlabel('Sobel result')
            ax[1][0].get_xaxis().set_ticks([])
            ax[1][0].get_yaxis().set_ticks([])

            samples = 300
            val = self.values
            rotation = -samples
            if len(self.values) < samples:
                val = self.values + [0] * (samples - len(self.values))
                rotation = -self.frame_counter

            v = collections.deque(val[-samples:])
            v.rotate(rotation)
            ax[0][1].plot(v, 'b', label=f'SI={si:03.2f}')

        return si


class Ti(Features):
    def __init__(self):
        super().__init__()
        self.previous_frame = None

    def calc(self, frame, destiny='diff\\', debug=False):
        self.frame_counter += 1
        ti = 0

        if self.previous_frame is not None:
            difference = frame - self.previous_frame
            ti = difference.std()

            if debug:
                global ax, fig
                ax[1][1].imshow(np.abs(difference), cmap='gray')

        self.values.append(ti)
        self.previous_frame = frame

        if debug:
            ax[1][1].set_xlabel('Diff result')
            ax[1][1].get_xaxis().set_ticks([])
            ax[1][1].get_yaxis().set_ticks([])

            samples = 300
            val = self.values
            rotation = -samples
            if len(self.values) < samples:
                val = val + [0] * (samples - len(self.values))
                rotation = -self.frame_counter

            v = collections.deque(val[-samples:])
            v.rotate(rotation)
            ax[0][1].plot(v, 'r', label=f'TI={ti:03.2f}')
            ax[0][1].set_xlabel('SI/TI')
            # ax[0][1].get_xaxis().set_ticks([])
            ax[0][1].legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
            ax[0][1].set_ylim(bottom=0)
            ax[0][1].set_xlim(left=0)
            fig.tight_layout()
            plt.savefig(f'{destiny}\\siti_{self.frame_counter}.jpg', dpi=150, cmap='gray')
            # plt.show()
            print('')

        return ti


def multi_plot(input_glob='*.csv', output_folder='graphs'):
    os.makedirs(output_folder, exist_ok=True)
    names = {}
    df = pd.DataFrame()
    for x in glob.glob(input_glob):
        print(f'Processando multiplot{x}.')
        name = x.replace(f'.{input_glob.split(".")[-1]}', '').split(f'{sl}')[-1]
        tmp = pd.read_csv(x, delimiter=',')

        names[name] = dict(si_med=tmp['si'].median(),
                           ti_med=tmp['ti'].median(),
                           yerr=np.array([[np.percentile(tmp['ti'], 25)], [np.percentile(tmp['ti'], 75)]]),
                           xerr=np.array([[np.percentile(tmp['si'], 25)], [np.percentile(tmp['si'], 75)]]))

    fig, ax_med = plt.subplots(1, 1, figsize=(10, 5), tight_layout=True, dpi=200)

    _x = []
    _y = []
    for name in names:
        yerr = names[name]['yerr']
        xerr = names[name]['xerr']
        x = names[name]['si_med']
        y = names[name]['ti_med']
        ax_med.errorbar(x=x, y=y, label=name, yerr=np.abs(yerr - y), xerr=np.abs(xerr - x), fmt='o')

        _x.append(x)
        _y.append(y)

    df['filename'] = list(glob.glob(input_glob))
    df['si_med'] = _x
    df['ti_med'] = _y

    ax_med.set_xlabel('SI')
    ax_med.set_ylabel('TI')
    ax_med.set_title('SI/TI - Median Values')
    ax_med.set_ylim(bottom=0)
    ax_med.set_xlim(left=0)
    ax_med.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))

    # plt.show()
    df.to_csv(f'{output_folder}{sl}scatter-err_siti.csv', index_label='frames')
    fig.savefig(f'{output_folder}{sl}scatter-err_siti')


def single_plot(input_glob='*.csv', output_folder='graphs'):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.DataFrame()

    for x in glob.glob(input_glob):
        print(f'Processando single{x}.')

        name = x.replace(f'.{input_glob.split(".")[-1]}', '').split(f'{sl}')[-1]
        tmp = pd.read_csv(x, delimiter=',')

        fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True, dpi=300)
        ax.plot(tmp['si'], label='si')
        ax.plot(tmp['ti'], label='ti')
        df[f'{name}_ti'] = tmp['ti']
        df[f'{name}_si'] = tmp['si']
        ax.set_xlabel('Frame')
        ax.set_ylabel('Information')
        ax.set_title(name)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
        # plt.show()
        fig.savefig(f'{output_folder}{sl}{name}')
    df.to_csv(f'{output_folder}{sl}videos_stats.csv', index_label='frames')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate SI/TI according ITU-T P.910',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", type=str, help="Video to analyze")
    parser.add_argument('--size', type=str, default='0x0', help='Dimensions in pixels')
    parser.add_argument('--num_frames', type=int, default=0, help='Process number of frames')
    parser.add_argument("--output", type=str, default="", help="Output CSV for si ti report")
    parser.add_argument("--form", type=str, default="", help="Force ffmpeg video format")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p", help="force ffmpeg pixel format")
    params = vars(parser.parse_args())

    siti(**params)
    multi_plot(params['output'])
