#!/bin/python3
import glob
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils import util

sl = '\\'
ffmpeg = 'ffmpeg.exe'
if platform.system() not in 'Windows':
    sl = '/'
    ffmpeg = 'ffmpeg'


def main():
    # stats()
    graph1()
    # kmeans()
    # graph2()
    # graph3()
    # graph4()
    # hist()
    pass


def hist():
    """
    Fazer um histograma para cada fator que estamos avaliando
    qualidade: 2000 kbps, 24000 kbps
    fmt: 1x1, 3x2, 6x4
    video: om_nom e rollercoaster
    Total: 2 x 3 x 2 = 12 histogramas

    :return:
    """
    pass


def graph4():
    """
    Este plot compara tile a tile a taxa e o tempo de decodificação para diferentes qualidades.

    :return:
    """
    config = util.Config('Config.json')
    dectime = util.load_json('times.json')

    dirname = 'graph4'
    os.makedirs(f'{dirname}', exist_ok=True)

    for fmt in config.tile_list:
        m, n = list(map(int, fmt.split('x')))

        for tile in range(1, m * n + 1):
            times = util.AutoDict()
            sizes = util.AutoDict()
            times_a_ld = []
            times_a_hd = []
            sizes_a_ld = []
            sizes_a_hd = []
            times_b_ld = []
            times_b_hd = []
            sizes_b_ld = []
            sizes_b_hd = []

            # for name in config.videos_list:
            #     for quality in config.rate_list:
            #         t = []
            #         s = []
            #         for chunk in range(1, config.duration + 1):
            #             t.append(dectime['ffmpeg'][name][fmt]['rate'][str(quality)][str(tile)][str(chunk)]['single']['times']['ut'])
            #             s.append(dectime['ffmpeg'][name][fmt]['rate'][str(quality)][str(tile)][str(chunk)]['single']['size'])
            #         times[name][str(quality)] = t
            #         times[name][str(quality)] = s

            for chunk in range(1, config.duration + 1):
                times_a_ld.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['times']['ut'])
                sizes_a_ld.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['size'])
                times_a_hd.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['times']['ut'])
                sizes_a_hd.append(dectime['ffmpeg']['om_nom'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['size'])

                times_b_ld.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['times']['ut'])
                sizes_b_ld.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(2000000)][str(tile)][str(chunk)]['single']['size'])
                times_b_hd.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['times']['ut'])
                sizes_b_hd.append(dectime['ffmpeg']['rollercoaster'][fmt]['rate'][str(24000000)][str(tile)][str(chunk)]['single']['size'])

            # a = plt.Axes()
            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
            ax[0].hist(times_a_ld, bins=10, histtype='step', label=f'Om_non_{fmt}_rate2000000')
            ax[0].hist(times_a_hd, bins=10, histtype='step', label=f'Om_non_{fmt}_rate24000000')
            ax[0].hist(times_b_ld, bins=10, histtype='step', label=f'rollercoaster_{fmt}_rate2000000')
            ax[0].hist(times_b_hd, bins=10, histtype='step', label=f'rollercoaster_{fmt}_rate24000000')
            ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[0].set_title(f'Tile {tile}')
            ax[0].set_xlabel('Times')
            ax[0].set_ylabel("Occurrence")

            ax[1].hist(times_a_ld, bins=10, density=True, cumulative=True, histtype='step', label=f'Om_non_{fmt}_rate2000000')
            ax[1].hist(times_a_hd, bins=10, density=True, cumulative=True, histtype='step', label=f'Om_non_{fmt}_rate24000000')
            ax[1].hist(times_b_ld, bins=10, density=True, cumulative=True, histtype='step', label=f'rollercoaster_{fmt}_rate2000000')
            ax[1].hist(times_b_hd, bins=10, density=True, cumulative=True, histtype='step', label=f'rollercoaster_{fmt}_rate24000000')
            ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[1].set_xlabel('Times')
            ax[1].set_ylabel("CDF")
            plt.tight_layout()
            plt.savefig(f'{dirname}{sl}hist_{fmt}_tile{tile}')
            # plt.show()
            print(f'hist_{fmt}_tile{tile}')

            # plt.hist(times, bins=20)
            plt.close()
            fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
            ax[0].bar(np.array(range(len(times_a_ld))) - 0.3, times_a_ld, width=0.2, label=f'om_nom-{fmt}-rate{2000000}')
            ax[0].bar(np.array(range(len(times_a_hd))) - 0.1, times_a_hd, width=0.2, label=f'om_nom-{fmt}-rate{24000000}')
            ax[0].bar(np.array(range(len(times_b_ld))) + 0.1, times_b_ld, width=0.2, label=f'rollercoaster-{fmt}-rate{2000000}')
            ax[0].bar(np.array(range(len(times_b_hd))) + 0.3, times_b_hd, width=0.2, label=f'rollercoaster-{fmt}-rate{24000000}')
            ax[0].set_title(f'Tile {tile} - Atrasos')
            ax[0].set_ylabel("Time")

            ax[1].plot(sizes_a_ld, label=f'om_nom-{fmt}-rate{2000000}')
            ax[1].plot(sizes_a_hd, label=f'om_nom-{fmt}-rate{24000000}')
            ax[1].plot(sizes_b_ld, label=f'rollercoaster-{fmt}-rate{2000000}')
            ax[1].plot(sizes_b_hd, label=f'rollercoaster-{fmt}-rate{24000000}')
            ax[1].set_title(f'Tile {tile} - Taxas')
            ax[1].set_xlabel("Chunk")
            ax[1].set_ylabel("Time")

            ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
            ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))

            plt.tight_layout()
            plt.savefig(f'{dirname}{sl}graph_{fmt}_tile{tile}')

            # plt.show()
            print(f'graph_{fmt}_tile{tile}')


def graph3() -> None:
    """
    bar
    fmt X average_dec_time (seconds) and fmt X average_rate (Bytes)
    :return: None
    """
    dirname = 'graph3'

    config = util.Config('config.json')
    dectime = util.load_json('times.json')

    # decoders = ['ffmpeg', 'mp4client']
    factors = ['rate']
    threads = ['single']

    # for decoder in decoders:
    for name in config.videos_list:
        for factor in factors:

            for thread in threads:
                df = pd.DataFrame()
                plt.close()
                fig, ax = plt.subplots(2, 1, figsize=(8, 5))
                quality_list = getattr(config, f'{factor}_list')
                offset = 0
                for quality in quality_list:
                    average_size = []
                    std_size = []
                    average_time = []
                    std_time = []
                    width = 0.8 / len(quality_list)
                    start_position = (0.8 - width) / 2

                    for fmt in config.tile_list:
                        m, n = list(map(int, fmt.split('x')))
                        size = []
                        time = []

                        for tile in range(1, m * n + 1):
                            for chunk in range(1, config.duration + 1):
                                size.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['size'])
                                time.append(dectime['ffmpeg'][name][fmt][factor][str(quality)][str(tile)][str(chunk)][thread]['times']['ut'])

                        average_size.append(np.average(size))
                        std_size.append(np.std(size))
                        average_time.append(np.average(time))
                        std_time.append(np.std(time))

                    x = np.array(range(1, len(average_time) + 1)) - start_position + offset
                    offset += width
                    ax[0].bar(x, average_time, width=width, yerr=std_time, label=f'rate_total={quality}')
                    ax[1].bar(x, average_size, width=width, yerr=std_size, label=f'rate_total={quality}')

                    df[f'times_{name}_{quality}'] = average_time

                ax[0].set_xticklabels(config.tile_list)
                ax[0].set_xticks(np.array(range(1, len(config.tile_list) + 1)))
                ax[1].set_xticklabels(config.tile_list)
                ax[1].set_xticks(np.array(range(1, len(config.tile_list) + 1)))

                ax[0].set_xlabel('Tile')
                ax[1].set_xlabel('Tile')
                ax[0].set_ylabel('Average Time')
                ax[1].set_ylabel('Average Rate')
                ax[0].set_title(f'{name} - Times by tiles, {factor}')
                ax[1].set_title(f'{name} - Rates by tiles, {factor}')
                ax[0].set_ylim(bottom=0)
                ax[1].set_ylim(bottom=0)
                ax[0].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
                ax[1].legend(loc='upper left', ncol=1, bbox_to_anchor=(1.01, 1.0))
                plt.tight_layout()
                os.makedirs(dirname, exist_ok=True)
                print(f'Salvando {dirname}{sl}{name}_{factor}.')
                fig.savefig(f'{dirname}{sl}{name}_{factor}')
                # plt.show()
                1


from scipy.spatial import Voronoi


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def kmeans() -> None:
    """
    kmeans and voronoi
    :return: None
    """
    dataset = pd.read_csv(f'all_spreadsheet.csv', index_col=0)
    for col in dataset:
        dataset[col] = dataset[col] / dataset[col].max()

    array = dataset.to_numpy()

    kmeans = KMeans(n_clusters=4, random_state=0)
    for arr in [2, 3]:  # [array[:, :2], array[:, :3], array[:, :4]]:
        kmeans = kmeans.fit(array[:, :arr])
        vor = Voronoi(kmeans.cluster_centers_[:, 0:2])
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # voronoi_plot_2d(vor, ax=ax)
        # print('')
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=3000)
        print("--")
        print(regions)
        print("--")
        print(vertices)
        #
        # colorize
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), '-', alpha=0.4, edgecolor='#000000', linestyle='-')
        # plt.show()

        for (index, row) in dataset[['si_med', 'ti_med']].iterrows():
            ax.plot(row[0], row[1], 'o', label=index)

        ax.set_xlabel('SI')
        ax.set_ylabel('TI')
        ax.set_title(f'SITI Voronoi')
        # plt.xlim(10, 140)
        # plt.ylim(0, 17)
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
        # ax.set_xticklabels(df.index, rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f'voronoi_{arr}', dpi=100)
        # plt.show()
        dataset[f'group_{arr}'] = kmeans.labels_
    dataset.to_csv('all_spreadsheet_kmeans.csv', index=0)

    print(kmeans.labels_)


def graph1() -> None:
    """
    chunks X dec_time (seconds) and chunks X file_size (Bytes)
    :return:
    """
    dirname = 'graph1'
    os.makedirs(dirname, exist_ok=True)
    df = pd.read_csv('all_spreadsheet-with_names.csv', encoding='utf-8', header=0, index_col=0)
    df['group_3'] = df['group_3'].replace(0, 'AT/BC').replace(1, 'AT/AC').replace(2, 'BT/BC').replace(3, 'BT/AC')
    df = df.sort_values(['u_avg_time', 'group_3'], ascending=[True, False])

    fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)

    ax.plot(df.index, df['u_avg_time'], 'o', color='#FF0000', label='Decode time')
    for group in ['BT/BC', 'BT/AC', 'AT/BC', 'AT/AC']:
        group_data = df[df['group_3'] == group]
        ax.bar(x=group_data.index, height=group_data['rate'], label=f'group_{group}')
    # ax.plot(df.index, df['u_avg_time'], color='#000000', label='_nolegend_')
    # [l] = ax.plot(df.index, df['u_avg_time'] / df['u_avg_time'].max(), color='#000000', label='_nolegend_')
    # l.label = None

    ax.set_ylabel('Size & Time (normalized)')
    ax.set_title(f'Decode time')
    ax.set_ylim(bottom=0)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    # plt.show()
    fig.savefig(f'{dirname}{sl}rates_and_times')
    print('')


def stats(yuv=f'..{sl}yuv-full', project='ffmpeg', bench_stamp='bench: utime'):
    name = []
    dectime = dict(rate=[],
                   u_avg_time=[],
                   u_std_time=[])

    for log_path in glob.glob(f'{project}{sl}*.txt'):
        name.append(os.path.splitext(os.path.basename(log_path))[0])
        video_path = f'{yuv}{sl}{name[-1]}.mp4'
        ut = []

        dur = 1800 / 30
        if log_path in 'ffmpeg\\ninja_turtles.csv':
            dur = 1729 / 30

        dectime['rate'].append(os.path.getsize(f'{video_path}') * 8 / 1000000 / dur)

        with open(f'{log_path}', 'r', encoding='utf-8') as f:
            for line in f:
                if line.find(bench_stamp) >= 0:
                    ut.append(float(line.strip().split(' ')[1][6:-1]) / 1000)

            dectime['u_avg_time'].append(np.average(ut))
            dectime['u_std_time'].append(np.std(ut))

    dectime = pd.DataFrame(dectime, index=name)
    dataset = pd.read_csv(f'todos_os_videos_60s_4k_comprimido{sl}scatter-err_siti.csv', index_col=0)
    dataset = pd.concat([dataset, dectime], axis=1)
    try:
        del (dataset['u_std_time'])
    except KeyError:
        pass

    dectime.to_csv('dectime.csv', encoding='utf-8', index_label='name')
    dataset.to_csv('all_spreadsheet.csv', encoding='utf-8', index_label='name')


if __name__ == "__main__":
    main()
