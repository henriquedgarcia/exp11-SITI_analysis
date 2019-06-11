#!/bin/env python3
import glob
import os
import platform
import subprocess
import time

sl = '\\'
ffmpeg = 'ffmpeg.exe'

if platform.system() not in 'Windows':
    sl = '/'
    ffmpeg = 'ffmpeg'


def main():
    yuv = f'..{sl}yuv-full'
    project = 'ffmpeg'
    os.makedirs(project, exist_ok=True)

    for n in range(5):
        print(f'Rodada {n}')

        for video in glob.glob(f'{yuv}{sl}*.mp4'):
            name = video.split(sl)[-1][:-4]
            command = (f'{ffmpeg} '
                       f'-hide_banner -threads 1 -nostats -benchmark -i {video} '
                       f'-f null -')

            run_bench(command, f'{project}{sl}{name}', 'txt')


def run_bench(command, log_path, ext, overwrite=True, log_mode='a'):
    if os.path.isfile(f'{log_path}.{ext}') and not overwrite:
        print(f'arquivo {log_path}.{ext} existe. Pulando.')
    else:
        attempts = 1
        while True:
            print(command)

            try:
                with open('temp.txt', 'w', encoding='utf-8') as f:
                    p = subprocess.run(command, shell=True, stdout=f, stderr=subprocess.STDOUT)
                    print(f'Returncode = {p.returncode}')
                    if p.returncode != 0:
                        print(f'Tentativa {attempts}. Erro. Exitcode == {p.returncode}. Tentando novamente.')
                        attempts += 1
                        continue

                with open('temp.txt', 'r', encoding='utf-8') as f1, \
                        open(f'{log_path}.{ext}', log_mode, encoding='utf-8') as f2:
                    f2.write(f1.read())
                    break

            except FileNotFoundError:
                print(f'Tentativa {attempts}. Erro ao abrir o arquivo {"temp.txt" + ".log"}')
                print('Tentando novamente em 5s.')
                attempts += 1
                time.sleep(5)


if __name__ == '__main__':
    main()
