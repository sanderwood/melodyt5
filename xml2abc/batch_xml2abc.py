import os
import math
import argparse
import subprocess
from tqdm import trange
from unidecode import unidecode
from multiprocessing import Pool

cmd = 'cd '+os.getcwd()
output = os.popen(cmd).read()
cmd = 'cmd /u /c python xml2abc.py -m 2 -c 6 -x '

def parse_args():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('path', type=str, help='The path to the directory containing the files')
    return parser.parse_args()

def run_filter(lines):
    score = ""
    for line in lines.replace("\r", "").split("\n"):
        if line[:2] in ['A:', 'B:', 'C:', 'D:', 'F:', 'G', 'H:', 'N:', 'O:', 'R:', 'r:', 'S:', 'T:', 'V:', 'W:', 'X:', 'Z:'] \
        or line=='\n' \
        or line.startswith('%'):
            continue
        else:
            if '%' in line:
                line = line.split('%')
                line = ''.join(line[:-1])
            score += line + '\n'
    score = score.strip()
    if score.endswith(" |"):
        score += "]"     
    return score

def convert_abc(file_list):
    for file_idx in trange(len(file_list)):
        file = file_list[file_idx]
        filename = os.path.basename(file)
        directory = os.path.dirname(file)
        path_parts = directory.split(os.sep)
        if path_parts:
            path_parts[0] = path_parts[0] + "_abc"
        new_directory = os.path.join(*path_parts)
        os.makedirs(new_directory, exist_ok=True)
        try:
            p = subprocess.Popen(cmd+'"'+file+'"', stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            result = p.communicate()
            output = result[0].decode('utf-8')
            output = run_filter(output)
            output = unidecode(output)

            if output=='':
                continue
            else:
                with open(new_directory+'/'+filename[:-4]+'.abc', 'w', encoding='utf-8') as f:
                    f.write(output)
        except Exception as e:
            print(e)
            pass
            
if __name__ == '__main__':
    args = parse_args()
    file_list = []
    abc_list = []
    
    for root, dirs, files in os.walk(args.path):
        for file in files:
            filename = os.path.join(root, file)
            file_list.append(filename)

    file_lists = []
    for i in range(os.cpu_count()):
        start_idx = int(math.floor(i*len(file_list)/os.cpu_count()))
        end_idx = int(math.floor((i+1)*len(file_list)/os.cpu_count()))
        file_lists.append(file_list[start_idx:end_idx])
    
    pool = Pool(processes=os.cpu_count())
    pool.map(convert_abc, file_lists)