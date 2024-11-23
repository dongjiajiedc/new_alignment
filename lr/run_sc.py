from core import *
import argparse
import warnings
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--cell_path1','-cp1', type=str)
parser.add_argument('--folder_path1','-f1', type=str)
parser.add_argument('--radius1','-r1', type=float,default=15)
parser.add_argument('--capacity1','-c1', type=float,default=0.1)
parser.add_argument('--epoches1','-e1', type=int,default=10)

parser.add_argument('--cell_path2','-cp2', type=str)
parser.add_argument('--folder_path2','-f2', type=str)
parser.add_argument('--radius2','-r2', type=float,default=15)
parser.add_argument('--capacity2','-c2', type=float,default=0.1)
parser.add_argument('--epoches2','-e2', type=int,default=10)

parser.add_argument('--contin', type=str,default='False')
parser.add_argument('--method', type=str,default='average')
parser.add_argument('--alignment', type=int,default=1)
parser.add_argument('--resolution', type=float,default=0.5)
parser.add_argument('--n_pca',type=int,default=50)
parser.add_argument('--meta_col',type=str,default='celltype')

args = parser.parse_args()

if(args.cell_path1 ==None):
    print("Please input the h5 file path for data 1")
    exit()
if(args.cell_path2 ==None):
    print("Please input the h5 file paht for data 2")
    exit()
if(os.path.exists(args.cell_path1)==False):
    print("Input correct path for data 1")
if(os.path.exists(args.cell_path2)==False):
    print("Input correct path for data 2")  
      
if(args.folder_path1 ==None):
    print("Please input folder path to save intermediate files of data1")
    exit()
if(args.folder_path2 ==None):
    print("Please input folder path to save intermediate files of data2")
    exit()
if(os.path.exists(args.folder_path1)==False):
    print("Input correct folder path for data 1")
if(os.path.exists(args.folder_path2)==False):
    print("Input correct folder path for data 2")    
    

cell_path1 = args.cell_path1
cell_path2= args.cell_path2
folder_path1 = args.folder_path1
folder_path2 = args.folder_path2
radius1 = args.radius1
radius2 = args.radius2
c1 = args.capacity1
c2 = args.capacity2
epoches1 = args.epoches1
epoches2 = args.epoches2
contin = str2bool(args.contin)
method = args.method
alignment = args.alignment
resolution = args.resolution
n_pca = args.n_pca
meta_col = args.meta_col
alignment_process(cell_path1,cell_path2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2,meta_col=meta_col,contin=contin,resolution=resolution,method=method,alignment=alignment,n_pca=n_pca)

# python run_sc.py -cp1 './datas/120/1/sample1_small.h5' -f1 "./datas/120/1/" -r1 52.48461374600768 -c1 0.1 -e1 10 -cp2 './datas/120/2/sample1_small.h5' -f2 "./datas/120/2/" -r2 52.43896907992145 -c2 0.1 -e2 10 --contin True --alignment 1 --resolution 1 --n_pca 100

