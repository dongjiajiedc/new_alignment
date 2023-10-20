from core import *
import argparse
import warnings
import os
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

parser.add_argument('--contin', type=bool,default=False)

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
contin = args.contin
alignment_process_st(cell_path1,cell_path2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2,contin,resolution=1)

# python run_sc.py -cp1 './datas/d1/sample.h5' -f1 "./datas/d1/" -r1 50 -c1 0.001 -e1 10 -cp2 './datas/d2/sample.h5' -f2 "./datas/d2/" -r2 50 -c2 0.001 -e2 10
