# Useful stuff
# Max Tegmark, March 2020

import csv, os
#import xlrd
import numpy as np

####################################################
##### Tools for loading and saving spreadsheets 
##### The csv tools require "import csv"
####################################################

def loadcsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f))

def loadtsv(filename):
    with open(filename, newline='') as f: 
        return list(csv.reader(f,delimiter="\t"))
        
def robustloadcsv(filename):
    if os.path.exists(filename):
        return loadcsv(filename)
    else:
        return []

def robustloadtsv(filename):
    if os.path.exists(filename):
        return loadtsv(filename)
    else:
        return []
	
def savecsv(filename,list):
    with open(filename,"w",newline="") as f:
        csv.writer(f).writerows(list)
        f.flush()

def savetsv(filename,list):
    with open(filename, "w", newline="") as f:
        csv.writer(f,delimiter="\t").writerows(list)
        f.flush()

def savetxt(filename,stringlist):
    with open(filename,"w",newline="") as f:
        f.writelines('\n'.join(stringlist)) 

def appendcsv(filename,list):
    with open(filename,"a", newline="") as f:
        csv.writer(f).writerows(list)
        f.flush()

def appendtsv(filename,list):
    with open(filename,"a", newline="") as f:
        csv.writer(f,delimiter="\t").writerows(list)
        f.flush()


#def loadxlsx(filename): ### Requires "import xlrd"
#    def strippa(x):
#        if isinstance(x,str):
#            return x.strip()
#        else:
#            return x
#    def trim(row):
#        k = len(row)
#        while k>0 and row[k-1] == "": k = k - 1
#        return row[:k]
#    wb = xlrd.open_workbook(filename, on_demand=True)
#    ws = wb.sheet_by_index(0)
#    return [trim([strippa(ws.cell_value(i,j)) for j in range(ws.ncols)]) for i in range(ws.nrows)] 

def flatten(listoflists): return list([j for i in listoflists for j in i])

def str2int(lst): return list(map(int,lst))

############################################
### These tools require import numpy as np
############################################

def log2(x): return np.log(x)/np.log(2)

# This version only works on numpy arrays:
#def mysort(arr,column): 
#    key = np.array([a[column] for a in arr])
#    return arr[key.argsort()]

def mysort(arr,column): 
    key = np.array([a[column] for a in arr])
    idx = key.argsort()
    return [arr[i] for i in idx]

# Returns integer positions of s in list:
def listPositions(lst,s): 
    return list(np.arange(len(lst))[list(L==s for L in lst)])

###############################################
### Sparse matrix multiplication without SciPy
###############################################

# A sparse vector svec = (idx,val)
def sparseDot(svec,vec):
    (idx,val) = svec
    sum([val[i]*vec[i] for i in idx])

def sparseMatVecMul(sMat,vec):
    (idx,val) = svec
    list([sparseDot(svec,vec) for svec in sMat])


