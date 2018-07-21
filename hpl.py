#!/usr/bin/env python
# -*- coding:utf-8 -*-
#import numpy as np
#from numpy import linalg as LA
#from numpy import *
import numpy as np
import scipy
import scipy.linalg
import time
import logging
logging.basicConfig(level=logging.INFO)

def iterate_func(nr,f,a,b,divsize):
    start = time.time()
    x = None
    for cnt in range(nr):
        msg=u"loop size= {} count={} divsize={}".format(nr,cnt,divsize)
        logging.info(msg)
        x = f(a,b)
    end = time.time() - start
    return x,end
def run_hpl(n,nr,tol=16):
    """
Run the High-performance  LINPACK test on a matrix of size n x n, nr number of times and ensures that the the maximum of the three residuals is strictly less than the prescribed tol erance (defaults to 16).
This function returns the  performance in GFlops/Sec.
    """
    a = np.random.rand(n, n);
    b = np.random.rand(n, 1);
    x,t = iterate_func(nr,np.linalg.solve, a, b,n)
    eps = np.finfo(np.float).eps
    r = np.dot(a, x)-b
    r0 = np.linalg.norm(r, np.inf)
    r1 = r0/(eps * np.linalg.norm(a, 1) * n)
    r2 = r0/(eps * np.linalg.norm(a, np.inf) * np.linalg.norm(x, np.inf) * n)
    performance  = (1e-9* (2.0/3.0 * n * n * n+ 3.0/2.0 * n * n) *nr/t)
    verified     = np.max((r0, r1, r2)) < 16
    if not verified:
        raise RuntimeError, "Solution did not meet the prescribed tolerance %d"% tol
    return performance

def solve_test():
    #A = numpy.array([[1.,2.]    # 行列Aの生成
    #              ,[3.,1.]])
    #B = numpy.array([14.,17.])   # 行列Bの生成
    
    #X = numpy.linalg.solve(A, B)
    # 計算結果の表示
    #print( "X=\n" + str(X) )
    U0 = np.array([[1.,-2.],
                     [0.,1]])
    L0= np.array([[1.,0.],
                     [-3.,0.]])
    B= np.array([14.,17.])

    A0 = np.array([[1,2],
                   [3,1]])
    Ainv = np.linalg.inv(A0)
    print(Ainv)
    x = np.dot(Ainv,B)
    print(x)
    
def main():
    r=run_hpl(10000,3)
    msg= "Linpack benchmark {0:.4f}  GFlops/Sec".format(r)
    logging.info(msg)
    #solve_test()
if __name__=='__main__':main()
