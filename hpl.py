#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cupy as cn
#import numpy as cn
import numpy as np
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
    accuracy=cn.float32
    a = cn.random.rand(n, n).astype(accuracy);
    b = cn.random.rand(n, 1).astype(accuracy);
    x,t = iterate_func(nr,cn.linalg.solve, a, b,n)
    eps = cn.finfo(accuracy).eps
    r = cn.dot(a, x)-b
    r0 = cn.linalg.norm(r, cn.inf)
    r1 = r0/(eps * cn.linalg.norm(a, 1) * n)
    r2 = r0/(eps * cn.linalg.norm(a, cn.inf) * cn.linalg.norm(x, cn.inf) * n)
    performance  = (1e-9* (2.0/3.0 * n * n * n+ 3.0/2.0 * n * n) *nr/t)
    verified     = np.max((r0.get(), r1.get(), r2.get())) < 16
    msg='performance={} verified={} r0={} r1={} r2={}'.format(performance,verified,r0,r1,r2)
    logging.info(msg)
    if not verified:
        err="Solution did not meet the prescribed tolerance {}".format(tol)
        raise RuntimeError(err)
    return performance
    

def main():
    r=run_hpl(15000,5)
    msg= "Linpack benchmark {0:.4f}  GFlops/Sec".format(r)
    logging.info(msg)
if __name__=='__main__':main()
