#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cupy as cn
#import numpy as cn
import numpy as np
import time
import logging
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", help="specify demensions",default=25000,type=int)
parser.add_argument("-l", help="loop count",default=20,type=int)
parser.add_argument("--type", help="specify fp32,fp64",default='fp32',choices=['fp32','fp64'])
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

def iterate_func(nr,f,a,b,divsize,mempool):
    start = time.time()
    x = None
    for cnt in range(nr):
        umem = 4 * mempool.used_bytes() // (1024*1024)
        msg=u"loop size= {} count={} divsize={} umem={} ".format(nr,cnt,divsize,umem)
        logging.info(msg)
        x = f(a,b)
    end = time.time() - start
    return x,end
def run_hpl(n,nr,tol=16):
    """
Run the High-performance  LINPACK test on a matrix of size n x n, nr number of times and ensures that the the maximum of the three residuals is strictly less than the prescribed tol erance (defaults to 16).
This function returns the  performance in GFlops/Sec.
    """
    mempool = cn.get_default_memory_pool()
    if args.type=='fp32':
        accuracy=cn.float32
    if args.type=='fp64':
        accuracy=cn.float64
    a = cn.random.rand(n, n).astype(accuracy);
    b = cn.random.rand(n, 1).astype(accuracy);
    x,t = iterate_func(nr,cn.linalg.solve, a, b,n,mempool)
    eps = cn.finfo(accuracy).eps
    r = cn.dot(a, x)-b
    r0 = cn.linalg.norm(r, cn.inf)
    r1 = r0/(eps * cn.linalg.norm(a, 1) * n)
    r2 = r0/(eps * cn.linalg.norm(a, cn.inf) * cn.linalg.norm(x, cn.inf) * n)
    performance  = (1e-9* (2.0/3.0 * n * n * n+ 3.0/2.0 * n * n) *nr/t)
    verified     = np.max((r0.get(), r1.get(), r2.get())) < 16
    umem = 4 * mempool.used_bytes() // (1024*1024)
    msg='performance={} umem={} verified={} r0={} r1={} r2={}'.format(performance,umem,verified,r0,r1,r2)
    logging.info(msg)
    if not verified:
        err="Solution did not meet the prescribed tolerance {}".format(tol)
        raise RuntimeError(err)
    return performance,umem
    

def main():
    r,umem=run_hpl(args.d,args.l)
    msg= f"Linpack benchmark dimensions={args.d}  gpu memory={umem} loop={args.l} type={args.type} {r:.4f}  GFlops/Sec"
    logging.info(msg)
if __name__=='__main__':main()
