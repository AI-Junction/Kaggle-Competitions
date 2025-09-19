# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 06:14:05 2017

@author: echtpar
"""
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, default = 1.0, help='what is the first number')
    parser.add_argument('--y', type=float, default = 1.0, help='what is the second number')
    parser.add_argument('--operation', type=str, default = 'add', help='what operation? add, mul, sub, or div')
    args = parser.parse_args()
    sys.stdout.write(str(calc(args)))


def calc(args):
    if args.operation =='add':
        return args.x+args.y
    elif args.operation == 'sub':
        return args.x-args.y
    elif args.operation == 'mul':
        return args.x*args.y
    elif args.operation == 'div':
        return args.x/args.y
        
#operation = calc(7, 3, 'div')
#print (operation)

if __name__ == '__main__':
    main()
    
    