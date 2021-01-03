#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is an implementation for a disassembler for the Four-Bit-Badge,
inspired by an implementation by Mike Szczys. There is no particular
advantage over Mike's version. This is just a complement to the
alternate assembler demonstrating regular expression parsing.
"""

__version__ = "0.1dev"

DESCRIPTION = """\
Reads binary files in maching code and produces a .s file with the
machine code in hex and assembly text, or just text with -C.

With no files given or name '-', reads from STDIN, and 
writes the assembly to STDOUT.
"""

ENCODING = 'UTF-8'
#-------------------------------------------------------------

import argparse
import os.path
import re
import string
import sys

from array import array

from typing import (Union, List, Pattern, Match, Dict, Tuple, BinaryIO,
    TextIO)

#-------------------------------------------------------------

def parse_args():
    """
    Parse sys.argv and return a Namespace object.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--version', action='version', version='%(prog)s '+__version__)
#    parser.add_argument('-v', '--verbose', action='store_true',
#                        help='enable verbose info printout')
#    parser.add_argument('-D', '--debug', action='store_true',
#                        help='enable debug logging')
# Not implemented
#    parser.add_argument('-l', '--labels', action='store_true',
#                        help="show labels for jump targets")
    parser.add_argument('-C', '--nocode', action='store_true',
                        help="don't show address and machine code")
    parser.add_argument('-o', '--outfile', type=str,
                        help="output file name else input with .s extension")
    parser.add_argument('binfile', nargs='?', default='-',
                        help='machine code binary file to be disassembled')

    args = parser.parse_args()
    return args

#-------------------------------------------------------------

registers="R0|R1|R2|R3|R4|R5|R6|R7|R8|R9|OUT|IN|JSR|PCL|PCM|PCH".split('|')
conds="C|NC|Z|NZ".split('|')
base_opcodes = "|ADD|ADC|SUB|SBB|OR|AND|XOR|MOV|MOV|MOV|MOV|MOV|MOV|MOV|JR".split('|')
ext_opcodes = "CP|ADD|INC|DEC|DSZ|OR|AND|XOR|EXR|BIT|BSET|BCLR|BTG|RRC|RET|SKIP".split('|')

#-------------------------------------------------------------

# Assembly by instruction type

def op_rxry(op:str,rx:int,ry:int)->str:
    return f"{op}\t{registers[rx]},{registers[ry]}"

def op_rxn(op:str,rx:int,n:int)->str:
    return f"{op}\t{registers[rx]},{n}"

def op_rxryr0(op:str,rx:int,ry:int)->str:
    return f"{op}\t[{registers[rx]}:{registers[ry]}],R0"

def op_r0rxry(op:str,rx:int,ry:int)->str:
    return f"{op}\tR0,[{registers[rx]}:{registers[ry]}]"

def op_nmr0(op:str,n:int,m:int)->str:
    return f"{op}\t[0x{n:x}:0x{m:x}],R0"

def op_r0nm(op:str,n:int,m:int)->str:
    return f"{op}\tR0,[0x{n:x}:0x{m:x}]"

def op_pcnm(op:str,n:int,m:int)->str:
    return f"{op}\tPC,[0x{n:x}:0x{m:x}]"

def op_jr(op:str,n:int,m:int)->str:
    return f"{op}\t[0x{n:x}:0x{m:x}]"

base_func = [
    None, op_rxry, op_rxry, op_rxry, op_rxry, op_rxry, op_rxry, op_rxry,
    op_rxry, op_rxn, op_rxryr0, op_r0rxry, op_nmr0, op_r0nm, op_pcnm, op_jr,
]

def op_r0n(op:str,n:int)->str:
    return f"{op}\tR0,{n}"

def op_rx(op:str,rx:int)->str:
    return f"{op}\t{registers[rx]}"

def op_n(op:str,n:int)->str:
    return f"{op}\t{n}"

def op_bit(op:str,n:int)->str:
    return f"{op}\t{registers[n>>2]},{n & 3}"

def op_skip(op:str,n:int)->str:
    return f"{op}\t{conds[n>>2]},{n & 3}"

ext_func = [
    op_r0n, op_r0n, op_rx, op_rx, op_rx, op_r0n, op_r0n, op_r0n,
    op_n, op_bit, op_bit, op_bit, op_bit, op_rx, op_r0n, op_skip,
]

#--------------------------------------------------------------

# Main program to process command line options

def main():
    global args
    global addr         # Current machine code address
    global lastcode     # Prior instruction (for jump)

    args = parse_args()

    f = args.binfile
    if f=='-':
        infile = sys.stdin
        size = 8192+8
    else:
        if not os.path.isfile(f):
            print(f"Input file {f} not found", file=sys.stderr)
            return
        size = os.path.getsize(f)
        if (size % 2)!=0 or size<10: # 3 word header + data + 1 word checksum
            print(f"Input file {f} has an invalid length", file=sys.stderr)
            return

        infile = open(f, 'rb')

    # Read the machine code input as halfword array
    machinecode = array('H')
    try:
        machinecode.fromfile(infile,int(size/2))
    except EOFError:
        pass
    infile.close()

    if machinecode[0]==0x00ff:
        machinecode.byteswap()
    if machinecode[0]!=0xff00 or machinecode[1]!=0xff00 or machinecode[2]!=0xc3a5:
        print(f"Input file {f} has an invalid header", file=sys.stderr)
        return

    checksum = machinecode[-1]
    machinecode = machinecode[3:-1]
    if checksum!=sum(machinecode) % 0xffff:
        print(f"Input file {f} has an invalid checksum", file=sys.stderr)
        return

    outname = args.outfile or ('-' if f=='-' else os.path.splitext(f)[0]+'.s')
    outfile = sys.stdout if outname=='-' else open(outname,'w')

    lastcode = 0
    for addr, code in enumerate(machinecode):
        n1 = (code>>8) & 15
        n2 = (code>>4) & 15
        n3 = code & 15
        asm = (base_func[n1](base_opcodes[n1],n2,n3)
                    if n1 else
               ext_func[n2](ext_opcodes[n2],n3))

        # Annotate jumps
        if n1==15:
            n = code & 0xff
            if n>127:
                n -= 256
            asm += f"\t; GOTO {(addr+n):03x}" if n!=0 else "\t; HCF"
        elif (code & 0xfe0)==0x9c0 and (lastcode & 0xf00)==0xe00:
            op = 'GOTO' if n2==13 else 'GOSUB'
            a = addr + ((lastcode&0xff)<<4)+n3
            asm += f"\t; {op} {a:03x}"
        if args.nocode:
            print(asm,file=outfile)
        else:
            print(f"{addr:03x} {code:03x} {asm}",file=outfile)

        lastcode = code

    outfile.close()

#-------------------------------------------------------------

if __name__ == '__main__':
    main()
