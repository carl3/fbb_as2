#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is an alternate implementation of an assembler for the Four-Bit-Badge,
inspired by an implementation by Mike Szczys. Mike's version uses a classic
pythonic approach, and seems to be well-written using ordinary token
string manipulation.

This version is an experiment using a more perlesque implementation based
on regular expression processing, to see how these 2 approaches compare.
"""

__version__ = "0.1dev"

DESCRIPTION = """\
Reads input files in assembly and produces a .bin file with 12 bit instructions
coded as pairs of 8 bit bytes, and a .lst file containing a human readable
hex version of the machine code with source code.

With no files given, reads from STDIN, writes the binary to STDOUT,
and listing file to STDERR.
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

#--------------------------------------------------------------

# Convenience routines

def index_dict(l:[str])->Dict[str,int]:
    """
    Returns the dictionary mapping list value to index
    (dict for l.index(s))
    """
    return {v:i for i,v in enumerate(l)}

#-------------------------------------------------------------

# Instruction definitions:
#
# Instructions are 12 bit words, 3 hex digits, coded:
# 
# Ixy - operator with 2 register operands: Opcode Rx,Ry
#       I is base_opcodes 1..8
# 9xn - move register immediate MOV Rx,n
# (?)[RAM Memory] read write, indirect address in registers or constants:
# Axy - move with 3 operands MOV [Rx:Ry],R0
# Bxy - move with 3 operands MOV R0,[Rx:Ry]
# Cnm - move with 2 immediate MOV [n:m],R0
# Dnm - move with 2 immediate MOV R0,[n:m]
# Enm - move with 2 immediate MOV PC,[n:m] (PCH=n,PCM=m)
# Fnm - Jump relative +-127 JR [n:m]
#
# [n:m] can be an integer 0<i<255, or [0<n<15:0<m<15]
# For jr [n:m] can be -128<i<127
#
# Extended opcodes:
# 0In - Operator with R0 and immediate: Opcode R0,n
#       I is ext_opcodes CP ADD OR AND XOR RET
# 0Iy - Operator with one register operand: Opcode Ry
#       I is INC DEC DSZ RRC
# 0In - Operator with immediate operand: Opcode n
#       I is EXR
# Extended opcodes with 2 2 bit (0..3) operands:
# 0Iy:i - Operator with register and bit index: Opcode Ry, i
#       I is BIT BSET BCLR BTG
#       y is R0..R3 i is 0..3
# 0Fc:n - Conditional skip SKIP c,n
#       c is C NC Z NZ
#
# Special Assembler Statements:
#  symbol opcode operands - Defines symbol as the current instruction address
#  symbol EQU value - Defines the symbol as value 
#         ORG value - Advances code memory to value > current addr
#
# Branch:
# GOTO  addr - Short for MOV PC,addr>>4; MOV PCL, addr % 16
# GOSUB addr - Short for MOV PC,addr>>4; MOV JSR, addr % 16
# 
# Values:
#    decimal number            
#    0xhhhh hex number
#    0bnnnn binary number
#    symbol                   symbol defined with EQU or address
#    register                 name of a register
#    value + value            added value left-right precidence
#    value - value            subtracted value left-right precidence
#    -value                   negated value
#    LOW value                low nibble of value (value&0xf)
#    MED value                middle nibble ((value>>4)&0xf)
#    HIGH value               upper nibble ((value>>8)&0xf)
#   

comment_delimiter = ";"     # Character to begin end of line comment
delimiters = ' ,:[]+-'      # Delimiters including space
linend_re = re.compile(r'\s*(;.*)?\s*$') # trim line end
spaces_re = re.compile(r'\s+')  #Convert whitespace to ' '
# Remove spaces around delimiters or split tokens and delimiters
delimiter_split_re = re.compile(r'\s*(['+re.escape(delimiters)+r'])\s*')
# Valid symbol
symbol_pat = r'\b(?!\d)\w+\b'    # Symbols are word characters starting with nondigit
number_pat = r'\b(?:\d+|0X[\da-f]+|0B[01]+)\b' # Decimal, hex, and binary constants

# Registers:
#                                            A   B  C   D   E   F
register_pat="R0|R1|R2|R3|R4|R5|R6|R7|R8|R9|OUT|IN|JSR|PCL|PCM|PCH"
#           0  1 2  3
cond_pat = "C|NC|Z|NZ" # for SKIP
#                 0  1   2   3   4   5  6   7   8   9   A   B   C   D   E   F 
base_opcode_all = "|ADD|ADC|SUB|SBB|OR|AND|XOR|MOV|MOV|MOV|MOV|MOV|MOV|MOV|JR"
#                  0  1   2   3   4   5  6   7   8   9   A     B   C   D   E   F 
ext_opcode_pat = "CP|ADD|INC|DEC|DSZ|OR|AND|XOR|EXR|BIT|BSET|BCLR|BTG|RRC|RET|SKIP"
base_oprxry_pat = "ADD|ADC|SUB|SBB|OR|AND|XOR|MOV" # 1..8 Rx,Ry operands
ext_opr0n_pat = "CP|ADD|OR|AND|XOR|RET" # R0,n operands
ext_opn_pat = "EXR" # Single 4 bit operand
opi8_pat = "JR"  # 8 bit signed operand -128..127
ext_opry_pat = "INC|DEC|DSZ|RRC" # Ry operand
ext_opryi_pat = "BIT|BSET|BCLR|BTG" # R0..R3 and 0..3 bit select
longjump_op_pat = "GOTO|GOSUB" # Assembler emits PC HIGH:MED,LOW in 2 instructions

val4_pat = f"(?:{symbol_pat}|{number_pat}|\\bLOW |\\bMED |\\bHIGH |[+\\-])+"
# [expr(0..255)] or [expr(0..15):expr(0..15)]:
val8_pat = f"\\[((?:{symbol_pat}|{number_pat}|[+\\-])+|({val4_pat}):({val4_pat}))\\]"
# expr(-128:127) or [expr(0..15):expr(0..15)]:
vals8_pat = f"((?:{symbol_pat}|{number_pat}|[+\\-])+|\\[({val4_pat}):({val4_pat})\\])"
val12_pat = f"(?:{symbol_pat}|{number_pat}|[+\\-])+"

# Move opcode values
mov_rxn =  0x9 # 9xn - move register immediate MOV Rx,n
mov_rxy0 = 0xA # Axy - move with 3 operands MOV [Rx:Ry],R0
mov_r0xy = 0xB # Bxy - move with 3 operands MOV R0,[Rx:Ry]
mov_nmr0 = 0xC # Cnm - move with 2 immediate MOV [n:m],R0
mov_r0nm = 0xD # Dnm - move with 2 immediate MOV R0,[n:m]
mov_pcnm = 0xE # Enm - move with 2 immediate MOV PC,[n:m]

nibble_selector_pat = "HIGH|MED|LOW" # Choose a 4 bit nibble from 12 bit value
nibble_select_shift=dict(
    HIGH=8,
    MED=4,
    LOW=0
)

# Additional instruction definitions, no operands:
builtin_opcodes=dict(
    NOP=0xf01,  # JR 1
    HCF=0xf00   # JR 0
)
builtin_opcodes_pat = '|'.join(builtin_opcodes.keys())

error_fill_inst = builtin_opcodes['HCF'] # Instruction for error lines
org_fill_inst = 0   # Fill gap from ORG (CP R0,0)

# Added "opcodes" that are assembler directives:
asm_def_pat = "ORG|EQU" 

# Combined known valid op codes
# Note regex does not allow user defined opcodes
all_opcode_pat = '|'.join([
    ext_opcode_pat+base_opcode_all,
    longjump_op_pat,
    builtin_opcodes_pat,
])

# Lists to map opcode (int) value to names
reg_names = register_pat.split('|')
cond_names = cond_pat.split('|')
ext_opcode_names = ext_opcode_pat.split('|')
base_opcode_names =  base_opcode_all.split('|')

# Dictionary to map opcode names to value
reg_name_dict=index_dict(reg_names)
cond_name_dict=index_dict(cond_names)
ext_opcode_dict=index_dict(ext_opcode_names)
base_opcode_dict=index_dict(base_opcode_names) # Has bogus '' for 0, E for MOV
base_opcode_dict['MOV'] = 8  # Reset MOV to Rx,Ry form, base_oprxry

skip_op = ext_opcode_dict['SKIP']

pcl_reg = reg_name_dict['PCL']
jsr_reg = reg_name_dict['JSR']

#-------------------------------------------------------------
# Regex definitions

# The regex patterns used for parsing are defined and compiled here
# for reference and efficiency. The matches are case insensitive.

# Helper to perform a case insensitive compile
def reI(pattern:str)->Pattern: return re.compile(pattern,flags=re.I)

# Operator operand styles:

oprxry_re = reI(f"^({base_oprxry_pat}) ({register_pat}),({register_pat})$")
opr0n_re = reI(f"^({ext_opr0n_pat}) R0,({val4_pat})$")
opn_re = reI(f"^({ext_opn_pat}) ({val4_pat})$")
opry_re = reI(f"^({ext_opry_pat}) ({register_pat})$")
opryi_re = reI(f"^({ext_opryi_pat}) (R0|R1|R2|R3),({val4_pat})$")
opi8_re = reI(f"^({opi8_pat})\\b ?{vals8_pat}$") # Signed/relative address
mov_rxn_re = reI(f"^MOV ({register_pat}),({val4_pat})$")
mov_r0xy_re = reI(f"^MOV R0,\\[({register_pat}):({register_pat})\\]$")
mov_rxy0_re = reI(f"^MOV\\[({register_pat}):({register_pat})\\],R0$")
mov_r0nm_re = reI(f"^MOV R0,{val8_pat}$")
mov_nmr0_re = reI(f"^MOV\\b ?{val8_pat},R0$")
mov_pcnm_re = reI(f"^MOV PC,{val8_pat}$")
skip_re = reI(f"^SKIP ({cond_pat}),({val4_pat})$")
builtin_opcodes_re = reI(f"^({builtin_opcodes_pat})$")

equline_re = reI(r"^EQU (.*)")
orgline_re = reI(r"^ORG (.*)")
longjump_line_re = reI(f"^({longjump_op_pat}) (.*)")
# We can use sub to trim a line label sym to save the address
# A misspelled opcode in 3 columns is also matched
symboltrim_re = reI(f"^({symbol_pat}) ((ORG|{all_opcode_pat})\\b.*|(?!LOW|MED|HIGH)[A-Z]+ .+)")

all_opcode_re = reI(f"^({all_opcode_pat})\\b")
invalid_opcode_re = reI(r'^([A-Z]\w+)')

help_rxry = "Rx,Ry"
help_r0n = "R0,val(0-f)"
help_ry = "Ry"
help_rxry_r0n = f"{help_rxry} or {help_r0n}"
help_ryi = "Ry,val(0-3) y=R0..R3"
help_addr = "val(0-fff)"
help_builtin = "(no operand)"

# We use a dict to provide the valid syntax for each op in error messages
opcode_help = dict(
    CP=help_r0n,
    ADD=help_rxry_r0n, ADC=help_rxry_r0n, SUB=help_rxry_r0n, SBB=help_rxry_r0n,
    OR=help_rxry_r0n, AND=help_rxry_r0n, XOR=help_rxry_r0n,
    MOV="one of: Rx,Ry Rx,n [Rx:Ry],R0 R0,[Rx:Ry] R0,[n:m] [n:m],R0 PC,[n:m]",
    INC=help_ry, DEC=help_ry, DSZ=help_ry, RRC=help_ry,
    EXR="val(0-f)",
    BIT=help_ryi, BSET=help_ryi, BCLR=help_ryi, BTG=help_ryi,
    JR="val(-127..128)",
    RET=help_r0n,
    SKIP="c,val(0-3) c=C|NC|Z|NZ",
    GOTO=help_addr, GOSUB=help_addr,
)
for w in builtin_opcodes.keys():
    opcode_help[w] = help_builtin


#-------------------------------------------------------------
# Globals:

# We use globals for the arguments and the IO file handles, and
# input state to simplify writing output.

args = {} # Set in main()

# Current file IO are set to stdio
infile = sys.stdin
binfile = sys.stdout
outfile = sys.stderr
msgfile = sys.stderr

#-------------------------------------------------------------

def parse_args():
    """
    Parse sys.argv and return a Namespace object.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION,
                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--version', action='version', version='%(prog)s '+__version__)
    #parser.add_argument('-v', '--verbose', action='store_true',
    #                    help='enable verbose info printout')
    #parser.add_argument('-D', '--debug', action='store_true',
    #                    help='enable debug logging')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="omit the listing output (errors only)")
    parser.add_argument('-n', '--linenumbers', action='store_true',
                        help="enable line numbers")
    parser.add_argument('-b', '--binfile', type=str,
                        help="binary file name else input with .bin extension")
    parser.add_argument('-o', '--outfile', type=str,
                        help="output file name else input with .lst extension")

    parser.add_argument('asmfile', nargs='?', default='-',
                        help='assembly language file to be processed')

    args = parser.parse_args()
    return args

#-------------------------------------------------------------

class InputContext:
    """
    We use an object to hold the contents of an input line
    both in the original raw text and trimmed simplified
    text for easy pattern matching. We include a regex
    test that returns the match status as well as saves
    the match groups for a search.

    This class holds the trimmed input line being processed,
    and can retain regex matches so we can use a regex search
    or substitution on the input in an if expression, then
    later reference the match groups (even in sub).

    Only one object is created and used.
    """
    def __init__(self):
        self.reset()
        self.lineno = 0     # Current line number
        self.addr = 0       # Current binary output address
        self.err_count = 0  # Number of error lines
        self.rawline = ''   # Original input line including \n
        self.l = ''         # Trimmed and simplified line


    def reset(self):
        self.errmsg = []    # Error messages for rawline
        self.m = None       # Saved match object
        self.mg = None      # Match groups or none
        self.inst = None    # Instruction word to emit
        self.instx = None   # Additional instruction to follow (GOTO)
        self.symbol = ''    # Symbol name at the line begin

    def newline(self, l:str):
        """
        Use newline to supply the next source input line. The string
        is saved in rawline, and a simplified version, l, with comments
        stripped, multiple whitespace converted to a single space, and
        whitespace around delimiters and the end of line stripped.
        The simplified version makes regex expressions less complex.

        If there is a symbol/label at the beginning of the line,
        we strip it and save it in symbol. Leading spaces are
        removed.

        The line number lineno is advanced.

        The errmsg[] is cleared to accumulate error messages for
        the new input line.

        The assembler is assumed to be case insensitive, so 
        the input line is converted to upper case when trimmed.
        """
        self.reset()
        self.lineno += 1
        self.rawline = l

        # Trim the source
        # Remove line end comment and spaces, convert to upper case
        l = linend_re.sub('',l).strip('\f').upper() # s/\s*(;.*)?\s*$//
        # Convert multiple whitespace, tabs, etc. to plain single space
        l = spaces_re.sub(' ',l) # s/\s+/ /g
        # Remove extra spaces around delimiters
        self.l = delimiter_split_re.sub(r'\1',l) # s/\s*([$delimiters])\s*/$1/g;

        if self.sub(symboltrim_re,"{1}"):
            # This does not support user defined opcodes
            self.symbol = self.mg[0] # Save symbol found at the line begin
        else:
            self.l = self.l.lstrip() # Trim leading space

    def search(self, pattern:Pattern)->Match:
        """
        The search does a re.search on l using the supplied pattern, and
        returns the resulting match object or None. The match
        object is saved in m, and mg set to the groups with null
        matches set to ''.
        """
        self.m = re.search(pattern, self.l)
        self.mg = self.m.groups('') if self.m else None
        return self.m

    def sub(self,
                pattern:Pattern,    # Pattern to match
                repl:str='',        # Repl str or format
                count=1,            # Number of substitutions
                formatted=True      # If "repl" is formatted {0},{1} on match groups
            )->int:                 # Returns number of substitutions
        """
        Performs a substitution on l using the specified pattern
        and replacement string. The last match is saved in m,
        and matched groups set in mg. The replacement string is
        actually a format string so instead of \1 \2, etc.,
        use {0} {1}, etc. 
        """
        def rfunc(m):
            self.m = m
            self.mg = m.groups('')
            if formatted:
                return repl.format(*self.mg)
            return repl

        self.m = None
        (self.l, self.nsubs) = re.subn(pattern, rfunc, self.l, count)
        return self.nsubs

    def error(self, msg:str):
        """
        Appends an error string with the message with input line to
        errmsg
        """
        self.errmsg.append(f"**{self.lineno} Error: {msg}\n")


inp = InputContext()
outwords = array('H') # Use python array to hold output words

symbols = {}      # Dictionary of defined symbols
symbolisaddr = set() # Symbol is defined as memory addr
symboldefline = {} # Line where a symbol has been defined

# Saved source input
inlines = []

#--------------------------------------------------------------

class ParserError(Exception):
    pass

# [lexp op][LOW|MED|HIGH ][-]rval
expr_re = reI(r'^(?:(.+)([\-\+]))?(?:(HIGH|MED|LOW) )?(\-)?(\w+)$')
value_re = reI(f"^(?:({symbol_pat})|(\\d+)|0X([\\da-f]+)|0B([01]+))$")


# Expression processing routines

def eval_expr(e:str)->Tuple[int,bool]:
    """
    Parse and evaluate en expression
    Returns the int value and boolean indicating this is an address
    If the value is an address, we can use it in a relative jump
    """
    isaddr = False

    # Parse the expression into [lexpr op] [LOW|MED|HIGH ][-]rvalue
    # Left to right precidence means evaluate the left hand side
    # expression recursively then apply to right side number or symbol
    m = re.match(expr_re, e)
    if not m: raise ParserError(f"Invalid expression '{e}'")
    lexpr, op, sel, sign, rval = m.groups('')

    m = re.match(value_re, rval)
    # rval must be a symbol or number in decimal, hex, or binary
    if not m: raise ParserError(f"Invalid value '{rval}'")
    sym, dval, hexval, bval = m.groups()

    if sym:
        v = symbols.get(sym,None)
        if v==None: raise ParserError(f"Undefined symbol '{sym}'")
        isaddr = sym in symbolisaddr
    elif bval:
        v = int(bval,2)
    elif hexval:
        v = int(hexval,16)
    else:
        v = int(dval)
    if sign:
        v = -v
    if sel:
        v = (v>>nibble_select_shift[sel]) & 15
        isaddr = False
        
    if lexpr:
        # Evaluate the expression up to the rightmost operator
        lv, lisaddr = eval_expr(lexpr)
        if lisaddr:
            isaddr = True
        if op=='+':
            v = lv + v
        elif op=='-':
            v = lv - v
    return v, isaddr

def validate_int(expr:Union[str,int],   #expression or int value
                 minv:int,              # Min allowed value
                 maxv:int,              # Max allowed value
                 rel:bool=False         # Address symbols are relative
                )->int:
    """
    Verifies the value is between min and max. If not,
    issue an error. If the value is none, an error has already
    been issued.
    """
    if isinstance(expr, int):
        v = expr
    else:
        try:
            v, isaddr = eval_expr(expr)
        except ParserError as E:
            inp.error(E.args[0])
            return 0

    if isaddr and rel:
        # Convert an address reference to relative
        if v<0 or v>= 4096:
            inp.error(f"The value of {expr} is an invalid address {v} (must be 0..4095)")
            v = inp.addr
        v = v - inp.addr
        # Skip is relative to addr+1
        if maxv==3:
            v -= 1
    elif rel:
        if not 0 <= v+inp.addr <= 4095:
            inp.error(f"The relative value of {expr} is an invalid address {v+inp.addr} (must be 0..4095)")

    if minv <= v <= maxv:
        if v < 0:
            v = 256 + v # Same as "v &= 0xff" with 2s complement
        return v
    # Issue range check
    inp.error(f"The value of {expr} is out of range: {v}, must be {minv}..{maxv}"
                       if isaddr or not re.match(r'-?\d+$',expr) else
                    f"The value {expr} must be {minv}..{maxv}" )
    return 0

def int2(expr:str)->int:
    return validate_int(expr, 0, 3)
    
def int2r(expr:str)->int:
    # Address relative with range 3 is addr+1
    return validate_int(expr, 0, 3, True)
    
def int4(expr:str)->int:
    return validate_int(expr, 0, 15)
    
def int8(expr:str)->int:
    return validate_int(expr, 0, 255)

def int12(expr:str)->int:
    return validate_int(expr, 0, 4095)

def ints(expr:str)->int:
    return validate_int(expr, -8192, 8192)

def ints8(expr:str)->int:
    return validate_int(expr, -128, 127, True)

#--------------------------------------------------------------

def op2(op:int,val8:int)->int:
    return (op<<8)+val8

def op3(op:int,val4a:int,val4b:int)->int:
     return (op<<8)+(val4a<<4)+val4b
   
def oprxry(op:int,rx:str,ry:str)->int:
    # Instructions with rx,ry
    return op3(op,reg_name_dict[rx], reg_name_dict[ry])
   
def opval8(op:int,val8:str,val4a:str,val4b:str)->int:
    """
    Instruction with n:m or 0..255 operand. val4a/b is null
    if an 8 bit expression is used.
    """
    return (op3(op, int4(val4a), int4(val4b))
               if val4a!='' else
             op2(op, int8(val8)))
   

#--------------------------------------------------------------

# Symbol and ORG processing

def reset_org(addrstr:str):
    """
    Set the address to the specified value that must be >=
    to the current address.
    """
    v = int12(addrstr)
    if inp.errmsg:
        return
    if v<inp.addr:
        inp.error(f"Origin must be >= {inp.addr:03x} ({inp.addr})")
    else:
        inp.addr = v

def symboldef(name:str, val:Union[int,str]):
    """
    Creates/validates a symbol definition. If the val is an int
    it is assumed to be an address. If a str, then it may evaluate
    to an int or address expression.
    """

    if isinstance(val, str):
        lasterr = len(inp.errmsg)
        try:
            val, isaddr = eval_expr(val)
        except ParserError as E:
            inp.error(E.args[0])
            val = None
            isaddr = False
        if lasterr > len(inp.errmsg):
            val = None
    else:
        isaddr = True

    if name in symboldefline:
        if inp.lineno != symboldefline[name]:
            inp.error(f"Symbol {name} is already defined on line {inp.lineno}")
            return
        elif val!=symbols[name] and symbols[name]!=None:
            inp.error(f"Symbol {name} definition error {val} !=  {symbols[name]}")
            return

    symbols[name] = val
    symboldefline[name]=inp.lineno
    if isaddr:
        symbolisaddr.add(name)

def processDefs(
        outwords:Union[None,array], # None for pass 1 or output binary
    ):
    """
    Handles the common processing for address (column 0 symbol),
    ORG and EQU definitions for pass 1 and pass 2 of the assembler.
    During pass 1 definitions are collected, in pass 2 definitions
    are checked and error messages issued. In the case of ORG,
    the output array is filled with padding during pass 2 when
    the origin is advanced.
    """
    # Handle ORG/EQU before assigning a symbol to addr
    if inp.search(orgline_re):
        # ORG val
        reset_org(inp.mg[0])
        if outwords:
            # Fill output instructions in the gap
            outwords.extend([org_fill_inst]*(inp.addr-len(outwords)))
        # Now we can assign the symbol
        # Clear l to indicate we already processed the op
        inp.l = ''
    elif  inp.search(equline_re):
        # EQU value
        # Define the symbol with right-hand-side value
        symboldef(inp.symbol,inp.mg[0])
        inp.symbol = inp.l = '' # Indicate we processed the symbol and op
    
    if inp.symbol:
        # Save the current address in the symbol
        # Todo: 
        symboldef(inp.symbol, inp.addr)



#--------------------------------------------------------------

# Top-level assembler parser

def assemble_file(
        inlines:[str],  # List of input text lines
        outwords:array, # halfword (16b) machine code output
        outfile:TextIO, # Listing output file
        ):
    """
    Reads the input line in 2 passes and creates the outwords
    array of machine instructions, and writes a listing file
    if outfile is not null, and copies error lines to stdout.

    The first pass collects symbol definitions, either machine
    code addresses, or constants defined with an EQU statement.

    During pass 2, the input lines are processed, symbol
    substitutions made, and the machine code is emitted
    into the outwords array. The machine code, assembly input,
    and error messages are written to the outfile listing if
    defined, and only lines with errors written to stderr.
    """

    # Pass 1, collect symbol definitions
    for inline in inlines:
        inp.newline(inline)
        processDefs(None)

        if inp.l=='': continue

        elif inp.search(longjump_line_re):
            inp.addr += 2 # GOTO/GOSUB emits 2 words
        else:
            inp.addr +=  1 # Assume all other lines emit 1 word
        
        if inp.addr >= 4096:
            break

    # Pass 2, emit machine code and listing
    inp.lineno = 0     # Current line number
    inp.addr = 0       # Current binary output address
    for inline in inlines:
        inp.newline(inline)
        processDefs(outwords)

        if inp.l=='':
            # Nothing to do, keep inst as none
            pass
        elif inp.search(longjump_line_re):
            # GOTO/GOSUB label
            opcode, addr = inp.mg
            addr = int12(addr)
            inp.inst = op2(mov_pcnm,addr>>4)
            inp.instx = op3(mov_rxn,
                            jsr_reg if opcode=='GOSUB' else pcl_reg,
                            addr & 15)
        elif inp.search(oprxry_re):
            # ADD Rx,Ry
            opcode, rx, ry = inp.mg
            inp.inst = oprxry(base_opcode_dict[opcode], rx, ry)
        elif inp.search(opry_re):
            # INC Ry
            opcode, ry = inp.mg
            inp.inst = op3(0, ext_opcode_dict[opcode], reg_name_dict[ry])
        elif inp.search(opr0n_re) or inp.search(opn_re):
            # ADD R0,n  EXR n
            opcode, n = inp.mg
            inp.inst = op3(0, ext_opcode_dict[opcode], int4(n))
        elif inp.search(opryi_re):
            # BSET R1,3
            opcode, ry, i = inp.mg
            inp.inst = op3(0, ext_opcode_dict[opcode],
                (reg_name_dict[ry]<<2)+int2(i))
        elif inp.search(opi8_re):
            # JR -20   JR Loopstart
            opcode, val8, n, m = inp.mg
            op = base_opcode_dict[opcode]
            # Signed 8 bit int can be a symbol relative to inp.addr
            inp.inst = (op3(op, int4(n), int4(m))
                            if n!='' else
                        op2(op, ints8(val8)))
        elif inp.search(mov_rxn_re):
            # MOV R2, 5
            rx, n = inp.mg
            inp.inst = op3(mov_rxn, reg_name_dict[rx], int4(n))
        elif inp.search(mov_r0xy_re):
            # MOV R0,[R2:R3]
            inp.inst = oprxry(mov_r0xy, *inp.mg)
        elif inp.search(mov_rxy0_re):
            # MOV [R2:R3],R0
            inp.inst = oprxry(mov_rxy0, *inp.mg)
        elif inp.search(mov_r0nm_re):
            # MOV R0,[9:6]
            inp.inst = opval8(mov_r0nm,  *inp.mg)
        elif inp.search(mov_nmr0_re):
            # MOV [9:6],R0
            inp.inst = opval8(mov_nmr0,  *inp.mg)
        elif inp.search(mov_pcnm_re):
            # MOV PC,[9:6]
            inp.inst = opval8(mov_pcnm,  *inp.mg)
        elif inp.search(skip_re):
            # SKIP CN, 3
            # Should we be able to use a label with addr?
            c, n = inp.mg
            inp.inst = op3(0, skip_op,  (cond_name_dict[c]>>2)+int2r(n))
        elif inp.search(builtin_opcodes_re):
            # HCF
            inp.inst = builtin_opcodes[inp.mg[0]]
        else:
            # There is an error, do some matching ti make error messages
            inp.inst = error_fill_inst # Assume error lines emit 1 word
            if inp.search(all_opcode_re):
                op = inp.mg[0]
                inp.error(f"Op {op} must have operands {opcode_help[op]}")
            elif inp.search(invalid_opcode_re):
                op = inp.mg[0]
                inp.error(f"Opcode {op} is invalid")
            else:
                inp.error("Invalid syntax - Unrecognized line")
                    

        # Prepare the listing hex columns and append the machine code inst
        if inp.instx!=None:
            if inp.errmsg:
                 inp.inst = inp.instx =  error_fill_inst  
            hexvals = f"{inp.addr:03x} {inp.inst:03x} {inp.instx:03x}"
            outwords.append(inp.inst)
            outwords.append(inp.instx)
            inp.addr += 2
        elif inp.inst!=None:
            if inp.errmsg:
                 inp.inst = error_fill_inst  
            hexvals = f"{inp.addr:03x} {inp.inst:03x}    "        
            outwords.append(inp.inst)
            inp.addr += 1
        else:
            hexvals = "           "

        if inp.addr >= 4096:
            inp.error("Instruction memory overflows 4096 words")
            outwords = outwords[:4096]

        # Format the listing line with hex, source, and errors
        if args.linenumbers:
            hexvals = f"{inp.lineno:4d} {hexvals}"
        outline = f"{hexvals}\t{inp.rawline}{''.join(inp.errmsg)}"
        if outfile:
            outfile.write(outline)
        if inp.errmsg and msgfile:
            msgfile.write(outline)

        if inp.addr >= 4096:
            break
        
#--------------------------------------------------------------

# Write the assembled binary output

def put_binfile(outwords:array, binfile:BinaryIO):
    """
    Computes the header and writes the output array to the file
    """
    checksum = array('H',[sum(outwords) % 0xffff])
    header = array('H',[0xff00,0xff00,0xc3a5])
    if sys.byteorder=='big':
        # Swap before writing
        checksum.byteswap()
        header.byteswap()
        outwords.byteswap()
    header.tofile(binfile)
    outwords.tofile(binfile)
    checksum.tofile(binfile)

#--------------------------------------------------------------

# Main program to process command line options

def main():
    global args, inlines, outfile, msgfile, outwords

    args = parse_args()

    msgfile = sys.stderr

    f = args.asmfile
    basename = os.path.splitext(f)[0]
    if f=='-':
        infile = sys.stdin
        args.binfile = args.binfile or '-'
        args.outfile = args.outfile
    else:
        if not os.path.isfile(f):
            print(f"Input file {f} not found", file=sys.stderr)
            return
        infile = open(f, encoding=ENCODING)
        args.binfile = args.binfile or basename+".bin"
        args.outfile = args.outfile or basename+".lst"

    # Read and store the input
    inlines = infile.readlines()
    infile.close()

    if not inlines:
        print(f"Input file {f} is empty", file=sys.stderr)
        return

    outfile = (None if args.quiet or not args.outfile
                else open(args.outfile, 'w', encoding=ENCODING))   

    assemble_file(inlines, outwords, outfile)
    if outfile:
        outfile.close()

    # Note we could omit the output if there are errors
    if outwords:
        binfile = (sys.stdout if args.binfile=='-'
                   else open(args.binfile, 'wb'))
        put_binfile(outwords, binfile)
        binfile.close()



if __name__ == '__main__':
    main()
