#!/bin/sh
#
# Test assembler & dissasembler

./fbb_as2.py test.s 2>test.err
# test2 is the reference copy of the binary output
echo diff test.bin test2.bin
diff test.bin test2.bin && echo No differences

# test2.s is disassembly with hex code
# test3.s is disassembly without hex code
./fbb_dis2.py test2.bin
./fbb_dis2.py -C -o test3.s test2.bin

# assemble dissassembled source and compare
./fbb_as2.py test3.s
echo diff test3.bin test2.bin && echo No differences
diff test3.bin test2.bin