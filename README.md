# Four Bit Badge Assembler/Dissassembler

This is an alternate implementation of an assembler for the Four-Bit-Badge, inspired by an implementation by Mike Szczys. Mike's version uses a classic pythonic approach, and seems to be well-written using ordinary token string manipulation.

This version is an experiment using a more perlesque implementation based on regular expression processing, to see how these 2 approaches compare.

## Incomplete Implementation

***Warning: This is an incomplete implementation. Some questions on
the requirements and specifications remain to be addressed.***

### To run

For command line help:
```
fbb_as2.py -h
fbb_dis2.py -h
```

### Testing

To run a test, execute the `test.sh` script. This will assemble test input with valid statements and error statements. The resulting output binary is compared with a reference implementation. The reference binary is dissassembled and the source code generated is assembled and compared with the
original.

Test files:

* `test.s` Assembly test with valid and error statements
* `test.lst` Assembly listing output
* `test.bin` Assembly binary output
* `test2.bin` Reference copy of test.bin
* `test2.s` Dissassembled output with hex code
* `test3.s` Dissassembled test2.bin without hex code
* `test3.bin` Re-assembled test3.s binary output
* `test3.lst` Listing output of test3.s assembly




