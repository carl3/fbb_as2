ADD	R0,R1
ADC	R2,R3
SUB	R4,R5
SBB	R6,R7
OR	R8,R9
AND	OUT,IN
XOR	JSR,PCL
MOV	PCM,PCH
MOV	R0,5
MOV	[R3:R4],R0
MOV	R0,[R5:R6]
MOV	[0x0:0x1],R0
MOV	R0,[0x2:0x3]
MOV	PC,[0x1:0x2]
MOV	PC,[0x6:0x4]
MOV	PC,[0x1:0x7]
JR	[0xf:0xa]	; GOTO 00a
JR	[0xf:0x7]	; GOTO 008
JR	[0x0:0x5]	; GOTO 017
JR	[0x0:0x5]	; GOTO 018
JR	[0x1:0x2]	; GOTO 026
JR	[0x2:0x3]	; GOTO 038
JR	[0xe:0xd]	; GOTO 003
CP	R0,5
ADD	R0,10
INC	R1
DEC	R2
DSZ	R3
OR	R0,9
AND	R0,12
XOR	R0,7
EXR	4
BIT	R0,3
BSET	R1,2
BCLR	R2,1
BTG	R3,0
RRC	R9
RET	R0,5
SKIP	C,0
SKIP	C,1
SKIP	C,2
SKIP	C,3
CP	R0,0
CP	R0,0
CP	R0,0
CP	R0,0
CP	R0,0
CP	R0,0
JR	[0x0:0x1]	; GOTO 031
JR	[0x0:0x0]	; HCF
MOV	PC,[0x0:0x1]
MOV	PCL,7	; GOTO 04a
MOV	PC,[0x0:0x0]
MOV	JSR,0	; GOSUB 035
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0xf:0xb]	; GOTO 03b
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
JR	[0x0:0x0]	; HCF
