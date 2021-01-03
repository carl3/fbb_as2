; This is a test file for the four-bit-badge assembler
;

Zero	org	0		; We should start at 0 anyway
Five	equ	5
later	equ	next2+1		; can reference addr or const below

; Base opcodes with 2 operands
start	add	R0,R1
	ADC	r2,r3		; Case insensitive
	sub	r4,r5
	sbb	r6,r7
	or	r8,r9
	and	out,in
	xor	jsr,pcl		; No problem with weird instructions
	mov	pcm,pch
; MOV variations
mov	mov	r0,Five		; Confusing symbol names are allowed
	mov	[r3:r4],r0
	mov	r0,[r5:r6]
	mov	[0:1],r0
	mov	r0,[2:3]
	mov	pc,[1:2]
	mov	pc,[100]
	mov	pc,[next]	; No automatic LOW:MED
	jr	-6
	jr	mov		; Address relative to .
	jr	next
	jr	prev3		; Can reference later equ if it has defined expr
	jr	[1:2]
	jr	0x23
	jr	start+3		; Address expressions compute relative
; Extended op codes
next	cp	R0,5		; Isn't cp a subset of mov?
	add	r0,10
	inc	r1
	dec	r2
	dsz	r3
	or	r0,Five+4
	and	r0,0b1100
	xor	r0,0x7
	exr	Five+start-1
	bit	r0,3
	bset	r1,2
	bclr	r2,Five-4
	btg	r3,start	; Address is used as literal
	rrc	r9
	ret	r0,5
	skip	c,0		; What does skip 0 mean?
	skip	nc,nocarry	; Relative address symbol is allowed
	skip	z,2
nocarry	skip	nz,3
; Test zero fill of instructions and symbol def
group3	org	0x030		; Set some fill
; Extra op codes
	nop			; Op that does nothing
stop	hcf			; Jump .
; Long jump
	goto	next
	gosub	start
;
prev3	equ	next+1		; Valid for reference above if next is defined

; Test error codes
	cp	R1,5
	add	r0,foo
	jr	next3		; EQU must be defined before ref
errdef	equ	next3+1		; reference in equ must be before def
next3	equ	next2+1		; Later is OK, but can be referenced in equ
	add	r0,22
	org	32
next2	goto	start-130
	jr	start-130	; An address gets relative, even if negative
	jr	Five-140	; An EQU is an offset
	jr	next2-130	; An address is converted to an offset
	jr	-100		; Valid offset but invalid address
	jr	next3		; next3 is ok here
	mov	pc,[HIGH next+15:MED next+15]	; HIGH/MED eval before +
	bit	pcl,3
	bit	R3,Five
	mov	R4,[3:4]
	skip	c,Five
nocarry	skip	nz,start	; Skip address is always relative

