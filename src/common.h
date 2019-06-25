#ifndef _COMMON_H_
#define _COMMON_H_

//data type
typedef float Dtype;

#define IMG_W 64
#define IMG_H 64
#define CONV_LAYER_NUM 3
#define FC_LAYER_NUM 3
#define CLASS_NUM 21

//input channel tile size
#define Tn 8
#define ITILESHIFT 3
//output channel tile size
#define Tm 16
#define OTILESHIFT 4

#define I_BUF_DEPTH_L 16*16
#define I_BUF_DEPTH_M 32*32
#define I_BUF_DEPTH_S 64*64

#define W_BUF_DEPTH 16

#define O_BUF_DEPTH_L 4*4
#define O_BUF_DEPTH_M 8*8
#define O_BUF_DEPTH_S 16*16

#define B_BUF_DEPTH 2

//fc params
#define BUFA_DEPTH 2688
#define BUFB_DEPTH 451 //65+129+257
#define QP 22.0 / 51.0
#define L_LENGTH 175314 //2688*64+64+65*48+48+49*1+1
#define M_LENGTH 357064 //2688*128+128+129*96+96+97*4+4
#define S_LENGTH 741024 //2688*256+256+257*192+192+193*16+16


#endif // !_COMMON_H_
