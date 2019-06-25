#include "common.h"

#pragma SDS data access_pattern(In:SEQUENTIAL, Param_l:SEQUENTIAL, Param_m:SEQUENTIAL, Param_s:SEQUENTIAL, Out:SEQUENTIAL)
#pragma SDS data mem_attribute(In:PHYSICAL_CONTIGUOUS, Param_l:PHYSICAL_CONTIGUOUS, Param_m:PHYSICAL_CONTIGUOUS, Param_s:PHYSICAL_CONTIGUOUS, Out:PHYSICAL_CONTIGUOUS)
#pragma SDS data copy(In[0:2688], Param_l[0:L_LENGTH], Param_m[0:M_LENGTH], Param_s[0:S_LENGTH], Out[0:CLASS_NUM])
void fc_fpga(Dtype *In, Dtype *Param_l, Dtype *Param_m, Dtype *Param_s, Dtype *Out);

void fc_lyr_l(
	Dtype BufferA[BUFA_DEPTH],
	Dtype *Param_l,
	Dtype BufferB[BUFB_DEPTH],
	int Inum_l,
	int Onum_l,
	int Wtiles_l,
	int Lyr
);

void fc_lyr_m(
	Dtype BufferA[BUFA_DEPTH],
	Dtype *Param_m,
	Dtype BufferB[BUFB_DEPTH],
	int Inum_l,
	int Onum_l,
	int Inum_m,
	int Onum_m,
	int Wtiles_m,
	int Lyr
);

void fc_lyr_s(
	Dtype BufferA[BUFA_DEPTH],
	Dtype *Param_s,
	Dtype BufferB[BUFB_DEPTH],
	int Inum_l,
	int Onum_l,
	int Inum_m,
	int Onum_m,
	int Inum_s,
	int Onum_s,
	int Wtiles_s,
	int Lyr
);

void fc_buf_read(Dtype *In, Dtype Buffer[BUFA_DEPTH]);

void fc_bias_read_l(Dtype *Param_l, Dtype *B_buf_l, int Onum_l);

void fc_bias_read_m(Dtype *Param_m, Dtype *B_buf_m, int Onum_m);

void fc_bias_read_s(Dtype *Param_s, Dtype *B_buf_s, int Onum_s);

void fc_weight_read_l(Dtype *Param_l, Dtype Wbuf_l[65][64], int Onum_l, int Len);

void fc_weight_read_m(Dtype *Param_m, Dtype Wbuf_m[65][128], int Onum_m, int Len);

void fc_weight_read_s(Dtype *Param_s, Dtype Wbuf_s[65][256], int Onum_s, int Len);

void fc_compute_l(Dtype BufferA[BUFA_DEPTH],
	Dtype BufferB[BUFB_DEPTH],
	Dtype Wbuf_l[65][64],
	Dtype B_buf_l[65],
	int Onum_l, int Wtil, int Lyr, int Ichnl, int Last, int Ichnl_real);

void fc_compute_m(Dtype BufferA[BUFA_DEPTH],
	Dtype BufferB[BUFB_DEPTH],
	Dtype Wbuf_m[65][128],
	Dtype B_buf_m[128],
	int Onum_m, int Wtil, int Lyr, int Ichnl, int Last, int Ichnl_real, int Ichnl_l, int Onum_l);

void fc_compute_s(Dtype BufferA[BUFA_DEPTH],
	Dtype BufferB[BUFB_DEPTH],
	Dtype Wbuf_s[65][256],
	Dtype B_buf_s[256],
	int Onum_s, int Wtil, int Lyr, int Ichnl, int Last, int Ichnl_real, int Ichnl_l, int Onum_l, int Ichnl_m, int Onum_m);

void fc_buf_write(Dtype Buffer[BUFB_DEPTH], Dtype *Out);

