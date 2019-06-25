#include "sds_lib.h"
#include "fc_fpga.h"
#include "common.h"
#include "check.h"

void fc_fpga(Dtype *In, Dtype *Param_l, Dtype *Param_m, Dtype *Param_s, Dtype *Out) {

	Dtype bufferA[BUFA_DEPTH];
	Dtype bufferB[BUFB_DEPTH];

	int i_num_l, o_num_l = 0;
	int i_num_m, o_num_m = 0;
	int i_num_s, o_num_s = 0;

	int wtiles_l, wtiles_m, wtiles_s = 0; 

	//load input
	fc_buf_read(In, bufferA);

	for (int lyr = 0; lyr < FC_LAYER_NUM; lyr++) {
		
		switch (lyr) {
		case 0: {i_num_l = 2688; o_num_l = 64; wtiles_l = 42; i_num_m = 2688; o_num_m = 128; wtiles_m = 42; i_num_s = 2688; o_num_s = 256; wtiles_s = 42; break; }
		case 1: {i_num_l = 65; o_num_l = 48; wtiles_l = 1; i_num_m = 129; o_num_m = 96; wtiles_m = 2; i_num_s = 257; o_num_s = 192; wtiles_s = 4; break; }
		case 2: {i_num_l = 49; o_num_l = 1; wtiles_l = 1; i_num_m = 97; o_num_m = 4; wtiles_m = 2; i_num_s = 193; o_num_s = 16; wtiles_s = 3; break; }
		default: {i_num_l = 0; o_num_l = 0; i_num_m = 0; o_num_m = 0; i_num_s = 0; o_num_s = 0; break;}
		}

		//fc in one layer
		fc_lyr_l(bufferA, Param_l, bufferB, i_num_l, o_num_l, wtiles_l, lyr);
		fc_lyr_m(bufferA, Param_m, bufferB, i_num_l, o_num_l, i_num_m, o_num_m, wtiles_m, lyr);
		fc_lyr_s(bufferA, Param_s, bufferB, i_num_l, o_num_l, i_num_m, o_num_m, i_num_s, o_num_s, wtiles_s, lyr);

		//buffer reset
		if (1 == lyr) {
			for (int n = 0; n < BUFB_DEPTH; n++) {
#pragma HLS pipeline
				bufferB[n] = 0;
			}
		}
		else {
			for (int n = 0; n < BUFA_DEPTH; n++) {
#pragma HLS pipeline
				bufferA[n] = 0;
			}
		}

		Param_l += i_num_l * o_num_l + o_num_l;
		Param_m += i_num_m * o_num_m + o_num_m;
		Param_s += i_num_s * o_num_s + o_num_s;
	}

	fc_buf_write(bufferB, Out);
	return;
}

void fc_lyr_l(
	Dtype BufferA[BUFA_DEPTH],
	Dtype *Param_l,
	Dtype BufferB[BUFB_DEPTH],
	int Inum_l,
	int Onum_l,
	int Wtiles_l,
	int Lyr
) 
{

	Dtype w_buf_l[65][64];
#pragma HLS array_partition variable=w_buf_l complete dim=1
	Dtype b_buf_l[65];
#pragma HLS array_partition variable=b_buf_l complete dim=1

	//load bias
	fc_bias_read_l(Param_l + Inum_l * Onum_l, b_buf_l, Onum_l);

	//l layer
	for (int wtil = 0; wtil < Wtiles_l; wtil++) {
#pragma HLS loop_tripcount min=1 max=42

		int read_len_l = 0;
		int ichnl_l = 0;

		switch (Lyr) {
		case 0: {read_len_l = 64; ichnl_l = 64; break; }
		case 1: {read_len_l = 65; ichnl_l = 64; break; }
		case 2: {read_len_l = 49; ichnl_l = 48; break; }
		}

		fc_weight_read_l(Param_l + wtil * read_len_l * Onum_l, w_buf_l, Onum_l, read_len_l);
		fc_compute_l(BufferA, BufferB, w_buf_l, b_buf_l, Onum_l, wtil, Lyr, read_len_l, wtil == Wtiles_l - 1, ichnl_l);
	}
}

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
)
{
	Dtype w_buf_m[65][128];
#pragma HLS array_partition variable=w_buf_m complete dim=1
	Dtype b_buf_m[128];
#pragma HLS array_partition variable=b_buf_m complete dim=1

	//load bias
	fc_bias_read_m(Param_m + Inum_m * Onum_m, b_buf_m, Onum_m);

	//m layer
	for (int wtil = 0; wtil < Wtiles_m; wtil++) {
#pragma HLS loop_tripcount min=2 max=42

		int read_len_m = 0;
		int ichnl_m = 0;

		if (0 == Lyr) {
			read_len_m = 64;
			ichnl_m = 64;
		}
		else if (1 == Lyr) {
			read_len_m = (wtil == Wtiles_m - 1) ? 65 : 64;
			ichnl_m = 64;
		}
		else {
			read_len_m = (wtil == Wtiles_m - 1) ? 33 : 64;
			ichnl_m = (wtil == Wtiles_m - 1) ? 32 : 64;
		}

		fc_weight_read_m(Param_m + wtil * 64 * Onum_m, w_buf_m, Onum_m, read_len_m);
		fc_compute_m(BufferA, BufferB, w_buf_m, b_buf_m, Onum_m, wtil, Lyr, read_len_m, wtil == Wtiles_m - 1, ichnl_m, Inum_l, Onum_l);
	}
}

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
)
{
	Dtype w_buf_s[65][256];
#pragma HLS array_partition variable=w_buf_s complete dim=1
	Dtype b_buf_s[256];
#pragma HLS array_partition variable=b_buf_s complete dim=1

	//load bias
	fc_bias_read_s(Param_s + Inum_s * Onum_s, b_buf_s, Onum_s);

	//s layer
	for (int wtil = 0; wtil < Wtiles_s; wtil++) {
#pragma HLS loop_tripcount min=3 max=42

		int read_len_s = 0;
		int ichnl_s = 0;

		if (0 == Lyr) {
			read_len_s = 64;
			ichnl_s = 64;
		}
		else {
			read_len_s = (wtil == Wtiles_s - 1) ? 65 : 64;
			ichnl_s = 64;
		}

		fc_weight_read_s(Param_s + wtil * 64 * Onum_s, w_buf_s, Onum_s, read_len_s);
		fc_compute_s(BufferA, BufferB, w_buf_s, b_buf_s, Onum_s, wtil, Lyr, read_len_s, wtil == Wtiles_s - 1, ichnl_s, Inum_l, Onum_l, Inum_m, Onum_m);
	}
	return;
}

//load conv_flat input *********************************************************************************************************************************

void fc_buf_read(Dtype *In, Dtype Buffer[BUFA_DEPTH]) {
	for (int n = 0; n < 2688; n++) {
#pragma HLS pipeline
		Buffer[n] = *In++;
	}
	return;
}

//load bias *********************************************************************************************************************************

void fc_bias_read_l(Dtype *Param_l, Dtype *B_buf_l, int Onum_l) {
	for (int m = 0; m < Onum_l; m++) {
#pragma HLS loop_tripcount min=1 max=64
#pragma HLS pipeline
		B_buf_l[m] = *Param_l++;
	}
	return;
}

void fc_bias_read_m(Dtype *Param_m, Dtype *B_buf_m, int Onum_m) {
	for (int m = 0; m < Onum_m; m++) {
#pragma HLS loop_tripcount min=4 max=128
#pragma HLS pipeline
		B_buf_m[m] = *Param_m++;
	}
	return;
}

void fc_bias_read_s(Dtype *Param_s, Dtype *B_buf_s, int Onum_s) {
	for (int m = 0; m < Onum_s; m++) {
#pragma HLS loop_tripcount min=16 max=256
#pragma HLS pipeline
		B_buf_s[m] = *Param_s++;
	}
	return;
}

//load weight *********************************************************************************************************************************

void fc_weight_read_l(Dtype *Param_l, Dtype Wbuf_l[65][64], int Onum_l, int Len) {

	for (int m = 0; m < Len; m++) {
#pragma HLS loop_tripcount min=49 max=65
		for (int n = 0; n < Onum_l; n++) {
#pragma HLS loop_tripcount min=1 max=64
#pragma HLS pipeline
			Wbuf_l[m][n] = *Param_l++;
		}
	}
	return;
}

void fc_weight_read_m(Dtype *Param_m, Dtype Wbuf_m[65][128], int Onum_m, int Len) {

	for (int m = 0; m < Len; m++) {
#pragma HLS loop_tripcount min=33 max=65
		for (int n = 0; n < Onum_m; n++) {
#pragma HLS loop_tripcount min=4 max=128
#pragma HLS pipeline
			Wbuf_m[m][n] = *Param_m++;
		}
	}
	return;
}

void fc_weight_read_s(Dtype *Param_s, Dtype Wbuf_s[65][256], int Onum_s, int Len) {

	for (int m = 0; m < Len; m++) {
#pragma HLS loop_tripcount min=64 max=65
		for (int n = 0; n < Onum_s; n++) {
#pragma HLS loop_tripcount min=16 max=256
#pragma HLS pipeline
			Wbuf_s[m][n] = *Param_s++;
		}
	}
	return;
}

//fc compute *********************************************************************************************************************************
void fc_compute_l(Dtype BufferA[BUFA_DEPTH], 
	Dtype BufferB[BUFB_DEPTH], 
	Dtype Wbuf_l[65][64], 
	Dtype B_buf_l[65], 
	int Onum_l, int Wtil, int Lyr, int Ichnl, int Last,int Ichnl_real) {

	Dtype part[65];
#pragma HLS array_partition variable=part complete dim=1

	Dtype *in_buf, *out_buf;

	if (1 == Lyr) {
		in_buf = BufferB;
		out_buf = BufferA;
	}
	else {
		in_buf = BufferA;
		out_buf = BufferB;
	}

	for (int n = 0; n < Ichnl_real; n++) {
#pragma HLS loop_tripcount min=48 max=64
#pragma HLS pipeline
		part[n] = in_buf[Wtil * 64 + n];
	}

	if ((1 == Lyr || 2 == Lyr) && (1 == Last)) {
		part[Ichnl_real] = QP;
	}

	for (int m = 0; m < Onum_l; m++) {
#pragma HLS loop_tripcount min=1 max=64
#pragma HLS pipeline
		Dtype result = (Dtype)0.0;
		for (int n = 0; n < Ichnl; n++) {
			Dtype muls = part[n] * Wbuf_l[n][m];
			result += muls;
		}
		Dtype partial = out_buf[m];
		partial += result;
		out_buf[m] = (Last) ? ((partial + B_buf_l[m]) < 0 ? (Dtype)(0.2 * (partial + B_buf_l[m])) : (partial + B_buf_l[m])) : partial;
	}
	return;
}

void fc_compute_m(Dtype BufferA[BUFA_DEPTH],
	Dtype BufferB[BUFB_DEPTH],
	Dtype Wbuf_m[65][128],
	Dtype B_buf_m[128],
	int Onum_m, int Wtil, int Lyr, int Ichnl, int Last, int Ichnl_real, int Ichnl_l, int Onum_l) {

	Dtype part[65];
#pragma HLS array_partition variable=part complete dim=1

	Dtype *in_buf, *out_buf;

	if (1 == Lyr) {
		in_buf = BufferB;
		out_buf = BufferA;
	}
	else {
		in_buf = BufferA;
		out_buf = BufferB;
	}

	int offset = (0 == Lyr) ? 0 : (Ichnl_l - 1);

	for (int n = 0; n < Ichnl_real; n++) {
#pragma HLS loop_tripcount min=32 max=64
#pragma HLS pipeline
		part[n] = in_buf[offset + Wtil * 64 + n];
	}

	if ((1 == Lyr || 2 == Lyr) && (1 == Last)) {
		part[Ichnl_real] = QP;
	}

	for (int m = 0; m < Onum_m; m++) {
#pragma HLS loop_tripcount min=4 max=64
#pragma HLS pipeline
		Dtype result = (Dtype)0.0;
		for (int n = 0; n < Ichnl; n++) {
			Dtype muls = part[n] * Wbuf_m[n][m];
			result += muls;
		}
		Dtype partial = out_buf[Onum_l + m];
		partial += result;
		out_buf[Onum_l + m] = (Last) ? ((partial + B_buf_m[m]) < 0 ? (Dtype)(0.2 * (partial + B_buf_m[m])) : (partial + B_buf_m[m])) : partial;
	}
	return;
}

void fc_compute_s(Dtype BufferA[BUFA_DEPTH],
	Dtype BufferB[BUFB_DEPTH],
	Dtype Wbuf_s[65][256],
	Dtype B_buf_s[256],
	int Onum_s, int Wtil, int Lyr, int Ichnl, int Last, int Ichnl_real, int Ichnl_l, int Onum_l, int Ichnl_m, int Onum_m) {

	Dtype part[65];
#pragma HLS array_partition variable=part complete dim=1

	Dtype *in_buf, *out_buf;

	if (1 == Lyr) {
		in_buf = BufferB;
		out_buf = BufferA;
	}
	else {
		in_buf = BufferA;
		out_buf = BufferB;
	}

	int offset = (0 == Lyr) ? 0 : (Ichnl_l + Ichnl_m - 2);

	for (int n = 0; n < Ichnl_real; n++) {
#pragma HLS loop_tripcount min=64 max=64
#pragma HLS pipeline
		part[n] = in_buf[offset + Wtil * 64 + n];
	}

	if ((1 == Lyr || 2 == Lyr) && (1 == Last)) {
		part[Ichnl_real] = QP;
	}

	for (int m = 0; m < Onum_s; m++) {
#pragma HLS loop_tripcount min=16 max=64
#pragma HLS pipeline
		Dtype result = (Dtype)0.0;
		for (int n = 0; n < Ichnl; n++) {
			Dtype muls = part[n] * Wbuf_s[n][m];
			result += muls;
		}
		Dtype partial = out_buf[Onum_l + Onum_m + m];
		partial += result;
		out_buf[Onum_l + Onum_m + m] = (Last) ? ((partial + B_buf_s[m]) < 0 ? (Dtype)(0.2 * (partial + B_buf_s[m])) : (partial + B_buf_s[m])) : partial;
	}
	return;
}

//write final result*********************************************************************************************************************************
void fc_buf_write(Dtype Buffer[BUFB_DEPTH], Dtype *Out) {
	for (int m = 0; m < CLASS_NUM; m++) {
#pragma HLS loop_tripcount min=10 max=10
#pragma HLS pipeline
		*Out++ = Buffer[m];
	}
}
