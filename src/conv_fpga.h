#ifndef _CONV_FPGA_H_
#define _CONV_FPGA_H_

#include "common.h"

//conv in fpga;include three modules
#pragma SDS data access_pattern(In_l:SEQUENTIAL, In_m:SEQUENTIAL, In_s:SEQUENTIAL)
#pragma SDS data access_pattern(Params_l:SEQUENTIAL, Params_m:SEQUENTIAL, Params_s:SEQUENTIAL)
#pragma SDS data access_pattern(Out_l:SEQUENTIAL, Out_m:SEQUENTIAL, Out_s:SEQUENTIAL)
#pragma SDS data mem_attribute(In_l:PHYSICAL_CONTIGUOUS, In_m:PHYSICAL_CONTIGUOUS, In_s:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(Params_l:PHYSICAL_CONTIGUOUS, Params_m:PHYSICAL_CONTIGUOUS,Params_s:PHYSICAL_CONTIGUOUS)
#pragma SDS data mem_attribute(Out_l:PHYSICAL_CONTIGUOUS, Out_m:PHYSICAL_CONTIGUOUS, Out_s:PHYSICAL_CONTIGUOUS)
#pragma SDS data copy(In_l[0:Ichnl*InRownum_l*InColnum_l], In_m[0:Ichnl*InRownum_m*InColnum_m], In_s[0:Ichnl*InRownum_s*InColnum_s])
#pragma SDS data copy(Params_l[0:Ochnl+Ichnl*Ochnl*Kern*Kern], Params_m[0:Ochnl+Ichnl*Ochnl*Kern*Kern], Params_s[0:Ochnl+Ichnl*Ochnl*Kern*Kern])
#pragma SDS data copy(Out_l[0:Ochnl*OutRownum_l*OutColnum_l], Out_m[0:Ochnl*OutRownum_m*OutColnum_m], Out_s[0:Ochnl*OutRownum_s*OutColnum_s])
void conv_fpga(Dtype *In_l,
	Dtype *In_m,
	Dtype *In_s,
	Dtype *Params_l,
	Dtype *Params_m,
	Dtype *Params_s,
	Dtype *Out_l,
	Dtype *Out_m,
	Dtype *Out_s,
	int Lyr,
	int InRownum_l,
	int InColnum_l,
	int InRownum_m,
	int InColnum_m,
	int InRownum_s,
	int InColnum_s,
	int OutRownum_l,
	int OutColnum_l,
	int OutRownum_m,
	int OutColnum_m,
	int OutRownum_s,
	int OutColnum_s,
	int Kern,
	int Ichnl,
	int Ochnl,
	int Isec,
	int Osec);

void conv_l(
	Dtype *In_l,
	Dtype *Params_l,
	Dtype *Out_l,
	int Ichnl,
	int Ochnl,
	int Kern,
	int InRownum_l,
	int InColnum_l,
	int OutRownum_l,
	int OutColnum_l,
	int IchnlTil,
	int ISec,
	int OSec,
	int Lyr);

void conv_ichnl_l(
	Dtype *In_l,
	Dtype *Params_l,
	Dtype Bbuf_l[Tm][B_BUF_DEPTH],
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	int IchnlTil,
	int OchnlTil,
	int Ochnl,
	int Kern,
	int Ni,
	int Lyr,
	int InRownum_l,
	int InColnum_l,
	int Rownum_l,
	int Colnum_l,
	int ISec
);

void conv_input_read_l(Dtype *In_l, Dtype InBuf_l[Tn][I_BUF_DEPTH_L], int IchnlTil, int InRownum_l, int InColnum_l, int Sec);

void conv_buf_write_l(
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	Dtype *Out_l,
	int OchnlTil,
	int Rownum_l,
	int Colnum_l
);

void conv_compute_l(
	Dtype InBuf_l[Tn][I_BUF_DEPTH_L],
	Dtype WBuf_l[Tn * Tm][W_BUF_DEPTH],
	Dtype BBuf_l[Tm][B_BUF_DEPTH],
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	int InRownum_l,
	int InColnum_l,
	int Rownum_l,
	int Colnum_l,
	int Kern,
	int IchnlTil,
	int OchnlTil,
	int OSec,
	int ISec
);

void conv_bias_read_l(Dtype *Params_l, Dtype Bbuf_l[Tm][B_BUF_DEPTH], int OSec);

void conv_weight_read_l(
	Dtype *Params_l,
	Dtype Wbuf_l[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern);

void conv_m(
	Dtype *In_m,
	Dtype *Params_m,
	Dtype *Out_m,
	int Ichnl,
	int Ochnl,
	int Kern,
	int InRownum_m,
	int InColnum_m,
	int OutRownum_m,
	int OutColnum_m,
	int IchnlTil,
	int ISec,
	int OSec,
	int Lyr);

void conv_ichnl_m(
	Dtype *In_m,
	Dtype *Params_m,
	Dtype Bbuf_m[Tm][B_BUF_DEPTH],
	Dtype OutBuf_m[Tm][O_BUF_DEPTH_M],
	int IchnlTil,
	int OchnlTil,
	int Ochnl,
	int Kern,
	int Ni,
	int Lyr,
	int InRownum_m,
	int InColnum_m,
	int Rownum_m,
	int Colnum_m,
	int ISec
);

void conv_input_read_m(Dtype *In_m, Dtype InBuf_m[Tn][I_BUF_DEPTH_M], int IchnlTil, int InRownum_m, int InColnum_m, int Sec);

void conv_buf_write_m(
	Dtype OutBuf_m[Tm][O_BUF_DEPTH_M],
	Dtype *Out_m,
	int OchnlTil,
	int Rownum_m,
	int Colnum_m
);

void conv_compute_m(
	Dtype InBuf_m[Tn][I_BUF_DEPTH_M],
	Dtype WBuf_m[Tn * Tm][W_BUF_DEPTH],
	Dtype BBuf_m[Tm][B_BUF_DEPTH],
	Dtype OutBuf_m[Tm][O_BUF_DEPTH_M],
	int InRownum_m,
	int InColnum_m,
	int Rownum_m,
	int Colnum_m,
	int Kern,
	int IchnlTil,
	int OchnlTil,
	int OSec,
	int ISec
);

void conv_bias_read_m(Dtype *Params_m, Dtype Bbuf_m[Tm][B_BUF_DEPTH], int OSec);

void conv_weight_read_m(
	Dtype *Params_m,
	Dtype Wbuf_m[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern);

void conv_s(
	Dtype *In_s,
	Dtype *Params_s,
	Dtype *Out_s,
	int Ichnl,
	int Ochnl,
	int Kern,
	int InRownum_s,
	int InColnum_s,
	int OutRownum_s,
	int OutColnum_s,
	int IchnlTil,
	int ISec,
	int OSec,
	int Lyr);

void conv_ichnl_s(
	Dtype *In_s,
	Dtype *Params_s,
	Dtype Bbuf_s[Tm][B_BUF_DEPTH],
	Dtype OutBuf_s[Tm][O_BUF_DEPTH_S],
	int IchnlTil,
	int OchnlTil,
	int Ochnl,
	int Kern,
	int Ni,
	int Lyr,
	int InRownum_s,
	int InColnum_s,
	int Rownum_s,
	int Colnum_s,
	int ISec
);

void conv_input_read_s(Dtype *In_s, Dtype InBuf_s[Tn][I_BUF_DEPTH_S], int IchnlTil, int InRownum_s, int InColnum_s, int Sec);

void conv_buf_write_s(
	Dtype OutBuf_s[Tm][O_BUF_DEPTH_S],
	Dtype *Out_s,
	int OchnlTil,
	int Rownum_s,
	int Colnum_s
);

void conv_compute_s(
	Dtype InBuf_s[Tn][I_BUF_DEPTH_S],
	Dtype WBuf_s[Tn * Tm][W_BUF_DEPTH],
	Dtype BBuf_s[Tm][B_BUF_DEPTH],
	Dtype OutBuf_s[Tm][O_BUF_DEPTH_S],
	int InRownum_s,
	int InColnum_s,
	int Rownum_s,
	int Colnum_s,
	int Kern,
	int IchnlTil,
	int OchnlTil,
	int OSec,
	int ISec
);

void conv_bias_read_s(Dtype *Params_s, Dtype Bbuf_s[Tm][B_BUF_DEPTH], int OSec);

void conv_weight_read_s(
	Dtype *Params_s,
	Dtype Wbuf_s[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern);



#endif // !_CONV_FPGA_H_
