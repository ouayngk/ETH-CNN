#include "sds_lib.h"

#include "conv_fpga.h"
#include "common.h"
#include "sub_sample.h"

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
	int Osec) {

	int itile = Lyr == 0 ? 1 : Tn;

	conv_l(In_l, Params_l, Out_l, Ichnl, Ochnl, Kern, InRownum_l, InColnum_l, OutRownum_l, OutColnum_l, itile, Isec, Osec, Lyr);

	conv_m(In_m, Params_m, Out_m, Ichnl, Ochnl, Kern, InRownum_m, InColnum_m, OutRownum_m, OutColnum_m, itile, Isec, Osec, Lyr);

	conv_s(In_s, Params_s, Out_s, Ichnl, Ochnl, Kern, InRownum_s, InColnum_s, OutRownum_s, OutColnum_s, itile, Isec, Osec, Lyr);

	return;

}

//layer l *********************************************************************************************************************************

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
	int Lyr) {

	//set on-chip buffer
	static Dtype b_buf_l[Tm][B_BUF_DEPTH];
#pragma HLS array_partition variable= b_buf_l complete dim=1

	static Dtype out_buf_l[Tm][O_BUF_DEPTH_L];
#pragma HLS array_partition variable= out_buf_l complete dim=1

	conv_bias_read_l(Params_l, b_buf_l, OSec);

	//output channel
	for (int n = 0; n < OSec; n++) {
//#pragma HLS dataflow
#pragma HLS loop_tripcount min=1 max=2
		int otile = Lyr == 1 ? ((n == 1) ? 8 : Tm) : Tm;

		conv_ichnl_l(In_l, Params_l, b_buf_l, out_buf_l, IchnlTil, otile, Ochnl, Kern, n, Lyr, InRownum_l, InColnum_l, OutRownum_l, OutColnum_l, ISec);

		conv_buf_write_l(
			out_buf_l,
			Out_l + n * Tm * OutRownum_l * OutColnum_l,
			otile,
			OutRownum_l,
			OutColnum_l
		);

	}
	return;

}

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
) {
	static Dtype in_buf_l[Tn][I_BUF_DEPTH_L];
#pragma HLS array_partition variable= in_buf_l complete dim=1

	static Dtype w_buf_l[Tn * Tm][W_BUF_DEPTH];
#pragma HLS array_partition variable= w_buf_l complete dim=1

	//input channel
	for (int m = 0; m < ISec; m++) {
#pragma HLS loop_tripcount min=1 max=3

		if (Ni == 0) {
			conv_input_read_l(In_l + (m << ITILESHIFT) * InRownum_l * InColnum_l, in_buf_l, IchnlTil, InRownum_l, InColnum_l, m);
		}

		int otileshift = Lyr == 1 ? ((Ni == 1) ? 3 : OTILESHIFT) : OTILESHIFT;
		int ochnl = Lyr == 1 ? ((Ni == 1) ? 32 : Ochnl) : Ochnl;
		if (Lyr == 2) {
			ochnl = (Ni == 1) ? 48 : Ochnl;
		}
		conv_weight_read_l(Params_l + Ochnl + ((m <<otileshift) * IchnlTil + (Ni << ITILESHIFT) * ochnl) * Kern * Kern, w_buf_l, IchnlTil, OchnlTil, Kern);

		conv_compute_l(
			in_buf_l,
			w_buf_l,
			Bbuf_l,
			OutBuf_l,
			InRownum_l,
			InColnum_l,
			Rownum_l,
			Colnum_l,
			Kern,
			IchnlTil,
			OchnlTil,
			Ni,
			m
		);
	}
	return;
}

void conv_input_read_l(Dtype *In_l, Dtype InBuf_l[Tn][I_BUF_DEPTH_L], int IchnlTil, int InRownum_l, int InColnum_l, int Sec) {
	for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
		for (int r = 0; r < InRownum_l; r++) {
#pragma HLS loop_tripcount min=2 max=16
			for (int c = 0; c < InColnum_l; c++) {
#pragma HLS loop_tripcount min=2 max=16
#pragma HLS pipeline
				InBuf_l[n][Sec * InRownum_l * InColnum_l + r * InColnum_l + c] = *In_l++;
			}
		}
	}
	return;
}

void conv_buf_write_l(
	Dtype OutBuf_l[Tm][O_BUF_DEPTH_L],
	Dtype *Out_l,
	int OchnlTil,
	int Rownum_l,
	int Colnum_l
) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int row = 0; row < Rownum_l; row++) {
#pragma HLS loop_tripcount min=1 max=4
			for (int col = 0; col < Colnum_l; col++) {
#pragma HLS loop_tripcount min=1 max=4
#pragma HLS pipeline
				Dtype data = OutBuf_l[m][row * Colnum_l + col];
				*Out_l++ = (data < 0) ? (0.2 * data) : data;
			}
		}
	}
	return;
}

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
) {
	static Dtype partial_l[Tm];
#pragma HLS array_partition variable=pratial_l complete

	for (int row = 0; row < Rownum_l; row++) {
#pragma HLS loop_tripcount min=1 max=4
		for (int col = 0; col < Colnum_l; col++) {
#pragma HLS loop_tripcount min=1 max=4
			for (int k1 = 0; k1 < Kern; k1++) {
#pragma HLS loop_tripcount min=2 max=4
				for (int k2 = 0; k2 < Kern; k2++) {
#pragma HLS loop_tripcount min=2 max=4
#pragma HLS pipeline
					for (int m = 0; m < Tm; m++) {
						if (ISec == 0 && 0 == k1 && 0 == k2) {
							partial_l[m] = ((OchnlTil - 1) < m) ? 0.0 : BBuf_l[m][OSec];
						}
						if ((ISec == 1 || ISec == 2) && 0 == k1 && 0 == k2) {
							partial_l[m] = 0.0;
						}
						for (int n = 0; n < Tn; n++) {
							Dtype input = InBuf_l[n][ISec * InRownum_l * InColnum_l + (Kern * row + k1) * InColnum_l + Kern * col + k2];
							Dtype weight = 0;
							if (m < OchnlTil && n < IchnlTil) {
								weight = WBuf_l[m * Tn + n][k1 * Kern + k2];
							}
							partial_l[m] += weight * input;
						}
						if (0 == ISec && (Kern - 1) == k1 && (Kern - 1) == k2) {
							OutBuf_l[m][row * Colnum_l + col] = partial_l[m];
						}
						if ((1 == ISec || 2 == ISec) && (Kern - 1) == k1 && (Kern - 1) == k2) {
							OutBuf_l[m][row * Colnum_l + col] += partial_l[m];
						}

					}
				}
			}
		}
	}
	return;
}

void conv_bias_read_l(Dtype *Params_l, Dtype Bbuf_l[Tm][B_BUF_DEPTH], int OSec) {
	for (int til = 0; til < OSec; til++) {
#pragma HLS loop_tripcount min=1 max=2
		for (int m = 0; m < Tm; m++) {
#pragma HLS pipeline
			Bbuf_l[m][til] = *Params_l++;
		}
	}

	return;
}

void conv_weight_read_l(
	Dtype *Params_l,
	Dtype Wbuf_l[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
			for (int k = 0; k < Kern * Kern; k++) {
#pragma HLS loop_tripcount min=4 max=16
#pragma HLS pipeline
				Wbuf_l[m * Tn + n][k] = *Params_l++;
			}
		}
	}
	return;
}


//layer m *********************************************************************************************************************************
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
	int Lyr) {

	//set on-chip buffer
	static Dtype b_buf_m[Tm][B_BUF_DEPTH];
#pragma HLS array_partition variable=b_buf_m complete dim=1

	static Dtype out_buf_m[Tm][O_BUF_DEPTH_M];
#pragma HLS array_partition variable=out_buf_m complete dim=1

	conv_bias_read_m(Params_m, b_buf_m, OSec);

	//output channel
	for (int n = 0; n < OSec; n++) {
#pragma HLS loop_tripcount min=1 max=2
//#pragma HLS dataflow
		int otile = Lyr == 1 ? ((n == 1) ? 8 : Tm) : Tm;

		conv_ichnl_m(In_m, Params_m, b_buf_m, out_buf_m, IchnlTil, otile, Ochnl, Kern, n, Lyr, InRownum_m, InColnum_m, OutRownum_m, OutColnum_m, ISec);

		conv_buf_write_m(
			out_buf_m,
			Out_m + n * Tm * OutRownum_m * OutColnum_m,
			otile,
			OutRownum_m,
			OutColnum_m
		);

	}
	return;

}

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
) {
	static Dtype in_buf_m[Tn][I_BUF_DEPTH_M];
#pragma HLS array_partition variable=in_buf_m complete dim=1

	static Dtype w_buf_m[Tn * Tm][W_BUF_DEPTH];
#pragma HLS array_partition variable=w_buf_m complete dim=1

	//input channel
	for (int m = 0; m < ISec; m++) {
#pragma HLS loop_tripcount min=1 max=3

		if (Ni == 0) {
			conv_input_read_m(In_m + (m << ITILESHIFT) * InRownum_m * InColnum_m, in_buf_m, IchnlTil, InRownum_m, InColnum_m, m);
		}

		int otileshift = Lyr == 1 ? ((Ni == 1) ? 3 : OTILESHIFT) : OTILESHIFT;
		int ochnl = Lyr == 1 ? ((Ni == 1) ? 32 : Ochnl) : Ochnl;
		if (Lyr == 2) {
			ochnl = (Ni == 1) ? 48 : Ochnl;
		}
		conv_weight_read_m(Params_m + Ochnl + ((m <<otileshift) * IchnlTil + (Ni << ITILESHIFT) * ochnl) * Kern * Kern, w_buf_m, IchnlTil, OchnlTil, Kern);

		conv_compute_m(
			in_buf_m,
			w_buf_m,
			Bbuf_m,
			OutBuf_m,
			InRownum_m,
			InColnum_m,
			Rownum_m,
			Colnum_m,
			Kern,
			IchnlTil,
			OchnlTil,
			Ni,
			m
		);
	}
	return;
}

void conv_input_read_m(Dtype *In_m, Dtype InBuf_m[Tn][I_BUF_DEPTH_M], int IchnlTil, int InRownum_m, int InColnum_m, int Sec) {
	for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
		for (int r = 0; r < InRownum_m; r++) {
#pragma HLS loop_tripcount min=4 max=32
			for (int c = 0; c < InColnum_m; c++) {
#pragma HLS loop_tripcount min=4 max=32
#pragma HLS pipeline
				InBuf_m[n][Sec * InRownum_m * InColnum_m + r * InColnum_m + c] = *In_m++;
			}
		}
	}
	return;
}

void conv_buf_write_m(
	Dtype OutBuf_m[Tm][O_BUF_DEPTH_M],
	Dtype *Out_m,
	int OchnlTil,
	int Rownum_m,
	int Colnum_m
) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int row = 0; row < Rownum_m; row++) {
#pragma HLS loop_tripcount min=2 max=8
			for (int col = 0; col < Colnum_m; col++) {
#pragma HLS loop_tripcount min=2 max=8
#pragma HLS pipeline
				Dtype data = OutBuf_m[m][row * Colnum_m + col];
				*Out_m++ = (data < 0) ? (0.2 * data) : data;
			}
		}
	}
	return;
}

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
) {
	static Dtype partial_m[Tm];
#pragma HLS array_partition variable=partial_m complete

	for (int row = 0; row < Rownum_m; row++) {
#pragma HLS loop_tripcount min=2 max=8
		for (int col = 0; col < Colnum_m; col++) {
#pragma HLS loop_tripcount min=2 max=8
			for (int k1 = 0; k1 < Kern; k1++) {
#pragma HLS loop_tripcount min=2 max=4
				for (int k2 = 0; k2 < Kern; k2++) {
#pragma HLS loop_tripcount min=2 max=4
#pragma HLS pipeline
					for (int m = 0; m < Tm; m++) {
						if (ISec == 0 && 0 == k1 && 0 == k2) {
							partial_m[m] = ((OchnlTil - 1) < m) ? 0.0 : BBuf_m[m][OSec];
						}
						if ((ISec == 1 || ISec == 2) && 0 == k1 && 0 == k2) {
							partial_m[m] = 0.0;
						}
						for (int n = 0; n < Tn; n++) {
							Dtype input = InBuf_m[n][ISec * InRownum_m * InColnum_m + (Kern * row + k1) * InColnum_m + Kern * col + k2];
							Dtype weight = 0;
							if (m < OchnlTil && n < IchnlTil) {
								weight = WBuf_m[m * Tn + n][k1 * Kern + k2];
							}
							partial_m[m] += weight * input;
						}
						if (0 == ISec && (Kern - 1) == k1 && (Kern - 1) == k2) {
							OutBuf_m[m][row * Colnum_m + col] = partial_m[m];
						}
						if ((1 == ISec || 2 == ISec) && (Kern - 1) == k1 && (Kern - 1) == k2) {
							OutBuf_m[m][row * Colnum_m + col] += partial_m[m];
						}

					}
				}
			}
		}
	}
	return;
}

void conv_bias_read_m(Dtype *Params_m, Dtype Bbuf_m[Tm][B_BUF_DEPTH], int OSec) {
	for (int til = 0; til < OSec; til++) {
#pragma HLS loop_tripcount min=1 max=2
		for (int m = 0; m < Tm; m++) {
#pragma HLS pipeline
			Bbuf_m[m][til] = *Params_m++;
		}
	}

	return;
}

void conv_weight_read_m(
	Dtype *Params_m,
	Dtype Wbuf_m[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
			for (int k = 0; k < Kern * Kern; k++) {
#pragma HLS loop_tripcount min=4 max=16
#pragma HLS pipeline
				Wbuf_m[m * Tn + n][k] = *Params_m++;
			}
		}
	}
	return;
}

//layer s *********************************************************************************************************************************
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
	int Lyr) {

	//set on-chip buffer
	static Dtype b_buf_s[Tm][B_BUF_DEPTH];
#pragma HLS array_partition variable=b_buf_s complete dim=1

	static Dtype out_buf_s[Tm][O_BUF_DEPTH_S];
#pragma HLS array_partition variable=out_buf_s complete dim=1

	conv_bias_read_s(Params_s, b_buf_s, OSec);

	//output channel
	for (int n = 0; n < OSec; n++) {
#pragma HLS loop_tripcount min=1 max=2
//#pragma HLS dataflow
		int otile = Lyr == 1 ? ((n == 1) ? 8 : Tm) : Tm;

		conv_ichnl_s(In_s, Params_s, b_buf_s, out_buf_s, IchnlTil, otile, Ochnl, Kern, n, Lyr, InRownum_s, InColnum_s, OutRownum_s, OutColnum_s, ISec);

		conv_buf_write_s(
			out_buf_s,
			Out_s + n * Tm * OutRownum_s * OutColnum_s,
			otile,
			OutRownum_s,
			OutColnum_s
		);

	}
	return;

}

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
) {
	static Dtype in_buf_s[Tn][I_BUF_DEPTH_S];
#pragma HLS array_partition variable=in_buf_s complete dim=1

	static Dtype w_buf_s[Tn * Tm][W_BUF_DEPTH];
#pragma HLS array_partition variable=w_buf_s complete dim=1

	//input channel
	for (int m = 0; m < ISec; m++) {
#pragma HLS loop_tripcount min=1 max=8

		if (Ni == 0) {
			conv_input_read_s(In_s + (m << ITILESHIFT) * InRownum_s * InColnum_s, in_buf_s, IchnlTil, InRownum_s, InColnum_s, m);
		}

		int otileshift = Lyr == 1 ? ((Ni == 1) ? 3 : OTILESHIFT) : OTILESHIFT;
		int ochnl = Lyr == 1 ? ((Ni == 1) ? 32 : Ochnl) : Ochnl;
		if (Lyr == 2) {
			ochnl = (Ni == 1) ? 48 : Ochnl;
		}
		conv_weight_read_s(Params_s + Ochnl + ((m <<otileshift) * IchnlTil + (Ni << ITILESHIFT) * ochnl) * Kern * Kern, w_buf_s, IchnlTil, OchnlTil, Kern);

		conv_compute_s(
			in_buf_s,
			w_buf_s,
			Bbuf_s,
			OutBuf_s,
			InRownum_s,
			InColnum_s,
			Rownum_s,
			Colnum_s,
			Kern,
			IchnlTil,
			OchnlTil,
			Ni,
			m
		);
	}
	return;
}

void conv_input_read_s(Dtype *In_s, Dtype InBuf_s[Tn][I_BUF_DEPTH_S], int IchnlTil, int InRownum_s, int InColnum_s, int Sec) {
	for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
		for (int r = 0; r < InRownum_s; r++) {
#pragma HLS loop_tripcount min=8 max=64
			for (int c = 0; c < InColnum_s; c++) {
#pragma HLS loop_tripcount min=8 max=64
#pragma HLS pipeline
				InBuf_s[n][Sec * InRownum_s * InColnum_s + r * InColnum_s + c] = *In_s++;
			}
		}
	}
	return;
}

void conv_buf_write_s(
	Dtype OutBuf_s[Tm][O_BUF_DEPTH_S],
	Dtype *Out_s,
	int OchnlTil,
	int Rownum_s,
	int Colnum_s
) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int row = 0; row < Rownum_s; row++) {
#pragma HLS loop_tripcount min=4 max=16
			for (int col = 0; col < Colnum_s; col++) {
#pragma HLS loop_tripcount min=4 max=16
#pragma HLS pipeline
				Dtype data = OutBuf_s[m][row * Colnum_s + col];
				*Out_s++ = (data < 0) ? (0.2 * data) : data;
			}
		}
	}
	return;
}

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
) {
	static Dtype partial_s[Tm];
#pragma HLS array_partition variable=partial_s complete

	for (int row = 0; row < Rownum_s; row++) {
#pragma HLS loop_tripcount min=4 max=16
		for (int col = 0; col < Colnum_s; col++) {
#pragma HLS loop_tripcount min=4 max=16
			for (int k1 = 0; k1 < Kern; k1++) {
#pragma HLS loop_tripcount min=2 max=4
				for (int k2 = 0; k2 < Kern; k2++) {
#pragma HLS loop_tripcount min=2 max=4
#pragma HLS pipeline
					for (int m = 0; m < Tm; m++) {
						if (ISec == 0 && 0 == k1 && 0 == k2) {
							partial_s[m] = ((OchnlTil - 1) < m) ? 0.0 : BBuf_s[m][OSec];
						}
						if ((ISec == 1 || ISec == 2) && 0 == k1 && 0 == k2) {
							partial_s[m] = 0.0;
						}
						for (int n = 0; n < Tn; n++) {
							Dtype input = InBuf_s[n][ISec * InRownum_s * InColnum_s + (Kern * row + k1) * InColnum_s + Kern * col + k2];
							Dtype weight = 0;
							if (m < OchnlTil && n < IchnlTil) {
								weight = WBuf_s[m * Tn + n][k1 * Kern + k2];
							}
							partial_s[m] += weight * input;
						}
						if (0 == ISec && (Kern - 1) == k1 && (Kern - 1) == k2) {
							OutBuf_s[m][row * Colnum_s + col] = partial_s[m];
						}
						if ((1 == ISec || 2 == ISec) && (Kern - 1) == k1 && (Kern - 1) == k2) {
							OutBuf_s[m][row * Colnum_s + col] += partial_s[m];
						}

					}
				}
			}
		}
	}
	return;
}

void conv_bias_read_s(Dtype *Params_s, Dtype Bbuf_s[Tm][B_BUF_DEPTH], int OSec) {
	for (int til = 0; til < OSec; til++) {
#pragma HLS loop_tripcount min=1 max=2
		for (int m = 0; m < Tm; m++) {
#pragma HLS pipeline
			Bbuf_s[m][til] = *Params_s++;
		}
	}

	return;
}

void conv_weight_read_s(
	Dtype *Params_s,
	Dtype Wbuf_s[Tn * Tm][W_BUF_DEPTH],
	int IchnlTil,
	int OchnlTil,
	int Kern) {
	for (int m = 0; m < OchnlTil; m++) {
#pragma HLS loop_tripcount min=8 max=16
		for (int n = 0; n < IchnlTil; n++) {
#pragma HLS loop_tripcount min=1 max=8
			for (int k = 0; k < Kern * Kern; k++) {
#pragma HLS loop_tripcount min=4 max=16
				Wbuf_s[m * Tn + n][k] = *Params_s++;
			}
		}
	}
	return;
}



