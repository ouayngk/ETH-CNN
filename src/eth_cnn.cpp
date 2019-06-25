#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>

#include "sds_lib.h"
#include "hls_half.h"

#include "get_param.h"
#include "check.h"
#include "eth_cnn.h"
#include "check.h"
#include "common.h"
#include "conv_fpga.h"
#include "performance.h"
#include "sub_sample.h"
#include "fc_fpga.h"

//shape of feature map in each layer
extern const int SHAPE_l[4];
extern const int SHAPE_m[4];
extern const int SHAPE_s[4];
//channel number in each layer
extern const int CHNEL_l[8];
extern const int CHNEL_m[8];
extern const int CHNEL_s[8];
//kernel size in each layer
extern const int KERNL[3];

void eth_cnn(Dtype *In, Dtype *Out, Dtype *Params) {

	std::cout << " [INFO] " << __FUNCTION__ << " , " << __LINE__ <<
		": Start ETH_CNN in FPGA..." << std::endl;

	//flat 
	int max_flat_size = 2688;
	Dtype *conv_flat = (Dtype *)sds_alloc(max_flat_size * sizeof(Dtype));

	//set three params off-chip memory
	Dtype *params_l = (Dtype *)sds_alloc(180250 * sizeof(Dtype));
	Dtype *params_m = (Dtype *)sds_alloc(362000 * sizeof(Dtype));
	Dtype *params_s = (Dtype *)sds_alloc(745960 * sizeof(Dtype));
	mem_check(params_l);
	mem_check(params_m);
	mem_check(params_s);

	//buf size should meet maxium memory requirment for each layer

	int max_buf_l = 32 * 16 * 16;
	int max_buf_m = 32 * 32 * 32;
	int max_buf_s = 32 * 64 * 64;
	Dtype *buffer_l_A = (Dtype *)sds_alloc(max_buf_l * sizeof(Dtype));
	Dtype *buffer_l_B = (Dtype *)sds_alloc(max_buf_l * sizeof(Dtype));
	Dtype *buffer_m_A = (Dtype *)sds_alloc(max_buf_m * sizeof(Dtype));
	Dtype *buffer_m_B = (Dtype *)sds_alloc(max_buf_m * sizeof(Dtype));
	Dtype *buffer_s_A = (Dtype *)sds_alloc(max_buf_s * sizeof(Dtype));
	Dtype *buffer_s_B = (Dtype *)sds_alloc(max_buf_s * sizeof(Dtype));

	mem_check(buffer_l_A);
	mem_check(buffer_l_B);
	mem_check(buffer_m_A);
	mem_check(buffer_m_B);
	mem_check(buffer_s_A);
	mem_check(buffer_s_B);

	//params mem rearrange;divide into three parts;
	params_mem_rearr(Params, params_l, params_m, params_s);

	Dtype *cur_params_l = params_l;
	Dtype *cur_params_m = params_m;
	Dtype *cur_params_s = params_s;

	float total_time = 0;

	for (int c_layer = 0; c_layer < CONV_LAYER_NUM; c_layer++) {
		int input_col_num_l = SHAPE_l[c_layer];
		int input_row_num_l = SHAPE_l[c_layer];
		int output_col_num_l = SHAPE_l[c_layer + 1];
		int output_row_num_l = SHAPE_l[c_layer + 1];

		int input_col_num_m = SHAPE_m[c_layer];
		int input_row_num_m = SHAPE_m[c_layer];
		int output_col_num_m = SHAPE_m[c_layer + 1];
		int output_row_num_m = SHAPE_m[c_layer + 1];

		int input_col_num_s = SHAPE_s[c_layer];
		int input_row_num_s = SHAPE_s[c_layer];
		int output_col_num_s = SHAPE_s[c_layer + 1];
		int output_row_num_s = SHAPE_s[c_layer + 1];

		if (0 == c_layer) {
			std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
				": " << c_layer << "th conv layer." << std::endl;
			int isec = 1;
			int osec = 1;

			sub_sample(In, buffer_l_A, buffer_m_A, buffer_s_A);
			perf_counter perf;
			perf.start();
			conv_fpga(buffer_l_A,
				      buffer_m_A,
				      buffer_s_A,
				      cur_params_l, 
				      cur_params_m, 
				      cur_params_s, 
				      buffer_l_B,
				      buffer_m_B,
				      buffer_s_B,
				      c_layer, 
				      input_row_num_l,
				      input_col_num_l,
				      input_row_num_m,
				      input_col_num_m,
				      input_row_num_s,
				      input_col_num_s,
				      output_row_num_l, 
				      output_col_num_l,
				      output_row_num_m, 
				      output_col_num_m, 
				      output_row_num_s, 
				      output_col_num_s,
				      KERNL[c_layer],
				      CHNEL_l[c_layer],
			          CHNEL_l[c_layer + 1],
			          isec,
				      osec);
			perf.stop();
			uint64_t cpu_cycles = perf.avg_cpu_cycles();
			float lyr_time = (float)cpu_cycles / 1.5e9;
			total_time += lyr_time;
			std::cout << "[INFO] " << __FUNCTION__ << "' " << __LINE__ <<
				": Finish in " << lyr_time << "s. " << std::endl;

			//update pointer to parameters
			cur_params_l += (CHNEL_l[0] * CHNEL_l[1] * KERNL[0] * KERNL[0] + CHNEL_l[1]);
			cur_params_m += (CHNEL_m[0] * CHNEL_m[1] * KERNL[0] * KERNL[0] + CHNEL_m[1]);
			cur_params_s += (CHNEL_s[0] * CHNEL_s[1] * KERNL[0] * KERNL[0] + CHNEL_s[1]);

		}
		else {
			std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
				": " << c_layer << "th conv layer." << std::endl;
			int isec = CHNEL_l[c_layer] / Tn;
			int osec = 2;
			mem_copy(buffer_l_B, buffer_m_B, buffer_s_B, buffer_l_A, buffer_m_A, buffer_s_A, c_layer);
			perf_counter perf;
			perf.start();
			conv_fpga(buffer_l_A,
				      buffer_m_A,
				      buffer_s_A,
				      cur_params_l,
				      cur_params_m,
				      cur_params_s,
				      buffer_l_B,
				      buffer_m_B,
				      buffer_s_B,
				      c_layer,
				      input_row_num_l,
				      input_col_num_l,
				      input_row_num_m,
				      input_col_num_m,
				      input_row_num_s,
				      input_col_num_s,
				      output_row_num_l,
				      output_col_num_l,
				      output_row_num_m,
				      output_col_num_m,
				      output_row_num_s,
				      output_col_num_s,
				      KERNL[c_layer],
				      CHNEL_l[c_layer],
			          CHNEL_l[c_layer + 1],
			          isec,
				      osec);
			perf.stop();
			uint64_t cpu_cycles = perf.avg_cpu_cycles();
			float lyr_time = (float)cpu_cycles / 1.5e9;
			total_time += lyr_time;
			std::cout << "[INFO] " << __FUNCTION__ << "' " << __LINE__ <<
				": Finish in " << lyr_time << "s. " << std::endl;
			if (c_layer == 1) {
				memcpy(conv_flat, buffer_l_B, 96 * sizeof(Dtype));
				memcpy(conv_flat + 96, buffer_m_B, 384 * sizeof(Dtype));
				memcpy(conv_flat + 96 + 384, buffer_s_B, 1536 * sizeof(Dtype));
			}
			else {
				memcpy(conv_flat + 2016, buffer_l_B, 32 * sizeof(Dtype));
				memcpy(conv_flat + 2016 + 32, buffer_m_B, 128 * sizeof(Dtype));
				memcpy(conv_flat + 2016 + 32 + 128, buffer_s_B, 512 * sizeof(Dtype));
			}
			cur_params_l += (CHNEL_l[c_layer] * CHNEL_l[c_layer + 1] * KERNL[c_layer] * KERNL[c_layer] + CHNEL_l[c_layer + 1]);
			cur_params_m += (CHNEL_m[c_layer] * CHNEL_m[c_layer + 1] * KERNL[c_layer] * KERNL[c_layer] + CHNEL_m[c_layer + 1]);
			cur_params_s += (CHNEL_s[c_layer] * CHNEL_s[c_layer + 1] * KERNL[c_layer] * KERNL[c_layer] + CHNEL_s[c_layer + 1]);
		}
	}
	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		": Total time in conv layer, " << total_time << "s. " << std::endl;

	//do fc layer 

	Dtype *fc_input = conv_flat;
	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		": FC layer. " << std::endl;
	perf_counter perf;
	perf.start();
	//fc layer function
	fc_fpga(fc_input, cur_params_l, cur_params_m, cur_params_s, Out);
	perf.stop();
	uint64_t cpu_cycles = perf.avg_cpu_cycles();
	float fc_time = (float)cpu_cycles / 1.5e9;
	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		": Finish in " << fc_time << "s. " << std::endl;

	//free memory
	sds_free(buffer_l_A);
	sds_free(buffer_m_A);
	sds_free(buffer_s_A);
	sds_free(buffer_l_B);
	sds_free(buffer_m_B);
	sds_free(buffer_s_B);
	sds_free(conv_flat);
	sds_free(params_l);
	sds_free(params_m);
	sds_free(params_s);
	return;
}

void params_mem_rearr(Dtype *In, Dtype *Out_l, Dtype *Out_m, Dtype *Out_s) {
	int offset_conv = 4 * 4 * 16 + 16 + 2 * 2 * 16 * 24 + 24 + 2 * 2 * 24 * 32 + 32;
	int offset_fc_l = 2688 * 64 + 64 + 65 * 48 + 48 + 49 * 1 + 1;
	int offset_fc_m = 2688 * 128 + 128 + 129 * 96 + 96 + 97 * 4 + 4;
	int offset_fc_s = 2688 * 256 + 256 + 257 * 192 + 192 + 193 * 16 + 16;

	//conv
	memcpy(Out_l, In, offset_conv * sizeof(Dtype));
	Out_l += offset_conv;
	
	memcpy(Out_m, In + offset_conv, offset_conv * sizeof(Dtype));
	Out_m += offset_conv;

	memcpy(Out_s, In + 2 * offset_conv, offset_conv * sizeof(Dtype));
	Out_s += offset_conv;

	In += 3 * offset_conv;

	//fc_l
	memcpy(Out_l, In, offset_fc_l * sizeof(Dtype));
	In += offset_fc_l;
	Out_l += offset_fc_l;

	//fc_m
	memcpy(Out_m, In, offset_fc_m * sizeof(Dtype));
	In += offset_fc_m;
	Out_m += offset_fc_m;

	//fc_s
	memcpy(Out_s, In, offset_fc_s * sizeof(Dtype));
	In += offset_fc_s;
	Out_s += offset_fc_s;
}

void mem_copy(Dtype *In_l, Dtype *In_m, Dtype *In_s, Dtype *Out_l, Dtype * Out_m, Dtype *Out_s, int Lyr) {
	int row_num_l = SHAPE_l[Lyr];
	int col_num_l = SHAPE_l[Lyr];
	int chnl_l = CHNEL_l[Lyr];

	int row_num_m = SHAPE_m[Lyr];
	int col_num_m = SHAPE_m[Lyr];
	int chnl_m = CHNEL_m[Lyr];

	int row_num_s = SHAPE_s[Lyr];
	int col_num_s = SHAPE_s[Lyr];
	int chnl_s = CHNEL_s[Lyr];

	int copy_num_l = chnl_l * row_num_l * col_num_l;
	int copy_num_m = chnl_m * row_num_m * col_num_m;
	int copy_num_s = chnl_s * row_num_s * col_num_s;

	memcpy(Out_l, In_l, copy_num_l * sizeof(Dtype));
	memcpy(Out_m, In_m, copy_num_m * sizeof(Dtype));
	memcpy(Out_s, In_s, copy_num_s * sizeof(Dtype));
}

