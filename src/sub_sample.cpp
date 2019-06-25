#include "sds_lib.h"

#include "sub_sample.h"
#include "common.h"

//sub_sample in FPGA

void sub_sample(Dtype *In, Dtype *Out_l, Dtype *Out_m, Dtype *Out_s) {

	Dtype sub_input_buffer[64][64];
	Dtype sub_l[16][16];
	Dtype sub_m[32][32];
	Dtype sub_s[64][64];

	for (int row = 0; row < 64; row++) {
		for (int col = 0; col < 64; col++) {
			sub_input_buffer[row][col] = 1.0 / 255.0 * In[row * 64 + col];
		}
	}

	sub_sample_l(sub_input_buffer, sub_l);
	sub_sample_m(sub_input_buffer, sub_m);
	sub_sample_s(sub_input_buffer, sub_s);

	sub_buf_write(sub_l, sub_m, sub_s, Out_l, Out_m, Out_s);

}

void sub_sample_l(Dtype In[64][64], Dtype Sub_l[16][16]) {
	Dtype value = 0;
	Dtype x_mean_reduce_64 = 0;
	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			value = In[row * 4][col * 4] + In[row * 4][col * 4 + 1] + In[row * 4][col * 4 + 2] + In[row * 4][col * 4 + 3]
				+ In[row * 4 + 1][col * 4] + In[row * 4 + 1][col * 4 + 1] + In[row * 4 + 1][col * 4 + 2] + In[row * 4 + 1][col * 4 + 3]
				+ In[row * 4 + 2][col * 4] + In[row * 4 + 2][col * 4 + 1] + In[row * 4 + 2][col * 4 + 2] + In[row * 4 + 2][col * 4 + 3]
				+ In[row * 4 + 3][col * 4] + In[row * 4 + 3][col * 4 + 1] + In[row * 4 + 3][col * 4 + 2] + In[row * 4 + 3][col * 4 + 3];
			Sub_l[row][col] = value / 16.0;
		}
	}

	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			x_mean_reduce_64 += Sub_l[row][col] / 256.0;
		}
	}

	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			Sub_l[row][col] = Sub_l[row][col] - x_mean_reduce_64;
		}
	}
}

void sub_sample_m(Dtype In[64][64], Dtype Sub_m[32][32]) {
	Dtype value = 0;
	Dtype x_mean_reduce_32[2][2] = { 0 };

	//sub-sample to 32*32
	for (int row = 0; row < 32; row++) {
		for (int col = 0; col < 32; col++) {
			value = In[row * 2][col * 2] + In[row * 2][col * 2 + 1]
				+ In[row * 2 + 1][col * 2] + In[row * 2 + 1][col * 2 + 1];
			Sub_m[row][col] = value / 4.0;
		}
	}

	for (int row = 0; row < 2; row++) {
		for (int col = 0; col < 2; col++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					x_mean_reduce_32[row][col] += Sub_m[row * 16 + i][col * 16 + j] / 256.0;
				}
			}
		}
	}

	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			Sub_m[row][col] = Sub_m[row][col] - x_mean_reduce_32[0][0];
		}
	}

	for (int row = 0; row < 16; row++) {
		for (int col = 16; col < 32; col++) {
			Sub_m[row][col] = Sub_m[row][col] - x_mean_reduce_32[0][1];
		}
	}

	for (int row = 16; row < 32; row++) {
		for (int col = 0; col < 16; col++) {
			Sub_m[row][col] = Sub_m[row][col] - x_mean_reduce_32[1][0];
		}
	}

	for (int row = 16; row < 32; row++) {
		for (int col = 16; col < 32; col++) {
			Sub_m[row][col] = Sub_m[row][col] - x_mean_reduce_32[1][1];
		}
	}
}

void sub_sample_s(Dtype In[64][64], Dtype Sub_s[64][64]) {
	Dtype x_mean_reduce_16[4][4] = { 0 };
	
	//sub-sample to 64*64
	for (int row = 0; row < 4; row++) {
		for (int col = 0; col < 4; col++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					x_mean_reduce_16[row][col] += In[row * 16 + i][col * 16 + j] / 256.0;
				}
			}
		}
	}

	for (int m = 0; m < 4; m++) {
		for (int n = 0; n < 4; n++) {
			for (int i = 0; i < 16; i++) {
				for (int j = 0; j < 16; j++) {
					Sub_s[i + m * 16][j + n * 16] = In[i + m * 16][j + n * 16] - x_mean_reduce_16[m][n];
				}
			}
		}
	}
}

void sub_buf_write(Dtype Sub_l[16][16], Dtype Sub_m[32][32], Dtype Sub_s[64][64], Dtype *Out_l, Dtype *Out_m, Dtype *Out_s) {
	for (int row = 0; row < 16; row++) {
		for (int col = 0; col < 16; col++) {
			*Out_l++ = Sub_l[row][col];
		}
	}

	for (int row = 0; row < 32; row++) {
		for (int col = 0; col < 32; col++) {
			*Out_m++ = Sub_m[row][col];
		}
	}

	for (int row = 0; row < 64; row++) {
		for (int col = 0; col < 64; col++) {
			*Out_s++ = Sub_s[row][col];
		}
	}
}
