#ifndef __SUB_SAMPLE__
#define __SUB_SAMPLE__

#include "common.h"

void sub_sample(Dtype *In, Dtype *Out_l, Dtype *Out_m, Dtype *Out_s);

void sub_sample_l(Dtype In[64][64], Dtype Sub_l[16][16]);
void sub_sample_m(Dtype In[64][64], Dtype Sub_m[32][32]);
void sub_sample_s(Dtype In[64][64], Dtype Sub_s[64][64]);

void sub_buf_write(Dtype Sub_l[16][16], Dtype Sub_m[32][32], Dtype Sub_s[64][64], Dtype *Out_l, Dtype *Out_m, Dtype *Out_s);

#endif // !__SUB_SAMPLE__

