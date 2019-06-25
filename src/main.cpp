#include <iostream>
#include <fstream>
#include <string.h>

#include "sds_lib.h"

#include "common.h"
#include "check.h"
#include "get_data.h"
#include "get_param.h"
#include "eth_cnn.h"

int main(int argc, char* arg[]) {
	
	std::cout << "ETH-CNN implementation in ZCU102" << std::endl;

	//get input data
	Dtype *in_img = (Dtype *)sds_alloc((IMG_W*IMG_H) * sizeof(Dtype));
	mem_check(in_img);
	memset(in_img, 0, IMG_H * IMG_W * sizeof(Dtype));
	//get_from_random(in_img);
	get_from_file(in_img);
	

	//alloc output memory
	Dtype * out = (Dtype *)sds_alloc(CLASS_NUM * sizeof(Dtype));
	mem_check(out);
	memset(out, 0, CLASS_NUM * sizeof(Dtype));

	//get parameters
	//include all weights and bias
	int param_buf_size = param_size();
	Dtype *params = (Dtype *)sds_alloc(param_buf_size * sizeof(Dtype));
	mem_check(params);
	get_params(params, param_buf_size);

	//eth-cnn
	eth_cnn(in_img, out, params);

	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		":out " << std::endl;
	for (int i = 0; i < CLASS_NUM; i++) {
		std::cout << out[i] << std::endl;
	}

	//check dataflow
	sds_free(in_img);
	sds_free(out);
	sds_free(params);
	return 0;
}
