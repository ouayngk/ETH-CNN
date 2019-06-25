#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "get_data.h"
#include "sds_lib.h"

//randomly init image
void get_from_random(Dtype * Img) {
	for (int row = 0; row < IMG_H; row++) {
		for (int col = 0; col < IMG_W; col++) {
			Img[row * IMG_W + col] = rand() % (IMG_W * IMG_H);
		}
	}
	return;
}

void get_from_file(Dtype *Img) {
	std::cout << "[INFO] " << __FUNCTION__ << ", " << __LINE__ <<
		": read image from file." << std::endl;
	std::ifstream in_bin("/mnt/Input_255.bin", std::ios::binary);
	char *in_buf = reinterpret_cast<char *>(Img);
	in_bin.read(in_buf, IMG_H * IMG_W * sizeof(Dtype));
	in_bin.close();
	return;
}
