#ifndef _CHECK_H_
#define _CHECK_H_

#include <iostream>
#include <stdlib.h>

#include "common.h"

inline void mem_check(Dtype *Mem) {
	if (NULL == Mem) {
		std::cout << " [ERROR] " << __FUNCTION__ << "," << __LINE__ <<
			": Memory check fail" << std::endl;
		exit(1);
	}
	return;
}

#endif // !_CHECK_H_




