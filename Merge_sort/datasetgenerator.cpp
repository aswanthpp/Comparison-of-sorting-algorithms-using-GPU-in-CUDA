#include<bits/stdc++.h>
using namespace std;

#define TEST_SIZE 10000 
#define RAND_RANGE 1000



static void write_data(char *file_name,unsigned int *data,unsigned int num) {
  FILE *handle = fopen(file_name, "w");
  fprintf(handle, "%d", num);
  for (int ii = 0; ii < num; ii++) {
    fprintf(handle, "\n%d", *data++);
  }
  fflush(handle);
  fclose(handle);
}

int main(){
	unsigned int *h_inVals;
	unsigned int *h_outVals;
	size_t memsize = sizeof(unsigned int) * TEST_SIZE;
	h_inVals = (unsigned int*)malloc(memsize);
	h_outVals = (unsigned int*)malloc(memsize);
	for(int i=0; i<TEST_SIZE; i++){ 
		h_inVals[i] = rand() % RAND_RANGE;
		h_outVals[i]=h_inVals[i]; 
	} 
	sort(h_outVals,h_outVals+TEST_SIZE);
	

	char *input_file_name  = "input.raw";
  char *output_file_name = "output.raw";
  write_data(input_file_name, h_inVals, TEST_SIZE);
  write_data(output_file_name, h_outVals, TEST_SIZE);


}