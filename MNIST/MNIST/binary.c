#include <stdio.h>
#include "Mnist_Fun.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

void Activation6(double *a2, double *z3, double *b, double **w)
{
	for (int i = 0; i < B_OUTPUT_LAYER; i++)
	{
		for (int j = 0; j < HIDDEN_LAYER; j++)
		{
			z3[i] += a2[j] * w[i][j];
		}
		z3[i] += b[i];
	}
}

void B_Transpose(double **w, double **w_t)
{
	for (int i = 0; i < HIDDEN_LAYER; i++)
		for (int j = 0; j < B_OUTPUT_LAYER; j++)
			w_t[i][j] = w[j][i];
}

void B_Gussian_Reset(double *bp2, double *bp3, double **wp2, double **wp3)
{
	for (int i = 0; i < HIDDEN_LAYER; i++) for (int j = 0; j < INPUT_LAYER; j++) wp2[i][j] = GaussianRandom();
	for (int i = 0; i < B_OUTPUT_LAYER; i++) for (int j = 0; j < HIDDEN_LAYER; j++) wp3[i][j] = GaussianRandom();

	for (int i = 0; i < HIDDEN_LAYER; i++) bp2[i] = GaussianRandom();
	for (int i = 0; i < B_OUTPUT_LAYER; i++) bp3[i] = GaussianRandom();
}

void TRAIN_BINARY(BINARY network, int high, int low)
{
	// practice model variable
	unsigned char output = 0;
	int  img_number = 0, n = 0, size = 0, epoch_count = 0;

	TRAIN_VARIABLE variable;

	// 1D memory allocation
	variable.ap1 = MemAlloc_1D(INPUT_LAYER); variable.ap2 = MemAlloc_1D(HIDDEN_LAYER); variable.ap_out = MemAlloc_1D(B_OUTPUT_LAYER); variable.a_out = MemAlloc_1D(B_OUTPUT_LAYER);
	variable.zp2 = MemAlloc_1D(HIDDEN_LAYER); variable.zp3 = MemAlloc_1D(B_OUTPUT_LAYER);
	variable.delta2 = MemAlloc_1D(HIDDEN_LAYER); variable.delta3 = MemAlloc_1D(B_OUTPUT_LAYER); variable.del_bp2 = MemAlloc_1D(HIDDEN_LAYER); variable.del_bp3 = MemAlloc_1D(B_OUTPUT_LAYER);
	variable.image = MemAlloc_1D(INPUT_LAYER);

	//2D memory allocation
	variable.del_wp2 = MemAlloc_2D(INPUT_LAYER, HIDDEN_LAYER); variable.del_wp3 = MemAlloc_2D(HIDDEN_LAYER, B_OUTPUT_LAYER); variable.wp3_t = MemAlloc_2D(B_OUTPUT_LAYER, HIDDEN_LAYER);


	while (epoch_count < 1){
		if (epoch_count == 0) B_Gussian_Reset(network.bp2, network.bp3, network.wp2, network.wp3);
		variable.f_label = fopen("train-labels.idx1-ubyte", "rb");
		variable.f_image = fopen("train-images.idx3-ubyte", "rb");

		// read header
		for (int i = 0; i < 4; i++){
			read_int_bigendian(variable.f_image);
			if (i < 2) read_int_bigendian(variable.f_label);
		}

		size = 0;
		img_number = 0;

		/////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////// train ////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////////////////////////
		while (img_number < TRAIN_SIZE){
			for (int i = 0; i < HIDDEN_LAYER; i++) for (int j = 0; j < INPUT_LAYER; j++) variable.del_wp2[i][j] = 0;
			for (int i = 0; i < B_OUTPUT_LAYER; i++) for (int j = 0; j < HIDDEN_LAYER; j++) variable.del_wp3[i][j] = 0;

			for (int i = 0; i < HIDDEN_LAYER; i++) variable.del_bp2[i] = 0;
			for (int i = 0; i < B_OUTPUT_LAYER; i++) variable.del_bp3[i] = 0;

			fread(variable.image, 28 * 28, 1, variable.f_image);
			fread(&(variable.val_label), 1, 1, variable.f_label);

			COORD Pos = { 0, 5 };
			SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), Pos);

			img_number++;
			printf("count of train image = %d   \n", img_number);

			if ((variable.val_label == high) || (variable.val_label == low)){
				for (int i = 0; i < INPUT_LAYER; i++) variable.ap1[i] = PIXEL_SCALE(variable.image[i]);
				for (int i = 0; i < B_OUTPUT_LAYER; i++) variable.a_out[i] = 0;
				if (variable.val_label == high) variable.a_out[1] = 1;
				else variable.a_out[0] = 1;

				// initialize z
				for (int i = 0; i < HIDDEN_LAYER; i++) variable.zp2[i] = 0;
				for (int i = 0; i < B_OUTPUT_LAYER; i++) variable.zp3[i] = 0;

				// feed forward
				Activation4(variable.ap1, variable.zp2, network.bp2, network.wp2);
				Sigmoid(variable.ap2, variable.zp2, HIDDEN_LAYER);
				Activation6(variable.ap2, variable.zp3, network.bp3, network.wp3);
				SOFTMAX(variable.ap_out, variable.zp3, B_OUTPUT_LAYER);

				B_Transpose(network.wp3, variable.wp3_t);

				// backpropa
				BP1(variable.ap_out, variable.a_out, variable.delta3, B_OUTPUT_LAYER);
				BP2(variable.wp3_t, variable.delta3, variable.zp2, variable.delta2, B_OUTPUT_LAYER);

				BP3_2(variable.del_bp3, variable.delta3, B_OUTPUT_LAYER);
				BP3_2(variable.del_bp2, variable.delta2, HIDDEN_LAYER);
				BP4_2(variable.del_wp3, variable.delta3, variable.ap2, HIDDEN_LAYER, B_OUTPUT_LAYER);
				BP4_2(variable.del_wp2, variable.delta2, variable.ap1, INPUT_LAYER, HIDDEN_LAYER);

				UPDATE_B(network.bp3, variable.del_bp3, B_OUTPUT_LAYER);
				UPDATE_B(network.bp2, variable.del_bp2, HIDDEN_LAYER);
				UPDATE_W(network.wp3, variable.del_wp3, HIDDEN_LAYER, B_OUTPUT_LAYER);
				UPDATE_W(network.wp2, variable.del_wp2, INPUT_LAYER, HIDDEN_LAYER);
			}
		}

		///test///
		n++;
		epoch_count++;

		fclose(variable.f_image);
		fclose(variable.f_label);
	}

	free(variable.image);
	free(variable.ap1), free(variable.ap2); free(variable.ap_out), free(variable.a_out);
	free(variable.zp2); free(variable.zp3);
	free(variable.delta2); free(variable.delta3); free(variable.del_bp2); free(variable.del_bp3);
	MemFree_2D(variable.del_wp2, HIDDEN_LAYER); MemFree_2D(variable.del_wp3, B_OUTPUT_LAYER);
	MemFree_2D(variable.wp3_t, HIDDEN_LAYER);
}


unsigned char TEST_BINARY(double *origin, BINARY network, int low, int high)
{
	unsigned char output = 0;

	TEST_VARIABLE test;

	// 1D memory allocation
	test.ap2 = MemAlloc_1D(HIDDEN_LAYER); test.ap_out = MemAlloc_1D(B_OUTPUT_LAYER);
	test.zp2 = MemAlloc_1D(HIDDEN_LAYER); test.zp3 = MemAlloc_1D(B_OUTPUT_LAYER);

	////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////// test ////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////
	// initialize z
	for (int i = 0; i < HIDDEN_LAYER; i++) test.zp2[i] = 0;
	for (int i = 0; i < B_OUTPUT_LAYER; i++) test.zp3[i] = 0;

	// feed forward
	Activation4(origin, test.zp2, network.bp2, network.wp2);
	Sigmoid(test.ap2, test.zp2, HIDDEN_LAYER);
	Activation6(test.ap2, test.zp3, network.bp3, network.wp3);
	SOFTMAX(test.ap_out, test.zp3, B_OUTPUT_LAYER);

	if (test.ap_out[0] < test.ap_out[1]) output = high;
	else output = low;

	free(test.ap2); free(test.ap_out);
	free(test.zp2); free(test.zp3);

	return output;
}