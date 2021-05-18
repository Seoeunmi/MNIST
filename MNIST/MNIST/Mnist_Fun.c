#include <stdio.h>
#include "Mnist_Fun.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

union BYTE4{   //(Big endian으로 기록된)4바이트 int 읽어 little endian 4byte int 로 바꾸는데 사용할 union 
	int v;
	unsigned char uc[4];
};

// big endian int 로 기록된 4바이트를 읽어 little endian int 로 해석하여 리턴
int read_int_bigendian(FILE *fp){
	unsigned char buf4[4] = { 0 };
	union BYTE4 b4;
	fread(buf4, 1, 4, fp);
	b4.uc[0] = buf4[3];   //바이트 순서 뒤집기
	b4.uc[1] = buf4[2];
	b4.uc[2] = buf4[1];
	b4.uc[3] = buf4[0];

	return b4.v;
}

void Gussian_Reset(double *bp2, double *bp3, double **wp2, double **wp3)
{
	for (int i = 0; i < HIDDEN_LAYER; i++) for (int j = 0; j < INPUT_LAYER; j++) wp2[i][j] = GaussianRandom();
	for (int i = 0; i < OUTPUT_LAYER; i++) for (int j = 0; j < HIDDEN_LAYER; j++) wp3[i][j] = GaussianRandom();

	for (int i = 0; i < HIDDEN_LAYER; i++) bp2[i] = GaussianRandom();
	for (int i = 0; i < OUTPUT_LAYER; i++) bp3[i] = GaussianRandom();
}

void Activation4(double *a1, double *z2, double *b, double **w)
{
	for (int i = 0; i < HIDDEN_LAYER; i++)
	{
		for (int j = 0; j < INPUT_LAYER; j++)
		{
			z2[i] += a1[j] * w[i][j];
		}
		z2[i] += b[i];
	}
}
void Activation5(double *a2, double *z3, double *b, double **w)
{
	for (int i = 0; i < OUTPUT_LAYER; i++)
	{
		for (int j = 0; j < HIDDEN_LAYER; j++)
		{
			z3[i] += a2[j] * w[i][j];
		}
		z3[i] += b[i];
	}
}


double** MemAlloc_2D(int width, int height)  // 2D memory allocation
{
	double **arr;
	int i = 0;

	arr = (double**)malloc(sizeof(double*)*height);
	for (i = 0; i < height; i++)
		arr[i] = (double*)malloc(sizeof(double)*width);

	return arr;
}
void MemFree_2D(double** arr, int height)  // 2D memory free
{
	int i = 0;

	for (i = 0; i < height; i++)
		free(arr[i]);
	free(arr);
}

int** MemAlloc_INT_2D(int width, int height)  // 2D memory allocation (int)
{
	int **arr;
	int i = 0;

	arr = (int**)malloc(sizeof(int*)*height);
	for (i = 0; i < height; i++)
		arr[i] = (int*)malloc(sizeof(int)*width);

	return arr;
}
void MemFree_INT_2D(int** arr, int height)  // 2D memory free (int)
{
	int i = 0;

	for (i = 0; i < height; i++)
		free(arr[i]);
	free(arr);
}

double* MemAlloc_1D(int size)
{
	double *arr;
	arr = (double *)malloc(sizeof(double)*size);
}

void Sigmoid(double *a, double *z, int layer)
{
	for (int i = 0; i < layer; i++)
	{
		a[i] = 1.0 / (1 + exp(-z[i]));
	}
}
double GaussianRandom(void) {
	double v1, v2, s;

	do {
		v1 = 2 * ((double)rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
		v2 = 2 * ((double)rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
		s = v1 * v1 + v2 * v2;
	} while (s >= 1 || s == 0);

	s = sqrt((-2 * log(s)) / s);

	return v1 * s;
}

double Delta_Deriv(double z)
{
	double result;

	result = (1.0 / (1 + exp(-z)))*(1 - (1.0 / (1 + exp(-z))));

	return result;
}


void Transpose(double **w, double **w_t)
{
	for (int i = 0; i < HIDDEN_LAYER; i++)
		for (int j = 0; j < OUTPUT_LAYER; j++)
			w_t[i][j] = w[j][i];
}

void BP1(double *ap_out, double *a_out, double *delta, int layer)
{
	for (int i = 0; i < layer; i++)
		delta[i] = ap_out[i] - a_out[i];
}
void BP2(double **wp3_t, double *delta3, double *zp2, double *delta2, int layer)
{
	double Dot_product[HIDDEN_LAYER] = { 0, }, Sig_deriv_z2[HIDDEN_LAYER];

	for (int i = 0; i < HIDDEN_LAYER; i++){
		for (int j = 0; j < layer; j++){
			Dot_product[i] += wp3_t[i][j] * delta3[j];
		}
	}

	for (int i = 0; i < HIDDEN_LAYER; i++){
		Sig_deriv_z2[i] = Delta_Deriv(zp2[i]);
		delta2[i] = Dot_product[i] * Sig_deriv_z2[i];
	}
}

void BP3_2(double *del_bp, double *delta, int layer)
{
	for (int i = 0; i < layer; i++)
		del_bp[i] += delta[i];
}

void BP4_2(double **del_wp, double *delta, double *ap_1, int layer1, int layer2)
{
	for (int j = 0; j < layer2; j++)
		for (int k = 0; k < layer1; k++)
			del_wp[j][k] += delta[j] * ap_1[k];
}

void UPDATE_B(double *bp, double *del_bp, int layer)
{
	for (int i = 0; i < layer; i++)
		bp[i] -= LR*del_bp[i];
}

void UPDATE_W(double  **wp, double **del_wp, int layer1, int layer2)
{
	for (int j = 0; j < layer2; j++)
		for (int k = 0; k < layer1; k++)
			wp[j][k] -= LR*del_wp[j][k];
}

void SOFTMAX(double *a_out, double *zp, int layer)
{
	double sum = 0, max = zp[0];

	for (int i = 0; i < layer; i++){
		if (zp[i] > max) max = zp[i];
	}

	for (int i = 0; i < layer; i++)
		sum += exp(zp[i] - max);

	for (int i = 0; i < layer; i++)
		a_out[i] = exp(zp[i] - max) / sum;
}

void TRAIN_NETWORK()
{
	//// practice model variable
	double Cost[EPOCH] = { 0, }, pre_cost = 0;
	int  img_number = 0, n = 0, size = 0, epoch_count = 0;

	TRAIN_VARIABLE variable;
	BINARY network[10];

	// 1D memory allocation
	variable.ap1 = MemAlloc_1D(INPUT_LAYER); variable.ap2 = MemAlloc_1D(HIDDEN_LAYER); variable.ap_out = MemAlloc_1D(OUTPUT_LAYER); variable.a_out = MemAlloc_1D(OUTPUT_LAYER);
	variable.zp2 = MemAlloc_1D(HIDDEN_LAYER); variable.zp3 = MemAlloc_1D(OUTPUT_LAYER);	variable.bp2 = MemAlloc_1D(HIDDEN_LAYER); variable.bp3 = MemAlloc_1D(OUTPUT_LAYER);
	variable.delta2 = MemAlloc_1D(HIDDEN_LAYER); variable.delta3 = MemAlloc_1D(OUTPUT_LAYER); variable.del_bp2 = MemAlloc_1D(HIDDEN_LAYER); variable.del_bp3 = MemAlloc_1D(OUTPUT_LAYER);
	variable.image = MemAlloc_1D(INPUT_LAYER);

	//2D memory allocation
	variable.wp2 = MemAlloc_2D(INPUT_LAYER, HIDDEN_LAYER); variable.wp3 = MemAlloc_2D(HIDDEN_LAYER, OUTPUT_LAYER);
	variable.del_wp2 = MemAlloc_2D(INPUT_LAYER, HIDDEN_LAYER); variable.del_wp3 = MemAlloc_2D(HIDDEN_LAYER, OUTPUT_LAYER); variable.wp3_t = MemAlloc_2D(OUTPUT_LAYER, HIDDEN_LAYER);

	for (int i = 0; i < 10; i++){
		network[i].bp2 = MemAlloc_1D(HIDDEN_LAYER);
		network[i].bp3 = MemAlloc_1D(B_OUTPUT_LAYER);

		network[i].wp2 = MemAlloc_2D(INPUT_LAYER, HIDDEN_LAYER);
		network[i].wp3 = MemAlloc_2D(HIDDEN_LAYER, B_OUTPUT_LAYER);
	}

	Gussian_Reset(variable.bp2, variable.bp3, variable.wp2, variable.wp3);

	// binary network train
	//TRAIN_BINARY(network[0], 2, 3); // network, real, predict
	//TRAIN_BINARY(network[1], 3, 5);
	//TRAIN_BINARY(network[2], 4, 9);
	//TRAIN_BINARY(network[3], 5, 3);
	//TRAIN_BINARY(network[4], 6, 5);
	//TRAIN_BINARY(network[5], 7, 2);
	//TRAIN_BINARY(network[6], 7, 9);
	//TRAIN_BINARY(network[7], 8, 5);
	//TRAIN_BINARY(network[8], 9, 4);
	//TRAIN_BINARY(network[9], 9, 5);


	while (epoch_count < EPOCH){
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
		while (size < TRAIN_SIZE / BATCH_SIZE){
			for (int i = 0; i < HIDDEN_LAYER; i++) for (int j = 0; j < INPUT_LAYER; j++) variable.del_wp2[i][j] = 0;
			for (int i = 0; i < OUTPUT_LAYER; i++) for (int j = 0; j < HIDDEN_LAYER; j++) variable.del_wp3[i][j] = 0;

			for (int i = 0; i < HIDDEN_LAYER; i++) variable.del_bp2[i] = 0;
			for (int i = 0; i < OUTPUT_LAYER; i++) variable.del_bp3[i] = 0;

			for (int batch = 0; batch < BATCH_SIZE; batch++){
				fread(variable.image, 28 * 28, 1, variable.f_image);
				fread(&(variable.val_label), 1, 1, variable.f_label);

				for (int i = 0; i < INPUT_LAYER; i++) variable.ap1[i] = PIXEL_SCALE(variable.image[i]);
				for (int i = 0; i < OUTPUT_LAYER; i++) variable.a_out[i] = 0;
				variable.a_out[variable.val_label] = 1;

				COORD Pos = { 0, 0 };
				SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), Pos);

				img_number++;
				printf("count of train image = %d   \n", img_number);

				// initialize z
				for (int i = 0; i < HIDDEN_LAYER; i++) variable.zp2[i] = 0;
				for (int i = 0; i < OUTPUT_LAYER; i++) variable.zp3[i] = 0;

				// feed forward
				Activation4(variable.ap1, variable.zp2, variable.bp2, variable.wp2);
				Sigmoid(variable.ap2, variable.zp2, HIDDEN_LAYER);
				Activation5(variable.ap2, variable.zp3, variable.bp3, variable.wp3);
				SOFTMAX(variable.ap_out, variable.zp3, OUTPUT_LAYER);

				pre_cost = 0;
				for (int i = 0; i < OUTPUT_LAYER; i++)
					pre_cost += ((-1)*variable.a_out[i] * log(variable.ap_out[i]));

				Cost[n] += pre_cost;

				Transpose(variable.wp3, variable.wp3_t);

				// backpropa
				BP1(variable.ap_out, variable.a_out, variable.delta3, OUTPUT_LAYER);
				BP2(variable.wp3_t, variable.delta3, variable.zp2, variable.delta2, OUTPUT_LAYER);

				BP3_2(variable.del_bp3, variable.delta3, OUTPUT_LAYER);
				BP3_2(variable.del_bp2, variable.delta2, HIDDEN_LAYER);
				BP4_2(variable.del_wp3, variable.delta3, variable.ap2, HIDDEN_LAYER, OUTPUT_LAYER);
				BP4_2(variable.del_wp2, variable.delta2, variable.ap1, INPUT_LAYER, HIDDEN_LAYER);
			}

			for (int i = 0; i < HIDDEN_LAYER; i++) for (int j = 0; j < INPUT_LAYER; j++) variable.del_wp2[i][j] /= BATCH_SIZE;
			for (int i = 0; i < OUTPUT_LAYER; i++) for (int j = 0; j < HIDDEN_LAYER; j++) variable.del_wp3[i][j] /= BATCH_SIZE;

			for (int i = 0; i < HIDDEN_LAYER; i++) variable.del_bp2[i] /= BATCH_SIZE;
			for (int i = 0; i < OUTPUT_LAYER; i++) variable.del_bp3[i] /= BATCH_SIZE;

			UPDATE_B(variable.bp3, variable.del_bp3, OUTPUT_LAYER);
			UPDATE_B(variable.bp2, variable.del_bp2, HIDDEN_LAYER);
			UPDATE_W(variable.wp3, variable.del_wp3, HIDDEN_LAYER, OUTPUT_LAYER);
			UPDATE_W(variable.wp2, variable.del_wp2, INPUT_LAYER, HIDDEN_LAYER);

			size++;
		}
		Cost[n] /= TRAIN_SIZE;

		///test///
		TEST_NETWORK(network, variable.bp2, variable.bp3, variable.wp2, variable.wp3, n);
		n++;
		epoch_count++;

		fclose(variable.f_image);
		fclose(variable.f_label);
	}

	/*printf("///////////////// Result_Cost_Epoch : %d //////////////////// \n", EPOCH);
	for (int i = 0; i < EPOCH; i++) printf("%lf   \n", Cost[i]);*/

	free(variable.image);
	free(variable.ap1), free(variable.ap2); free(variable.ap_out), free(variable.a_out);
	free(variable.zp2); free(variable.zp3);
	free(variable.bp2); free(variable.bp3);
	free(variable.delta2); free(variable.delta3); free(variable.del_bp2); free(variable.del_bp3);
	MemFree_2D(variable.wp2, HIDDEN_LAYER); MemFree_2D(variable.wp3, OUTPUT_LAYER);
	MemFree_2D(variable.del_wp2, HIDDEN_LAYER); MemFree_2D(variable.del_wp3, OUTPUT_LAYER);
	MemFree_2D(variable.wp3_t, HIDDEN_LAYER);
	for (int i = 0; i < 10; i++){
		free(network[i].bp2); free(network[i].bp3);
		MemFree_2D(network[i].wp2, HIDDEN_LAYER);
		MemFree_2D(network[i].wp3, B_OUTPUT_LAYER);
	}
}

void TEST_NETWORK(BINARY *network, double *bp2, double *bp3, double **wp2, double **wp3, int n)
{
	int img_number = 0, first = 0, second = 0, high, low, accuracy = 0, best_accuracy = 0;
	unsigned char output = 0, revise = 0;
	int **confusion_matrix;

	TEST_VARIABLE test;

	// 1D memory allocation
	test.ap1 = MemAlloc_1D(INPUT_LAYER); test.ap2 = MemAlloc_1D(HIDDEN_LAYER); test.ap_out = MemAlloc_1D(OUTPUT_LAYER); test.a_out = MemAlloc_1D(OUTPUT_LAYER);
	test.zp2 = MemAlloc_1D(HIDDEN_LAYER); test.zp3 = MemAlloc_1D(OUTPUT_LAYER);
	test.image = MemAlloc_1D(INPUT_LAYER);

	// 2D memory allocation
	confusion_matrix = MemAlloc_INT_2D(10, 10);

	for (int i = 0; i < 10; i++) for (int j = 0; j < 10; j++) confusion_matrix[i][j] = 0;

	test.f_test_label = fopen("t10k-labels.idx1-ubyte", "rb");
	test.f_test_image = fopen("t10k-images.idx3-ubyte", "rb");
	// read header
	for (int i = 0; i < 4; i++){
		read_int_bigendian(test.f_test_image);
		if (i < 2) read_int_bigendian(test.f_test_label);
	}

	////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////// test ////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////
	img_number = 0;

	while (img_number < TEST_SIZE){
		output = 0; img_number++;
		fread(test.image, 28 * 28, 1, test.f_test_image);
		fread(&(test.val_label), 1, 1, test.f_test_label);

		for (int i = 0; i < INPUT_LAYER; i++) test.ap1[i] = PIXEL_SCALE(test.image[i]);
		for (int i = 0; i < OUTPUT_LAYER; i++) test.a_out[i] = 0;
		test.a_out[test.val_label] = 1;

		// initialize z
		for (int i = 0; i < HIDDEN_LAYER; i++) test.zp2[i] = 0;
		for (int i = 0; i < OUTPUT_LAYER; i++) test.zp3[i] = 0;

		// feed forward
		Activation4(test.ap1, test.zp2, bp2, wp2);
		Sigmoid(test.ap2, test.zp2, HIDDEN_LAYER);
		Activation5(test.ap2, test.zp3, bp3, wp3);
		SOFTMAX(test.ap_out, test.zp3, OUTPUT_LAYER);

		for (int i = 1; i < OUTPUT_LAYER; i++){
			if (test.ap_out[i] > test.ap_out[output]) output = i;
		}

		COORD Pos = { 0, 2 };
		SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), Pos);

		printf("Epoch : %d \n", n);
		printf("count of test image = %d   \n", img_number);
		printf("accuracy = %f %%     ", (float)accuracy / 100);

		if ((test.val_label == 2) && (output == 3)) output = TEST_BINARY(test.ap1, network[0], output, test.val_label);
		if ((test.val_label == 3) && (output == 5)) output = TEST_BINARY(test.ap1, network[1], output, test.val_label);
		if ((test.val_label == 4) && (output == 9)) output = TEST_BINARY(test.ap1, network[2], output, test.val_label);
		if ((test.val_label == 5) && (output == 3)) output = TEST_BINARY(test.ap1, network[3], output, test.val_label);
		if ((test.val_label == 6) && (output == 5)) output = TEST_BINARY(test.ap1, network[4], output, test.val_label);
		if ((test.val_label == 7) && (output == 2)) output = TEST_BINARY(test.ap1, network[5], output, test.val_label);
		if ((test.val_label == 7) && (output == 9)) output = TEST_BINARY(test.ap1, network[6], output, test.val_label);
		if ((test.val_label == 8) && (output == 5)) output = TEST_BINARY(test.ap1, network[7], output, test.val_label);
		if ((test.val_label == 9) && (output == 4)) output = TEST_BINARY(test.ap1, network[8], output, test.val_label);
		if ((test.val_label == 9) && (output == 5)) output = TEST_BINARY(test.ap1, network[9], output, test.val_label);

		if (test.val_label == output) accuracy++;

		confusion_matrix[test.val_label][output]++;
	}
	if (best_accuracy < accuracy) best_accuracy = accuracy;

	printf("best_accuracy = %f %% \n", (float)best_accuracy / 100);

	Print_Matrix(confusion_matrix);

	free(test.image);
	free(test.ap1), free(test.ap2); free(test.ap_out), free(test.a_out);
	free(test.zp2); free(test.zp3);
	MemFree_INT_2D(confusion_matrix, 10);

	fclose(test.f_test_image); fclose(test.f_test_label);
}

void Print_Matrix(int **matrix)
{
	int actual_sum[10] = { 0, }, predict_sum[10] = { 0, }, img_number = 0;
	float Recall[10] = { 0, }, Precision[10] = { 0, }, accuracy = 0, best_accuracy = 0;

	for (int i = 0; i < 10; i++){
		for (int j = 0; j < 10; j++){
			actual_sum[i] += matrix[i][j];
			predict_sum[i] += matrix[j][i];
		}
		img_number += actual_sum[i];
	}

	// print result
	printf("\n\n ////////////////////Confusion Matrix/////////////////// \n  P");
	for (int i = 0; i < 10; i++) printf("%4d ", i);
	printf("\n R  -------------------------------------------------- \n");
	for (int i = 0; i < 10; i++){
		printf(" %d |", i);
		for (int j = 0; j < 10; j++){
			printf("%4d ", matrix[i][j]);
		}
		printf("| %4d \n", actual_sum[i]);
	}
	printf("    --------------------------------------------------   \n    ");
	for (int i = 0; i < 10; i++) printf("%4d ", predict_sum[i]);

	// Find the value
	for (int i = 0; i < 10; i++) Recall[i] = (float)matrix[i][i] / actual_sum[i];
	for (int i = 0; i < 10; i++) Precision[i] = (float)matrix[i][i] / predict_sum[i];
	for (int i = 0; i < 10; i++) accuracy += (float)matrix[i][i];
	accuracy /= (float)img_number;
	if (best_accuracy < accuracy) best_accuracy = accuracy;

	// print analysis
	printf("\n\n ////////////////////Analysis Matrix/////////////////// \n");
	printf("Table_accuracy = %f %%     Table_best_accuracy = %f %%\n           ", accuracy*100.0, best_accuracy * 100);
	for (int i = 0; i < 10; i++) printf("%6d ", i);
	printf("\n            ---------------------------------------------------------------------- \n");
	printf("   Recall  | ");
	for (int i = 0; i < 10; i++) printf("%.4f ", Recall[i]);
	printf("|\n Precision | ");
	for (int i = 0; i < 10; i++) printf("%.4f ", Precision[i]);
	printf("|\n            ----------------------------------------------------------------------   \n");
}