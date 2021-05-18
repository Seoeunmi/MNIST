#ifndef Mnist_Fun
#define Mnist_Fun

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define INPUT_LAYER 784  //28*28
#define HIDDEN_LAYER 30
#define OUTPUT_LAYER 10

#define B_OUTPUT_LAYER 2

#define LR 3 //learning_rate

#define BATCH_SIZE 10
#define EPOCH 100

#define PIXEL_SCALE(x) (((double) (x)) / 255.0f)

typedef struct TRAIN_VARIABLE{
	FILE *f_image;
	FILE *f_label;

	unsigned char val_label;
	unsigned char *image;


	double *ap1;
	double *ap2;
	double *ap_out;
	double *a_out;

	double *zp2;
	double *zp3;

	double *bp2;
	double *bp3;

	double *delta2;
	double *delta3;
	double *del_bp2;
	double *del_bp3;

	double **wp2;
	double **wp3;
	double **wp3_t;
	double **del_wp2;
	double **del_wp3;

}TRAIN_VARIABLE;

typedef struct TEST_VARIABLE{
	FILE *f_test_image;
	FILE *f_test_label;

	unsigned char val_label;
	unsigned char *image;

	double *ap1;
	double *ap2;
	double *ap_out;
	double *a_out;

	double *zp2;
	double *zp3;
}TEST_VARIABLE;

typedef struct BINARY{
	double *bp2;
	double *bp3;

	double **wp2;
	double **wp3;
}BINARY;

int read_int_bigendian(FILE *fp);
void Gussian_Reset(double *bp2, double *bp3, double **wp2, double **wp3);
void B_Gussian_Reset(double *bp2, double *bp3, double **wp2, double **wp3);

void Activation4(double *a1, double *z2, double *b, double **w);
void Activation5(double *a2, double *z3, double *b, double **w);
void Activation6(double *a2, double *z3, double *b, double **w);

double **MemAlloc_2D(int width, int height);
void MemFree_2D(double** arr, int height);

int** MemAlloc_INT_2D(int width, int height);
void MemFree_INT_2D(int** arr, int height);

double* MemAlloc_1D(int size);

void Sigmoid(double *z, double *a, int layer);

double GaussianRandom(void);

double Delta_Deriv(double z);

void Transpose(double **w, double **w_t);
void B_Transpose(double **w, double **w_t);

void BP1(double *ap3, double *a3, double *delta3, int layer);
void BP2(double **wp3_t, double *delta3, double *zp2, double *delta2, int layer);

void BP3_2(double *del_bp, double *delta, int layer);
void BP4_2(double **del_wp, double *delta, double *ap_1, int layer1, int layer2);

void UPDATE_B(double *bp, double *del_bp, int layer);
void UPDATE_W(double  **wp, double **del_wp, int layer1, int layer2);

void SOFTMAX(double *a_out, double *zp, int layer);

void TRAIN_NETWORK();
void TEST_NETWORK(BINARY *network, double *bp2, double *bp3, double **wp2, double **wp3, int n);

void TRAIN_BINARY(BINARY network, int high, int low);
unsigned char TEST_BINARY(double *origin, BINARY network, int low, int high);

// printf
void Print_Matrix(int **matrix);
#endif Mnist_Fun