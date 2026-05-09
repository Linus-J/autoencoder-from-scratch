#pragma once

typedef struct {
	double *data;      /* flat row-major storage: data[i*cols + j] */
	double **entries;  /* row pointers into data: entries[i] = &data[i*cols] */
	int rows;
	int cols;
} Matrix;

Matrix* matrix_create(int row, int col);
void matrix_fill(Matrix *m, double n);
void matrix_free(Matrix *m);
void matrix_print(Matrix *m);
Matrix* matrix_copy(Matrix *m);
void matrix_save(Matrix* m, char* file_string);
Matrix* matrix_load(char* file_string);
void matrix_init(Matrix* m, double n, int k);
int matrix_argmax(Matrix* m);
Matrix* matrix_flatten(Matrix* m, int axis);
Matrix* matrix_unflatten(Matrix* m);