#include "activations.h"

#include <math.h>
#include "../matrix/ops.h"

double sigmoid(double input) {
	return 1.0 / (1 + exp(-1 * input));
}

Matrix* sigmoidPrime(Matrix* m) {
	Matrix* ones = matrix_create(m->rows, m->cols);
	matrix_fill(ones, 1);
	Matrix* subtracted = subtract(ones, m, 0);
	Matrix* multiplied = multiply(m, subtracted, 0);
	matrix_free(ones);
	matrix_free(subtracted);
	return multiplied;
}

double relu(double input) {
	return (input>=0)*input;
}

Matrix* reluPrime(Matrix* m) {
	Matrix* mrelu = matrix_create(m->rows, m->cols);
	int i,j;
	for (i = 0; i < m->rows; i++){	
		for (j = 0; j < m->cols; j++){
			mrelu -> entries[i][j] = (m -> entries[i][j] >= 0);
		}
	}
	return mrelu;
}

Matrix* softmax(Matrix* m) {
	double total = 0;
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			total += exp(m->entries[i][j]);
		}
	}
	Matrix* mat = matrix_create(m->rows, m->cols);
	for (int i = 0; i < mat->rows; i++) {
		for (int j = 0; j < mat->cols; j++) {
			mat->entries[i][j] = exp(m->entries[i][j]) / total;
		}
	}
	return mat;
}