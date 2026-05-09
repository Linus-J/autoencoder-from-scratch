#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../util/ziggurat_inline.h"

#define MAXCHAR 100

Matrix* matrix_create(int row, int col) {
	Matrix *matrix = malloc(sizeof(Matrix));
	matrix->rows = row;
	matrix->cols = col;
	/* Single contiguous allocation — cache-friendly and BLAS-ready */
	matrix->data    = malloc((size_t)row * col * sizeof(double));
	matrix->entries = malloc((size_t)row * sizeof(double*));
	for (int i = 0; i < row; i++) {
		matrix->entries[i] = &matrix->data[(size_t)i * col];
	}
	return matrix;
}

void matrix_fill(Matrix *m, double n) {
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->entries[i][j] = n;
		}
	}
}

void matrix_free(Matrix *m) {
	free(m->data);
	m->data = NULL;
	free(m->entries);
	m->entries = NULL;
	free(m);
}

void matrix_print(Matrix* m) {
	printf("Rows: %d Columns: %d\n", m->rows, m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%1.3f ", m->entries[i][j]);
		}
		printf("\n");
	}
}

Matrix* matrix_copy(Matrix* m) {
	Matrix* mat = matrix_create(m->rows, m->cols);
	size_t n = (size_t)m->rows * m->cols;
	for (size_t k = 0; k < n; k++) mat->data[k] = m->data[k];
	return mat;
}

void matrix_save(Matrix* m, char* file_string) {
	FILE* file = fopen(file_string, "w");
	if (!file) { perror(file_string); return; }
	fprintf(file, "%d\n", m->rows);
	fprintf(file, "%d\n", m->cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			fprintf(file, "%.6f\n", m->entries[i][j]);
		}
	}
	fclose(file);
}

Matrix* matrix_load(char* file_string) {
	FILE* file = fopen(file_string, "r");
	if (!file) { perror(file_string); return NULL; }
	char entry[MAXCHAR];
	fgets(entry, MAXCHAR, file);
	int rows = atoi(entry);
	fgets(entry, MAXCHAR, file);
	int cols = atoi(entry);
	Matrix* m = matrix_create(rows, cols);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			fgets(entry, MAXCHAR, file);
			m->entries[i][j] = strtod(entry, NULL);
		}
	}
	fclose(file);
	return m;
}

double uniform_distribution(double low, double high) {
	double difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}

void matrix_init(Matrix* m, double n, int k) {
	/* He initialisation: N(0,1) * sqrt(2 / fan_in)
	 * Caller must have invoked r4_nor_setup() first. */
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			m->entries[i][j] = r4_nor_value() * sqrt((k + 1) / n);
		}
	}
}

int matrix_argmax(Matrix* m) {
	// Expects a Mx1 matrix
	double max_score = 0;
	int max_idx = 0;
	for (int i = 0; i < m->rows; i++) {
		if (m->entries[i][0] > max_score) {
			max_score = m->entries[i][0];
			max_idx = i;
		}
	}
	return max_idx;
}

Matrix* matrix_flatten(Matrix* m, int axis) {
	// Axis = 0 -> Column Vector, Axis = 1 -> Row Vector
	Matrix* mat;
	if (axis == 0) {
		mat = matrix_create(m->rows * m->cols, 1);
	} else if (axis == 1) {
		mat = matrix_create(1, m->rows * m->cols);
	} else {
		printf("Argument to matrix_flatten must be 0 or 1");
		exit(EXIT_FAILURE);
	}
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			if (axis == 0) mat->entries[i * m->cols + j][0] = m->entries[i][j];
			else if (axis == 1) mat->entries[0][i * m->cols + j] = m->entries[i][j];
		}
	}
	return mat;
}

Matrix* matrix_unflatten(Matrix* m) {
	// Takes 784x1 col vector and returns 28x28 matrix
	Matrix* mat = matrix_create(28, 28);

	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			mat->entries[i][j] = m->entries[i * 28 + j][0];
		}
	}
	return mat;
}