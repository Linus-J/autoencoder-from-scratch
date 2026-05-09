#include "ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

int check_dimensions(Matrix *m1, Matrix *m2) {
	if (m1->rows == m2->rows && m1->cols == m2->cols) return 1;
	return 0;
}

Matrix* multiply(Matrix *m1, Matrix *m2, unsigned short int comp) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] * m2->entries[i][j];
			}
		}
		if (comp == 1){
			matrix_free(m1);
		}
		else if (comp == 2){
			matrix_free(m2);
		}
		else if (comp == 3){
			matrix_free(m1);
			matrix_free(m2);
		}
		return m;
	} else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* divide(Matrix *m1, Matrix *m2, unsigned short int comp) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] / m2->entries[i][j];
			}
		}
		if (comp == 1){
			matrix_free(m1);
		}
		else if (comp == 2){
			matrix_free(m2);
		}
		else if (comp == 3){
			matrix_free(m1);
			matrix_free(m2);
		}
		return m;
	} else {
		printf("Dimension mistmatch multiply: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* add(Matrix *m1, Matrix *m2, unsigned short int comp) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
			}
		}
		if (comp == 1){
			matrix_free(m1);
		}
		else if (comp == 2){
			matrix_free(m2);
		}
		else if (comp == 3){
			matrix_free(m1);
			matrix_free(m2);
		}
		return m;
	} else if (m1->rows == m2->rows && m2->cols == 1) {
		/* Broadcast: add a column vector m2 to every column of m1 */
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m1->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] + m2->entries[i][0];
			}
		}
		if (comp == 1) matrix_free(m1);
		else if (comp == 2) matrix_free(m2);
		else if (comp == 3) { matrix_free(m1); matrix_free(m2); }
		return m;
	} else {
		printf("Dimension mismatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* subtract(Matrix *m1, Matrix *m2, unsigned short int comp) {
	if (check_dimensions(m1, m2)) {
		Matrix *m = matrix_create(m1->rows, m1->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				m->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
			}
		}
		if (comp == 1){
			matrix_free(m1);
		}
		else if (comp == 2){
			matrix_free(m2);
		}
		else if (comp == 3){
			matrix_free(m1);
			matrix_free(m2);
		}
		return m;
	} else {
		printf("Dimension mistmatch subtract: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* apply(double (*func)(double), Matrix* m, unsigned short int comp) {
	if (comp == 1) {
		size_t n = (size_t)m->rows * m->cols;
		for (size_t k = 0; k < n; k++) m->data[k] = func(m->data[k]);
		return m;
	}
	Matrix *mat = matrix_copy(m);
	size_t n = (size_t)m->rows * m->cols;
	for (size_t k = 0; k < n; k++) mat->data[k] = func(m->data[k]);
	return mat;
}

Matrix* dot(Matrix *m1, Matrix *m2, unsigned short int comp) {
	if (m1->cols != m2->rows) {
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
	Matrix *m = matrix_create(m1->rows, m2->cols);
	/* Use CBLAS dgemm: m = 1.0 * m1 * m2 + 0.0 * m
	 * All matrices are row-major (CblasRowMajor). */
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	            m1->rows, m2->cols, m1->cols,
	            1.0, m1->data, m1->cols,
	            m2->data, m2->cols,
	            0.0, m->data, m->cols);
	if (comp == 1)       matrix_free(m1);
	else if (comp == 2)  matrix_free(m2);
	else if (comp == 3) { matrix_free(m1); matrix_free(m2); }
	return m;
}

Matrix* scale(double n, Matrix* m, unsigned short int comp) {
	if (comp == 1) {
		size_t sz = (size_t)m->rows * m->cols;
		for (size_t k = 0; k < sz; k++) m->data[k] *= n;
		return m;
	}
	Matrix* mat = matrix_copy(m);
	size_t sz = (size_t)m->rows * m->cols;
	for (size_t k = 0; k < sz; k++) mat->data[k] *= n;
	return mat;
}

Matrix* sqrm(Matrix* m, unsigned short int comp) {
	if (comp == 1) {
		size_t n = (size_t)m->rows * m->cols;
		for (size_t k = 0; k < n; k++) { double v = m->data[k]; m->data[k] = v * v; }
		return m;
	}
	Matrix* mat = matrix_copy(m);
	size_t n = (size_t)m->rows * m->cols;
	for (size_t k = 0; k < n; k++) { double v = m->data[k]; mat->data[k] = v * v; }
	return mat;
}

Matrix* sqrtm(Matrix* m, unsigned short int comp) {
	if (comp == 1) {
		size_t n = (size_t)m->rows * m->cols;
		for (size_t k = 0; k < n; k++) m->data[k] = sqrt(m->data[k]);
		return m;
	}
	Matrix* mat = matrix_copy(m);
	size_t n = (size_t)m->rows * m->cols;
	for (size_t k = 0; k < n; k++) mat->data[k] = sqrt(m->data[k]);
	return mat;
}

Matrix* addScalar(double n, Matrix* m, unsigned short int comp) {
	if (comp == 1) {
		size_t sz = (size_t)m->rows * m->cols;
		for (size_t k = 0; k < sz; k++) m->data[k] += n;
		return m;
	}
	Matrix* mat = matrix_copy(m);
	size_t sz = (size_t)m->rows * m->cols;
	for (size_t k = 0; k < sz; k++) mat->data[k] += n;
	return mat;
}

Matrix* transpose(Matrix* m, unsigned short int comp) {
	Matrix* mat = matrix_create(m->cols, m->rows);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[j][i] = m->entries[i][j];
		}
	}
	if (comp == 1){
		matrix_free(m);
	}
	return mat;
}