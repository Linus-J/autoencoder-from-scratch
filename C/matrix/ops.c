#include "ops.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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
	} else {
		if (m1->rows == m2->rows){
			Matrix *m = matrix_create(m1->rows, m1->cols);
			for (int i = 0; i < m1->rows; i++) {
				for (int j = 0; j < m2->cols; j++) {
					m->entries[i][j] = m1->entries[i][j] + m2->entries[i][0];
				}
			}
		}
		else{
			printf("Dimension mistmatch add: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
			exit(1);
		}
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
	Matrix *mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = (*func)(m->entries[i][j]);
		}
	}
	if (comp == 1){
		matrix_free(m);
	}
	return mat;
}

Matrix* dot(Matrix *m1, Matrix *m2, unsigned short int comp) {
	if (m1->cols == m2->rows) {
		Matrix *m = matrix_create(m1->rows, m2->cols);
		for (int i = 0; i < m1->rows; i++) {
			for (int j = 0; j < m2->cols; j++) {
				double sum = 0;
				for (int k = 0; k < m2->rows; k++) {
					sum += m1->entries[i][k] * m2->entries[k][j];
				}
				m->entries[i][j] = sum;
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
		printf("Dimension mistmatch dot: %dx%d %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
		exit(1);
	}
}

Matrix* scale(double n, Matrix* m, unsigned short int comp) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] *= n;
		}
	}
	if (comp == 1){
		matrix_free(m);
	}
	return mat;
}

Matrix* sqrm(Matrix* m, unsigned short int comp) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = pow(mat->entries[i][j],2);
		}
	}
	if (comp == 1){
		matrix_free(m);
	}
	return mat;
}

Matrix* sqrtm(Matrix* m, unsigned short int comp) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] = sqrt(mat->entries[i][j]);
		}
	}
	if (comp == 1){
		matrix_free(m);
	}
	return mat;
}

Matrix* addScalar(double n, Matrix* m, unsigned short int comp) {
	Matrix* mat = matrix_copy(m);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			mat->entries[i][j] += n;
		}
	}
	if (comp == 1){
		matrix_free(m);
	}
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