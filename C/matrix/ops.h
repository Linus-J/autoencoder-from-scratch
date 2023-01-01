#pragma once

#include "matrix.h"

Matrix* multiply(Matrix* m1, Matrix* m2, unsigned short int comp);
Matrix* divide(Matrix *m1, Matrix *m2, unsigned short int comp);
Matrix* add(Matrix* m1, Matrix* m2, unsigned short int comp);
Matrix* subtract(Matrix* m1, Matrix* m2, unsigned short int comp);
Matrix* dot(Matrix* m1, Matrix* m2, unsigned short int comp);
Matrix* apply(double (*func)(double), Matrix* m, unsigned short int comp);
Matrix* scale(double n, Matrix* m, unsigned short int comp);
Matrix* addScalar(double n, Matrix* m, unsigned short int comp);
Matrix* sqrtm(Matrix* m, unsigned short int comp);
Matrix* sqrm(Matrix* m, unsigned short int comp);
Matrix* transpose(Matrix* m, unsigned short int comp);
