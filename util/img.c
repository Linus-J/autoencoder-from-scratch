#include "img.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCHAR 10000

Img** csv_to_imgs(char* file_string, int number_of_imgs) {
	FILE *fp;
	Img** imgs = malloc(number_of_imgs * sizeof(Img*));
	char row[MAXCHAR];
	fp = fopen(file_string, "r");

	// Read the first line 
	fgets(row, MAXCHAR, fp);
	int i = 0;
	while (feof(fp) != 1 && i < number_of_imgs) {
		imgs[i] = malloc(sizeof(Img));

		int j = 0;
		fgets(row, MAXCHAR, fp);
		char* token = strtok(row, ",");
		imgs[i]->img_data = matrix_create(28, 28);
		while (token != NULL) {
			if (j == 0) {
				imgs[i]->label = atoi(token);
			} else {
				imgs[i]->img_data->entries[(j-1) / 28][(j-1) % 28] = atoi(token) / 256.0;
			}
			token = strtok(NULL, ",");
			j++;
		}
		i++;
	}
	fclose(fp);
	return imgs;
}

void img_print(Img* img) {
	matrix_print(img->img_data);
	printf("Img Label: %d\n", img->label);
}

void img_save(Img* img) {
	float image[28][28];
	int i,j=0;
	for (i = 0; i < 28; i++){	
		for (j = 0; j < 28; j++){
			image[i][j] = img -> img_data -> entries[i][j];
		}
	}
	float temp=0;
	int width = 28, height = 28;
	FILE* pgmimg;
	pgmimg = fopen("pgmimg.pgm", "wb");
	// Write Magic Number to the File
	fprintf(pgmimg, "P2\n"); 
	
	// Write Width and Height
	fprintf(pgmimg, "%d %d\n", width, height); 
	
	// Writing the maximum gray value
	fprintf(pgmimg, "255\n"); 
	int count = 0;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			temp = image[i][j];
			// Writing the gray values in the 2D array to the file
			fprintf(pgmimg, "%d ", (int)(temp*255));
		}
		fprintf(pgmimg, "\n");
	}
	fclose(pgmimg);
}

void img_save_new(Img **imgs, int n) {
	int i = 0, j = 0;
    int a = 28, b =28*n;
	float **image = malloc(sizeof(int*)*a);

	for (i = 0; i < a; i++){
		image[i] = malloc(sizeof(float)*b);
	}

	for (int h = 1; h < n+1; h++){	
		for (i = 0; i < a; i++){	
			for (j = a*(h-1); j < a*h; j++){
				image[i][j] = imgs[h-1] -> img_data -> entries[i][j-a*(h-1)];
			}
		}
	}
	float temp = 0;
	FILE* pgmimg;
	pgmimg = fopen("nimages.pgm", "wb");
	// Write Magic Number to the File
	fprintf(pgmimg, "P2\n"); 
	
	// Write Width and Height
	fprintf(pgmimg, "%d %d\n", b, a); 
	
	// Writing the maximum gray value
	fprintf(pgmimg, "255\n"); 
	int count = 0;
	for (i = 0; i < a; i++) {
		for (j = 0; j < b; j++) {
			temp = image[i][j];
			// Writing the gray values in the 2D array to the file
			fprintf(pgmimg, "%d ", (int)(temp*255));
		}
		fprintf(pgmimg, "\n");
	}
	fclose(pgmimg);

	for (i = 0; i < 28; i++){
		free(image[i]);
		image[i] = NULL;
	}
	free(image);
	image = NULL;
}

void img_free(Img* img) {
	matrix_free(img->img_data);
	free(img);
	img = NULL;
}

void imgs_free(Img** imgs, int n) {
	for (int i = 0; i < n; i++) {
		img_free(imgs[i]);
	}
}
