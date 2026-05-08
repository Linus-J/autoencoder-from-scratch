C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c) main.c
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h)
OBJ = ${C_SOURCES:.c=.o}
CFLAGS = -O2 -Wall -Wextra -Wpedantic

MAIN = main
CC = /usr/bin/gcc

main: ${OBJ}
	${CC} ${CFLAGS} $^ -o $@ -lm

%.o: %.c ${HEADERS}
	${CC} ${CFLAGS} -c $< -o $@ -lm

LIB_OBJ = $(filter-out main.o, ${OBJ})

valgrind_test: ${LIB_OBJ} valgrind_test.o
	${CC} ${CFLAGS} $^ -o $@ -lm

valgrind: valgrind_test
	valgrind --leak-check=full --track-origins=yes --error-exitcode=1 ./valgrind_test

clean:
	rm -f matrix/*.o neural/*.o util/*.o *.o ${MAIN} valgrind_test valgrind_test.o
