CC = gcc
CFLAGS = -shared -fPIC -O3 -march=native
TARGET = libtriinterp.so

all: $(TARGET)

$(TARGET): triangle_interp.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)

.PHONY: all clean
