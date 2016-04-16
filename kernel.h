#ifndef KERNEL_H
#define KERNEL_H


struct uchar4;

void kernelLauncher(uchar4 *d_out, float *d_u, float *d_v, int w, int h,
                    int setup);
void resetgs(float *d_u, float *d_v, int w, int h, int setup);

#endif

