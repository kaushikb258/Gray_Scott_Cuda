#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#include <GL/freeglut.h>
#endif
#define W 640
#define H 640

float *d_u = 0;
float *d_v = 0;
int iterationCount = 0;
int setup = 0;

void keyboard(unsigned char key, int x, int y) {
  if (key == 'c') 
   {
    setup = (setup == 0)? 1 : 0;
   }
  if (key == 'z') resetgs(d_u, d_v, W, H, setup);
  if (key == 27) exit(0);
  glutPostRedisplay();
}

void mouse(int button, int state, int x, int y) {
  // do nothing
  return ;
}

void idle(void) {
  ++iterationCount;
  glutPostRedisplay();
}

void printInstructions() {
  printf("Gray-Scott Visualizer:\n"
         "Mouse click inactive\n"
         "Change running case : c\n"
         "Reset case          : z\n"
         "Exit                : Esc\n");
}

#endif

