void funcParallelBeam(float *reconstruction_array, float *transformation, float *sinogram_array, int nx, int ny, int nz, int nu, int nv, int na);
void funcConeBeam    (float *reconstruction_array, float *transformation, float *sinogram_array, int nx, int ny, int nz, int nu, int nv, int na, float su, float sv, float du, float dv, float s2d);