#include "forward.h"

__global__ 
void kernel_parallel(float* proj, cudaTextureObject_t texObjImg, float* transformation, int nw)
{
	int nu = gridDim.x;
	int nv = gridDim.y;
	int iu = blockIdx.x;
	int iv = blockIdx.y;
	int ia = threadIdx.x;

	float u = -1. + (float) (1 + iu * 2) / nu;
	float v = -1. + (float) (1 + iv * 2) / nv;
	float w = -1. + (float) 1/nw;
	float dw = (float) 2/nw;

	float t00 = transformation[0 + 0*4 + ia*4*4];
	float t01 = transformation[1 + 0*4 + ia*4*4];
	float t02 = transformation[2 + 0*4 + ia*4*4];
	float t03 = transformation[3 + 0*4 + ia*4*4];

	float t10 = transformation[0 + 1*4 + ia*4*4];
	float t11 = transformation[1 + 1*4 + ia*4*4];
	float t12 = transformation[2 + 1*4 + ia*4*4];
	float t13 = transformation[3 + 1*4 + ia*4*4];

	float t20 = transformation[0 + 2*4 + ia*4*4];
	float t21 = transformation[1 + 2*4 + ia*4*4];
	float t22 = transformation[2 + 2*4 + ia*4*4];
	float t23 = transformation[3 + 2*4 + ia*4*4];

	float xx = t00 * u + t01 * v + t03;
	float yy = t10 * u + t11 * v + t13;
	float zz = t20 * u + t21 * v + t23;

	float sum = 0;
	float x, y, z;

	for (int i = 0; i < nw; i++)
	{
		x = xx + t02 * w;
		y = yy + t12 * w;
		z = zz + t22 * w;
		sum += tex3D<float>(texObjImg, x+.5, y+.5, z+.5);
		w += dw;
	}
	int idx = iu + iv*nu + ia*nu*nv;
	proj[idx] = sum;
}

void funcParallelBeam(float *detector_array, float *transformation, float *object_array, int nx, int ny, int nz, int nu, int nv, int nw, int na)
{
	// object array >> texture memory
	const cudaExtent objSize = make_cudaExtent(nx, ny, nz);
	cudaArray *d_object_array = 0;
	cudaTextureObject_t tex_object_array = 0;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMalloc3DArray(&d_object_array, &channelDesc, objSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)object_array, objSize.width * sizeof(float), objSize.width, objSize.height);
	copyParams.dstArray = d_object_array;
	copyParams.extent = objSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_object_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[1] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[2] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&tex_object_array, &texRes, &texDescr, NULL);

	//
	float *d_transformation;
	cudaMalloc(&d_transformation, na * 4 * 4 * sizeof(float));
	cudaMemcpy(d_transformation, transformation, na * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
	//
	float *d_detector_array;
	cudaMalloc(&d_detector_array, na * nu * nv * sizeof(float));
	//
	kernel_parallel <<< dim3(nu,nv,1), dim3(na,1,1) >>> (d_detector_array, tex_object_array, d_transformation, nw);
	cudaMemcpy(detector_array, d_detector_array, na*nu*nv*sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(d_detector_array);
	cudaFree(d_transformation);
	cudaFreeArray(d_object_array);
	cudaDestroyTextureObject(tex_object_array);
}


__global__ 
void kernel_cone(float* proj, cudaTextureObject_t texObjImg, float* transformation, int nw, float su, float sv, float s2d, float near, float far)
{
	int nu = gridDim.x;
	int nv = gridDim.y;
	int iu = blockIdx.x;
	int iv = blockIdx.y;
	int ia = threadIdx.x;

	float rx = -su/2 + su/2/nu + iu*su/nu;
	float ry = -sv/2 + sv/2/nv + iv*sv/nv;
	float rz = -s2d;
	float magnitude = powf((powf(rx,2.) + powf(ry,2.) + powf(rz,2.)), .5);
	rx /= magnitude;
	ry /= magnitude;
	rz /= magnitude;

	float dt = (far - near) / nw;
	float t = near;

	float t00 = transformation[0 + 0*4 + ia*4*4];
	float t01 = transformation[1 + 0*4 + ia*4*4];
	float t02 = transformation[2 + 0*4 + ia*4*4];
	float t03 = transformation[3 + 0*4 + ia*4*4];

	float t10 = transformation[0 + 1*4 + ia*4*4];
	float t11 = transformation[1 + 1*4 + ia*4*4];
	float t12 = transformation[2 + 1*4 + ia*4*4];
	float t13 = transformation[3 + 1*4 + ia*4*4];

	float t20 = transformation[0 + 2*4 + ia*4*4];
	float t21 = transformation[1 + 2*4 + ia*4*4];
	float t22 = transformation[2 + 2*4 + ia*4*4];
	float t23 = transformation[3 + 2*4 + ia*4*4];

	float sum = 0;
	float x, y, z;

	for (int i = 0; i < nw; i++)
	{
		x = t00*rx*t + t01*ry*t + t02*rz*t + t03;
		y = t10*rx*t + t11*ry*t + t12*rz*t + t13;
		z = t20*rx*t + t21*ry*t + t22*rz*t + t23;
		sum += tex3D<float>(texObjImg, x+.5, y+.5, z+.5);
		t += dt;
	}
	int idx = iu + iv*nu + ia*nu*nv;
	proj[idx] = sum;
}

void funcConeBeam(float *detector_array, float *transformation, float *object_array, int nx, int ny, int nz, int nu, int nv, int nw, int na, float su, float sv, float s2d, float near, float far)
{
	// object array >> texture memory
	const cudaExtent objSize = make_cudaExtent(nx, ny, nz);
	cudaArray *d_object_array = 0;
	cudaTextureObject_t tex_object_array = 0;

	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMalloc3DArray(&d_object_array, &channelDesc, objSize);

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*)object_array, objSize.width * sizeof(float), objSize.width, objSize.height);
	copyParams.dstArray = d_object_array;
	copyParams.extent = objSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	cudaResourceDesc            texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_object_array;

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false; // access with normalized texture coordinates
	texDescr.filterMode = cudaFilterModeLinear; // linear interpolation
	texDescr.addressMode[0] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[1] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.addressMode[2] = cudaAddressModeBorder; // wrap texture coordinates
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&tex_object_array, &texRes, &texDescr, NULL);

	//
	float *d_transformation;
	cudaMalloc(&d_transformation, na * 4 * 4 * sizeof(float));
	cudaMemcpy(d_transformation, transformation, na * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
	//
	float *d_detector_array;
	cudaMalloc(&d_detector_array, na * nu * nv * sizeof(float));
	//
	kernel_cone <<< dim3(nu,nv,1), dim3(na,1,1) >>> (d_detector_array, tex_object_array, d_transformation, nw, su, sv, s2d, near, far);
	cudaMemcpy(detector_array, d_detector_array, na*nu*nv*sizeof(float), cudaMemcpyDeviceToHost);


	cudaFree(d_detector_array);
	cudaFree(d_transformation);
	cudaFreeArray(d_object_array);
	cudaDestroyTextureObject(tex_object_array);
}