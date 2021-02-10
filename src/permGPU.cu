// Copyright (C) 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 by Ivo D. Shterev
#include <cuda.h>
#include "ranker.h"

#include <R.h>
#include <Rmath.h>

#include <Rinternals.h>
#include <R_ext/Rdynload.h>

using namespace std;

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 16

#define BLOCKSIZE_Tmax 512

__constant__ float Y_d[1000];
__constant__ float Ydelta_d[1000];

extern "C"{
void R_init_permGPU(DllInfo* info) {
  R_registerRoutines(info, NULL, NULL, NULL, NULL);
  R_useDynamicSymbols(info, TRUE);
}

void Yperm(float *Y, const int N)
{
  for (int n = 1; n < N; n++){
    int idx = n*unif_rand();   // uniform over [0,n]
    float val = Y[n];
    Y[n] = Y[idx];
    Y[idx] = val;
  }
}

void YpermSU(float *Ytime, float *Ydelta, const int N)
{
  for (int n = 1; n < N; n++){
    int idx = n*unif_rand();   // uniform over [0,n]
    float val_time = Ytime[n];
    float val_delta = Ydelta[n];

    Ytime[n] = Ytime[idx];
    Ydelta[n] = Ydelta[idx];

    Ytime[idx] = val_time;
    Ydelta[idx] = val_delta;
  }
}

void YpermSUindex(float *Ytime, float *Ydelta, float *index, const int N)
{
  for (int n = 1; n < N; n++){
    int idx = n*unif_rand();   // uniform over [0,n]

    float val_time = Ytime[n];
    float val_delta = Ydelta[n];
    float val_index = index[n];

    Ytime[n] = Ytime[idx];
    Ydelta[n] = Ydelta[idx];
    index[n] = index[idx];

    Ytime[idx] = val_time;
    Ydelta[idx] = val_delta;
    index[idx] = val_index;
  }
}
}

// common prototypes
__global__ void Tmax(const float *Ttilde, const int K, float *Tmax)
{
  __shared__ float zeta [BLOCKSIZE_Tmax];

  int idx = threadIdx.x;
  zeta[idx] = 0.0;
  for (int k = idx; k < K; k += BLOCKSIZE_Tmax){
    if (zeta[idx] < abs(Ttilde[k]))
      zeta[idx] = abs(Ttilde[k]);
  }

  __syncthreads();

  // compute the global maximum
  if (BLOCKSIZE_Tmax >= 512){
    if (idx < 256){
      if (zeta[idx] < zeta[idx+256])
	zeta[idx] = zeta[idx+256];
    }
    __syncthreads();
  }

  if (BLOCKSIZE_Tmax >= 256){
    if (idx < 128){
      if (zeta[idx] < zeta[idx+128])
	zeta[idx] = zeta[idx+128];
    }
    __syncthreads();
  }

  if (BLOCKSIZE_Tmax >= 128){
    if (idx < 64){
      if (zeta[idx] < zeta[idx+64])
	zeta[idx] = zeta[idx+64];
    }
    __syncthreads();
  }

  if (BLOCKSIZE_Tmax >= 64){
    if (idx < 32){
      if (zeta[idx] < zeta[idx+32])
	zeta[idx] = zeta[idx+32];
    }
  }

  if (idx < 16){
    if (BLOCKSIZE_Tmax >= 32){
      if (zeta[idx] < zeta[idx+16])
	zeta[idx] = zeta[idx+16];
    }
 
    if (BLOCKSIZE_Tmax >= 16){
      if (zeta[idx] < zeta[idx+8])
	zeta[idx] = zeta[idx+8];
    }
  
    if (BLOCKSIZE_Tmax >= 8){
      if (zeta[idx] < zeta[idx+4])
	zeta[idx] = zeta[idx+4];
    }
 
    if (BLOCKSIZE_Tmax >= 4){
      if (zeta[idx] < zeta[idx+2])
	zeta[idx] = zeta[idx+2];
    }
   
    if (BLOCKSIZE_Tmax >= 2){
      if (zeta[idx] < zeta[idx+1])
	zeta[idx] = zeta[idx+1];
    }
  }

  // store result to global variable
  if (idx == 0)
    Tmax[0] = zeta[0];
}

__global__ void Pvalue(const float *T, const float *Ttilde, float *p, float *P, const float *Tmax, const int K){
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
 
  if(idx < K){
    if(abs(Ttilde[idx]) >= abs(T[idx]))
      p[idx]++;
    if(Tmax[0] >= abs(T[idx]))
      P[idx]++;
  }
}

//****************t-test**************************************
extern "C"{
 void Index0(const float *Y, int *N0, const int N)
{
  int count = 0;
  for (int n = 0; n < N; n++){
    if (Y[n] == 0.0){
      N0[count] = n;
      count++;
    }
  }
}
}

__global__ void SumSumSq(const float *X, float *Sum, float *SumSq, const int N, const int K, const size_t pitch_x){
  int k = blockIdx.x*blockDim.x + threadIdx.x;

  if (k < K){
    float sum = 0.0;
    float sumsq = 0.0;

    for (int n = 0; n < N; n++){
      float val = *((float*)((char*)X + k * pitch_x) + n);
      sum += val;
      sumsq += val * val;
    }

    Sum[k] = sum;
    SumSq[k] = sumsq;
  }
}

__global__ void Tstat(const float *X, float *T, const float *Sum, const float *SumSq, const int *N0, const int n0, const int n1, const size_t pitch_x, const size_t pitch_n0, const int N, const int K)
{
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(k < K){
    float x0 = 0.0;
    float s0 = 0.0;

    float xj;
    for (int n = 0; n < n0; n++){    
      xj = *((float*)((char*)X + k * pitch_x) + (*(int*)((char*)N0+n*pitch_n0)));
      x0 += xj;
      s0 += xj * xj;
    }

    x0 /= n0;
    float x1 = (Sum[k] - n0*x0) / n1;
     
    float s1 = (SumSq[k] - s0 - n1*x1*x1) / (n1-1);
    s0 = (s0 - n0*x0*x0) / (n0-1);

    T[k] = (x0-x1) / sqrt(s0/n0+s1/n1);
  }
}

//**************wilcoxon-test***************************************************
extern "C"{
 void Index1(const float *Y, int *N1, const int N)
{
  int count = 0;
  for (int n = 0; n < N; n++){
    if (Y[n] == 1.0){
      N1[count] = n;
      count++;
    }
  }
}
}

__global__ void Wstat(const float *X, float *T, const int *N1, const int K, const float mean, const float std, const int n1, const size_t pitch_x, const size_t pitch_n1)
{
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(k < K){
    float stat = 0.0;
    for (int n = 0; n < n1; n++)
      stat += *((float*)((char*)X + k * pitch_x) + (*(int*)((char*)N1+n*pitch_n1)));

    T[k] = abs(stat-mean) / std;
  }
}

//******************pearson-test***************************************************
__global__ void MeanX(const float *X, float *meanX, const int N, const int K, const size_t pitch_x)
{
  __shared__ float mean[BLOCKSIZE_Y][BLOCKSIZE_X];

  int idx = threadIdx.x;
  int idy = threadIdx.y;

  int colId = blockIdx.x*BLOCKSIZE_Y + idy;

  if (colId < K){
    mean[idy][idx] = 0.0;

    // compute mean
    for (int n = idx; n < N; n += BLOCKSIZE_X)   
      mean[idy][idx] += *((float*)((char*)X + colId * pitch_x) + n);
    __syncthreads();

    // add partial quantities
    if (BLOCKSIZE_X >= 512){
      if (idx < 256)
	mean[idy][idx] += mean[idy][idx + 256];
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 256){
      if (idx < 128)
	mean[idy][idx] += mean[idy][idx + 128];
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 128){
      if (idx < 64)
	mean[idy][idx] += mean[idy][idx + 64];
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 64){
      if (idx < 32)
	mean[idy][idx] += mean[idy][idx + 32];
    }

    if (idx < 16){
      if (BLOCKSIZE_X >= 32)
	mean[idy][idx] += mean[idy][idx + 16];

      if (BLOCKSIZE_X >= 16)
	mean[idy][idx] += mean[idy][idx +  8];

      if (BLOCKSIZE_X >= 8)
	mean[idy][idx] += mean[idy][idx +  4];

      if (BLOCKSIZE_X >= 4)
	mean[idy][idx] += mean[idy][idx +  2];

      if (BLOCKSIZE_X >= 2)
	mean[idy][idx] += mean[idy][idx +  1];
    }

    if (idx == 0)
      meanX[colId] = mean[idy][0] / N;
  }
}

__global__ void SumQuadX(const float *X, float *meanX, float *sumquadX, const int N, const int K, const size_t pitch_x)
{
  __shared__ float aux[BLOCKSIZE_Y][BLOCKSIZE_X];

  int idx = threadIdx.x;
  int idy = threadIdx.y;

  int colId = blockIdx.x*BLOCKSIZE_Y + idy;

  float xn;
  float xk;

  if (colId < K){
    aux[idy][idx] = 0.0;

    // compute mean
    xk = meanX[colId];
    for (int n = idx; n < N; n += BLOCKSIZE_X){   
      xn = *((float*)((char*)X + colId * pitch_x) + n);
      aux[idy][idx] += (xn-xk) * (xn-xk);
    }

    __syncthreads();

    // add partial quantities
    if (BLOCKSIZE_X >= 512){
      if (idx < 256)
	aux[idy][idx] += aux[idy][idx + 256];
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 256){
      if (idx < 128)
	aux[idy][idx] += aux[idy][idx + 128];
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 128){
      if (idx < 64)
	aux[idy][idx] += aux[idy][idx + 64];
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 64){
      if (idx < 32)
	aux[idy][idx] += aux[idy][idx + 32];
    }

    if (idx < 16){
      if (BLOCKSIZE_X >= 32)
	aux[idy][idx] += aux[idy][idx + 16];

      if (BLOCKSIZE_X >= 16)
	aux[idy][idx] += aux[idy][idx +  8];

      if (BLOCKSIZE_X >= 8)
	aux[idy][idx] += aux[idy][idx +  4];

      if (BLOCKSIZE_X >= 4)
	aux[idy][idx] += aux[idy][idx +  2];

      if (BLOCKSIZE_X >= 2)
	aux[idy][idx] += aux[idy][idx +  1];
    }

    if (idx == 0)
      sumquadX[colId] = aux[idy][0];
  }
}

__global__ void Pstat(const float *X, float *T, const float *meanX, const float *sumquadX, const int N, const int K, const float meanY, const float sumquadY, const size_t pitch_x)
{
  int colId = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(colId < K){
    float stat = 0.0;
    // compute the test statistic 
    for (int n = 0; n < N; n ++)    
      stat += Y_d[n] * (*((float*)((char*)X + colId * pitch_x) + n));

    stat = (stat-N*meanY*meanX[colId])/sqrt(sumquadX[colId]*sumquadY);

    // store result to global variable
    T[colId] = stat*sqrt(N-2.0)/sqrt(1.0-stat*stat);
  }
}

//******************************spearman-test***************************************************
__global__ void SPstat(const float *X, float *T, const int N, const int K, const float mean, const float std, const size_t pitch_x)
{
  int k = blockIdx.x*blockDim.x + threadIdx.x;
  
  if(k < K){
    float stat = 0.0;
    for (int n = 0; n < N; n++)
      stat += Y_d[n] * (*((float*)((char*)X + k * pitch_x) + n));

    T[k] = (stat-mean)*sqrt(N-1.0)/std;
  }
}

//********************survival-test******************************************************
__global__ void Index0(int *N0, const int N)
{
  int count = 0;
  for (int n = 0; n < N; n++){
    if (Ydelta_d[n] == 1.0){
      N0[count] = n;
      count++;
    }
  }
}

__global__ void Yperm(float *Ytime, float *Ydelta, const float *Rand, const int N)
{
  for (int n = 1; n < N; n++){
    int idx = n * Rand[n]; // uniform over [0,n]

    // time
    float val = Ytime[n];
    Ytime[n] = Ytime[idx];
    Ytime[idx] = val;

    // delta
    val = Ydelta[n];
    Ydelta[n] = Ydelta[idx];
    Ydelta[idx] = val;
  }
}

__global__ void Den(float *den, const int *N0, const int N, const int n0)
{ 
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < n0){
    float val = 0.0;
    int ind = N0[idx];
    den[idx] = 0.0;
    for (int j = 0; j < N; j++){
      if(Y_d[j] >= Y_d[ind])
	val++;
    }

    den[idx] = val;
  }
}

__global__ void SUstat(const float *X, float *T, const int *N0, const float *Den, const int N, const int K, const int n0, const size_t pitch_x, const int stand)
{
  __shared__ float stat[BLOCKSIZE_Y][BLOCKSIZE_X];
  __shared__ float se[BLOCKSIZE_Y][BLOCKSIZE_X];

  int idx = threadIdx.x;
  int idy = threadIdx.y;

  int k = blockIdx.x*BLOCKSIZE_Y + idy;

  if(k < K){
    // survival test statistic
    stat[idy][idx] = 0.0;
    se[idy][idx] = 0.0;

    float den = 0.0;
    float *x = (float *)((char*)X+pitch_x*k);

    for (int n = idx; n < n0; n += BLOCKSIZE_X){
      int ind = N0[n];
      if(!isnan(x[ind])){
        float nom = 0.0;
        float nom2 = 0.0;
        for (int j = 0; j < N; j++){
	  float aux = x[j];
	  if (Y_d[j] >= Y_d[ind] && !isnan(aux)){
	    nom += aux;
            nom2 += aux*aux;
          }
        }
      
        den = Den[n];
        stat[idy][idx] += x[ind] - nom/den;
        se[idy][idx] += nom2/den - nom*nom/den/den;
      }
    }

    __syncthreads();

    if (BLOCKSIZE_X >= 512){
      if (idx < 256){
        stat[idy][idx] += stat[idy][idx+256];
        se[idy][idx] += se[idy][idx+256];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 256){
      if (idx < 128){
        stat[idy][idx] += stat[idy][idx+128];
        se[idy][idx] += se[idy][idx+128];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 128){
      if (idx < 64){
        stat[idy][idx] += stat[idy][idx+64];
        se[idy][idx] += se[idy][idx+64];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 64){
      if (idx < 32){
        stat[idy][idx] += stat[idy][idx+32];
        se[idy][idx] += se[idy][idx+32];
      }
    }

    if (idx < 16){
      if (BLOCKSIZE_X >= 32){
        stat[idy][idx] += stat[idy][idx+16];
        se[idy][idx] += se[idy][idx+16];
      }
 
      if (BLOCKSIZE_X >= 16){
        stat[idy][idx] += stat[idy][idx+8];
        se[idy][idx] += se[idy][idx+8];
      }
  
      if (BLOCKSIZE_X >= 8){
        stat[idy][idx] += stat[idy][idx+4];
        se[idy][idx] += se[idy][idx+4];
      }
 
      if (BLOCKSIZE_X >= 4){
        stat[idy][idx] += stat[idy][idx+2];
        se[idy][idx] += se[idy][idx+2];
      }
   
      if (BLOCKSIZE_X >= 2){
       	stat[idy][idx] += stat[idy][idx+1];
        se[idy][idx] += se[idy][idx+1];
      }
    }
  
    if (idx == 0){
      if (stand == 1)
        T[k] = stat[idy][0]*stat[idy][0]/se[idy][0];
      else
        T[k] = stat[idy][0]/sqrt(se[idy][0]);
    }
  }
}

__global__ void SUstat_new(const float *X, float *T, const int *N0, const float *Den, const int N, const int K, const int n0, const size_t pitch_x, const int stand)
{
  __shared__ float stat[BLOCKSIZE_Y][BLOCKSIZE_X];
  __shared__ float se[BLOCKSIZE_Y][BLOCKSIZE_X];

  int idx = threadIdx.x;
  int idy = threadIdx.y;

  int k = blockIdx.x*BLOCKSIZE_Y + idy;

  if(k < K){
    // survival test statistic
    stat[idy][idx] = 0.0;
    se[idy][idx] = 0.0;

    float den = 0.0;
    float *x = (float *)((char*)X+pitch_x*k);

    for (int n = idx; n < n0; n += BLOCKSIZE_X){
      int ind = N0[n];
      if(!isnan(x[ind])){
        float nom = 0.0;
        float nom2 = 0.0;
        for (int j = 0; j < N; j++){
	  float aux = x[j];
	  if (Y_d[j] >= Y_d[ind] && !isnan(aux)){
	    nom += aux;
            nom2 += aux*aux;
          }
        }
      
        den = Den[n];
        stat[idy][idx] += x[ind] - nom/den;
        se[idy][idx] += nom2/den - nom*nom/den/den;
      }
    }

    __syncthreads();

    if (BLOCKSIZE_X >= 512){
      if (idx < 256){
        stat[idy][idx] += stat[idy][idx+256];
        se[idy][idx] += se[idy][idx+256];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 256){
      if (idx < 128){
        stat[idy][idx] += stat[idy][idx+128];
        se[idy][idx] += se[idy][idx+128];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 128){
      if (idx < 64){
        stat[idy][idx] += stat[idy][idx+64];
        se[idy][idx] += se[idy][idx+64];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 64){
      if (idx < 32){
        stat[idy][idx] += stat[idy][idx+32];
        se[idy][idx] += se[idy][idx+32];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 32){
      if (idx < 16){
        stat[idy][idx] += stat[idy][idx+16];
        se[idy][idx] += se[idy][idx+16];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 16){
      if (idx < 8){
        stat[idy][idx] += stat[idy][idx+8];
        se[idy][idx] += se[idy][idx+8];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 8){
      if (idx < 4){
        stat[idy][idx] += stat[idy][idx+4];
        se[idy][idx] += se[idy][idx+4];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 4){
      if (idx < 2){
        stat[idy][idx] += stat[idy][idx+2];
        se[idy][idx] += se[idy][idx+2];
      }
      __syncthreads();
    }

    if (BLOCKSIZE_X >= 2){
      if (idx < 1){
        stat[idy][idx] += stat[idy][idx+1];
        se[idy][idx] += se[idy][idx+1];
      }
      __syncthreads();
    }
  
    if (idx == 0){
      if (stand == 1)
        T[k] = stat[idy][0]*stat[idy][0]/se[idy][0];
      else
        T[k] = stat[idy][0]/sqrt(se[idy][0]);
    }
  }
}

extern "C"{
void scoregpu(float *X_h, float *Y_h, float *Ydelta_h, const int *n, const int *k, const char *t[], const int *b, const char *ind, const int *stand, const int *pval, float *T)
{
  int B = *b;
  int N = *n;
  int K = *k;
  string test = *t;
  int index = *ind;

  // common data
  float *T_h;      
  T_h = (float*)malloc(K*(B+1)*sizeof(float));
  for (int k = 0; k < K*(B+1); k++)
    T_h[k] = 0.0;

  float *Taux_h;      
  Taux_h = (float*)malloc(K*sizeof(float));
  for (int k = 0; k < K; k++)
    Taux_h[k] = 0.0;

  float *index_all;  
  if (index == 1){    
    index_all = (float*)malloc(N*(B+1)*sizeof(float));
    for (int n = 0; n < N*(B+1); n++)
      index_all[n] = 0.0;
  }

  float *index_n;
  if (index == 1){    
    index_n = (float*)malloc(N*sizeof(float));
    for (int n = 0; n < N; n++){
      index_n[n] = n + 1.0;
      index_all[n] = n + 1.0;
    }
  }

  // allocate memory on device
  size_t pitch_x;
  size_t width = K;
  size_t height = N;

  float *X_d;
  cudaError_t err = cudaMallocPitch((void **) &X_d, &pitch_x, height*sizeof(float), width);
  if (err != cudaSuccess)
    error("Failed at malloc:X_d");

  err = cudaMemcpy2D(X_d, pitch_x, X_h, height*sizeof(float), height*sizeof(float), width, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:X_d");

  err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(float), 0, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:Y_d");

  float *T_d;                                                                                                                                                
  err = cudaMalloc((void **) &T_d, K*(B+1)*sizeof(float));
  if (err != cudaSuccess)
    error("Failed at malloc:T_d");

  err = cudaMemcpy(T_d, T_h, sizeof(float)*K, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:T_d");

 // survival test
  if (test == "cox" || test == "npcox"){
    if (test =="npcox"){
      // compute ranks of X_h;
      string method = "average";  // Can also be "min" or "max" or "default"
      for (int k = 0; k < K; k++){
        // determine non-missing values
        int nm = 0;
        for (int n = 0; n < N; n++){
          if (!isnan(X_h[k*N+n]))
            nm++;
        } 

        if (nm != 0){
          vector<double> a(nm);
          vector<int> mindex(N);
          int c = 0;
          for (int n = 0; n < N; n++){
            double aux = X_h[k*N+n];
            if (!isnan(aux)){
              a[c] = aux;
              c++;
              mindex[n] = 1; // not miss
            }
            else
              mindex[n] = 0; // miss
          }

          vector<double> ranks;
          rank(a, ranks, method);

          c = 0;
          for (int n = 0; n < N; n++){
            if (mindex[n] == 1.0){  
              X_h[k*N+n] = ranks[c];
              c++;
            }
            else 
              X_h[k*N+n] = NA_REAL;
          }
        }
      }
    }

    err = cudaMemcpy2D(X_d, pitch_x, X_h, height*sizeof(float), height*sizeof(float), width, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at memcpy:X_d");

    // compute n0
    int n0 = 0;
    for (int n = 0; n < N; n++){
      if (Ydelta_h[n] == 1.0)
	n0++;
    }

    err = cudaMemcpyToSymbol(Ydelta_d, Ydelta_h, N*sizeof(float), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at copy to symbol:Ydelta_d");

    int *N0_d;
    err = cudaMalloc((void **) &N0_d, n0*sizeof(int));
    if (err != cudaSuccess)
      error("Failed at malloc:N0_d");

    float *den_d;
    err = cudaMalloc((void **) &den_d, n0*sizeof(float));
    if (err != cudaSuccess)
      error("Failed at malloc:den_d");
  
      // configuration
    int blockSize_n0 = 512;
    int nBlocks_n0 = n0/blockSize_n0 + (n0%blockSize_n0 == 0?0:1);

    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 grid(K/BLOCKSIZE_Y + (K%BLOCKSIZE_Y == 0?0:1), 1, 1);

    // compute N0
    Index0<<<1, 1>>>(N0_d, N);

    // compute T
    Den<<<nBlocks_n0, blockSize_n0>>>(den_d, N0_d, N, n0);
    SUstat_new<<<grid, threads>>>(X_d, T_d, N0_d, den_d, N, K, n0, pitch_x, *stand);

    err = cudaMemcpy(Taux_h, T_d, sizeof(float)*K, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
      error("Failed at memcpy:Taux_h");
    
    for(int k = 0; k < K; k++)
      T_h[k] = Taux_h[k];

    // random number seed
    GetRNGstate();

    for (int b = 0; b < B; b++){
      // permute Y
      if (index == 1){
        YpermSUindex(Y_h, Ydelta_h, index_n, N);
        for (int n = 0; n < N; n++)
          index_all[(b+1)*N+n] = index_n[n];
      }
      else
        YpermSU(Y_h, Ydelta_h, N);

      err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at copy to symbol:Ytime_d");
     
      err = cudaMemcpyToSymbol(Ydelta_d, Ydelta_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at copy to symbol:Ydelta_d");

      // compute N0
      Index0<<<1, 1>>>(N0_d, N);

      // compute test statistic
      Den<<<nBlocks_n0, blockSize_n0>>>(den_d, N0_d, N, n0);
      SUstat_new<<<grid, threads>>>(X_d, T_d, N0_d, den_d, N, K, n0, pitch_x, *stand);
      err = cudaMemcpy(Taux_h, T_d, sizeof(float)*K, cudaMemcpyDeviceToHost);
      if (err != cudaSuccess)
        error("Failed at memcpy:Taux_h");
      
      for(int k = 0; k < K; k++)
        T_h[(b+1)*K+k] = Taux_h[k];
    }

    // random seed
    PutRNGstate();

    // cleanup
    err = cudaFree(N0_d);
    if (err != cudaSuccess)
      error("Failed at free:N0_d");

    err = cudaFree(den_d);
    if (err != cudaSuccess)
      error("Failed at free:den_d");
  }

  // test not known
  else
    error("Test not known");

  // write to output
  if (*pval == 1){
    for (int k = 0; k < K*(B+1); k++)
      T[k] = pchisq(fabs(T_h[k]), 1.0, 0, 0);
  }
  else{
    for (int k = 0; k < K*(B+1); k++)
      T[k] = T_h[k];
  }

  if (index == 1){
    for (int n = 0; n < N*(B+1); n++)
      T[K*(B+1)+n] = index_all[n];
  }

  // cleanup
  free(T_h);
  free(Taux_h);

  if (index == 1){
    free(index_all);
    free(index_n);
  }

  err = cudaFree(X_d);
  if (err != cudaSuccess)
    error("Failed at free:X_d");
     
  err = cudaFree(T_d);
  if (err != cudaSuccess)
    error("Failed at free:T_d");
}
}

extern "C"{
void permgpu(float *X_h, float *Y_h, float *Ydelta_h, const int *n, const int *k, const int *b, const char *t[], float *pPT)
{
  int N = *n;
  int K = *k;
  string test = *t;
  int B = *b;

  // common data
  float *T_h;      T_h      = (float *)malloc(  K*sizeof(float));
  float *Ttilde_h; Ttilde_h = (float *)malloc(  K*sizeof(float));
  float *Tmax_h;   Tmax_h   = (float *)malloc(    sizeof(float)); Tmax_h[0] = 0.0;
  float *p_h;      p_h      = (float *)malloc(  K*sizeof(float));
  float *P_h;      P_h      = (float *)malloc(  K*sizeof(float));

  for (int k = 0; k < K; k++){
    p_h[k]      = 0.0;
    P_h[k]      = 0.0;
    T_h[k]      = 0.0;
    Ttilde_h[k] = 0.0;
  }

  // allocate memory on device
  size_t pitch_x;
  size_t width = K;
  size_t height = N;

  float *X_d;
  cudaError_t err = cudaMallocPitch((void **) &X_d, &pitch_x, height*sizeof(float), width);
  if (err != cudaSuccess)
    error("Failed at malloc:X_d");
  
  err = cudaMemcpy2D(X_d, pitch_x, X_h, height*sizeof(float), height*sizeof(float), width, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:X_d");

  err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(float), 0, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:Y_d");

  float *p_d;                                                                                                                                                
  err = cudaMalloc((void **) &p_d, K*sizeof(float));
  if (err != cudaSuccess)
    error("Failed at malloc:p_d");
  
  err = cudaMemcpy(p_d, p_h, sizeof(float)*K, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:p_d");

  float *P_d;                                                                                                                                                
  err = cudaMalloc((void **) &P_d, K*sizeof(float));
  if (err != cudaSuccess)
    error("Failed at malloc:P_d");
  
  err = cudaMemcpy(P_d, P_h, sizeof(float)*K, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:P_d");

  float *T_d;                                                                                                                                                
  err = cudaMalloc((void **) &T_d, K*sizeof(float));
  if (err != cudaSuccess)
    error("Failed at malloc:T_d");
  
  err = cudaMemcpy(T_d, T_h, sizeof(float)*K, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:T_d");
  
  float *Ttilde_d;                                                                                                                                              
  err = cudaMalloc((void **) &Ttilde_d, K*sizeof(float));
  if (err != cudaSuccess)
    error("Failed at malloc:Ttilde_d");
  
  err = cudaMemcpy(Ttilde_d, Ttilde_h, sizeof(float)*K, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:Ttilde_d");

  float *Tmax_d;
  err = cudaMalloc((void **) &Tmax_d, sizeof(float));
  if (err != cudaSuccess)
    error("Failed at malloc:Tmax_d");
  
  err = cudaMemcpy(Tmax_d, Tmax_h, sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    error("Failed at memcpy:Tmax_d");

  // random number seed
  GetRNGstate();

  // t-test
  if (test == "ttest"){
      // allocate memory on device    
      float *Sum_d;
      err = cudaMalloc((void **) &Sum_d, K*sizeof(float));
      if (err != cudaSuccess)
	error("Failed at malloc:Sum_d");

      float *SumSq_d;
      err = cudaMalloc((void **) &SumSq_d, K*sizeof(float));
      if (err != cudaSuccess)
	error("Failed at malloc:SumSq_d");
                                                                                                                          
      // start execution
      //compute n0 and n1 on the host
      int n0 = 0;
      for (int n = 0; n < N; n++){
	if (Y_h[n] == 0)
	  n0++;
      }
      int n1 = N - n0;

      // configuration
      int blockSize = 512;
      int kBlocks = K/blockSize + (K%blockSize == 0?0:1);

      // compute N0 on the host
      int *N0_h;
      N0_h = (int*)malloc(n0*sizeof(int));
      Index0(Y_h, N0_h, N);

      size_t pitch_n0;
      int *N0_d;
      err = cudaMallocPitch((void **) &N0_d, &pitch_n0, sizeof(int), n0);
      if (err != cudaSuccess)
	error("Failed at malloc:N0_d");

      err = cudaMemcpy2D(N0_d, pitch_n0, N0_h, sizeof(int), sizeof(int), n0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at memcpy:N0_d");

      // compute Sum and SumSq
      SumSumSq<<<kBlocks, blockSize>>> (X_d, Sum_d, SumSq_d, N, K, pitch_x);

      // compute T
      Tstat<<<kBlocks, blockSize>>>(X_d, T_d, Sum_d, SumSq_d, N0_d, n0, n1, pitch_x, pitch_n0, N, K);

      for (int b = 0; b < B; b++){
	// permute Y
       	Yperm(Y_h, N);

	// compute index of Y
	Index0(Y_h, N0_h, N);
	err = cudaMemcpy2D(N0_d, pitch_n0, N0_h, sizeof(int), sizeof(int), n0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	  error("Failed at memcpy:N0_d");

	// compute test statistic
       	Tstat<<<kBlocks, blockSize>>>(X_d, Ttilde_d, Sum_d, SumSq_d, N0_d, n0, n1, pitch_x, pitch_n0, N, K);

	// compute maximum of test statistic
       	Tmax<<<1, BLOCKSIZE_Tmax>>> (Ttilde_d, K, Tmax_d);

	// compute p-values
       	Pvalue <<<kBlocks, blockSize>>> (T_d, Ttilde_d, p_d, P_d, Tmax_d, K);
      }

      // cleanup
      free(N0_h);

      err = cudaFree(Sum_d);
      if (err != cudaSuccess)
	error("Failed at free:Sum_d");

      err = cudaFree(SumSq_d);
      if (err != cudaSuccess)
	error("Failed at free:SumSq_d");

      err = cudaFree(N0_d);
      if (err != cudaSuccess)
	error("Failed at free:N0_d");
  }
     
  // wilcoxon test
  else if (test == "wilcoxon"){
    // compute n0 and n1
    float n0 = 0.0;
    for (int n = 0; n < N; n++){
      if (Y_h[n] == 0.0)
	n0++;
    }
    int n1 = N - n0;

    // compute mean and variance
    float mean = n1 * (n0+n1+1.0) / 2.0; // summing over n1
    float std = sqrt(n0*n1*(n0+n1+1.0)/12.0);

    // compute N1 on the host
    int *N1_h;
    N1_h = (int*)malloc(n1*sizeof(int));
    Index1(Y_h, N1_h, N);

    size_t pitch_n1;
    int *N1_d;
    err = cudaMallocPitch((void **) &N1_d, &pitch_n1, sizeof(int), n1);
    if (err != cudaSuccess)
      error("Failed at malloc:N1_d");
    
    err = cudaMemcpy2D(N1_d, pitch_n1, N1_h, sizeof(int), sizeof(int), n1, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at memcpy:N1_d");

    // compute ranks of X_h;
    string method = "average";  // Can also be "min" or "max" or "default"
    for (int k = 0; k < K; k++){
      vector<double> a(N);
      for (int n = 0; n < N; n++)
	a[n] = X_h[k*N+n];

      vector<double> ranks;
      rank(a, ranks, method);

      for (int n = 0; n < N; n++)
	X_h[k*N+n] = ranks[n];
    }

    err = cudaMemcpy2D(X_d, pitch_x, X_h, height*sizeof(float), height*sizeof(float), width, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at memcpy:X_d");

    // configuration
    int blockSize = 512;
    int nBlocks = K/blockSize + (K%blockSize == 0?0:1);
                                                                                                                          
    // start execution
    // compute T
    Wstat <<<nBlocks, blockSize>>> (X_d, T_d, N1_d, K, mean, std, n1, pitch_x, pitch_n1);

    for (int b = 0; b < B; b++){
      // permute Y
      Yperm(Y_h, N);

      // compute index of Y
      Index1(Y_h, N1_h, N);
      err = cudaMemcpy2D(N1_d, pitch_n1, N1_h, sizeof(int), sizeof(int), n1, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at memcpy:N1_d");

      // compute test statistic
      Wstat <<<nBlocks, blockSize>>> (X_d, Ttilde_d, N1_d, K, mean, std, n1, pitch_x, pitch_n1);

      // compute maximum of test statistic
      Tmax<<<1, BLOCKSIZE_Tmax>>> (Ttilde_d, K, Tmax_d);

      // compute p-values
      Pvalue <<<nBlocks, blockSize>>> (T_d, Ttilde_d, p_d, P_d, Tmax_d, K);
    }

    // cleanup
    free(N1_h);

    err = cudaFree(N1_d);
    if (err != cudaSuccess)
      error("Failed at free:N1_d");
  }

  // pearson test
  else if (test == "pearson"){
      // compute mean of Y
      float meanY = 0.0;
      for (int n = 0; n < N; n++)
	meanY += Y_h[n];
      meanY /= N;

      // compute sumquadY
      float sumquadY = 0.0;
      for (int n = 0; n < N; n++)
	sumquadY += (Y_h[n]-meanY) * (Y_h[n]-meanY);

      float *meanX_d;
      err = cudaMalloc((void **) &meanX_d, K*sizeof(float));
      if (err != cudaSuccess)
	error("Failed at malloc:meanX_d");

      float *sumquadX_d;
      err = cudaMalloc((void **) &sumquadX_d, K*sizeof(float));
      if (err != cudaSuccess)
	error("Failed at malloc:sumquadX_d");
  
      // configuration
      int blockSize = 512;
      int nBlocks = K/blockSize + (K%blockSize == 0?0:1);

      // for Tstat
      dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
      dim3 grid(K/BLOCKSIZE_Y + (K%BLOCKSIZE_Y == 0?0:1), 1, 1);
                                                                                                                          
      // start execution
      //compute meanX
      MeanX<<<grid, threads>>>(X_d, meanX_d, N, K, pitch_x);

      // compute sumquadX
      SumQuadX<<<grid, threads>>>(X_d, meanX_d, sumquadX_d, N, K, pitch_x);

      // compute T
      Pstat<<<nBlocks, blockSize>>>(X_d, T_d, meanX_d, sumquadX_d, N, K, meanY, sumquadY, pitch_x);

      for (int b = 0; b < B; b++){
	// permute Y
       	Yperm(Y_h, N);
	err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	  error("Failed at copy to symbol:Y_d");

	// compute test statistic
	Pstat<<<nBlocks, blockSize>>>(X_d, Ttilde_d, meanX_d, sumquadX_d, N, K, meanY, sumquadY, pitch_x);

	// compute maximum of test statistic
       	Tmax <<<1, BLOCKSIZE_Tmax>>> (Ttilde_d, K, Tmax_d);

	// compute p-values
       	Pvalue <<<nBlocks, blockSize>>> (T_d, Ttilde_d, p_d, P_d, Tmax_d, K);
      }

      // cleanup
      err = cudaFree(meanX_d);
      if (err != cudaSuccess)
	error("Failed at free:meanX_d");

      err = cudaFree(sumquadX_d);
      if (err != cudaSuccess)
	error("Failed at free:sumquadX_d");
  }

  // spearman test
  else if (test == "spearman"){
    // compute mean and standard deviation
    float mean = N*(N+1.0)*(N+1.0)/4.0;

    float std = N*(N*N-1.0) / 12.0;

    // compute ranks of Y
    string method = "average";  // Can also be "min" or "max" or "default"
    vector<double> a(N);
    for (int n = 0; n < N; n++)
      a[n] = Y_h[n];
    vector<double> ranks;
    rank(a, ranks, method);
    for (int n = 0; n < N; n++)
      Y_h[n] = ranks[n];
    err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at copy to symbol:Y_d");

    // compute ranks of X_h;
    for (int k = 0; k < K; k++){
      vector<double> a(N);
      for (int n = 0; n < N; n++)
	a[n] = X_h[k*N+n];

      vector<double> ranks;
      rank(a, ranks, method);

      for (int n = 0; n < N; n++)
	X_h[k*N+n] = ranks[n];
    }
    err = cudaMemcpy2D(X_d, pitch_x, X_h, height*sizeof(float), height*sizeof(float), width, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at memcpy:X_d");
  
    // configuration
    int blockSize = 512;
    int nBlocks = K/blockSize + (K%blockSize == 0?0:1);
                                                                                                                          
    // start execution
    // compute T
    SPstat <<<nBlocks, blockSize>>> (X_d, T_d, N, K, mean, std, pitch_x);

    for (int b = 0; b < B; b++){
      // permute Y
      Yperm(Y_h, N);
      err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at copy to symbol:Y_d");

      // compute test statistic
      SPstat <<<nBlocks, blockSize>>> (X_d, Ttilde_d, N, K, mean, std, pitch_x);

      // compute maximum of test statistic
      Tmax <<<1, BLOCKSIZE_Tmax>>> (Ttilde_d, K, Tmax_d);

      // compute p-values
      Pvalue <<<nBlocks, blockSize>>> (T_d, Ttilde_d, p_d, P_d, Tmax_d, K);
    }
  }

  // survival test
  else if (test == "cox" || test == "npcox"){
    if (test =="npcox"){
      // compute ranks of X_h;
      string method = "average";  // Can also be "min" or "max" or "default"
      for (int k = 0; k < K; k++){
        // determine non-missing values
        int nm = 0;
        for (int n = 0; n < N; n++){
          if (!isnan(X_h[k*N+n]))
            nm++;
        } 

        if (nm != 0){
          vector<double> a(nm);
          vector<int> mindex(N);
          int c = 0;
          for (int n = 0; n < N; n++){
            float aux = X_h[k*N+n];
            if (!isnan(aux)){
              a[c] = aux;
              c++;
              mindex[n] = 1; // not miss
            }
            else
              mindex[n] = 0; // miss
          }

          vector<double> ranks;
          rank(a, ranks, method);

          c = 0;
          for (int n = 0; n < N; n++){
            if (mindex[n] == 1.0){  
              X_h[k*N+n] = ranks[c];
              c++;
            }
            else 
              X_h[k*N+n] = NA_REAL;
          }
        }
      }
    }

    err = cudaMemcpy2D(X_d, pitch_x, X_h, height*sizeof(float), height*sizeof(float), width, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at memcpy:X_d");

    // compute n0
    int n0 = 0;
    for (int n = 0; n < N; n++){
      if (Ydelta_h[n] == 1)
	n0++;
    }

    err = cudaMemcpyToSymbol(Ydelta_d, Ydelta_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      error("Failed at copy to symbol:Ydelta_d");

    int *N0_d;
    err = cudaMalloc((void **) &N0_d, n0*sizeof(int));
    if (err != cudaSuccess)
      error("Failed at malloc:N0_d");

    float *den_d;
    err = cudaMalloc((void **) &den_d, n0*sizeof(float));
    if (err != cudaSuccess)
      error("Failed at malloc:den_d");
  
      // configuration
    int blockSize_K = 512;
    int nBlocks_K = K/blockSize_K + (K%blockSize_K == 0?0:1);

    int blockSize_n0 = 512;
    int nBlocks_n0 = n0/blockSize_n0 + (n0%blockSize_n0 == 0?0:1);

    dim3 threads(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 grid(K/BLOCKSIZE_Y + (K%BLOCKSIZE_Y == 0?0:1), 1, 1);

    // compute N0
    Index0<<<1, 1>>>(N0_d, N);

    // compute T
    Den<<<nBlocks_n0, blockSize_n0>>>(den_d, N0_d, N, n0);
    SUstat_new<<<grid, threads>>>(X_d, T_d, N0_d, den_d, N, K, n0, pitch_x, 1);

    for (int b = 0; b < B; b++){
      // permute Y
      YpermSU(Y_h, Ydelta_h, N);
      err = cudaMemcpyToSymbol(Y_d, Y_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at copy to symbol:Ytime_d");
      
      err = cudaMemcpyToSymbol(Ydelta_d, Ydelta_h, N*sizeof(int), 0, cudaMemcpyHostToDevice);
      if (err != cudaSuccess)
	error("Failed at copy to symbol:Ydelta_d");

      // compute N0
      Index0<<<1, 1>>>(N0_d, N);

      // compute test statistic
      Den<<<nBlocks_n0, blockSize_n0>>>(den_d, N0_d, N, n0);
      SUstat_new<<<grid, threads>>>(X_d, Ttilde_d, N0_d, den_d, N, K, n0, pitch_x, 1);

      // compute maximum of test statistic
      Tmax <<<1, BLOCKSIZE_Tmax>>> (Ttilde_d, K, Tmax_d);

      // compute p-values
      Pvalue <<<nBlocks_K, blockSize_K>>> (T_d, Ttilde_d, p_d, P_d, Tmax_d, K);
    }

    // cleanup
    err = cudaFree(N0_d);
    if (err != cudaSuccess)
      error("Failed at free:N0_d");

    err = cudaFree(den_d);
    if (err != cudaSuccess)
      error("Failed at free:den_d");
  }

  // test not known
  else
    error("Test not known");

  // random seed
  PutRNGstate();

  // retrieve result from device and store                                                                                                                                          
  err = cudaMemcpy(p_h, p_d, sizeof(float)*K, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    error("Failed at memcpy:p_h");

  err = cudaMemcpy(P_h, P_d, sizeof(float)*K, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    error("Failed at memcpy:P_h");

  err = cudaMemcpy(T_h, T_d, sizeof(float)*K, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
    error("Failed at memcpy:T_h");

  // write to output
  int count = 0;
  for (int k = 0; k < K; k++){
    pPT[count] = p_h[k]/B;
    count++;
  }
  for (int k = 0; k < K; k++){
    pPT[count] = P_h[k]/B;
    count++;
  }
  for (int k = 0; k < K; k++){
    pPT[count] = T_h[k];
    count++;
  }

  // cleanup
  free(p_h     );
  free(P_h     );
  free(T_h     );
  free(Ttilde_h);
  free(Tmax_h);

  err = cudaFree(X_d);
  if (err != cudaSuccess)
    error("Failed at free:X_d");
     
  err = cudaFree(p_d);
  if (err != cudaSuccess)
    error("Failed at free:p_d");
      
  err = cudaFree(P_d);
  if (err != cudaSuccess)
    error("Failed at free:P_d");
     
  err = cudaFree(T_d);
  if (err != cudaSuccess)
    error("Failed at free:T_d");
     
  err = cudaFree(Ttilde_d);
  if (err != cudaSuccess)
    error("Failed at free:Ttilde_d");
    
  err = cudaFree(Tmax_d);
  if (err != cudaSuccess)
    error("Failed at free:Tmax_d");
}
}


