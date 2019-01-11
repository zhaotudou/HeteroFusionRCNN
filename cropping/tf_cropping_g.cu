#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

__device__ bool pts_in_box(float px, float py, float pz, 
                           float cx, float cy, float cz,
                           float dx, float dy, float dz) {
  float min_x = cx - 0.5 * dx;
  float max_x = cx + 0.5 * dx;
  float min_y = cy - dy;
  float max_y = cy;
  float min_z = cz - 0.5 * dz;
  float max_z = cz + 0.5 * dz;
  if (px >= min_x && px <= max_x &&
      py >= min_y && py <= max_y &&
      pz >= min_z && pz <= max_z)
    return true;
  return false;
}

__global__ void pccropandsampleKernel(
    const float* pts_data, const float* fts_data, const float* boxes_data, const int* box_ind_data,
    int num_boxes, int batch, int npts, int resize, int channel,
    float* crop_pts_data, float* crop_fts_data, int* crop_ind_data, bool* non_empty_box_data) {
  
  __shared__ unsigned int box_pts_num;
  for (int b=blockIdx.x; b<num_boxes; b+=gridDim.x) {
    if (threadIdx.x == 0)
      box_pts_num = 0;
    __syncthreads();

    float cx = boxes_data[b*6];
    float cy = boxes_data[b*6+1];
    float cz = boxes_data[b*6+2];
    float dx = boxes_data[b*6+3];
    float dy = boxes_data[b*6+4];
    float dz = boxes_data[b*6+5];
    int bch = box_ind_data[b];
    const float *pts = pts_data + bch * npts * 3;
    const float *fts = fts_data + bch * npts * channel;
    float *crop_pts = crop_pts_data + b * resize * 3;
    float *crop_fts = crop_fts_data + b * resize * channel;
    int *crop_ind = crop_ind_data + b * resize;
    for (int p=threadIdx.x; p<npts; p+=blockDim.x) {
      float px = pts[p*3]; 
      float py = pts[p*3+1]; 
      float pz = pts[p*3+2];
      if (pts_in_box(px, py, pz, cx, cy, cz, dx, dy, dz)) {
        int pos = atomicInc(&box_pts_num, UINT_MAX);
        if (pos < resize) {
          crop_pts[pos*3] = px;  
          crop_pts[pos*3 + 1] = py;  
          crop_pts[pos*3 + 2] = pz; 
          for (int c=0; c < channel; c++) {
            crop_fts[pos * channel + c] = fts[p * channel + c];
          }
          crop_ind[pos] = p;
        } else {
          break;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      if (box_pts_num <= 0) {
        non_empty_box_data[b] = false;
        //printf("[EMPTY] PcCropAndSample: box = %d, box_pts_num = %d\n", b, box_pts_num);
      } else if (box_pts_num < resize) {
        //printf("[LESS] PcCropAndSample: box = %d, box_pts_num = %d\n", b, box_pts_num);
        int max = box_pts_num - 1;
        curandState state;
        curand_init((unsigned long long)clock(), 0, 0, &state);
        while (box_pts_num < resize) {
          float r = curand_uniform(&state);
          r *= (max + 0.999999);
          int idx = (int)truncf(r);
          crop_pts[box_pts_num*3] = crop_pts[idx*3];
          crop_pts[box_pts_num*3 + 1] = crop_pts[idx*3 + 1];
          crop_pts[box_pts_num*3 + 2] = crop_pts[idx*3 + 2];
          for(int c=0; c < channel; c++) {
            crop_fts[box_pts_num * channel + c] = crop_fts[idx * channel + c];
          }
          crop_ind[box_pts_num] = crop_ind[idx];
          box_pts_num++;
        }
      } else {
        //printf("[FULL] PcCropAndSample: box = %d, box_pts_num = %d\n", b, box_pts_num);
      }
    }
  }
}

__global__ void pccropandsamplegradftsKernel(
    const int* box_ind_data, const int* crop_ind_data, const float* grad_crop_fts_data,
    int num_boxes, int npts, int resize, int channel, float* grad_fts_data) {
  
  for (int b=blockIdx.x; b<num_boxes; b+=gridDim.x) {
    int bch = box_ind_data[b];
    const int* crop_ind = crop_ind_data + b * resize;
    const float* grad_crop_fts = grad_crop_fts_data + b * resize * channel;
    float* grad_fts = grad_fts_data + bch * npts * channel;
    for (int p=threadIdx.x; p<resize; p+=blockDim.x) {
      int idx = crop_ind[p];
      for(int c=0; c<channel; c++) {
        atomicAdd(&grad_fts[idx*channel + c], grad_crop_fts[p*channel + c]);
      }
    }
  }
}

void pccropandsample_gpu(
    const float* pts_data, const float* fts_data, const float* boxes_data, const int* box_ind_data,
    int num_boxes, int batch, int npts, int resize, int channel,
    float* crop_pts_data, float* crop_fts_data, int* crop_ind_data, bool* non_empty_box_data) {
  pccropandsampleKernel<<<32, 512>>>(pts_data, fts_data, boxes_data, box_ind_data, 
                                   num_boxes, batch, npts, resize, channel,
                                   crop_pts_data, crop_fts_data, crop_ind_data, non_empty_box_data);
}

void pccropandsamplegradfts_gpu(
    const int* box_ind_data, const int* crop_ind_data, const float* grad_crop_fts_data,
    int num_boxes, int npts, int resize, int channel,
    float* grad_fts_data) {
  pccropandsamplegradftsKernel<<<32, 512>>>(box_ind_data, crop_ind_data, grad_crop_fts_data,
                                   num_boxes, npts, resize, channel,
                                   grad_fts_data);
}
