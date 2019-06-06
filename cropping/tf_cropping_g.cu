#include <stdio.h>

__device__ float dot(float x1, float y1, float z1, float x2, float y2, float z2) {
    return (x1 * x2 + y1 * y2 + z1 * z2);
}

__device__ bool is_point_inside(float px, float py, float pz, 
                                float p1x, float p1y, float p1z,
                                float p2x, float p2y, float p2z,
                                float p4x, float p4y, float p4z,
                                float p5x, float p5y, float p5z) {
  float ux = p2x - p1x;
  float uy = p2y - p1y;
  float uz = p2z - p1z;
  
  float vx = p4x - p1x;
  float vy = p4y - p1y;
  float vz = p4z - p1z;
  
  float wx = p5x - p1x;
  float wy = p5y - p1y;
  float wz = p5z - p1z;

  float u_dot_x = dot(ux, uy, uz, px, py, pz);
  float u_dot_p1 = dot(ux, uy, uz, p1x, p1y, p1z);
  float u_dot_p2 = dot(ux, uy, uz, p2x, p2y, p2z);

  float v_dot_x = dot(vx, vy, vz, px, py, pz);
  float v_dot_p1 = dot(vx, vy, vz, p1x, p1y, p1z);
  float v_dot_p4 = dot(vx, vy, vz, p4x, p4y, p4z);
  
  float w_dot_x = dot(wx, wy, wz, px, py, pz);
  float w_dot_p1 = dot(wx, wy, wz, p1x, p1y, p1z);
  float w_dot_p5 = dot(wx, wy, wz, p5x, p5y, p5z);

  if (u_dot_p1 < u_dot_x && u_dot_x < u_dot_p2 &&
      v_dot_p1 < v_dot_x && v_dot_x < v_dot_p4 &&
      w_dot_p1 < w_dot_x && w_dot_x < w_dot_p5)
    return true;
  return false;
}

__global__ void pccropandsampleKernel(
    const float* pts_data, const float* fts_data, const float* intensities_data, const bool* mask_data, const float* boxes_data, const int* box_ind_data,
    int num_boxes, int batch, int npts, int resize, int channel, int intensity_channel,
    float* crop_pts_data, float* crop_fts_data, float* crop_intensities_data, bool* crop_mask_data, int* crop_ind_data, bool* non_empty_box_data) {
  
  __shared__ unsigned int box_pts_num;
  for (int b=blockIdx.x; b<num_boxes; b+=gridDim.x) {
    if (threadIdx.x == 0)
      box_pts_num = 0;
    __syncthreads();

    float p1x = boxes_data[b*24];
    float p1y = boxes_data[b*24 + 8];
    float p1z = boxes_data[b*24 + 16];
    
    float p2x = boxes_data[b*24 + 1];
    float p2y = boxes_data[b*24 + 8  + 1];
    float p2z = boxes_data[b*24 + 16 + 1];
    
    float p4x = boxes_data[b*24 + 3];
    float p4y = boxes_data[b*24 + 8  + 3];
    float p4z = boxes_data[b*24 + 16 + 3];
    
    float p5x = boxes_data[b*24 + 4];
    float p5y = boxes_data[b*24 + 8  + 4];
    float p5z = boxes_data[b*24 + 16 + 4];
    
    int bch = box_ind_data[b];
    const float *pts = pts_data + bch * npts * 3;
    const float *fts = fts_data + bch * npts * channel;
    const float *intensities = intensities_data + bch * npts * intensity_channel;
    const bool *mask = mask_data + bch * npts;
    float *crop_pts = crop_pts_data + b * resize * 3;
    float *crop_fts = crop_fts_data + b * resize * channel;
    float *crop_intensities = crop_intensities_data + b * resize * intensity_channel;
    bool *crop_mask = crop_mask_data + b * resize;
    int *crop_ind = crop_ind_data + b * resize;
    for (int p=threadIdx.x; p<npts; p+=blockDim.x) {
      float px = pts[p*3]; 
      float py = pts[p*3+1]; 
      float pz = pts[p*3+2];
      if (is_point_inside(px, py, pz, p1x, p1y, p1z, p2x, p2y, p2z, p4x, p4y, p4z, p5x, p5y, p5z)) {
        int pos = atomicInc(&box_pts_num, UINT_MAX);
        if (pos < resize) {
          crop_pts[pos*3] = px;  
          crop_pts[pos*3 + 1] = py;  
          crop_pts[pos*3 + 2] = pz; 
          for (int c=0; c < channel; c++) {
            crop_fts[pos * channel + c] = fts[p * channel + c];
          }
          for (int c=0; c < intensity_channel; c++) {
            crop_intensities[pos * intensity_channel + c] = intensities[p * intensity_channel + c];
          }
          crop_mask[pos] = mask[p];
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
        int counter = 0;
        int init_box_pts_num = box_pts_num;
        while (box_pts_num < resize) {
          int idx = counter++ % init_box_pts_num;
          crop_pts[box_pts_num*3] = crop_pts[idx*3];
          crop_pts[box_pts_num*3 + 1] = crop_pts[idx*3 + 1];
          crop_pts[box_pts_num*3 + 2] = crop_pts[idx*3 + 2];
          for(int c=0; c < channel; c++) {
            crop_fts[box_pts_num * channel + c] = crop_fts[idx * channel + c];
          }
          for(int c=0; c < intensity_channel; c++) {
            crop_intensities[box_pts_num * intensity_channel + c] = crop_intensities[idx * intensity_channel + c];
          }
          crop_mask[box_pts_num] = crop_mask[idx];
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
    const float* pts_data, const float* fts_data, const float* intensities_data, const bool* mask_data, const float* boxes_data, const int* box_ind_data,
    int num_boxes, int batch, int npts, int resize, int channel, int intensity_channel,
    float* crop_pts_data, float* crop_fts_data, float* crop_intensities_data, bool* crop_mask_data, int* crop_ind_data, bool* non_empty_box_data) {
  pccropandsampleKernel<<<32, 512>>>(pts_data, fts_data, intensities_data, mask_data, boxes_data, box_ind_data, 
                                   num_boxes, batch, npts, resize, channel, intensity_channel,
                                   crop_pts_data, crop_fts_data, crop_intensities_data, crop_mask_data, crop_ind_data, non_empty_box_data);
}

void pccropandsamplegradfts_gpu(
    const int* box_ind_data, const int* crop_ind_data, const float* grad_crop_fts_data,
    int num_boxes, int npts, int resize, int channel,
    float* grad_fts_data) {
  pccropandsamplegradftsKernel<<<32, 512>>>(box_ind_data, crop_ind_data, grad_crop_fts_data,
                                   num_boxes, npts, resize, channel,
                                   grad_fts_data);
}
