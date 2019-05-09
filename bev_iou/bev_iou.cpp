
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace tensorflow;

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

REGISTER_OP("OrientedNMS")
  .Attr("nms_threshold: float")
  .Input("boxes: float32")
  .Output("keep: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle boxes_dims; // (N, 5)
      c->WithRank(c->input(0), 2, &boxes_dims);
    
      ::tensorflow::shape_inference::ShapeHandle keep_dims = c->MakeShape({c->Dim(boxes_dims, 0)});
      c->set_output(0, keep_dims);
      return Status::OK();
  });

void oriented_nms_gpu(const float* boxes, unsigned long long *mask,
                const int boxes_num, const float nms_overlap_thresh);

class OrientedNMSOp: public OpKernel{
  public:
    explicit OrientedNMSOp(OpKernelConstruction* context):OpKernel(context) { 
        OP_REQUIRES_OK(context,
                    context->GetAttr("nms_threshold", &nms_threshold));
        // Check that nms_threshold is positive
        OP_REQUIRES(context, nms_threshold >= 0,
                    errors::InvalidArgument("Need nms_threshold >= 0, got ",
                                            nms_threshold));
    }
    
    void Compute(OpKernelContext * context) override {

      const Tensor& boxes = context->input(0);  // (N, 5)
      
      OP_REQUIRES(context, boxes.dims()==2 && boxes.dim_size(0) > 0 && boxes.dim_size(1) == 5, errors::InvalidArgument("OrientendNMS expects (N, 5) boxes shape"));
      const int boxes_num = boxes.dim_size(0);

      auto boxes_flat = boxes.flat<float>();
      const float * boxes_data = &(boxes_flat(0));

      Tensor* keep = NULL;
      // // allocate the memory on host, i.e., CPU
      // AllocatorAttributes attr;
      // attr.set_on_host(true);

      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{boxes_num}, &keep));

      auto keep_flat = keep->flat<int32>();
      int* keep_data = &(keep_flat(0));

      const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

      unsigned long long *mask_data = NULL;
      CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
      oriented_nms_gpu(boxes_data, mask_data, boxes_num, nms_threshold);

      std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

      CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));

      cudaFree(mask_data);

      unsigned long long remv_cpu[col_blocks];
      memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

      int num_to_keep = 0;

      // temp cpu data for output
      std::vector<int> keep_data_cpu(boxes_num, -1);

      for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;
        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data_cpu[num_to_keep] = i;
            num_to_keep++;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
      }

      // copy output from cpu to gpu
      CHECK_ERROR(cudaMemcpy(keep_data, &keep_data_cpu[0], boxes_num * sizeof(int),
                            cudaMemcpyHostToDevice));
      if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );
    }

    private:
    float nms_threshold;
};

REGISTER_KERNEL_BUILDER(Name("OrientedNMS").Device(DEVICE_GPU),OrientedNMSOp);

REGISTER_OP("ComputeBevIOU")
  .Input("proposals: float32")
  .Input("gt_bboxes: float32")
  .Output("overlap_area: float32")
  .Output("bev_iou: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle proposals_dims; // (N, 5)
      c->WithRank(c->input(0), 2, &proposals_dims);
      ::tensorflow::shape_inference::ShapeHandle gt_bboxes_dims; // (M, 5)
      c->WithRank(c->input(1), 2, &gt_bboxes_dims);
    
      ::tensorflow::shape_inference::ShapeHandle overlap_area = c->MakeShape({c->Dim(proposals_dims, 0), c->Dim(gt_bboxes_dims, 0)});
      c->set_output(0, overlap_area);
      ::tensorflow::shape_inference::ShapeHandle bev_iou = c->MakeShape({c->Dim(proposals_dims, 0), c->Dim(gt_bboxes_dims, 0)});
      c->set_output(1, bev_iou);
      return Status::OK();
  });

void compute_bev_iou_gpu(const int num_a, const float* boxes_a, 
                const int num_b, const float* boxes_b, float* ans_overlap, float* ans_iou);

class ComputeBevIOUOp: public OpKernel{
  public:
    explicit ComputeBevIOUOp(OpKernelConstruction* context):OpKernel(context) { }
    
    void Compute(OpKernelContext * context) override {

      const Tensor& proposals = context->input(0);  // (N, 5)
      const Tensor& gt_bboxes = context->input(1);  // (M, 5)
      
      OP_REQUIRES(context, proposals.dims()==2 && proposals.dim_size(0) > 0 && proposals.dim_size(1) == 5, errors::InvalidArgument("ComputeIOU3D expects (N, 7) proposals shape"));
      OP_REQUIRES(context, gt_bboxes.dims()==2 && gt_bboxes.dim_size(0) > 0 && gt_bboxes.dim_size(1) == 5, errors::InvalidArgument("ComputeIOU3D expects (M, 7) gt_bboxes shape"));
      const int num_proposals = proposals.dim_size(0);
      const int num_gt = gt_bboxes.dim_size(0);

      auto proposals_flat = proposals.flat<float>();
      auto gt_bboxes_flat = gt_bboxes.flat<float>();
      const float * proposals_data = &(proposals_flat(0));
      const float * gt_bboxes_data = &(gt_bboxes_flat(0));

      Tensor* overlap_area = nullptr;
      Tensor* bev_iou = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{num_proposals, num_gt}, &overlap_area));
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{num_proposals, num_gt}, &bev_iou));

      auto overlap_area_flat = overlap_area->flat<float>();
      auto bev_iou_flat = bev_iou->flat<float>();
      float* overlap_area_data = &(overlap_area_flat(0));
      float* bev_iou_data = &(bev_iou_flat(0));
      cudaMemset(overlap_area_data, 0, num_proposals * num_gt * sizeof(float));
      cudaMemset(bev_iou_data, 0, num_proposals * num_gt * sizeof(float));
      compute_bev_iou_gpu(num_proposals, proposals_data, num_gt, gt_bboxes_data, overlap_area_data, bev_iou_data);
    }
};

REGISTER_KERNEL_BUILDER(Name("ComputeBevIOU").Device(DEVICE_GPU),ComputeBevIOUOp);

