/* Furthest point sampling
 * Original author: Haoqiang Fan
 * Modified by Charles R. Qi
 * All Rights Reserved. 2017. 
 */
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

REGISTER_OP("PcCropAndSample")
  .Attr("resize: int")
  .Input("pts: float32")
  .Input("fts: float32")
  .Input("boxes: float32")
  .Input("box_ind: int32")
  .Output("crop_pts: float32")
  .Output("crop_fts: float32")
  .Output("crop_ind: int32")
  .Output("non_non_empty_box: bool")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle pts_dims; // B * P * 3
      c->WithRank(c->input(0), 3, &pts_dims);
      ::tensorflow::shape_inference::ShapeHandle fts_dims; // B * P * C
      c->WithRank(c->input(1), 3, &fts_dims);
      ::tensorflow::shape_inference::ShapeHandle boxes_dims; // N * 6
      c->WithRank(c->input(2), 2, &boxes_dims);
      ::tensorflow::shape_inference::ShapeHandle box_ind_dims; // N
      c->WithRank(c->input(3), 1, &box_ind_dims);
      int resize;
      TF_RETURN_IF_ERROR(c->GetAttr("resize", &resize));
    
      ::tensorflow::shape_inference::ShapeHandle crop_pts_dims = c->MakeShape({c->Dim(boxes_dims, 0), resize, c->Dim(pts_dims, 2)});
      ::tensorflow::shape_inference::ShapeHandle crop_fts_dims = c->MakeShape({c->Dim(boxes_dims, 0), resize, c->Dim(fts_dims, 2)});
      ::tensorflow::shape_inference::ShapeHandle crop_ind_dims = c->MakeShape({c->Dim(boxes_dims, 0), resize});
      ::tensorflow::shape_inference::ShapeHandle non_non_empty_box_dims = c->MakeShape({c->Dim(boxes_dims, 0)});
      c->set_output(0, crop_pts_dims);
      c->set_output(1, crop_fts_dims);
      c->set_output(2, crop_ind_dims);
      c->set_output(3, non_non_empty_box_dims);
      return Status::OK();
  });

REGISTER_OP("PcCropAndSampleGradFts")
  .Input("fts: float32")
  .Input("box_ind: int32")
  .Input("crop_ind: int32")
  .Input("grad_crop_fts: float32")
  .Output("grad_fts: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
  });

static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
                                           const Tensor& box_index,
                                           int* num_boxes) {
  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    *num_boxes = 0;
    return Status::OK();
  }
  // The shape of 'boxes' is [num_boxes, 6].
  if (boxes.dims() != 2) {
    return errors::InvalidArgument("boxes must be 2-D",
                                    boxes.shape().DebugString());
  }
  *num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 6) {
    return errors::InvalidArgument("boxes must have 6 columns");
  }
  // The shape of 'box_index' is [num_boxes].
  if (box_index.dims() != 1) {
    return errors::InvalidArgument("box_index must be 1-D",
                                    box_index.shape().DebugString());
  }
  if (box_index.dim_size(0) != *num_boxes) {
    return errors::InvalidArgument("box_index has incompatible shape");
  }
  return Status::OK();
}

void pccropandsample_gpu(
    const float* pts_data, const float* fts_data, const float* boxes_data, const int* box_ind_data,
    int num_boxes, int batch, int npts, int resize, int channel,
    float* crop_pts_data, float* crop_fts_data, int* crop_ind_data, bool* non_empty_box_data);

class PcCropAndSampleGpuOp: public OpKernel{
  public:
    explicit PcCropAndSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
      OP_REQUIRES_OK(context, context->GetAttr("resize", &resize_));
      OP_REQUIRES(context, resize_ > 0, errors::InvalidArgument("PcCropAndSample expects positive resize"));
    }
    
    void Compute(OpKernelContext * context) override {

      const Tensor& pts = context->input(0);  // B * P * 3
      const Tensor& fts = context->input(1);  // B * P * C
      const Tensor& boxes = context->input(2);// N * 6
      const Tensor& box_index = context->input(3);  // N
      
      OP_REQUIRES(context, pts.dims()==3 && pts.dim_size(1) > 0 && pts.dim_size(2) == 3, errors::InvalidArgument("PcCropAndSample expects (B, P, 3) pts shape"));
      OP_REQUIRES(context, fts.dims()==3 && fts.dim_size(1) > 0 && fts.dim_size(2) > 0,  errors::InvalidArgument("PcCropAndSample expects (B, P, C) fts shape"));
      OP_REQUIRES(context, pts.dim_size(0) == fts.dim_size(0) && pts.dim_size(1) == fts.dim_size(1), errors::InvalidArgument("PcCropAndSample expects pts & fts has same (B, P, ...) shape"));
      const int batch_size = pts.dim_size(0);
      const int npts = pts.dim_size(1);
      const int channel = fts.dim_size(2);
      int num_boxes = 0;
      OP_REQUIRES_OK(context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes));

      auto pts_flat = pts.flat<float>();
      auto fts_flat = fts.flat<float>();
      auto boxes_flat = boxes.flat<float>();
      auto box_ind_flat = box_index.flat<int>();
      const float * pts_data = &(pts_flat(0));
      const float * fts_data = &(fts_flat(0));
      const float * boxes_data = &(boxes_flat(0));
      const int * box_ind_data = &(box_ind_flat(0));

      Tensor* crop_pts = nullptr;
      Tensor* crop_fts = nullptr;
      Tensor* crop_ind = nullptr;
      Tensor* non_empty_box = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{num_boxes, resize_, 3}, &crop_pts));
      OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{num_boxes, resize_, channel}, &crop_fts));
      OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{num_boxes, resize_}, &crop_ind));
      OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{num_boxes}, &non_empty_box));
      
      auto crop_pts_flat = crop_pts->flat<float>();
      auto crop_fts_flat = crop_fts->flat<float>();
      auto crop_ind_flat = crop_ind->flat<int>();
      auto non_empty_box_flat = non_empty_box->flat<bool>();
      float* crop_pts_data = &(crop_pts_flat(0));
      float* crop_fts_data = &(crop_fts_flat(0));
      int* crop_ind_data = &(crop_ind_flat(0));
      bool* non_empty_box_data = &(non_empty_box_flat(0));
      cudaMemset(crop_pts_data, 0, num_boxes * resize_ * 3 * sizeof(float));
      cudaMemset(crop_fts_data, 0, num_boxes * resize_ * channel * sizeof(float));
      cudaMemset(crop_ind_data, 0, num_boxes * resize_ * sizeof(int));
      cudaMemset(non_empty_box_data, 1, num_boxes * sizeof(bool));

      //Tensor temp_tensor;
      //OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      //auto temp_flat=temp_tensor.flat<float>();
      //float * temp=&(temp_flat(0));
      pccropandsample_gpu(pts_data, fts_data, boxes_data, box_ind_data, 
                            num_boxes, batch_size, npts, resize_, channel,
                            crop_pts_data, crop_fts_data, crop_ind_data, non_empty_box_data);
    }
    private:
        int resize_;
};

REGISTER_KERNEL_BUILDER(Name("PcCropAndSample").Device(DEVICE_GPU),PcCropAndSampleGpuOp);

void pccropandsamplegradfts_gpu(
    const int* box_ind_data, const int* crop_ind_data, const float* grad_crop_fts_data,
    int num_boxes, int npts, int resize, int channel,
    float* grad_fts_data);

class PcCropAndSampleGradFtsGpuOp: public OpKernel{
  public:
    explicit PcCropAndSampleGradFtsGpuOp(OpKernelConstruction* context):OpKernel(context) {}
    
    void Compute(OpKernelContext * context) override {

      const Tensor& fts = context->input(0);          // B * P * C
      const Tensor& box_ind = context->input(1);      // N
      const Tensor& crop_ind = context->input(2);     // N * R
      const Tensor& grad_crop_fts = context->input(3);// N * R * C
      
      OP_REQUIRES(context, fts.dims()==3 && grad_crop_fts.dims() == 3 && fts.dim_size(2) == grad_crop_fts.dim_size(2),  errors::InvalidArgument("PcCropAndSampleGradFts expects fts and grad_crop_fts has same shape(2)"));
      OP_REQUIRES(context, box_ind.dim_size(0) == crop_ind.dim_size(0) && crop_ind.dim_size(0) == grad_crop_fts.dim_size(0), errors::InvalidArgument("PcCropAndSampleGradFts expects box_ind, crop_ind and grad_crop_fts has same shape(0)"));
      OP_REQUIRES(context, crop_ind.dim_size(1) == grad_crop_fts.dim_size(1), errors::InvalidArgument("PcCropAndSampleGradFts expects crop_ind and grad_crop_fts has same shape(1)"));
      const int batch_size = fts.dim_size(0);
      const int npts = fts.dim_size(1);
      const int channel = fts.dim_size(2);
      const int num_boxes = box_ind.dim_size(0);
      const int resize = crop_ind.dim_size(1);

      auto box_ind_flat = box_ind.flat<int>();
      auto crop_ind_flat = crop_ind.flat<int>();
      auto grad_crop_fts_flat = grad_crop_fts.flat<float>();
      const int * box_ind_data = &(box_ind_flat(0));
      const int * crop_ind_data = &(crop_ind_flat(0));
      const float * grad_crop_fts_data = &(grad_crop_fts_flat(0));

      Tensor* grad_fts = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{batch_size, npts, channel}, &grad_fts));
      
      auto grad_fts_flat = grad_fts->flat<float>();
      float* grad_fts_data = &(grad_fts_flat(0));
      cudaMemset(grad_fts_data, 0, batch_size * npts * channel * sizeof(float));

      //Tensor temp_tensor;
      //OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      //auto temp_flat=temp_tensor.flat<float>();
      //float * temp=&(temp_flat(0));
      pccropandsamplegradfts_gpu(box_ind_data, crop_ind_data, grad_crop_fts_data,
                            num_boxes, npts, resize, channel,
                            grad_fts_data);
    }
    private:
        int resize_;
};

REGISTER_KERNEL_BUILDER(Name("PcCropAndSampleGradFts").Device(DEVICE_GPU),PcCropAndSampleGradFtsGpuOp);
