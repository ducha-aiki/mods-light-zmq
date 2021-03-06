#ifndef SYNTHDETECTION_HPP
#define SYNTHDETECTION_HPP
/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/
#undef __STRICT_ANSI__

#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//
#include <sys/time.h>
inline long getMilliSecs()
{
  timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec*1000 + t.tv_usec/1000;
}
#include "detectors/detectors_parameters.hpp"
#include "descriptors_parameters.hpp"
#include "detectors/mser/utls/matrix.h"

#include "detectors/helpers.h"
#include <sys/time.h>
#include "detectors/structures.hpp"

inline long getMilliSecs1()
{
  timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec*1000 + t.tv_usec/1000;
}
void rectifyTransformation(double &a11, double &a12, double &a21, double &a22);

/// Functions
void rectifyTransformation(double &a11, double &a12, double &a21, double &a22);

int SetVSPars (const std::vector <double> &scale_set,
               const std::vector <double> &tilt_set,
               const double phi_base,
               const std::vector <double> &FGINNThreshold,
               const std::vector <double> &DistanceThreshold,
               const std::vector <std::string> descriptors,
               std::vector<ViewSynthParameters> &par,
               std::vector<ViewSynthParameters> &prev_par,
               const double InitSigma=0.5,
               const int doBlur=1, const int dsplevels = 0,
               const double mixSigma=1.0, const double maxSigma=1.0);
//Function generates parameters for view synthesis based on gived scale, tilt and rotation sets, avoiding duplicates with previous synthesis.

//void GenerateSynthImage(const cv::Mat &in_img, SynthImage &out_img,const char* in_img_name, const double tilt,const double phi, const double zoom, const double InitSigma=0.5,const int doBlur=1, const int img_id = 0);
int ReprojectRegionsAndRemoveTouchBoundary(AffineRegionList &keypoints, double *H, int orig_w, int orig_h, const double mrSize = 3.0*sqrt(3.0), const bool dontRemove = false) ;
//Function reprojects detected regions to other image ("original") using H matrix (H is from original to tilted).
//Then all regions that are outside original image (fully or partially) are deleted.
bool HIsEye(double* H);
void GenerateSynthImageCorr(const cv::Mat &in_img,
                            SynthImage &out_img,
                            const std::string in_img_name,
                            double tilt,
                            const double phi,
                            const double zoom,
                            const double InitSigma=0.5,
                            const int doBlur=1,
                            const int img_id = 0,
                            const bool convert2gray = true);
//Function generates scaled, rotated and tilted image with homography from original to generated image and places all this into SynthImage structure
//Phi is rotation angle in radians
//Tilt - is scale factor in horizontal direction (to simulate real tilt)
//Zoom - scale factor
//InitSigma (= 0.5 by default). Bluring is done with sigma_aa = InitSigma * tilt / 2 for tilting and sigma_aa = InitSigma / (4*zoom) for downscaling.
//doBlur - to make gaussian convolution before scaling or no

void GenerateSynthImageByH(const cv::Mat &in_img, SynthImage &out_img,const double* H,const double InitSigma = 0.5,const int doBlur =1,const int img_id = 0);
//Function generates scaled, rotated and tilted image from image and homography matrix from original to generated image and places all this into SynthImage structure


template<typename T, typename params>
int DetectAffineRegions(SynthImage &img, AffineRegionList &keypoints, params par, detector_type det_type,
                        int (*detector)(cv::Mat &input, std::vector<T> &out,const params par,ScalePyramid &scale_pyramid,
                                        const double tilt,const double zoom))
//Function detects affine regions using detector function and writes them into AffineRegionList structure
{
  keypoints.clear();
  int RegionsNumber=0;
  std::vector<T> out1;
  RegionsNumber=detector(img.pixels, out1, par,img.pyramid, img.tilt, img.zoom);
  typename std::vector<T>::iterator ptr = out1.begin();
  keypoints.reserve(RegionsNumber);
  AffineRegion AffRegTmp;
  AffRegTmp.img_id=img.id;
  AffRegTmp.img_reproj_id= 0;
  AffRegTmp.type= det_type;

  for (int i = 0; i < RegionsNumber; i++, ptr++)
    {
      AffRegTmp.id = i;
      AffRegTmp.det_kp.s=ptr->s * sqrt(fabs(ptr->a11 * ptr->a22 - ptr->a12 * ptr->a21));
      rectifyTransformation(ptr->a11,ptr->a12,ptr->a21,ptr->a22);
      AffRegTmp.det_kp.x = ptr->x;
      AffRegTmp.det_kp.y = ptr->y;
      AffRegTmp.det_kp.a11 = ptr->a11;
      AffRegTmp.det_kp.a12 = ptr->a12;
      AffRegTmp.det_kp.a21 = ptr->a21;
      AffRegTmp.det_kp.a22 = ptr->a22;
      AffRegTmp.det_kp.response = ptr->response;
      AffRegTmp.det_kp.sub_type = ptr->sub_type;
      keypoints.push_back(AffRegTmp);
    }
  return RegionsNumber;
}


void rectifyTransformation(double &a11, double &a12, double &a21, double &a22);
//Rotates ellipse vertically(not the shape, just orientation) and normalizes matrix determinant to one

int ReprojectRegions(AffineRegionList &keypoints, double *H, int orig_w, int orig_h);
//Function reprojects detected regions to other image ("original") using H matrix (H is from original to tilted).
//Then all regions that are outside original image (fully or partially) are deleted.

double ellipseOverlap(AffineKeypoint ref_kp, AffineKeypoint test_kp, const double max_error=10000.);
//Computes overlap error between two ellipses in one image
//E=0.5||I-AB^-1||f +d(a,b),
//I = [1 0; 0 1], A,B - ellipse matrices, d(a,b) - distance between ellipse centers in canonical coordinate frame.
//d(a,b) is computed first. if  d(a,b)> max_error, other part doesn`t computed - for speed reasons.
//E=0 means that ellipses are the same.

int ReprojectRegionsBack(AffineRegionList &keypoints, double *H);
int ReprojectRegionsBackReal(AffineRegionList &keypoints, double *H, const int width2, const int height2);

void ReprojectByH(AffineKeypoint in_kp, AffineKeypoint &out_kp, double* H);
void ReprojectByHReal(AffineKeypoint in_kp, AffineKeypoint &out_kp, double* H);
//Reprojects ellipse matrix and point coordinates using homography matrix H
//For H=[h11 h12 h13; h21 h22 h23; 0 0 1] only;

int DetectOrientation(AffineRegionList &in_kp_list,
                      AffineRegionList &out_kp_list1,
                      SynthImage &img,
                      const  double mrSize = 3.0*sqrt(3.0),
                      const int patchSize = 41,
                      const int doHalfSIFT = 0,
                      const int maxAngNum= 0,
                      const double th = 0.8,
                      const bool addUpRight = false);

int DetectOrientationExt(AffineRegionList &in_kp_list,
                      AffineRegionList &out_kp_list,
                      SynthImage &img,
                      const double mrSize,
                      const int patchSize,
                      std::string exernal_command);
int DetectAffineShapeExt(AffineRegionList &in_kp_list,
                      AffineRegionList &out_kp_list1,
                      SynthImage &img,
                      const AffineShapeParams par);
int DetectAffineShape(AffineRegionList &in_kp_list,
                      AffineRegionList &out_kp_list1,
                      SynthImage &img,
                      const AffineShapeParams par);

//Detects orientation of the affine region and adds regions with detected orientation to the list.
//All points that derived from one have the same parent_id


void ExtractPatchesColumn(const AffineRegionList &in_kp_list,
                    const  SynthImage &img, cv::Mat& patches,  double mrSize = 3.0*sqrt(3.0),
                           int patchSize = 41, bool fast_extraction = false, bool photoNorm = false, bool export_and_read = true, bool do_mask = true);

template <typename FuncType>
void DescribeRegions(AffineRegionList &in_kp_list,
                     SynthImage &img, FuncType descriptor,
                     double mrSize = 3.0*sqrt(3.0), int patchSize = 41, bool fast_extraction = false, bool photoNorm = false, bool export_and_read = false)
//Describes region with SIFT or other descriptor
{
  // std::cerr << "photonorm=" << photoNorm << std::endl;
  std::vector<unsigned char> workspace;
  unsigned int i;
  // patch size in the image / patch size -> amount of down/up sampling
  cv::Mat patch(patchSize, patchSize, CV_32FC1);
  unsigned int n_descs = in_kp_list.size();
  cv::Mat mask(patchSize,patchSize,CV_32F);
  computeCircularGaussMask(mask);
  cv::Mat all_patches;
  if ( !fast_extraction) {
      for (i = 0; i < n_descs; i++) {
          float mrScale = ceil(in_kp_list[i].det_kp.s * mrSize); // half patch size in pixels of image

          int patchImageSize = 2 * int(mrScale) + 1; // odd size
          float imageToPatchScale = float(patchImageSize) / float(patchSize);  // patch size in the image / patch size -> amount of down/up sampling
          // is patch touching boundary? if yes, ignore this feature
          if (imageToPatchScale > 0.4) {
              // the pixels in the image are 0.4 apart + the affine deformation
              // leave +1 border for the bilinear interpolation
              patchImageSize += 2;
              size_t wss = patchImageSize * patchImageSize * sizeof(float);
              if (wss >= workspace.size())
                workspace.resize(wss);

              Mat smoothed(patchImageSize, patchImageSize, CV_32FC1, (void *) &workspace.front());
              // interpolate with det == 1
              interpolate(img.pixels,
                          (float) in_kp_list[i].det_kp.x,
                          (float) in_kp_list[i].det_kp.y,
                          (float) in_kp_list[i].det_kp.a11,
                          (float) in_kp_list[i].det_kp.a12,
                          (float) in_kp_list[i].det_kp.a21,
                          (float) in_kp_list[i].det_kp.a22,
                          smoothed);

              gaussianBlurInplace(smoothed, 1.5f * imageToPatchScale);
              // subsample with corresponding scale
              interpolate(smoothed, (float) (patchImageSize >> 1), (float) (patchImageSize >> 1),
                          imageToPatchScale, 0, 0, imageToPatchScale, patch);
            } else {
              // if imageToPatchScale is small (i.e. lot of oversampling), affine normalize without smoothing
              interpolate(img.pixels,
                          (float) in_kp_list[i].det_kp.x,
                          (float) in_kp_list[i].det_kp.y,
                          (float) in_kp_list[i].det_kp.a11 * imageToPatchScale,
                          (float) in_kp_list[i].det_kp.a12 * imageToPatchScale,
                          (float) in_kp_list[i].det_kp.a21 * imageToPatchScale,
                          (float) in_kp_list[i].det_kp.a22 * imageToPatchScale,
                          patch);

            }

          if (photoNorm) {
              float mean, var;
              photometricallyNormalize(patch, mask, mean, var);
            }

              descriptor(patch, in_kp_list[i].desc.vec);

          ///
          in_kp_list[i].desc.type = descriptor.type;

        }

    } else {
      for (i = 0; i < n_descs; i++) {
          double mrScale = (double) mrSize * in_kp_list[i].det_kp.s; // half patch size in pixels of image
          int patchImageSize = 2 * int(mrScale) + 1; // odd size
          double imageToPatchScale = double(patchImageSize) / (double) patchSize;
          float curr_sc = imageToPatchScale;

          interpolate(img.pixels,
                      (float) in_kp_list[i].det_kp.x,
                      (float) in_kp_list[i].det_kp.y,
                      (float) in_kp_list[i].det_kp.a11 * curr_sc,
                      (float) in_kp_list[i].det_kp.a12 * curr_sc,
                      (float) in_kp_list[i].det_kp.a21 * curr_sc,
                      (float) in_kp_list[i].det_kp.a22 * curr_sc,
                      patch);
          if (photoNorm) {
              float mean, var;
              photometricallyNormalize(patch, mask, mean, var);
            }
          descriptor(patch, in_kp_list[i].desc.vec);
          in_kp_list[i].desc.type = descriptor.type;
        }
    }
}
void AddRegionsToList(AffineRegionList &kp_list, AffineRegionList& new_kps);
//Function for getting new regions ID right (original IDs are changed to new ones to ensure no collisions in kp_list)

void AddRegionsToListByType(AffineRegionList &kp_list, AffineRegionList& new_kps, int type);
//Function for getting new regions ID right AND only given type

void SynthDetectDescribeKeypoints (const SynthImage &orig_img,
                                   IterationViewsynthesisParam &synth_par,
                                   DetectorsParameters &det_par,
                                   DescriptorsParameters &desc_par,
                                   std::map<std::string, AffineRegionList> &regs,
                                   TimeLog &times1);


void WriteKPs(AffineRegionList &keys, std::ostream &out1);
//Function writes keypoints to stream in format:
//descriptor_size(default = 128) keys_number
//x y scale a11 a12 a21 a22 desc[descriptor_size]

void ReadKPs(AffineRegionList &keys, std::istream &in1);
//Function reads keypoints from stream in format:
//descriptor_size(default = 128) keys_number
//x y scale a11 a12 a21 a22 desc[descriptor_size]

void ReadKPsMik(AffineRegionList &keys, std::istream &in1, const int det_type1 = DET_UNKNOWN, const double scale_factor=1.0);
//Function reads keypoints from stream in Mikolajczuk format:
//descriptor_size(default = 128) keys_number
//x y scale a b c desc[descriptor_size]

void linH(const double x, const double y, double *H, double *linearH);
//Function linearizes homography matrix to affine


#endif // SYNTHDETECTION_HPP
