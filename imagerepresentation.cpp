#include "imagerepresentation.h"
#include "synth-detection.hpp"
#include "detectors/mser/extrema/extrema.h"
#include <fstream>
#include <opencv2/features2d/features2d.hpp>
#include "cnpy/cnpy.h"
#ifdef _OPENMP
#include <omp.h>
#endif


#define VERBOSE 1
#include <zmq.hpp>

inline static bool endsWith(const std::string& str, const std::string& suffix)
{
  return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}
std::vector<std::vector<float> > DescribeWithZmq(zmqDescriptorParams par,
                                                 AffineRegionVector &kps,
                                                 SynthImage &temp_img1){

  std::vector<std::vector<float> > out;
  cv::Mat patches;
  int odd_patch_size = par.patchSize ;
  if (kps.size() > 0){
      ExtractPatchesColumn(kps,temp_img1, patches,
                           par.mrSize,
                           odd_patch_size,
                           false,
                           false,
                           true);
    }

  std::vector<uchar> bufff1;
  cv::imencode(".png",patches,bufff1);

  zmq::context_t context (1);
  int socket_mode = ZMQ_REQ;

  zmq::socket_t   socket(context, socket_mode);
  socket.connect(par.port);

  zmq::message_t mods_to_cnn(bufff1.size()) ;
  memcpy ((void *) mods_to_cnn.data (), bufff1.data(), bufff1.size());
  zmq::message_t cnn_to_mods;
#pragma omp critical
  {
    socket.send(mods_to_cnn);
    //  Get the reply.
    socket.recv(&cnn_to_mods);
  }
  std::vector<float> inMsg(cnn_to_mods.size() / sizeof(float));
  std::memcpy(inMsg.data(), cnn_to_mods.data(), cnn_to_mods.size());
  const int desc_size = inMsg.size() /kps.size() ;

  for (int img_num=0; img_num<kps.size(); img_num++)
    {
      std::vector<float> curr_desc(desc_size);
      int offset = img_num*desc_size;
      for (int i = 0; i < desc_size; ++i) {
          const float v1 = inMsg[i+offset];
          curr_desc[i] = v1;
        }
      out.push_back(curr_desc);
    }
  socket.close();
  return out;
}

void saveKP(AffineKeypoint &ak, std::ostream &s) {
  s << ak.x << " " << ak.y << " " << ak.a11 << " " << ak.a12 << " " << ak.a21 << " " << ak.a22 << " ";
  s << ak.pyramid_scale << " " << ak.octave_number << " " << ak.s << " " << ak.sub_type << " ";
}
void saveKPBench(AffineKeypoint &ak, std::ostream &s) {
  s << ak.x << " " << ak.y << " "  << ak.s << " " << ak.a11 << " " << ak.a12 << " " << ak.a21 << " " << ak.a22;
}

void saveKP_KM_format(AffineKeypoint &ak, std::ostream &s) {
  double sc = ak.s * sqrt(fabs(ak.a11*ak.a22 - ak.a12*ak.a21))*3.0*sqrt(3.0);
  rectifyAffineTransformationUpIsUp(ak.a11,ak.a12,ak.a21,ak.a22);

  Mat A = (Mat_<float>(2,2) << ak.a11, ak.a12, ak.a21, ak.a22);
  SVD svd(A, SVD::FULL_UV);

  float *d = (float *)svd.w.data;
  d[0] = 1.0f/(d[0]*d[0]*sc*sc);
  d[1] = 1.0f/(d[1]*d[1]*sc*sc);

  A = svd.u * Mat::diag(svd.w) * svd.u.t();
  s << ak.x << " " << ak.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1) << " ";
}

void saveKPMichal(AffineKeypoint &ak, std::ostream &s) {
  ak.s *= sqrt(fabs(ak.a11*ak.a22 - ak.a12*ak.a21))*3.0*sqrt(3.0);

  rectifyAffineTransformationUpIsUp(ak.a11,ak.a12,ak.a21,ak.a22);

  s << ak.x << " " << ak.y << " " << ak.s << " " << ak.a11 << " " << ak.a12 << " " << ak.a21 << " " << ak.a22 << " ";
  s << ak.sub_type << " " << ak.response << " ";
}
void saveKPMichalBin(AffineKeypoint &ak, std::ostream &s) {
  //float x, y, s, a11, a12, a21, a22, int type, float response, unsigned char desc[128]
  ak.s *= sqrt(fabs(ak.a11*ak.a22 - ak.a12*ak.a21))*3.0*sqrt(3.0);
  rectifyAffineTransformationUpIsUp(ak.a11,ak.a12,ak.a21,ak.a22);

  float x = (float)ak.x;
  s.write((char *)&x, sizeof(float));

  float y = (float)ak.y;
  s.write((char *)&y, sizeof(float));
  //float scale = ak.s*mrSize;
  float scale = (float)ak.s;
  s.write((char *)&scale, sizeof(float));

  float a11 = (float)ak.a11;
  s.write((char *)&a11, sizeof(float));

  float a12 = (float)ak.a12;
  s.write((char *)&a12, sizeof(float));

  float a21 = (float)ak.a21;
  s.write((char *)&a21, sizeof(float));

  float a22 = (float)ak.a22;
  s.write((char *)&a22, sizeof(float));

  s.write((char *)&ak.sub_type, sizeof(int));

  float resp = (float)ak.response;
  s.write((char *)&resp, sizeof(float));

}

void saveKP_KM_format_binary(AffineKeypoint &ak, std::ostream &s) {
  double sc = ak.s * sqrt(fabs(ak.a11*ak.a22 - ak.a12*ak.a21))*3.0*sqrt(3.0);
  rectifyAffineTransformationUpIsUp(ak.a11,ak.a12,ak.a21,ak.a22);

  Mat A = (Mat_<float>(2,2) << ak.a11, ak.a12, ak.a21, ak.a22);
  SVD svd(A, SVD::FULL_UV);

  float *d = (float *)svd.w.data;
  d[0] = 1.0f/(d[0]*d[0]*sc*sc);
  d[1] = 1.0f/(d[1]*d[1]*sc*sc);

  A = svd.u * Mat::diag(svd.w) * svd.u.t();
  float x = (float)ak.x;
  s.write((char *)&x, sizeof(float));

  float y = (float)ak.y;
  s.write((char *)&y, sizeof(float));

  float a = (float)A.at<float>(0,0);
  s.write((char *)&a, sizeof(float));

  float b = (float)A.at<float>(0,1);
  s.write((char *)&b, sizeof(float));

  float c = (float)A.at<float>(1,1);
  s.write((char *)&c, sizeof(float));

}

void saveAR(AffineRegion &ar, std::ostream &s) {
  saveKPBench(ar.reproj_kp,s);
  s << " " << ar.desc.vec.size() << " ";
  for (unsigned int i = 0; i < ar.desc.vec.size(); ++i) {
      s << ar.desc.vec[i] << " ";
    }
}
void saveAR_KM_format(AffineRegion &ar, std::ostream &s) {
  saveKP_KM_format(ar.reproj_kp,s);
  for (unsigned int i = 0; i < ar.desc.vec.size(); i++) {
      s << ar.desc.vec[i] << " ";
    }
  s << std::endl;
}
void saveARBench(AffineRegion &ar, std::ostream &s, std::ostream &s2) {
  saveKPBench(ar.det_kp,s2);
  saveKPBench(ar.reproj_kp,s);
}
void saveARMichal(AffineRegion &ar, std::ostream &s) {
  saveKPMichal(ar.reproj_kp,s);
  for (unsigned int i = 0; i < ar.desc.vec.size(); ++i) {
      s << ar.desc.vec[i] << " ";
    }
}
void saveARMichalBinary(AffineRegion &ar, std::ostream &s) {
  saveKPMichalBin(ar.reproj_kp,s);
  for (unsigned int i = 0; i < ar.desc.vec.size(); ++i) {
      int desc = (int)MAX(0,MIN(ar.desc.vec[i], 255));
      unsigned char desc1 = (unsigned char) (desc);
      s.write((char *)&desc1, sizeof(unsigned char));
    }
}
void saveARMikBinary(AffineRegion &ar, std::ostream &s) {
  saveKP_KM_format_binary(ar.reproj_kp, s);
  for (unsigned int i = 0; i < ar.desc.vec.size(); ++i) {
      float desc = (float)MAX(0,MIN(ar.desc.vec[i], 255));
      s.write((char *)&desc, sizeof(float));
    }
}
void loadKP(AffineKeypoint &ak, std::istream &s) {
  s >> ak.x >> ak.y >> ak.a11 >> ak.a12 >>ak.a21 >> ak.a22 >> ak.pyramid_scale >> ak.octave_number >> ak.s >> ak.sub_type;
}

void loadAR(AffineRegion &ar, std::istream &s) {
  s >> ar.id >> ar.img_id >> ar.img_reproj_id;
  s >> ar.parent_id;
  loadKP(ar.det_kp,s);
  loadKP(ar.reproj_kp,s);
  //  s >> ar.desc.type;
  int size1;
  s >> size1;
  ar.desc.vec.resize(size1);
  for (unsigned int i = 0; i < ar.desc.vec.size(); ++i) {
      s >> ar.desc.vec[i];
    }
}

void L2normalize(const float* input_arr, int size, std::vector<float> &output_vect)
{
  double norm = 0.0;
  for (int i = 0; i < size; ++i) {
      norm+=input_arr[i] * input_arr[i];
    }
  const double norm_coef = 1.0/sqrt(norm + 1e-10);
  for (int i = 0; i < size; ++i) {
      const float v1 = norm_coef*input_arr[i] ;
      output_vect[i] = v1;
    }
}
void L1normalize(const float* input_arr, int size, std::vector<float> &output_vect)
{
  double norm=0.0;
  for (int i = 0; i < size; ++i) {
      norm+=input_arr[i];
    }
  const double norm_coef = 1.0/norm;
  for (int i = 0; i < size; ++i) {
      const float v1 = floor(norm_coef*input_arr[i]);
      output_vect[i] = v1;
    }
}
void RootNormalize(const float* input_arr, int size, std::vector<float> &output_vect)
{
  L2normalize(input_arr,size,output_vect);
  double norm=0.0;
  for (int i = 0; i < size; ++i) {
      norm+=input_arr[i];
    }
  const double norm_coef = 1.0/norm;
  for (int i = 0; i < size; ++i) {
      const float v1 = sqrt(norm_coef*input_arr[i]);
      output_vect[i] = v1;
    }
}

ImageRepresentation::ImageRepresentation(cv::Mat _in_img, std::string _name)
{
  if (_in_img.channels() ==3) {
      _in_img.convertTo(OriginalImg,CV_32FC3);

    } else {
      _in_img.convertTo(OriginalImg,CV_32F);
    }
  Name = _name;
}
ImageRepresentation::ImageRepresentation()
{
}

ImageRepresentation::~ImageRepresentation()
{
  RegionVectorMap.clear();
}

void ImageRepresentation::Clear()
{
  RegionVectorMap.clear();
}
void ImageRepresentation::SetImg(cv::Mat _in_img, std::string _name)
{
  if (_in_img.channels() ==3) {
      _in_img.convertTo(OriginalImg,CV_32FC3);

    } else {
      _in_img.convertTo(OriginalImg,CV_32F);
    }
  Name = _name;
}

descriptor_type ImageRepresentation::GetDescriptorType(std::string desc_name)
{
  for (unsigned int i=0; i< DescriptorNames.size(); i++)
    if (DescriptorNames[i].compare(desc_name)==0)
      return static_cast<descriptor_type>(i);
  return DESC_UNKNOWN;
}

detector_type ImageRepresentation::GetDetectorType(std::string det_name)
{
  for (unsigned int i=0; i< DetectorNames.size(); i++)
    if (DetectorNames[i].compare(det_name)==0)
      return static_cast<detector_type>(i);
  return DET_UNKNOWN;
}

TimeLog ImageRepresentation::GetTimeSpent()
{
  return TimeSpent;
}

int ImageRepresentation::GetRegionsNumber(std::string det_name)
{
  int reg_number = 0;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  if (det_name.compare("All") == 0)
    {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        {
          AffineRegionVectorMap::iterator desc_it;
          if ( (desc_it = regions_it->second.find("None")) != regions_it->second.end() )
            reg_number +=  desc_it->second.size();
        }
    }
  else
    {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          AffineRegionVectorMap::iterator desc_it;
          if ( (desc_it = regions_it->second.find("None")) != regions_it->second.end() )
            reg_number +=  desc_it->second.size();
        }
    }
  return reg_number;
}
int ImageRepresentation::GetDescriptorsNumber(std::string desc_name, std::string det_name)
{
  int reg_number = 0;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  if (det_name.compare("All") == 0)
    {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        if (desc_name.compare("All") == 0)
          {
            for (desc_it = regions_it->second.begin();
                 desc_it != regions_it->second.end(); desc_it++)
              reg_number +=  desc_it->second.size();
          }
        else
          {
            desc_it = regions_it->second.find(desc_name);
            if (desc_it != regions_it->second.end() )
              reg_number +=  desc_it->second.size();

          }
    }
  else
    {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          if (desc_name.compare("All") == 0)
            {
              for (desc_it = regions_it->second.begin();
                   desc_it != regions_it->second.end(); desc_it++)
                reg_number +=  desc_it->second.size();
            }
          else
            {
              desc_it = regions_it->second.find(desc_name);
              if (desc_it != regions_it->second.end() )
                reg_number +=  desc_it->second.size();

            }
        }
    }
  return reg_number;
}
int ImageRepresentation::GetDescriptorDimension(std::string desc_name)
{
  int dim = 0;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  for (regions_it = RegionVectorMap.begin();regions_it != RegionVectorMap.end(); regions_it++)
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        if (desc_it->second.size() > 0)
          {
            dim = desc_it->second[0].desc.vec.size();
            break;
          }
    }
  return dim;
}
cv::Mat ImageRepresentation::GetDescriptorsMatByDetDesc(const std::string desc_name,const std::string det_name)
{
  unsigned int dim = GetDescriptorDimension(desc_name);
  unsigned int n_descs = GetDescriptorsNumber(desc_name,det_name);

  cv::Mat descriptors(dim, n_descs, CV_32F);
  int reg_number = 0;

  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  if (det_name.compare("All") == 0)
    {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        {
          desc_it = regions_it->second.find(desc_name);
          if (desc_it != regions_it->second.end() )
            {
              AffineRegionVector *currentDescVector = &(desc_it->second);
              unsigned int curr_size = currentDescVector->size();
              for (unsigned int i = 0; i<curr_size; i++, reg_number++)
                {
                  float* Row = descriptors.ptr<float>(reg_number);
                  AffineRegion curr_region = (*currentDescVector)[i];
                  for (unsigned int j = 0; j<dim; j++)
                    Row[j] = curr_region.desc.vec[j];
                }
            }
        }
    }
  else
    {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          desc_it = regions_it->second.find(desc_name);
          if (desc_it != regions_it->second.end() )
            {
              AffineRegionVector *currentDescVector = &(desc_it->second);
              unsigned int curr_size = currentDescVector->size();
              for (unsigned int i = 0; i<curr_size; i++, reg_number++)
                {
                  float* Row = descriptors.ptr<float>(reg_number);
                  AffineRegion curr_region = (*currentDescVector)[i];
                  for (unsigned int j = 0; j<dim; j++)
                    Row[j] = curr_region.desc.vec[j];
                }
            }
        }
    }
  return descriptors;
}

cv::Mat ImageRepresentation::GetDescriptorsMatByDetDesc(std::vector<Point2f> &coordinates, const std::string desc_name,const std::string det_name)
{
  unsigned int dim = GetDescriptorDimension(desc_name);
  unsigned int n_descs = GetDescriptorsNumber(desc_name,det_name);

  cv::Mat descriptors(dim, n_descs, CV_32F);
  coordinates.clear();
  coordinates.reserve(n_descs);
  int reg_number = 0;

  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  if (det_name.compare("All") == 0)
    {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        {
          desc_it = regions_it->second.find(desc_name);
          if (desc_it != regions_it->second.end() )
            {
              AffineRegionVector *currentDescVector = &(desc_it->second);
              unsigned int curr_size = currentDescVector->size();
              for (unsigned int i = 0; i<curr_size; i++, reg_number++)
                {
                  float* Row = descriptors.ptr<float>(reg_number);
                  AffineRegion curr_region = (*currentDescVector)[i];
                  Point2f curr_point;
                  curr_point.x = curr_region.reproj_kp.x;
                  curr_point.y = curr_region.reproj_kp.y;
                  coordinates.push_back(curr_point);
                  for (unsigned int j = 0; j<dim; j++)
                    Row[j] = curr_region.desc.vec[j];
                }
            }
        }
    }
  else
    {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          desc_it = regions_it->second.find(desc_name);
          if (desc_it != regions_it->second.end() )
            {
              AffineRegionVector *currentDescVector = &(desc_it->second);
              unsigned int curr_size = currentDescVector->size();
              for (unsigned int i = 0; i<curr_size; i++, reg_number++)
                {
                  float* Row = descriptors.ptr<float>(reg_number);
                  AffineRegion curr_region = (*currentDescVector)[i];
                  Point2f curr_point;
                  curr_point.x = curr_region.reproj_kp.x;
                  curr_point.y = curr_region.reproj_kp.y;
                  coordinates.push_back(curr_point);

                  for (unsigned int j = 0; j<dim; j++)
                    Row[j] = curr_region.desc.vec[j];
                }
            }
        }
    }
  return descriptors;
}

AffineRegion ImageRepresentation::GetAffineRegion(std::string desc_name, std::string det_name, int idx)
{
  AffineRegion curr_region;
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          curr_region = (*currentDescVector)[idx];
          return curr_region;
        }
    }
  return curr_region;
}
AffineRegionVector ImageRepresentation::GetAffineRegionVector(std::string desc_name, std::string det_name, std::vector<int> idxs)
{
  unsigned int n_regs = idxs.size();
  AffineRegionVector regions;
  regions.reserve(n_regs);

  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;


  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          for (unsigned int i = 0; i < n_regs; i++)
            regions.push_back((*currentDescVector)[idxs[i]]);
        }
    }

  return regions;
}
AffineRegionVector ImageRepresentation::GetAffineRegionVector(std::string desc_name, std::string det_name)
{
  unsigned int n_regs = GetDescriptorsNumber(desc_name,det_name);
  AffineRegionVector regions;
  regions.reserve(n_regs);

  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;
  if (det_name.compare("All") == 0)  {
      for (regions_it = RegionVectorMap.begin();
           regions_it != RegionVectorMap.end(); regions_it++)
        {
          desc_it = regions_it->second.find(desc_name);
          if (desc_it != regions_it->second.end() )
            {
              AffineRegionVector *currentDescVector = &(desc_it->second);
              for (unsigned int i = 0; i < n_regs; i++)
                regions.push_back((*currentDescVector)[i]);
            }
        }
    }
  else {
      regions_it = RegionVectorMap.find(det_name);
      if ( regions_it != RegionVectorMap.end())
        {
          desc_it = regions_it->second.find(desc_name);
          if (desc_it != regions_it->second.end() )
            {
              AffineRegionVector *currentDescVector = &(desc_it->second);
              for (unsigned int i = 0; i < n_regs; i++)
                regions.push_back((*currentDescVector)[i]);
            }
        }
    }
  return regions;
}

void ImageRepresentation::AddRegions(AffineRegionVector &RegionsToAdd, std::string det_name, std::string desc_name)
{
  std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
  AffineRegionVectorMap::iterator desc_it;

  regions_it = RegionVectorMap.find(det_name);
  if ( regions_it != RegionVectorMap.end())
    {
      desc_it = regions_it->second.find(desc_name);
      if (desc_it != regions_it->second.end() )
        {
          AffineRegionVector *currentDescVector = &(desc_it->second);
          ImageRepresentation::AddRegionsToList(*currentDescVector,RegionsToAdd);
        }
      else
        {
          regions_it->second[desc_name] = RegionsToAdd;
        }
    }
  else
    {
      std::map<std::string, AffineRegionVector> new_desc;
      new_desc[desc_name] = RegionsToAdd;
      RegionVectorMap[det_name] = new_desc;
    }
}
void ImageRepresentation::AddRegions(AffineRegionVectorMap &RegionsMapToAdd, std::string det_name)
{
  AffineRegionVectorMap::iterator desc_it;

  for (desc_it = RegionsMapToAdd.begin();
       desc_it != RegionsMapToAdd.end(); desc_it++)
    AddRegions(desc_it->second,det_name,desc_it->first);
}

void ImageRepresentation::AddRegionsToList(AffineRegionList &kp_list, AffineRegionList &new_kps)
{
  int size = (int)kp_list.size();
  unsigned int new_size = size + new_kps.size();
  AffineRegionList::iterator ptr = new_kps.begin();
  for (unsigned int i=size; i< new_size; i++, ptr++)
    {
      AffineRegion temp_reg = *ptr;
      temp_reg.id += size;
      temp_reg.parent_id +=size;
      kp_list.push_back(temp_reg);
    }
}

void ImageRepresentation::SynthDetectDescribeKeypoints (IterationViewsynthesisParam &synth_par,
                                                        DetectorsParameters &det_par,
                                                        DescriptorsParameters &desc_par,
                                                        DominantOrientationParams &dom_ori_par)
{
  double time1 = 0;
#ifdef _OPENMP
  omp_set_nested(1);
#endif
#pragma omp parallel for schedule (dynamic,1)
  for (unsigned int det=0; det < DetectorNames.size(); det++)
    {
      std::string curr_det = DetectorNames[det];
      unsigned int n_synths = synth_par[curr_det].size();

      std::vector<AffineRegionVectorMap> OneDetectorKeypointsMapVector;
      OneDetectorKeypointsMapVector.resize(n_synths);

#pragma omp parallel for schedule (dynamic,1)
      for (unsigned int synth=0; synth<n_synths; synth++)
        {
          ///Synthesis
          long s_time = getMilliSecs1();
          AffineRegionVector temp_kp1;
          AffineRegionVectorMap temp_kp_map;
          SynthImage temp_img1;
          GenerateSynthImageCorr(OriginalImg, temp_img1, Name.c_str(),
                                 synth_par[curr_det][synth].tilt,
                                 synth_par[curr_det][synth].phi,
                                 synth_par[curr_det][synth].zoom,
                                 synth_par[curr_det][synth].InitSigma,
                                 synth_par[curr_det][synth].doBlur, synth);

          bool doExternalAffineAdaptation = false;

          time1 = ((double)(getMilliSecs1() - s_time))/1000;
          TimeSpent.SynthTime += time1;

          bool SIFT_like_desc = true;
          bool HalfSIFT_like_desc = false;

          for (unsigned int i_desc=0; i_desc < synth_par[curr_det][synth].descriptors.size();i_desc++) {
              std::string curr_desc = synth_par[curr_det][synth].descriptors[i_desc];
              if (curr_desc.find("Half") != std::string::npos) {
                  HalfSIFT_like_desc = true;
                }
            }
          /// Detection
          s_time = getMilliSecs1();
          if (curr_det.compare("HessianAffine")==0)
            {
              doExternalAffineAdaptation = (det_par.BaumbergParam.external_command.size() > 0) || (det_par.BaumbergParam.useZMQ);

              DetectAffineRegions(temp_img1, temp_kp1,det_par.HessParam,DET_HESSIAN,DetectAffineKeypoints);
            }
          else if (curr_det.compare("ReadAffs") == 0) {
              if (endsWith(det_par.ReadAffsFromFileParam.fname,".npz")){

                  temp_kp1 =  ImageRepresentation::PreLoadRegionsNPZ(det_par.ReadAffsFromFileParam.fname);
                }  else {
                  std::ifstream focikp(det_par.ReadAffsFromFileParam.fname);
                  if (focikp.is_open()) {
                      int kp_size;
                      focikp >> kp_size;
                      std::cerr << kp_size << std::endl;
                      temp_kp1.reserve(kp_size);
                      for (int kp_num = 0; kp_num < kp_size; kp_num++) {
                          AffineRegion temp_region;
                          temp_region.det_kp.pyramid_scale = -1;
                          temp_region.det_kp.octave_number = -1;
                          temp_region.det_kp.sub_type = 101;
                          focikp >> temp_region.det_kp.x;
                          focikp >> temp_region.det_kp.y;
                          focikp >> temp_region.det_kp.s;
                          focikp >> temp_region.det_kp.a11;
                          focikp >> temp_region.det_kp.a12;
                          focikp >> temp_region.det_kp.a21;
                          focikp >> temp_region.det_kp.a22;
                          temp_region.det_kp.response = 100;
                          temp_region.type = DET_READ;
                          temp_kp1.push_back(temp_region);
                        }
                    }
                  focikp.close();
                }
            }
          else if (curr_det.compare("DoG")==0)
            {
              DetectAffineRegions(temp_img1, temp_kp1,det_par.DoGParam,DET_DOG,DetectAffineKeypoints);
            }
          else if (curr_det.compare("HarrisAffine")==0)
            {
              DetectAffineRegions(temp_img1, temp_kp1,det_par.HarrParam,DET_HARRIS,DetectAffineKeypoints);
            }
          else if (curr_det.compare("MSER")==0)
            {
              DetectAffineRegions(temp_img1, temp_kp1,det_par.MSERParam,DET_MSER,DetectMSERs);
            }

          //Baumberg iteration
          if (doExternalAffineAdaptation) {
              AffineRegionVector temp_kp_aff;
              AffineShapeParams afShPar = det_par.BaumbergParam;
              afShPar.affBmbrgMethod = det_par.HessParam.AffineShapePars.affBmbrgMethod;
              // std::cout << "bmbg method: " << (int)afShPar.affBmbrgMethod;
              if (temp_kp1.size() > 0) {
                  if (afShPar.external_command.size() > 0) {
                      DetectAffineShapeExt(temp_kp1,
                                           temp_kp_aff,
                                           temp_img1,
                                           afShPar);

                    } else if (afShPar.useZMQ) {
                      //det_par.AffNetParam.mrSize = afShPar.mrSize;
                      std::vector<std::vector<float> > a11a21a22 = DescribeWithZmq(det_par.AffNetParam,
                                                                                   temp_kp1,
                                                                                   temp_img1);
                      AffineRegion temp_region,  const_temp_region;
                      temp_kp_aff.clear();
                      temp_kp_aff.reserve(a11a21a22.size());

                      for (int kp_idx = 0; kp_idx < a11a21a22.size(); kp_idx ++) {
                          const_temp_region=temp_kp1[kp_idx];
                          temp_region=const_temp_region;
                          temp_region.det_kp.a11 = a11a21a22[kp_idx][0];
                          temp_region.det_kp.a12 = 0;
                          temp_region.det_kp.a21 = a11a21a22[kp_idx][1];
                          temp_region.det_kp.a22 = a11a21a22[kp_idx][2];
                          rectifyAffineTransformationUpIsUp(temp_region.det_kp.a11,
                                                            temp_region.det_kp.a12,
                                                            temp_region.det_kp.a21,
                                                            temp_region.det_kp.a22 );
                          float l1 = 1.0f, l2 = 1.0f;
                          if (!getEigenvalues(temp_region.det_kp.a11,
                                              temp_region.det_kp.a12,
                                              temp_region.det_kp.a21,
                                              temp_region.det_kp.a22, l1, l2)) {
                              continue;
                            }

                          // leave on too high anisotropyb
                          if ((l1/l2>6) || (l2/l1>6)) {
                              continue;
                            }
                          if (interpolateCheckBorders(temp_img1.pixels.cols,temp_img1.pixels.rows,
                                                      (float) temp_region.det_kp.x,
                                                      (float) temp_region.det_kp.y,
                                                      (float) temp_region.det_kp.a11,
                                                      (float) temp_region.det_kp.a12 ,
                                                      (float) temp_region.det_kp.a21,
                                                      (float) temp_region.det_kp.a22 ,
                                                      det_par.AffNetParam.mrSize * temp_region.det_kp.s,
                                                      det_par.AffNetParam.mrSize * temp_region.det_kp.s) ) {
                              continue;
                            }

                          temp_kp_aff.push_back(temp_region);

                        }

                    }
                  else {
                      DetectAffineShape(temp_kp1,
                                        temp_kp_aff,
                                        temp_img1,
                                        afShPar);
                    }

                  temp_kp1 = temp_kp_aff;
                }
            }
          //
          /// Orientation estimation

          time1 = ((double)(getMilliSecs1() - s_time))/1000;
          TimeSpent.DetectTime += time1;
          s_time = getMilliSecs1();

          AffineRegionVector temp_kp1_SIFT_like_desc;
          AffineRegionVector temp_kp1_HalfSIFT_like_desc;
          AffineRegionVector temp_kp1_upright;
          ReprojectRegionsAndRemoveTouchBoundary(temp_kp1, temp_img1.H, OriginalImg.cols, OriginalImg.rows, desc_par.RootSIFTParam.PEParam.mrSize + 0.01 , true);

          if (curr_det.compare("ReadAffs") == 0){

            } else {
              ////////////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Change it!s
              if (SIFT_like_desc) {
                  if (dom_ori_par.external_command.size() > 0) {
                      DetectOrientationExt(temp_kp1, temp_kp1_SIFT_like_desc, temp_img1,
                                           dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,dom_ori_par.external_command);
                    } else if (dom_ori_par.useZMQ) {
                      std::vector<std::vector<float> > yx = DescribeWithZmq(det_par.OriNetParam,
                                                                            temp_kp1,
                                                                            temp_img1);
                      AffineRegion temp_region,  const_temp_region;

                      temp_kp1_SIFT_like_desc.clear();
                      temp_kp1_SIFT_like_desc.reserve(yx.size());

                      for (int kp_idx = 0; kp_idx < yx.size(); kp_idx ++) {
                          const_temp_region=temp_kp1[kp_idx];

                          double angle = atan2(yx[kp_idx][0], yx[kp_idx][1]);
                          double ci = cos(angle);
                          double si = sin(angle);

                          temp_region=const_temp_region;
                          temp_region.det_kp.a11 = const_temp_region.det_kp.a11*ci-const_temp_region.det_kp.a12*si;
                          temp_region.det_kp.a12 = const_temp_region.det_kp.a11*si+const_temp_region.det_kp.a12*ci;
                          temp_region.det_kp.a21 = const_temp_region.det_kp.a21*ci-const_temp_region.det_kp.a22*si;
                          temp_region.det_kp.a22 = const_temp_region.det_kp.a21*si+const_temp_region.det_kp.a22*ci;
                          temp_kp1_SIFT_like_desc.push_back(temp_region);
                        }

                    } else
                    {
                      DetectOrientation(temp_kp1, temp_kp1_SIFT_like_desc, temp_img1,
                                        dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,
                                        false, dom_ori_par.maxAngles,
                                        dom_ori_par.threshold, false);
                    }
                }
              if (HalfSIFT_like_desc) {
                  DetectOrientation(temp_kp1, temp_kp1_HalfSIFT_like_desc, temp_img1,
                                    dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,
                                    true, dom_ori_par.maxAngles,
                                    dom_ori_par.threshold, false);
                }
              if (dom_ori_par.addUpRight) {
                  DetectOrientation(temp_kp1, temp_kp1_upright, temp_img1,
                                    dom_ori_par.PEParam.mrSize, dom_ori_par.PEParam.patchSize,
                                    false, 0, 1.0, true);
                }
            }
          temp_kp_map["None"] = temp_kp1;

          for (unsigned int i_desc=0; i_desc < synth_par[curr_det][synth].descriptors.size();i_desc++) {
              std::string curr_desc = synth_par[curr_det][synth].descriptors[i_desc];
              AffineRegionVector temp_kp1_desc;
              AffineRegionVector dsp_desc;
              if (dom_ori_par.addUpRight) {
                  temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1_upright.begin(), temp_kp1_upright.end());
                }
              //             ReprojectRegions(temp_kp1_desc, temp_img1.H, OriginalImg.cols, OriginalImg.rows);
              if (curr_det.compare("ReadAffs") == 0) {

                  temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1.begin(), temp_kp1.end());
                  std::cerr << "Read detections from provided file" << std::endl;
                }  else {
                  //Add oriented and upright keypoints if any
                  if (HalfSIFT_like_desc) {
                      temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1_HalfSIFT_like_desc.begin(),
                                           temp_kp1_HalfSIFT_like_desc.end());
                    }
                  if (SIFT_like_desc && (!HalfSIFT_like_desc)) {

                      temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1_SIFT_like_desc.begin(),
                                           temp_kp1_SIFT_like_desc.end());

                    }
                  if (!SIFT_like_desc) {
                      temp_kp1_desc.insert(temp_kp1_desc.end(), temp_kp1.begin(),
                                           temp_kp1.end());
                    }
                  ReprojectRegions(temp_kp1_desc, temp_img1.H, OriginalImg.cols, OriginalImg.rows);
                }

              ///Description
              ///
              time1 = ((double) (getMilliSecs1() - s_time)) / 1000;
              TimeSpent.OrientTime += time1;
              s_time = getMilliSecs1();

              if (curr_desc.compare("RootSIFT") == 0) //RootSIFT
                {
                  SIFTDescriptor RootSIFTdesc(desc_par.RootSIFTParam);
                  DescribeRegions(temp_kp1_desc,
                                  temp_img1, RootSIFTdesc,
                                  desc_par.RootSIFTParam.PEParam.mrSize,
                                  desc_par.RootSIFTParam.PEParam.patchSize,
                                  desc_par.RootSIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.RootSIFTParam.PEParam.photoNorm);
                }
              else if (curr_desc.compare("HalfRootSIFT") == 0) //HalfRootSIFT
                {
                  SIFTDescriptor HalfRootSIFTdesc(desc_par.HalfRootSIFTParam);
                  DescribeRegions(temp_kp1_desc,
                                  temp_img1, HalfRootSIFTdesc,
                                  desc_par.HalfRootSIFTParam.PEParam.mrSize,
                                  desc_par.HalfRootSIFTParam.PEParam.patchSize,
                                  desc_par.HalfRootSIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.HalfRootSIFTParam.PEParam.photoNorm);
                }
              else if (curr_desc.compare("HalfSIFT") == 0) //HalfSIFT
                {
                  ///Description
                  SIFTDescriptor HalfSIFTdesc(desc_par.HalfSIFTParam);
                  DescribeRegions(temp_kp1_desc,
                                  temp_img1, HalfSIFTdesc,
                                  desc_par.HalfSIFTParam.PEParam.mrSize,
                                  desc_par.HalfSIFTParam.PEParam.patchSize,
                                  desc_par.HalfSIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.HalfSIFTParam.PEParam.photoNorm);
                }

              else if (curr_desc.compare("ZMQ")==0)
                {
                  if (temp_kp1_desc.size() > 0) {
                      std::vector<std::vector<float> > descrs = DescribeWithZmq(desc_par.zmqDescParam,
                                                                                temp_kp1_desc,
                                                                                temp_img1);
                      for (unsigned int kp_idx = 0; kp_idx < temp_kp1_desc.size(); kp_idx++) {
                          temp_kp1_desc[kp_idx].desc.vec.resize(descrs[kp_idx].size());
                          for (unsigned int di = 0; di < descrs[kp_idx].size(); di ++) {
                              temp_kp1_desc[kp_idx].desc.vec[di] = descrs[kp_idx][di];
                            };
                          temp_kp1_desc[kp_idx].desc.type=DESC_ZMQ;
                        }
                    }
                }
              else if (curr_desc.compare("SIFT") == 0) //SIFT
                {
                  SIFTDescriptor SIFTdesc(desc_par.SIFTParam);
                  DescribeRegions(temp_kp1_desc,
                                  temp_img1, SIFTdesc,
                                  desc_par.SIFTParam.PEParam.mrSize,
                                  desc_par.SIFTParam.PEParam.patchSize,
                                  desc_par.SIFTParam.PEParam.FastPatchExtraction,
                                  desc_par.SIFTParam.PEParam.photoNorm);
                }
              else if (curr_desc.compare("CLIDescriptor") == 0) //ResSIFT
                {
                  cv::Mat patches;
                  if (temp_kp1_desc.size() > 0){
                      ExtractPatchesColumn(temp_kp1_desc,temp_img1,patches,
                                           desc_par.CLIDescParam.PEParam.mrSize,
                                           desc_par.CLIDescParam.PEParam.patchSize,
                                           desc_par.CLIDescParam.PEParam.FastPatchExtraction,
                                           desc_par.CLIDescParam.PEParam.photoNorm,
                                           true);
                    }
                  std::cerr << patches.rows << " " <<   patches.cols << std::endl;
                  if (patches.cols > 0) {
                      int n_descs = patches.rows / patches.cols;
                      if ( desc_par.CLIDescParam.hardcoded_run_string) {

                          cv::imwrite( desc_par.CLIDescParam.hardcoded_input_fname, patches);
                          std::string command =  desc_par.CLIDescParam.runfile;
                          std::cerr << command << std::endl;
                          system(command.c_str());
                          std::ifstream focikp( desc_par.CLIDescParam.hardcoded_output_fname);
                          if (focikp.is_open()) {
                              int dim1 = 0;
                              focikp >> dim1;
                              std::cerr << dim1 << " " << n_descs << " " << temp_kp1_desc.size() << std::endl;
                              for (int i = 0; i < n_descs; i++) {
                                  temp_kp1_desc[i].desc.vec.resize(dim1);
                                  for (int dd = 0; dd < dim1; dd++) {
                                      focikp >> temp_kp1_desc[i].desc.vec[dd];
                                    }
                                  temp_kp1_desc[i].desc.type = DESC_CLI;
                                }
                            }
                          focikp.close();
                          std::string rm_command = "rm " +  desc_par.CLIDescParam.hardcoded_input_fname;
                          system(rm_command.c_str());
                          rm_command = "rm " +  desc_par.CLIDescParam.hardcoded_output_fname;
                          system(rm_command.c_str());
                        } else {

                          int rnd1 = (int) getMilliSecs() + (std::rand() % (int)(1001));
                          std::string img_fname = "CLIDESC"+std::to_string(rnd1)+".bmp";
                          cv::imwrite(img_fname,patches );
                          std::string desc_fname = "CLIDESC"+std::to_string(rnd1)+".txt";
                          std::string command = desc_par.CLIDescParam.runfile + " " + img_fname + " " +desc_fname;

                          system(command.c_str());
                          std::ifstream focikp(desc_fname);
                          if (focikp.is_open()) {
                              int dim1 = 0;
                              focikp >> dim1;
                              std::cerr << dim1 << " " << n_descs << " " << temp_kp1_desc.size() << std::endl;

                              for (int i = 0; i < n_descs; i++) {
                                  temp_kp1_desc[i].desc.vec.resize(dim1);
                                  for (int dd = 0; dd < dim1; dd++) {
                                      focikp >> temp_kp1_desc[i].desc.vec[dd];
                                    }
                                  temp_kp1_desc[i].desc.type = DESC_CLI;
                                }
                            }
                          focikp.close();
                          std::string rm_command = "rm " + img_fname;
                          system(rm_command.c_str());
                          rm_command = "rm " + desc_fname;
                          system(rm_command.c_str());
                        }
                    } else {

                      for (int i = 0; i < temp_kp1_desc.size(); i++) {
                          temp_kp1_desc[i].desc.vec.resize(128);
                        }

                    };
                }
              temp_kp_map[curr_desc] = temp_kp1_desc;
              time1 = ((double)(getMilliSecs1() - s_time)) / 1000;
              TimeSpent.DescTime += time1;
              s_time = getMilliSecs1();

              //   std::cerr << "storing" << std::endl;
              OneDetectorKeypointsMapVector[synth] = temp_kp_map;
            }
          for (unsigned int synth=0; synth<n_synths; synth++)
            AddRegions(OneDetectorKeypointsMapVector[synth],curr_det);
        }
    }
}
void ImageRepresentation::SaveRegionsMichal(std::string fname, int mode) {
  std::vector<std::string> desc_names;
  for (std::map<std::string, AffineRegionVectorMap>::const_iterator
       reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
      for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
           desc_it != reg_it->second.end(); ++desc_it) {
          if (desc_it->first == "None") {
              continue;
            }
          desc_names.push_back(desc_it->first);
        }
    }
  for (unsigned int desc_num = 0; desc_num < desc_names.size(); desc_num++) {
      std::string current_desc_name = desc_names[desc_num];
      std::ofstream kpfile(fname + current_desc_name);
      if (mode == ios::binary) {
          if (kpfile.is_open()) {
              //   int magic = '\1ffa';
              //   kpfile.write((char *) &magic, sizeof(int));

              int num_keys = GetDescriptorsNumber(current_desc_name);
              //  kpfile.write((char *) &num_keys, sizeof(int));
              if (num_keys == 0)
                {
                  std::cerr << "No keypoints detected" << std::endl;
                  kpfile.close();
                  continue;
                }
              //    std::cerr << num_keys << std::endl;
              int desc_dim;

              for (std::map<std::string, AffineRegionVectorMap>::const_iterator
                   reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end(); ++reg_it) {
                  for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
                       desc_it != reg_it->second.end(); ++desc_it) {
                      if (desc_it->first != current_desc_name) {
                          continue;
                        }
                      if (desc_it->second.size() == 0)
                        continue;
                      desc_dim = desc_it->second[0].desc.vec.size();
                    }
                }
              if (desc_dim == 0) {
                  std::cerr << "All descriptors are empty" << std::endl;
                  kpfile.close();
                  continue;
                }
              //  kpfile.write((char *) &desc_dim, sizeof(int));
              //  std::cerr << desc_dim << std::endl;
              //              int img_w = OriginalImg.cols;
              //            kpfile.write((char *) &img_w, sizeof(int));
              // std::cerr << img_w << std::endl;
              //          int img_h = OriginalImg.rows;
              //        kpfile.write((char *) &img_h, sizeof(int));
              // std::cerr << img_h << std::endl;

              for (std::map<std::string, AffineRegionVectorMap>::const_iterator
                   reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end(); ++reg_it) {
                  for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
                       desc_it != reg_it->second.end(); ++desc_it) {
                      if (desc_it->first != current_desc_name) {
                          continue;
                        }
                      int n_desc = desc_it->second.size();

                      for (int i = 0; i < n_desc; i++) {
                          AffineRegion ar = desc_it->second[i];
                          saveARMikBinary(ar, kpfile);
                        }
                    }
                }
            }
          else {
              std::cerr << "Cannot open file " << fname << " to save keypoints" << endl;
            }
          kpfile.close();
          //      std::cerr << "END OF FILE" << std::endl;
        } else {


          if (kpfile.is_open()) {

              int num_keys = GetDescriptorsNumber(current_desc_name);
              kpfile << "128" << std::endl;
              kpfile << num_keys << std::endl;
              if (num_keys == 0)
                {
                  std::cerr << "No keypoints detected" << std::endl;
                  kpfile.close();
                  continue;
                }
              int desc_dim;

              for (std::map<std::string, AffineRegionVectorMap>::const_iterator
                   reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end(); ++reg_it) {
                  for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
                       desc_it != reg_it->second.end(); ++desc_it) {
                      if (desc_it->first != current_desc_name) {
                          continue;
                        }
                      int n_desc = desc_it->second.size();

                      for (int i = 0; i < n_desc; i++) {
                          AffineRegion ar = desc_it->second[i];
                          saveAR_KM_format(ar, kpfile);
                        }
                    }
                }
            }
        }
    }
}

void ImageRepresentation::SaveRegions(std::string fname, int mode) {
  std::ofstream kpfile(fname);
  if (mode == ios::binary) {

    } else {
      if (kpfile.is_open()) {
          //    std::map<std::string, AffineRegionVectorMap>::iterator regions_it;
          //    AffineRegionVectorMap::iterator desc_it;
          kpfile << RegionVectorMap.size() << std::endl;
          for (std::map<std::string, AffineRegionVectorMap>::const_iterator
               reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
              kpfile << reg_it->first << " " << reg_it->second.size() << std::endl;
              std::cerr << reg_it->first << " " << reg_it->second.size() << std::endl;

              for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
                   desc_it != reg_it->second.end(); ++desc_it) {
                  kpfile << desc_it->first << " " << desc_it->second.size() << std::endl;
                  int n_desc = desc_it->second.size();
                  if (n_desc > 0) {
                      kpfile << (desc_it->second)[0].desc.vec.size() << std::endl;
                    } else {
                      std::cerr << "No descriptor " << desc_it->first << std::endl;
                    }
                  for (int i = 0; i < n_desc ; i++ ) {
                      AffineRegion ar = desc_it->second[i];
                      saveAR(ar, kpfile);
                      kpfile << std::endl;
                    }
                }
            }
        }
      else {
          std::cerr << "Cannot open file " << fname << " to save keypoints" << endl;
        }
      kpfile.close();
    }
}

void ImageRepresentation::SaveRegionsNPZ(std::string fname) {

  int desc_dim = -1;

  int num_desc_dets = 0;
  for (std::map<std::string, AffineRegionVectorMap>::const_iterator
       reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
      for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
           desc_it != reg_it->second.end(); ++desc_it) {
          if (desc_it->first == "None") continue;
          num_desc_dets+=desc_it->second.size();
          if (desc_dim == -1){
              desc_dim = desc_it->second[0].desc.vec.size();
            }
          assert (desc_dim ==desc_it->second[0].desc.vec.size() );
        }
    }

  std::vector<double> xy(2*num_desc_dets);
  std::vector<double> scales(num_desc_dets);
  std::vector<double> responses(num_desc_dets);
  std::vector<double> A(4*num_desc_dets);
  std::vector<uchar> descs(desc_dim*num_desc_dets);

  int count = 0;
  for (std::map<std::string, AffineRegionVectorMap>::const_iterator
       reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {

      for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
           desc_it != reg_it->second.end(); ++desc_it) {
          if (desc_it->first == "None") continue;
          int n_desc = desc_it->second.size();
          for (int i = 0; i < n_desc ; i++ ) {
              AffineRegion ar = desc_it->second[i];
              xy[2*count] = ar.reproj_kp.x;
              xy[2*count+1] = ar.reproj_kp.y;
              scales[count] = ar.reproj_kp.s;
              A[4*count] = ar.reproj_kp.a11;
              A[4*count+1] = ar.reproj_kp.a12;
              A[4*count+2] = ar.reproj_kp.a21;
              A[4*count+3] = ar.reproj_kp.a22;

              responses[count] = ar.det_kp.response;

              for (int di = 0; di < desc_dim ; di++ ) {
                  descs[count*desc_dim + di] = (uchar)ar.desc.vec[di];
                }
              count++;
            }
        }
    }

  cnpy::npz_save(fname,"xy",&xy[0],{num_desc_dets,2},"w"); //"w" overwrites any existing file
  cnpy::npz_save(fname,"scales",&scales[0],{num_desc_dets,1},"a"); //"a" appends to the file we created above
  cnpy::npz_save(fname,"responses",&responses[0],{num_desc_dets,1},"a"); //"a" appends to the file we created above
  cnpy::npz_save(fname,"A",&A[0],{num_desc_dets,4},"a"); //"a" appends to the file we created above
  cnpy::npz_save(fname,"descs",&descs[0],{num_desc_dets,desc_dim},"a"); //"a" appends to the file we created above


}
void ImageRepresentation::LoadRegions(std::string fname) {
  std::ifstream kpfile(fname);
  if (kpfile.is_open()) {
      int numberOfDetectors = 0;
      kpfile >> numberOfDetectors;
      //    std::cerr << "numberOfDetectors=" <<numberOfDetectors << std::endl;
      for (int det = 0; det < numberOfDetectors; det++) {
          std::string det_name;
          int num_of_descs = 0;
          kpfile >> det_name;
          kpfile >> num_of_descs;
          //      std::cerr << det_name << " " << num_of_descs << std::endl;

          //reg_it->first << " " << reg_it->second.size() << std::endl;
          for (int desc = 0; desc < num_of_descs; desc++)  {
              AffineRegionVector desc_regions;
              std::string desc_name;
              kpfile >> desc_name;

              int num_of_kp = 0;
              kpfile >> num_of_kp;
              int desc_size;
              kpfile >> desc_size;
              //        std::cerr << desc_name << " " << num_of_kp << " " << desc_size << std::endl;
              for (int kp = 0; kp < num_of_kp; kp++)  {
                  AffineRegion ar;
                  loadAR(ar, kpfile);
                  desc_regions.push_back(ar);
                }
              AddRegions(desc_regions,det_name,desc_name);
            }
        }
    }
  else {
      std::cerr << "Cannot open file " << fname << " to save keypoints" << endl;
    }
  kpfile.close();
}
 AffineRegionVector ImageRepresentation::PreLoadRegionsNPZ(std::string fname) {
  cnpy::npz_t my_npz = cnpy::npz_load(fname);
  std::vector<std::string> keys;
  bool A_is_here = false;
  bool angle_is_here = false;

  for(map<std::string,cnpy::NpyArray>::iterator it = my_npz.begin(); it != my_npz.end(); ++it) {
      keys.push_back(it->first);
      if (it->first == "A") {
          A_is_here = true;
        }
      if (it->first == "angles") {
          angle_is_here = true;
        }
    }


  cnpy::NpyArray arr_xy = my_npz["xy"];
  double* xy_ = arr_xy.data<double>();

  cnpy::NpyArray arr_scales = my_npz["scales"];
  double* scales_ = arr_scales.data<double>();


  cnpy::NpyArray arr_resps = my_npz["responses"];
  double* responses_ = arr_resps.data<double>();

  cnpy::NpyArray arr_descs = my_npz["descs"];
  uchar* descs_ = arr_descs.data<uchar>();

  int num_of_kp = arr_xy.shape[0];
  int desc_dim = arr_descs.shape[1];

  assert( arr_xy.shape[0] ==  arr_scales.shape[0]);
  assert( arr_scales.shape[0] ==  arr_descs.shape[0]);



  AffineRegionVector desc_regions;
  std::string det_name = "ReadAffs";
  std::string desc_name = "ZMQ";
  if (A_is_here) { // save affine matrix
      cnpy::NpyArray arr_A = my_npz["A"];
      assert( arr_A.shape[0] ==  arr_descs.shape[0]);

      double* A_ = arr_A.data<double>();

      for (int kp = 0; kp < num_of_kp; kp++)  {
          AffineRegion ar;
          ar.det_kp.x = xy_[2*kp];
          ar.det_kp.y = xy_[2*kp+1];

          ar.det_kp.s = scales_[kp];
          ar.det_kp.a11 = A_[4*kp];
          ar.det_kp.a12 = A_[4*kp+1];
          ar.det_kp.a21 = A_[4*kp+2];
          ar.det_kp.a22 = A_[4*kp+3];

          ar.det_kp.response = responses_[kp];
          ar.type = DET_READ;
          ar.reproj_kp.x = xy_[2*kp];
          ar.reproj_kp.y = xy_[2*kp+1];

          ar.reproj_kp.s = scales_[kp];
          ar.reproj_kp.a11 = A_[4*kp];
          ar.reproj_kp.a12 = A_[4*kp+1];
          ar.reproj_kp.a21 = A_[4*kp+2];
          ar.reproj_kp.a22 = A_[4*kp+3];
          ar.reproj_kp.response = responses_[kp];
          ar.type = DET_READ;

          ar.desc.type = DESC_ZMQ;
          ar.desc.vec.resize(desc_dim);
          for (int dd=0; dd < desc_dim; dd++){
              ar.desc.vec[dd] = descs_[kp*desc_dim + dd];
            }
          desc_regions.push_back(ar);
        }

    } else if (angle_is_here){ //save orientation
      cnpy::NpyArray arr_angles = my_npz["angles"];
      assert( arr_angles.shape[0] ==  arr_descs.shape[0]);

      double* angles_ = arr_angles.data<double>();

      for (int kp = 0; kp < num_of_kp; kp++)  {
          AffineRegion ar;
          double angle = angles_[kp]*M_PI/180.0;
          ar.det_kp.x = xy_[2*kp];
          ar.det_kp.y = xy_[2*kp+1];

          ar.det_kp.s = scales_[kp];
          ar.det_kp.a11 = cos(angle);
          ar.det_kp.a12 = sin(angle);
          ar.det_kp.a21 = -sin(angle);
          ar.det_kp.a22 = cos(angle);
          ar.det_kp.response = responses_[kp];
          ar.type = DET_READ;
          ar.reproj_kp.x = xy_[2*kp];
          ar.reproj_kp.y = xy_[2*kp+1];

          ar.reproj_kp.s = scales_[kp];
          ar.reproj_kp.a11 = cos(angle);
          ar.reproj_kp.a12 = sin(angle);
          ar.reproj_kp.a21 = -sin(angle);
          ar.reproj_kp.a22 = cos(angle);
          ar.reproj_kp.response = responses_[kp];
          ar.type = DET_READ;

          ar.desc.type = DESC_ZMQ;
          ar.desc.vec.resize(desc_dim);
          for (int dd=0; dd < desc_dim; dd++){
              ar.desc.vec[dd] = descs_[kp*desc_dim + dd];
            }
          desc_regions.push_back(ar);
        }
    } else { //circular upright
      for (int kp = 0; kp < num_of_kp; kp++)  {
          AffineRegion ar;
          double angle = 0;
          ar.det_kp.x = xy_[2*kp];
          ar.det_kp.y = xy_[2*kp+1];

          ar.det_kp.s = scales_[kp];
          ar.det_kp.a11 = cos(angle);
          ar.det_kp.a12 = sin(angle);
          ar.det_kp.a21 = -sin(angle);
          ar.det_kp.a22 = cos(angle);
          ar.det_kp.response = responses_[kp];
          ar.type = DET_READ;
          ar.reproj_kp.x = xy_[2*kp];
          ar.reproj_kp.y = xy_[2*kp+1];

          ar.reproj_kp.s = scales_[kp];
          ar.reproj_kp.a11 = cos(angle);
          ar.reproj_kp.a12 = sin(angle);
          ar.reproj_kp.a21 = -sin(angle);
          ar.reproj_kp.a22 = cos(angle);
          ar.reproj_kp.response = responses_[kp];
          ar.type = DET_READ;

          ar.desc.type = DESC_ZMQ;
          ar.desc.vec.resize(desc_dim);
          for (int dd=0; dd < desc_dim; dd++){
              ar.desc.vec[dd] = descs_[kp*desc_dim + dd];
            }
          desc_regions.push_back(ar);
        }
    }
  return desc_regions;


}
void ImageRepresentation::LoadRegionsNPZ(std::string fname) {

  AffineRegionVector avr = PreLoadRegionsNPZ(fname);
  AddRegions(avr,"ReadAffs", "ZMQ");

}

void ImageRepresentation::SaveDescriptorsBenchmark(std::string fname1) {
  std::vector<std::string> desc_names;
  int num_keys  = 0;
  std::ofstream kpfile(fname1);
  if (kpfile.is_open()) {
      for (std::map<std::string, AffineRegionVectorMap>::const_iterator
           reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
          for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
               desc_it != reg_it->second.end(); ++desc_it) {

              if (desc_it->first == "None") {
                  continue;
                }
              num_keys += desc_it->second.size();
            }
        }

      std::cerr << num_keys << std::endl;
      for (std::map<std::string, AffineRegionVectorMap>::const_iterator
           reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
          for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
               desc_it != reg_it->second.end(); ++desc_it) {

              if (desc_it->first == "None") {
                  continue;
                }
              //   int num_keys = desc_it->second.size();
              for (int i = 0; i < num_keys ; i++ ) {
                  AffineRegion ar = desc_it->second[i];
                  for (int ddd = 0; ddd < ar.desc.vec.size(); ++ddd){
                      kpfile << ar.desc.vec[ddd] << " ";
                    }
                  kpfile << std::endl;
                }
            }
        }
    }  else {
      std::cerr << "Cannot open file " << fname1 << " to save keypoints" << endl;
    }
  kpfile.close();
}
void ImageRepresentation::SaveRegionsBenchmark(std::string fname1, std::string fname2) {
  std::vector<std::string> desc_names;

  std::ofstream kpfile(fname1);
  std::ofstream kpfile2(fname2);
  int num_keys = 0;
  if (kpfile.is_open() && kpfile2.is_open() ) {
      for (std::map<std::string, AffineRegionVectorMap>::const_iterator
           reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
          for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
               desc_it != reg_it->second.end(); ++desc_it) {

              if (desc_it->first != "None") {
                  continue;
                }
              num_keys += desc_it->second.size();
            }
        }
      kpfile << num_keys << std::endl;
      kpfile2 << num_keys << std::endl;

      for (std::map<std::string, AffineRegionVectorMap>::const_iterator
           reg_it = RegionVectorMap.begin(); reg_it != RegionVectorMap.end();  ++reg_it) {
          for (AffineRegionVectorMap::const_iterator desc_it = reg_it->second.begin();
               desc_it != reg_it->second.end(); ++desc_it) {

              if (desc_it->first != "None") {
                  continue;
                }
              int num_keys1 = desc_it->second.size();

              for (int i = 0; i < num_keys1 ; i++ ) {
                  AffineRegion ar = desc_it->second[i];
                  saveARBench(ar, kpfile,kpfile2);
                  kpfile << std::endl;
                  kpfile2 << std::endl;
                }
            }

        }
    }
  else {
      std::cerr << "Cannot open file " << fname1 << " to save keypoints" << endl;
    }
  kpfile.close();
  kpfile2.close();

}
//}


