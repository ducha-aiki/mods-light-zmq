/*------------------------------------------------------*/
/* Copyright 2013, Dmytro Mishkin  ducha.aiki@gmail.com */
/*------------------------------------------------------*/

#undef __STRICT_ANSI__
#include <fstream>
#include <string>
#include <iomanip>
#include <sys/time.h>
#include <map>

#include "io_mods.h"

#include "detectors/mser/extrema/extrema.h"
#include "detectors/helpers.h"
#include "matching/siftdesc.h"
#include "synth-detection.hpp"

#include "detectors/affinedetectors/scale-space-detector.hpp"
#include "detectors/detectors_parameters.hpp"
#include "descriptors_parameters.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "matching.hpp"

#include "configuration.hpp"
#include "imagerepresentation.h"
#include "correspondencebank.h"

#ifdef WITH_ORSA
#include "orsa.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <sys/stat.h>
#include <unistd.h>

inline bool exists_test3 (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

using namespace std;

const int nn_n = 50; //number of nearest neighbours retrieved to get 1st inconsistent
inline static bool endsWith(const std::string& str, const std::string& suffix)
{
  return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}


int main(int argc, char **argv)
{
  if ((argc < 4))
    {
      std::cerr << " ************************************************************************** " << std::endl
                << " ******** Two-view Matching with On-Demand Synthesis ********************** " << std::endl
                << " ************************************************************************** " << std::endl
                << "Usage: " << argv[0] << " imfnames.txt out_keys.txt config_iter.ini iters.ini" << std::endl
                << "- imfnames.txt: input images list, one line per imge " << std::endl
                << "- out_keys.txt: file with output file names, one per input image." << std::endl
                << "- config_iter.ini: input file with detectors and descriptors paramaters [optional, default = 'config_iter.ini'] " << std::endl
                << "- iters.ini: input file with parameters of iterative view synthesis [optional, default= 'iters.ini']" << std::endl
                << " ******************************************************************************* " << std::endl;
      return 1;
    }
  long c_start = getMilliSecs();
  double time1;
  TimeLog TimingLog;
  logs log1;
  /// Parameters reading
  configs Config1;
  if (getCLIparamExtractFeatures(Config1,argc,argv)) return 1;
  int VERB = Config1.OutputParam.verbose;

  ///
  std::map<std::string, torch::jit::script::Module> CNN_models;

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load(Config1.DetectorsPars.AffNetParam.path_to_model);

    torch::DeviceType device_type;
    if (Config1.DetectorsPars.AffNetParam.onGPU ) {
      device_type = torch::kCUDA;
    } else {
      device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    module.to(device);
    CNN_models["AffNet"] = module;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the Affnet model" <<Config1.DetectorsPars.AffNetParam.path_to_model <<  "\n";

  }




  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load(Config1.DetectorsPars.OriNetParam.path_to_model);
    torch::DeviceType device_type;
    if (Config1.DetectorsPars.OriNetParam.onGPU ) {
      device_type = torch::kCUDA;
    } else {
      device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    module.to(device);
    CNN_models["OriNet"] = module;
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the OriNet model" <<Config1.DetectorsPars.OriNetParam.path_to_model <<  "\n";

  }



  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    torch::jit::script::Module module = torch::jit::load(Config1.DescriptorPars.torchDescParam.path_to_model);
    torch::DeviceType device_type;
    if (Config1.DescriptorPars.torchDescParam.onGPU ) {
      device_type = torch::kCUDA;
    } else {
      device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    module.to(device);
    CNN_models["TorchScriptDescriptor"] = module;
  }
  catch (const c10::Error& e) {
      std::cerr << "error loading the TorchScriptDescriptor model" <<Config1.DescriptorPars.torchDescParam.path_to_model <<  "\n";


  }
  std::vector<std::string> input_img_fnames;
  std::vector<std::string> output_fnames;

  std::ifstream imfile(Config1.CLIparams.img1_fname);
  std::string str;

  while (std::getline(imfile, str))
    {
      input_img_fnames.push_back(str);
    }
  std::ifstream outfile(Config1.CLIparams.k1_fname);
  std::string str2;
  while (std::getline(outfile, str2))
    {
      output_fnames.push_back(str2);
    }
  if (output_fnames.size() != input_img_fnames.size()) {
      std::cerr <<  "Length of input and output file lists are not equal" << input_img_fnames.size() << " " << output_fnames.size() << std::endl;
      return 1;
    }
      /// Data structures preparation
      ImageRepresentation ImgRep1;
  for (int file_idx = 0; file_idx < output_fnames.size(); file_idx++) {
      std::string curr_img_fname = input_img_fnames[file_idx];
      std::string out_fname = output_fnames[file_idx];
      std::cout <<  file_idx << " " << curr_img_fname << " " << out_fname << " " << std::endl;
      if (exists_test3(out_fname + "ZMQ")) {
         std::cout <<   out_fname << " exists, skip" << std::endl;
         continue;
       }
      if (exists_test3(out_fname )) {
         
         std::cout <<   out_fname << " exists, skip" << std::endl;
         continue;
       }
      /// Input images reading
      cv::Mat img1;
      SynthImage tilt_img1;

      tilt_img1.id=0;
      img1 = cv::imread(curr_img_fname,Config1.LoadColor); // load grayscale; Try RGB?
      if(!img1.data) {
          std::cerr <<  "Could not open or find the image1 " << curr_img_fname<< std::endl;
          continue;
        }
      ///
     ImgRep1.Clear();
     ImgRep1.SetImg(img1, curr_img_fname);

      /// Affine regions detection
      std::cerr << "View synthesis, detection and description..." << endl;
#ifdef _OPENMP
      omp_set_nested(1);
#endif
      ImgRep1.SynthDetectDescribeKeypoints(Config1.ItersParam[0],
          Config1.DetectorsPars,
          Config1.DescriptorPars,
          Config1.DomOriPars,CNN_models);

      TimeLog img1time = ImgRep1.GetTimeSpent();
      /// Writing images and logs
      std::cerr << "Writing files... " << endl;




      if (Config1.OutputParam.outputMikFormat) {
          ImgRep1.SaveRegionsMichal(out_fname, 123);
        }  else {
          if (endsWith(out_fname,".npz")){
              ImgRep1.SaveRegionsNPZ(out_fname);
            } else {
              ImgRep1.SaveRegions(out_fname,0);

            }

//          ImgRep1.SaveRegionsAMatrix(out_fname);
        }
    }
  return 0;
}


