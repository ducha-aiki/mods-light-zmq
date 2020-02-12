#ifndef DESCRIPTORS_PARAMETERS_HPP
#define DESCRIPTORS_PARAMETERS_HPP

#include "detectors/structures.hpp"
#include "matching/siftdesc.h"

struct DominantOrientationParams {

  int maxAngles;
  float threshold;
  bool addUpRight;
  bool halfSIFTMode;
  bool useTS;
  bool addMirrored;
  std::string  external_command;
  PatchExtractionParams PEParam;
  DominantOrientationParams() {
    maxAngles = -1;
    threshold = 0.8;
    addUpRight = false;
    halfSIFTMode = false;
    useTS = false;
    addMirrored = false;
    external_command = "";
  }
};

struct CLIDescriptorParams {
    PatchExtractionParams PEParam;
    std::string runfile;
    std::string hardcoded_input_fname;
    std::string hardcoded_output_fname;
    bool hardcoded_run_string;
    CLIDescriptorParams() {
     hardcoded_run_string = true;
    }


};



struct DescriptorsParameters {
  CLIDescriptorParams CLIDescParam;
  SIFTDescriptorParams SIFTParam;  
  SIFTDescriptorParams RootSIFTParam;
  SIFTDescriptorParams HalfSIFTParam;
  SIFTDescriptorParams HalfRootSIFTParam;
  //zmqDescriptorParams zmqDescParam;
  torchscriptDescriptorParams torchDescParam;

};

#endif // DESCRIPTORS_PARAMETERS_HPP
