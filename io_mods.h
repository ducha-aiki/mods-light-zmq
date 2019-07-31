//
// Created by old-ufo on 3/30/15.
//
#ifndef MODS_NEW_IO_MODS_H
#define MODS_NEW_IO_MODS_H

#include "configuration.hpp"
#include "detectors/detectors_parameters.hpp"
#include "descriptors_parameters.hpp"
#include "matching.hpp"
#include "inih/cpp/INIReader.h"

const int Tmin = 9;//minimum number of command-line parameters

struct configs
{
    int n_threads;
    DescriptorsParameters DescriptorPars;
    DetectorsParameters DetectorsPars;
    DominantOrientationParams DomOriPars;
    int LoadColor;
    MatchPars Matchparam;
    RANSACPars RANSACParam;
    std::vector<IterationViewsynthesisParam> ItersParam;
    parameters CLIparams;
    filteringParams FilterParam;
    drawingParams DrawParam;
    outputParams OutputParam;

    bool read_pre_extracted;
    bool match_one_to_many;
    string descriptor, matching_lib, verification_type;
    configs()
    {
        n_threads = 1;
        LoadColor = 0;
        read_pre_extracted = false;
        match_one_to_many = false;

    }
};

void WriteLog(logs log, ostream& out);
void WriteTimeLog(TimeLog log, ostream &out,
                  const int writeRelValues = 1,
                  const int writeAbsValues = 0,
                  const int writeDescription = 0);
void GetMSERPars(extrema::ExtremaParams &MSERPars, INIReader &reader,const char* section="MSER");
void GetCLIDescPars(CLIDescriptorParams &pars, INIReader &reader,const char* section = "CLIDescriptor");
void GetReadPars(ReadAffsFromFileParams &pars, INIReader &reader,const char* section="ReadAffs");
void GetHessPars(ScaleSpaceDetectorParams &HessPars, INIReader &reader,const char* section="HessianAffine");
void GetORBPars(ORBParams &pars, INIReader &reader,const char* section="ORB");
void GetPatchExtractionPars(PatchExtractionParams &PEPars, INIReader &reader,const char* section);
void GetHarrPars(ScaleSpaceDetectorParams &HarrPars, INIReader &reader,const char* section="HarrisAffine");
void GetDoGPars(ScaleSpaceDetectorParams &DoGPars, INIReader &reader,const char* section="DoG");
void GetDomOriPars(DominantOrientationParams &DomOriPars, INIReader &reader,const char* section="DominantOrientation");
void GetBaumbergPars(AffineShapeParams &pars, INIReader &reader,const char* section="AffineAdaptation");
void GetSaddlePars(SaddleParams &pars, INIReader &reader,const char* section="Saddle");
void GetZMQPars(zmqDescriptorParams &pars, INIReader &reader,const char* section="zmqDescriptor");
void GetMatchPars(MatchPars &pars, INIReader &reader, INIReader &iter_reader, const char* section="Matching");
void GetSIFTDescPars(SIFTDescriptorParams &pars, INIReader &reader,const char* section="SIFTDescriptor");
void GetRANSACPars(RANSACPars &pars, INIReader &reader,const char* section="RANSAC");
void GetIterPars(std::vector<IterationViewsynthesisParam> &pars, INIReader &reader);
int getCLIparam(configs &conf1,int argc, char **argv);
int getCLIparamExtractFeatures(configs &conf1,int argc, char **argv);
int getCLIparamExtractFeaturesBenchmark(configs &conf1,int argc, char **argv);
int getCLIparamExportDescriptorsBenchmark(configs &conf1,int argc, char **argv);


#endif //MODS_NEW_IO_MODS_H
