#ifndef DETECTORS_PARAMETERS_HPP
#define DETECTORS_PARAMETERS_HPP

#include "structures.hpp"
#include "mser/extrema/extremaParams.h"
#include "affinedetectors/scale-space-detector.hpp"

struct ReadAffsFromFileParams {
    std::string fname;
    ReadAffsFromFileParams() {
        fname="";
    }
};

struct ORBParams
{
    int nfeatures;
    float scaleFactor;
    int nlevels;
    int edgeThreshold;
    int firstLevel;
    int WTA_K;
    PatchExtractionParams PEParam;
    bool doBaumberg;
    int doNMS;
    //  int patchSize;
    //  double mrSize;
    //  bool FastPatchExtraction;
    //  bool photoNorm;
    ORBParams()
    {
        doBaumberg = false;
        nfeatures = 500;
        scaleFactor = 1.2;
        nlevels = 8;
        edgeThreshold = 31;
        firstLevel = 0;
        WTA_K=2;
        doNMS=1;
        //    patchSize=31;
        //    mrSize = 3.0*sqrt(3.0);
        //    FastPatchExtraction = false;
        //    photoNorm =false;
    }
};


struct DetectorsParameters
{
    extrema::ExtremaParams MSERParam;
    ScaleSpaceDetectorParams HessParam;
    ScaleSpaceDetectorParams HarrParam;
    ScaleSpaceDetectorParams DoGParam;
    zmqDescriptorParams AffNetParam;
    zmqDescriptorParams OriNetParam;
    ReadAffsFromFileParams ReadAffsFromFileParam;
    AffineShapeParams BaumbergParam;
    ORBParams ORBParam;
};


#endif // DETECTORS_PARAMETERS_HPP
