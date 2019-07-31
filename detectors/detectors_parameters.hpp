#ifndef DETECTORS_PARAMETERS_HPP
#define DETECTORS_PARAMETERS_HPP

#include "structures.hpp"
#include "mser/extrema/extremaParams.h"
#include "affinedetectors/scale-space-detector.hpp"
#include "detectors/saddle/lbq.h"

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


struct SaddleParams{
    bool doBaumberg;
    int doNMS;
    double respThreshold;
    int epsilon;
    int pyrLevels;
    double scalefac;
    int deltaThr;
    int edgeThreshold;
    int descSize;
    int WTA_K;
    int nfeatures;
    int scoreType;
    int gab;
    bool allC1feats;
    bool strictMaximum;
    int subPixPrecision;
    bool gravityCenter;
    int innerTstType;
    int minArcLength;
    int maxArcLength;
    short ringsType;
    int binPattern;
    float saddle_perc; //alpha
    PatchExtractionParams PEParam;
    SaddleParams() {
        allC1feats = false;
        doBaumberg = false;
        strictMaximum = false;
        subPixPrecision = 1;
        gravityCenter = false;
        PEParam.patchSize = descSize;
        PEParam.mrSize = 3.0;
        innerTstType = 1;
        doNMS = 1;
        gab = edgeThreshold;
        respThreshold = 0;
        epsilon = 1;
        pyrLevels = 8;
        scalefac = 1.3;
        deltaThr = 0;
        edgeThreshold = 3;
        descSize = 31;
        WTA_K = 2;
        nfeatures = 5000;
        scoreType = 1;
        minArcLength = 2;
        maxArcLength = 8;
        ringsType = 4;
         binPattern = Binpat::OCV;
         saddle_perc = 0.5;
    }
};


struct DetectorsParameters
{
    extrema::ExtremaParams MSERParam;
    ScaleSpaceDetectorParams HessParam;
    ScaleSpaceDetectorParams HarrParam;
    ScaleSpaceDetectorParams DoGParam;
    SaddleParams SaddleParam;
    zmqDescriptorParams AffNetParam;
    zmqDescriptorParams OriNetParam;
    ReadAffsFromFileParams ReadAffsFromFileParam;
    AffineShapeParams BaumbergParam;
    ORBParams ORBParam;
};


#endif // DETECTORS_PARAMETERS_HPP
