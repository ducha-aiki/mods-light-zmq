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
};


#endif // DETECTORS_PARAMETERS_HPP
