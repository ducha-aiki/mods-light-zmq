/*
 * fast.hpp
 *
 *  Created on: Oct 13, 2015
 *      Author: aldanjav
 */

#ifndef SRC_FAST_HPP_
#define SRC_FAST_HPP_

#include <opencv2/features2d/features2d.hpp>
#include "sorb.h"

using namespace cv;

namespace cmp{

class CV_EXPORTS_W FastFeatureDetector
{
public:

    enum
    { // Define it in old class to simplify migration to 2.5
      TYPE_5_8 = 0,
      TYPE_7_12 = 1,
      TYPE_9_16 = 2,
      TYPE_SADDLE_CENTRAL_PIXEL = 3,
      TYPE_SADDLE_INNER_PATTERN = 4,
      TYPE_SHADDLE = 5,
      TYPE_SADDLE_CONDITIONED = 6,
      TYPE_BLOB_CONDITIONED = 7,
      TYPE_SADDLE_BLOB = 8,
      TYPE_BLOB_CONDITIONED_OUTERTEST = 9
    };

    CV_WRAP void detect2( const Mat& image, CV_OUT vector<SadKeyPoint>& keypoints, Mat & resp, const Mat& mask=Mat() ) const;

    CV_WRAP FastFeatureDetector( int threshold=10, int nonmaxSuppression=2 );
   // AlgorithmInfo* info() const;

protected:
    virtual void detectImpl ( const Mat& image, vector<SadKeyPoint>& keypoints, const Mat& mask=Mat() ) const = 0;
    virtual void detectImpl2( const Mat& image, vector<SadKeyPoint>& keypoints, Mat& resp, const Mat& mask=Mat() ) const = 0;

    int threshold;
    int nonmaxSuppression;
};

class FastFeatureDetector2 : public cmp::FastFeatureDetector
{
public:
    CV_WRAP FastFeatureDetector2( int threshold=10, int nonmaxSuppression=2);
    CV_WRAP FastFeatureDetector2( int threshold, int nonmaxSuppression, int type);
    CV_WRAP FastFeatureDetector2( int threshold, int nonmaxSuppression, int type, float scale);
    CV_WRAP FastFeatureDetector2( int threshold, int nonmaxSuppression, int type, float scale, double thr);
    CV_WRAP FastFeatureDetector2( int threshold, int nonmaxSuppression, int type, float scale, double responsethr, uchar deltaThr);
    CV_WRAP FastFeatureDetector2( int threshold, int nonmaxSuppression, int type, float scale, double responsethr, uchar deltaThr, int scoreType);
    CV_WRAP FastFeatureDetector2( int threshold, int nonmaxSuppression, int type, float scale, double responsethr, uchar deltaThr, int scoreType,
                                  bool allC1feats, bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType, int minArcLength, int maxArcLength, uchar blobThr );
   // cv::AlgorithmInfo* info() const;


protected:
    virtual void detectImpl ( const Mat& image, vector<SadKeyPoint>& keypoints, const Mat& mask=Mat() ) const;
    virtual void detectImpl2( const Mat& image, vector<SadKeyPoint>& keypoints, Mat& resp, const Mat& mask=Mat() ) const;

    short type;
    float scale;
    double responsethr;
    uchar deltaThr;
    int scoreType;
    bool allC1feats;
    bool strictMaximum;
    int subPixPrecision;
    bool gravityCenter;
    int innerTstType;
    int minArcLength;
    int maxArcLength;
    uchar blobThr;

};

}//namespace cmp

#endif /* SRC_FAST_HPP_ */
