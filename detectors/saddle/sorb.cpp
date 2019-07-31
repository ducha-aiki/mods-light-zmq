/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/** Authors: Ethan Rublee, Vincent Rabaud, Gary Bradski */

#include <iterator>
#include "fast_score.hpp"
#include "fast.hpp"
#include <stdio.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "sorb.h"
#include "lbq.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace cmp
{

const float HARRIS_K = 0.04f;
const int DESCRIPTOR_SIZE = 32;
const bool VERBOSE = true;

/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in an image
 */
static void
HarrisResponses(const Mat& img, vector<SadKeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step/img.elemSize1());
    int r = blockSize/2;

    float scale = (1 << 2) * blockSize * 255.0f;
    scale = 1.0f / scale;
    float scale_sq_sq = scale * scale * scale * scale;

    AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ )
            ofs[i*blockSize + j] = (int)(i*step + j);

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = cvRound(pts[ptidx].pt.x - r);
        int y0 = cvRound(pts[ptidx].pt.y - r);

        const uchar* ptr0 = ptr00 + y0*step + x0;
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
        	const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;
        }
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static float IC_Angle(const Mat& image, const int half_k, Point2f pt,
                      const vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;
    // float mag_centroid;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -half_k; u <= half_k; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= half_k; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    // mag_centroid = sqrt((float)m_01*(float)m_01+(float)m_10*(float)m_10);
    // printf("%.3f\n", mag_centroid);
    return fastAtan2((float)m_01, (float)m_10);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void computeOrbDescriptor(const SadKeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc, int dsize, int WTA_K)
{
    float angle = kpt.angle;
    //angle = cvFloor(angle/12)*12.f;
    angle *= (float)(CV_PI/180.f);
    float a = (float)cos(angle), b = (float)sin(angle);

    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    int step = (int)img.step;

    float x, y;
    int ix, iy;
#if 1
    #define GET_VALUE(idx) \
           (x = pattern[idx].x*a - pattern[idx].y*b, \
            y = pattern[idx].x*b + pattern[idx].y*a, \
            ix = cvRound(x), \
            iy = cvRound(y), \
            *(center + iy*step + ix) )
#else
    #define GET_VALUE(idx) \
        (x = pattern[idx].x*a - pattern[idx].y*b, \
        y = pattern[idx].x*b + pattern[idx].y*a, \
        ix = cvFloor(x), iy = cvFloor(y), \
        x -= ix, y -= iy, \
        cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
#endif

    if( WTA_K == 2 )
    {
        for (int i = 0; i < dsize; ++i, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            desc[i] = (uchar)val;
        }
    }
    else if( WTA_K == 3 )
    {
        for (int i = 0; i < dsize; ++i, pattern += 12)
        {
            int t0, t1, t2, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
            val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

            t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

            t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

            t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
            val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

            desc[i] = (uchar)val;
        }
    }
    else if( WTA_K == 4 )
    {
        for (int i = 0; i < dsize; ++i, pattern += 16)
        {
            int t0, t1, t2, t3, u, v, k, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            t2 = GET_VALUE(2); t3 = GET_VALUE(3);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val = k;

            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            t2 = GET_VALUE(6); t3 = GET_VALUE(7);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val |= k << 2;

            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            t2 = GET_VALUE(10); t3 = GET_VALUE(11);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val |= k << 4;

            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            t2 = GET_VALUE(14); t3 = GET_VALUE(15);
            u = 0, v = 2;
            if( t1 > t0 ) t0 = t1, u = 1;
            if( t3 > t2 ) t2 = t3, v = 3;
            k = t0 > t2 ? u : v;
            val |= k << 6;

            desc[i] = (uchar)val;
        }
    }
    else
        CV_Error( CV_StsBadSize, "Wrong WTA_K. It can be only 2, 3 or 4." );

    #undef GET_VALUE
}


static void initializeOrbPattern( const Point* pattern0, vector<Point>& pattern, int ntuples, int tupleSize, int poolSize )
{
    RNG rng(0x12345678);
    int i, k, k1;
    pattern.resize(ntuples*tupleSize);

    for( i = 0; i < ntuples; i++ )
    {
        for( k = 0; k < tupleSize; k++ )
        {
            for(;;)
            {
                int idx = rng.uniform(0, poolSize);
                Point pt = pattern0[idx];
                for( k1 = 0; k1 < k; k1++ )
                    if( pattern[tupleSize*i + k1] == pt )
                        break;
                if( k1 == k )
                {
                    pattern[tupleSize*i + k] = pt;
                    break;
                }
            }
        }
    }
}


static void makeRandomPattern(int patchSize, Point* pattern, int npoints)
{
    RNG rng(0x34985739); // we always start with a fixed seed,
                         // to make patterns the same on each run
    for( int i = 0; i < npoints; i++ )
    {
        pattern[i].x = rng.uniform(-patchSize/2, patchSize/2+1);
        pattern[i].y = rng.uniform(-patchSize/2, patchSize/2+1);
    }
}


static inline float getScale(int level, int firstLevel, double scaleFactor)
{
    return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}

static inline double getScaleDouble(int level, int firstLevel, double scaleFactor)
{
    return std::pow(scaleFactor, (double)(level - firstLevel));
}

/** Constructor
 * @param detector_params parameters to use
 */
SORB::SORB(double _responseThr, float _scaleFactor, int _nlevels, int _edgeThreshold,
         int _epsilon, int _WTA_K, int _scoreType, int _patchSize, int _doNMS, int _descSize, uchar _deltaThr, int _nfeatures,
		 bool _allC1feats , bool _strictMaximum, int _subPixPrecision , bool _gravityCenter, int _innerTstType, int _minArcLength,
		 int _maxArcLength, short _ringsType, int _binPattern, float _alpha ) :
		 responseThr(_responseThr), scaleFactor(_scaleFactor), nlevels(_nlevels),
		 edgeThreshold(_edgeThreshold), epsilon(_epsilon), WTA_K(_WTA_K),
		 scoreType(_scoreType), patchSize(_patchSize), doNMS(_doNMS),
		 descSize(_descSize), deltaThr(_deltaThr), nfeatures(_nfeatures),
		 allC1feats(_allC1feats), strictMaximum(_strictMaximum), subPixPrecision(_subPixPrecision), gravityCenter(_gravityCenter),
		 innerTstType(_innerTstType), minArcLength(_minArcLength), maxArcLength(_maxArcLength), ringsType(_ringsType), 
         binPattern(_binPattern), alpha(_alpha)
{}

int SORB::descriptorSize() const
{
    return K_BYTES;
}

void SORB::setDescriptorSize(int dsize)
{
	descSize = dsize;
}

int SORB::descriptorType() const
{
    return CV_8U;
}

/** Compute the ORB features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 */
void SORB::operator()(InputArray image, InputArray mask, vector<SadKeyPoint>& keypoints) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}


/** Compute the ORB keypoint orientations
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the iamge (can be empty, but the computation will be slower)
 * @param scale the scale at which we compute the orientation
 * @param keypoints the resulting keypoints
 */
static void computeOrientation(const Mat& image, vector<SadKeyPoint>& keypoints,
                               int halfPatchSize, const vector<int>& umax)
{
    // Process each keypoint
    for (vector<SadKeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, halfPatchSize, keypoint->pt, umax);
    }
}

static void nmsVectorwiseDown(vector<SadKeyPoint>& kpquery, Mat& response, float sf)
{
	// Create the response image
	int xc, yc, numpts = (int)kpquery.size();
	double score, *p1, *p2, *p3;

    for (int iPt=0; iPt<numpts; iPt++)
    {
    	xc = (int)round(kpquery[iPt].pt.x/sf);
    	yc = (int)round(kpquery[iPt].pt.y/sf);
    	score = kpquery[iPt].response;

    	p1 = response.ptr<double>(yc-1);
    	p2 = response.ptr<double>(yc);
    	p3 = response.ptr<double>(yc+1);

    	if (score <= p1[xc-1] || score <= p1[xc] || score <= p1[xc+1] ||
    		score <= p2[xc-1] || score <= p2[xc] || score <= p2[xc+1] ||
			score <= p3[xc-1] || score <= p3[xc] || score <= p3[xc+1] )
    	{
    		kpquery.erase(kpquery.begin() + iPt );
    	}
    }
}

static void nmsVectorwiseUp(vector<SadKeyPoint>& kpcurr, vector<SadKeyPoint>& kpup, Mat& response, float sf )
{
	// Create the response image
	int xc, yc, numpts = kpcurr.size();
	double score, *p1, *p2, *p3;

	Mat r = Mat::zeros(response.rows, response.cols, CV_64F);

    for (int iPt=0; iPt<(int)kpup.size(); iPt++)
    {
    	xc = (int)round(kpup[iPt].pt.x/sf);
    	yc = (int)round(kpup[iPt].pt.y/sf);
    	score = kpup[iPt].response;

    	p2 = r.ptr<double>(yc);
    	if (score > p2[xc])
    		p2[xc] = score;
    }

    for (int iPt=0; iPt<(int)kpcurr.size(); iPt++)
	{
		xc = (int)round(kpcurr[iPt].pt.x);
		yc = (int)round(kpcurr[iPt].pt.y);
		score = kpcurr[iPt].response;

		p1 = r.ptr<double>(yc-1);
		p2 = r.ptr<double>(yc);
		p3 = r.ptr<double>(yc+1);

		if (score <= p1[xc-1] || score <= p1[xc] || score <= p1[xc+1] ||
			score <= p2[xc-1] || score <= p2[xc] || score <= p2[xc+1] ||
			score <= p3[xc-1] || score <= p3[xc] || score <= p3[xc+1] )
		{
			kpcurr.erase( kpcurr.begin() + iPt );
		}
	}
}

/** Compute the NMS along the image pyramid
 * @param
 */
static void nmsPyramid(vector<Mat>& respPyramid, vector<vector<SadKeyPoint> >& allKeypoints, float scaleFactor)
{
    // Process each keypoint
	int nlevels = (int)allKeypoints.size();

    // Intermediate levels
	for (int level = 1; level < nlevels-1; ++level)
	{
		vector<SadKeyPoint> & kptsCurr = allKeypoints[level];
		vector<SadKeyPoint> & kptsDown = allKeypoints[level+1];
		vector<SadKeyPoint> & kptsUp   = allKeypoints[level-1];
        
        if ((int)kptsCurr.size())
        {
            // One level DOWN 
            if ((int)kptsDown.size())
            	nmsVectorwiseDown(kptsCurr, respPyramid[level+1], scaleFactor);
            // One level UP
            if ((int)kptsUp.size())
            	nmsVectorwiseUp(kptsCurr, kptsUp, respPyramid[level], scaleFactor);
        }
	}

	// For the last level
	vector<SadKeyPoint> & kptsCurr2 = allKeypoints[nlevels-1];
	vector<SadKeyPoint> & kptsUp   = allKeypoints[nlevels-2];
	if ( nlevels>1 && (int)kptsCurr2.size() && (int)kptsUp.size() )
		nmsVectorwiseUp(kptsCurr2, kptsUp, respPyramid[nlevels-1], scaleFactor);

}

static void computeFeatResponse(vector<vector<SadKeyPoint> >& allKeypoints, int respType)
{
	int nOctaves = allKeypoints.size(), nFeats;
	switch (respType)
	{
		case SORB::SUMOFABS_SCORE:

			for (int iOctave = 0; iOctave<nOctaves; iOctave++)
			{
				nFeats = allKeypoints[iOctave].size();
				uchar lbs[16];
				uchar inten[16];
				double v;
				for (int iFeat = 0; iFeat<nFeats; iFeat++)
				{
					memcpy(lbs, allKeypoints[iOctave][iFeat].labels, 16*sizeof(uchar));
					memcpy(inten, allKeypoints[iOctave][iFeat].intensityPixels, 16*sizeof(uchar));

					double greenredsum=0;
					v = allKeypoints[iOctave][iFeat].intensityCenter;

					for (int iElem=0; iElem<16; iElem++)
					{
						if (lbs[iElem] != 0)
							greenredsum += abs((double)inten[iElem]-v);
					}
					allKeypoints[iOctave][iFeat].response = greenredsum;
				}
			}

			break;

		case SORB::AVGOFABS_SCORE:
			for (int iOctave = 0; iOctave<nOctaves; iOctave++)
			{
				nFeats = allKeypoints[iOctave].size();
				uchar lbs[16];
				uchar inten[16];
				double v;
				for (int iFeat = 0; iFeat<nFeats; iFeat++)
				{
					memcpy(lbs, allKeypoints[iOctave][iFeat].labels, 16*sizeof(uchar));
					memcpy(inten, allKeypoints[iOctave][iFeat].intensityPixels, 16*sizeof(uchar));

					double greenredsum=0;
					int greenrednum = 0;
					v = allKeypoints[iOctave][iFeat].intensityCenter;

					for (int iElem=0; iElem<16; iElem++)
					{
						if (lbs[iElem] != 0)
						{
							greenredsum += abs((double)inten[iElem]-v);
							greenrednum++;
						}
					}
					allKeypoints[iOctave][iFeat].response = greenredsum/greenrednum;
				}
			}
			break;
	}
}


/** Compute the ORB keypoints on an image
 * @param image_pyramid the image pyramid to compute the features and descriptors on
 * @param mask_pyramid the masks to apply at every level
 * @param keypoints the resulting keypoints, clustered per level
 *
 * Javier Aldana
 * IMPORTANT NOTE: For this function the parameter "nfeatures" is not any more the
 * maximum number of features, it is the square hessian response threshold.
 * The variable "firstLevel" is not anymore the index of the first level, now is
 * the threshold for significant brighter and darker so called epsilon.
 */

struct KeypointResponseGreater
{
    inline bool operator()(const SadKeyPoint& kp1, const SadKeyPoint& kp2) const
    {
        return kp1.response > kp2.response;
    }
};

struct KeypointResponseGreaterThanThreshold
{
    KeypointResponseGreaterThanThreshold(float _value) :
    value(_value)
    {
    }
    inline bool operator()(const SadKeyPoint& kpt) const
    {
        return kpt.response >= value;
    }
    float value;
};

static void retainBest(vector<SadKeyPoint>& keypoints, int n_points)
{
    //this is only necessary if the keypoints size is greater than the number of desired points.
    if( n_points >= 0 && keypoints.size() > (size_t)n_points )
    {
        if (n_points==0)
        {
            keypoints.clear();
            return;
        }
        //first use nth element to partition the keypoints into the best and worst.
        std::nth_element(keypoints.begin(), keypoints.begin() + n_points, keypoints.end(), cmp::KeypointResponseGreater());
        //this is the boundary response, and in the case of FAST may be ambigous
        float ambiguous_response = keypoints[n_points - 1].response;
        //use std::partition to grab all of the keypoints with the boundary response.
        vector<SadKeyPoint>::const_iterator new_end =
        std::partition(keypoints.begin() + n_points, keypoints.end(),
                       cmp::KeypointResponseGreaterThanThreshold(ambiguous_response));
        //resize the keypoints, given this new end point. nth_element and partition reordered the points inplace
        keypoints.resize(new_end - keypoints.begin());
    }
}

struct RoiPredicate
{
    RoiPredicate( const Rect& _r ) : r(_r)
    {}

    bool operator()( const SadKeyPoint& keyPt ) const
    {
        return !r.contains( keyPt.pt );
    }

    Rect r;
};

static void runByImageBorder( vector<SadKeyPoint>& keypoints, Size imageSize, int borderSize )
{
    if( borderSize > 0)
    {
        if (imageSize.height <= borderSize * 2 || imageSize.width <= borderSize * 2)
            keypoints.clear();
        else
            keypoints.erase( std::remove_if(keypoints.begin(), keypoints.end(),
                                       RoiPredicate(Rect(Point(borderSize, borderSize),
                                                         Point(imageSize.width - borderSize, imageSize.height - borderSize)))),
                             keypoints.end() );
    }
}

inline void mergeSaddlesAndBlobs(std::vector< SadKeyPoint > &  keypoints, int nFeatures, float alpha)
{
    vector<SadKeyPoint> saddleKeypoints, blobKeypoints; 

    saddleKeypoints.reserve(nFeatures*2);
    blobKeypoints.reserve(nFeatures*2);
    saddleKeypoints.clear();
    blobKeypoints.clear();

    for (vector<SadKeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        switch (keypoint->class_id)
        {
            case 0:
                saddleKeypoints.push_back(*keypoint);
                break;
            case 1:
            case 2:
                blobKeypoints.push_back(*keypoint);
                break;
            default:
                std::cerr << "Unknown region type (0,1,2)" << std::endl;
        }

    }
    // float alpha = (float)saddleKeypoints.size()/keypoints.size();
    int saddleNum = (int)round(nFeatures*alpha);
    int blobNum = (int)round(nFeatures*(1-alpha));

    if (saddleKeypoints.size()<saddleNum)
        blobNum += saddleNum - saddleKeypoints.size();

    if (blobKeypoints.size()<blobNum)
        saddleNum += blobNum - blobKeypoints.size();

    retainBest(saddleKeypoints, saddleNum);
    retainBest(blobKeypoints, blobNum);
    
    keypoints.clear();
    
    for (vector<SadKeyPoint>::iterator keypoint = saddleKeypoints.begin(),
         keypointEnd = saddleKeypoints.end(); keypoint != keypointEnd; ++keypoint)
        keypoints.push_back(*keypoint);

    for (vector<SadKeyPoint>::iterator keypoint = blobKeypoints.begin(),
         keypointEnd = blobKeypoints.end(); keypoint != keypointEnd; ++keypoint)
        keypoints.push_back(*keypoint);
}

void computeKeyPoints(const vector<Mat>& imagePyramid,
                             const vector<Mat>& maskPyramid,
							 vector<Mat>& respPyramid,
                             vector<vector<SadKeyPoint> >& allKeypoints,
                             double responseThr, int epsilon, float scaleFactor,
                             int edgeThreshold, int patchSize, int scoreType, int doNMS, uchar deltaThr, int nfeatures,
							 bool allC1feats, bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
							 int minArcLength, int maxArcLength, short ringsType, float alpha )
{

	int nlevels = (int)imagePyramid.size();
#if false
    printf("\nSADDLE detector parameters: \n   nLevels: %d, scaleFactor: %.1f, epsilon: %d, responseThr: %.2f, borderGab: %d, doNMS: %d\n   deltaThr: %d, nFeats: %d, allC1features: %d, strictMaxNMS: %d, subpixelMethod: %d\n   C1C2gravityCenter: %d, InnerTstMethod: %d, ScoreType: %d, minArc: %d, maxArc: %d\n   ringsType: %d, alpha: %3.2f\n",
				nlevels, scaleFactor, epsilon, responseThr, edgeThreshold, doNMS, deltaThr, nfeatures, allC1feats, strictMaximum, subPixPrecision, gravityCenter, innerTstType, scoreType, minArcLength, maxArcLength, ringsType, alpha );
#endif
    vector<int> nfeaturesPerLevel(nlevels);

    // fill the extractors and descriptors for the corresponding scales
    float factor = (float)(1.0 / scaleFactor);
    float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
        sumFeatures += nfeaturesPerLevel[level];
        ndesiredFeaturesPerScale *= factor;
    }
    nfeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // Pre-compute the end of a row in a circular patch
    int halfPatchSize = patchSize / 2;
    vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    allKeypoints.resize(nlevels);
    int num_for_take_more = 0;
    int taken_sum = 0;
    int needed_sum = 0;

    for (int level = nlevels - 1; level >= 0; level--)
    {
        int featuresNum = nfeaturesPerLevel[level] + num_for_take_more;
        allKeypoints[level].reserve(featuresNum*2);

        float sf = getScale(level, 0, scaleFactor);

        vector<SadKeyPoint> & keypoints = allKeypoints[level];

        // Detect SADDLE features
        FastFeatureDetector2 fd( epsilon, doNMS, ringsType, level,
                                 responseThr, deltaThr, scoreType,
								 allC1feats, strictMaximum, subPixPrecision,
                                 gravityCenter, innerTstType, minArcLength,
                                 maxArcLength );
        fd.detect2(imagePyramid[level], keypoints, respPyramid[level], maskPyramid[level]);

        
        // Remove keypoints very close to the border
        runByImageBorder(keypoints, imagePyramid[level].size(), edgeThreshold);

        //cull to the final desired level, using the new Harris scores or the original FAST scores.
        if (level == 0) {
            featuresNum = nfeatures - taken_sum;
          }
        
        mergeSaddlesAndBlobs(keypoints, featuresNum, alpha);

        taken_sum += (int)keypoints.size();
        needed_sum += nfeaturesPerLevel[level];
        if (taken_sum < needed_sum) {
            num_for_take_more = needed_sum  - taken_sum;
          }

        // Set the level of the coordinates
        for (vector<SadKeyPoint>::iterator keypoint = keypoints.begin(),
             keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->octave = level;
            keypoint->size *= sf;
        }
        computeOrientation(imagePyramid[level], keypoints, halfPatchSize, umax);

    }

    // Compute Non Maximum Suppression along the pyramid
    if (doNMS==2)
    	nmsPyramid( respPyramid, allKeypoints, scaleFactor);

}


/** Compute the ORB decriptors
 * @param image the image to compute the features and descriptors on
 * @param integral_image the integral image of the image (can be empty, but the computation will be slower)
 * @param level the scale at which we compute the orientation
 * @param keypoints the keypoints to use
 * @param descriptors the resulting descriptors
 */
static void computeDescriptors(const Mat& image, vector<SadKeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern, int dsize, int WTA_K)
{
    //convert to grayscale if more than one color
    CV_Assert(image.type() == CV_8UC1);
    //create the descriptor mat, keypoints.size() rows, BYTES cols
    descriptors = Mat::zeros((int)keypoints.size(), dsize, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i), dsize, WTA_K);
}


/** Compute the ORB features and descriptors on an image
 * @param img the image to compute the features and descriptors on
 * @param mask the mask to apply
 * @param keypoints the resulting keypoints
 * @param descriptors the resulting descriptors
 * @param do_keypoints if true, the keypoints are computed, otherwise used as an input
 * @param do_descriptors if true, also computes the descriptors
 *
 * Javier Aldana
 * IMPORTANT NOTES: The global variable "firstLevel" was suppressed from the constructor
 * and it is fixed inside this function
 */


void substract_images( Mat iMatlab )
{

	char imgpath[] = "/home/aldanjav/Work/Matlab_projects/Benchmark_Saddle/chessboard.ppm";
	Mat iMemory = cv::imread( imgpath, IMREAD_GRAYSCALE );

	Mat imDif;
  std::cout << "iMatlab, cols: " << iMatlab.cols << ", rows: " << iMatlab.rows << std::endl;
	std::cout << "iMemory, cols: " << iMemory.cols << ", rows: " << iMemory.rows << std::endl;

	if (iMatlab.cols==iMemory.cols && iMatlab.rows==iMemory.rows)
	{
		cv::absdiff(iMatlab, iMemory, imDif);
		Scalar errorSum = cv::sum(imDif);
		std::cout << "Error pixel-wise: " << errorSum.val[0] << std::endl;
	}
	else
		std::cout << "Dimension mismatch between images" << std::endl;

}


void SORB::operator()( InputArray _image, InputArray _mask, vector<SadKeyPoint>& _keypoints,
                      OutputArray _descriptors, bool useProvidedKeypoints) const
{
    CV_Assert(patchSize >= 2);

    int firstLevel = 0;
    bool do_keypoints = !useProvidedKeypoints;
    bool do_descriptors = _descriptors.needed();
    vector<float> errorResize( this->nlevels );

    if( (!do_keypoints && !do_descriptors) || _image.empty() )
        return;

    //ROI handling
    const int HARRIS_BLOCK_SIZE = 9;
    int halfPatchSize = patchSize / 2;
    int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE/2))+1;

    Mat image = _image.getMat(), mask = _mask.getMat();

    if( image.type() != CV_8UC1 )
        cvtColor(_image, image, CV_BGR2GRAY);

    int levelsNum = this->nlevels;

    if( !do_keypoints )
    {
        levelsNum = 0;
        for( size_t i = 0; i < _keypoints.size(); i++ )
            levelsNum = std::max(levelsNum, std::max(_keypoints[i].octave, 0));
        levelsNum++;
    }

    // float sigma = 1.5;

    // Pre-compute the scale pyramids
    vector<Mat> imagePyramid(levelsNum), maskPyramid(levelsNum), respPyramid(levelsNum);
    for (int level = 0; level < levelsNum; ++level)
    {
        float scale = 1/getScale(level, firstLevel, scaleFactor);
        Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));
        Size wholeSize(sz.width + border*2, sz.height + border*2);
        Mat temp(wholeSize, image.type()), masktemp;

        imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));
        respPyramid[level] = Mat::zeros(sz, CV_64F);
        errorResize[level] = image.cols*scale - cvRound(image.cols*scale);

// #if false
//         resize(image, imagePyramid[level], sz, 0, 0, INTER_AREA);
// 		copyMakeBorder(imagePyramid[level], imagePyramid[level], border, border, border, border, BORDER_REFLECT_101+BORDER_ISOLATED);
// 		imagePyramid[level] = imagePyramid[level](Rect(border, border, sz.width, sz.height));
// #else
        if( !mask.empty() )
        {
            masktemp = Mat(wholeSize, mask.type());
            maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
        }

        // Compute the resized image
        if( level != firstLevel )
        {
            if( level < firstLevel )
            {
                if (!mask.empty())
            	resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
                    resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
            }
            else
            {
                resize(imagePyramid[level-1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);
//				Mat blurredImg;
//				cv::GaussianBlur(imagePyramid[level-1], blurredImg, Size(0,0), sigma, 0);
//				resize(blurredImg, imagePyramid[level], sz, 0, 0, INTER_AREA);
                resize(imagePyramid[level-1], imagePyramid[level], sz, 0, 0, INTER_AREA);
                if (!mask.empty())
                {
                    resize(maskPyramid[level-1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
                    threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
                }
            }

            copyMakeBorder(imagePyramid[level], temp, border, border, border, border,
                           BORDER_REFLECT_101+BORDER_ISOLATED);
            if (!mask.empty())
                copyMakeBorder(maskPyramid[level], masktemp, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp, border, border, border, border,
                           BORDER_REFLECT_101);
            if( !mask.empty() )
                copyMakeBorder(mask, masktemp, border, border, border, border,
                               BORDER_CONSTANT+BORDER_ISOLATED);
        }
    }

    
    // Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
    vector < vector<SadKeyPoint> > allKeypoints;
    if( do_keypoints )
    {   
        // Get keypoints, those will be far enough from the border that no check will be required for the descriptor
        computeKeyPoints(imagePyramid, maskPyramid, respPyramid, allKeypoints,
                         responseThr, epsilon, scaleFactor, edgeThreshold,
						 patchSize, scoreType, doNMS, deltaThr, nfeatures,
						 allC1feats, strictMaximum, subPixPrecision, gravityCenter,
						 innerTstType, minArcLength, maxArcLength, ringsType, alpha);
    }
    else
    {
        
        // Cluster the input keypoints depending on the level they were computed at
        allKeypoints.resize(levelsNum);
        for (vector<SadKeyPoint>::iterator keypoint = _keypoints.begin(),
             keypointEnd = _keypoints.end(); keypoint != keypointEnd; ++keypoint)
            allKeypoints[keypoint->octave].push_back(*keypoint);

        // Make sure we rescale the coordinates
        for (int level = 0; level < levelsNum; ++level)
        {
            if (level == firstLevel)
                continue;

            vector<SadKeyPoint> & keypoints = allKeypoints[level];
            double scale = 1.0/getScaleDouble(level, firstLevel, scaleFactor);
            for (vector<SadKeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                keypoint->pt *= scale;
        }
    }


    Mat descriptors;
    vector<Point> pattern;
    if( do_descriptors )
    {
        int nkeypoints = 0;
        for (int level = 0; level < levelsNum; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, descSize, CV_8U);
            descriptors = _descriptors.getMat();
        }

        const int npoints = 512;
        Point patternbuf[npoints];
        Binpat::BitPatterns learn_bin_patterns(binPattern);
        const Point* pattern0 = (const Point*)learn_bin_patterns.get_pattern();

        if( patchSize != 31 )
        {
            pattern0 = patternbuf;
            makeRandomPattern(patchSize, patternbuf, npoints);
        }

        CV_Assert( WTA_K == 2 || WTA_K == 3 || WTA_K == 4 );

        if( WTA_K == 2 )
            std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
        else
        {
            int ntuples = descriptorSize()*4;
            initializeOrbPattern(pattern0, pattern, ntuples, WTA_K, npoints);
        }
    }

    _keypoints.clear();
    int offset = 0;
    for (int level = 0; level < levelsNum; ++level)
    {
        // Get the features and compute their orientation
        vector<SadKeyPoint>& keypoints = allKeypoints[level];
        int nkeypoints = (int)keypoints.size();

        // Compute the descriptors
        if (do_descriptors)
        {
            Mat desc;
            if (!descriptors.empty())
            {
                desc = descriptors.rowRange(offset, offset + nkeypoints);
            }

            offset += nkeypoints;

            // Preprocess the resized image
            Mat& workingMat = imagePyramid[level];

            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
            computeDescriptors(workingMat, keypoints, desc, pattern, descSize, WTA_K);
        }

        // Copy to the output data
        if (level != firstLevel)
        {
            double scale = getScaleDouble(level, firstLevel, scaleFactor);
            for (vector<SadKeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint) {
                keypoint->pt *= scale;
              }
        }
        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }

}

void SORB::detectImpl( const Mat& image, vector<SadKeyPoint>& keypoints, const Mat& mask ) const
{
    (*this)(image, mask, keypoints, noArray(), false);
}

void SORB::computeImpl( const Mat& image, vector<SadKeyPoint>& keypoints, Mat& descriptors) const
{
    (*this)(image, Mat(), keypoints, descriptors, true);
}

struct ResponseComparator
{
    bool operator() (const SadKeyPoint& a, const SadKeyPoint& b)
    {
        return std::abs(a.response) > std::abs(b.response);
    }
};

static void keepStrongest( int N, vector<SadKeyPoint>& keypoints )
{
    if( (int)keypoints.size() > N )
    {
        vector<SadKeyPoint>::iterator nth = keypoints.begin() + N;
        std::nth_element( keypoints.begin(), nth, keypoints.end(), ResponseComparator() );
        keypoints.erase( nth, keypoints.end() );
    }
}

class GridAdaptedFeatureDetectorInvoker : public ParallelLoopBody
{
private:
    int gridRows_, gridCols_;
    int maxPerCell_;
    vector<SadKeyPoint>& keypoints_;
    const Mat& image_;
    const Mat& mask_;
    const Ptr<FeatureDetector>& detector_;
    Mutex* kptLock_;

    GridAdaptedFeatureDetectorInvoker& operator=(const GridAdaptedFeatureDetectorInvoker&); // to quiet MSVC

public:

    GridAdaptedFeatureDetectorInvoker(const Ptr<FeatureDetector>& detector, const Mat& image, const Mat& mask,
                                      vector<SadKeyPoint>& keypoints, int maxPerCell, int gridRows, int gridCols,
                                      cv::Mutex* kptLock)
        : gridRows_(gridRows), gridCols_(gridCols), maxPerCell_(maxPerCell),
          keypoints_(keypoints), image_(image), mask_(mask), detector_(detector),
          kptLock_(kptLock)
    {
    }

    void operator() (const Range& range) const
    {
        for (int i = range.start; i < range.end; ++i)
        {
            int celly = i / gridCols_;
            int cellx = i - celly * gridCols_;

            Range row_range((celly*image_.rows)/gridRows_, ((celly+1)*image_.rows)/gridRows_);
            Range col_range((cellx*image_.cols)/gridCols_, ((cellx+1)*image_.cols)/gridCols_);

            Mat sub_image = image_(row_range, col_range);
            Mat sub_mask;
            if (!mask_.empty()) sub_mask = mask_(row_range, col_range);

            vector<SadKeyPoint> sub_keypoints;
            sub_keypoints.reserve(maxPerCell_);

            detector_->detectKeypoints( sub_image, sub_keypoints );
            keepStrongest( maxPerCell_, sub_keypoints );

            std::vector<cmp::SadKeyPoint>::iterator it = sub_keypoints.begin(),
                                                end = sub_keypoints.end();
            for( ; it != end; ++it )
            {
                it->pt.x += col_range.start;
                it->pt.y += row_range.start;
            }

            cv::AutoLock join_keypoints(*kptLock_);
            keypoints_.insert( keypoints_.end(), sub_keypoints.begin(), sub_keypoints.end() );
        }
    }
};

GridAdaptedFeatureDetector::GridAdaptedFeatureDetector( const Ptr<cmp::FeatureDetector>& _detector,
                                        int _maxTotalKeypoints,
                                        int _gridRows, int _gridCols ) : FeatureDetector(),
	detector(_detector), maxTotalKeypoints(_maxTotalKeypoints), gridRows(_gridRows), gridCols(_gridCols)
{};


void GridAdaptedFeatureDetector::detectKeypoints(const Mat& image, vector<SadKeyPoint>& keypoints, cv::Mat mask) const
{
	if (image.empty() || maxTotalKeypoints < gridRows * gridCols)
	    {
	        keypoints.clear();
	        return;
	    }
	    keypoints.reserve(maxTotalKeypoints);
	    int maxPerCell = maxTotalKeypoints / (gridRows * gridCols);

	    cv::Mutex kptLock;
	    cv::parallel_for_(cv::Range(0, gridRows * gridCols),
	    		cmp::GridAdaptedFeatureDetectorInvoker(detector, image, mask, keypoints, maxPerCell, gridRows, gridCols, &kptLock));
};


}//End namespace cmp
