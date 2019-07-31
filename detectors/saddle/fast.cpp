/* This is FAST corner detector, contributed to OpenCV by the author, Edward Rosten.
   Below is the original copyright and the references */

/*
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
The references are:
 * Machine learning for high-speed corner detection,
   E. Rosten and T. Drummond, ECCV 2006
 * Faster and better: A machine learning approach to corner detection
   E. Rosten, R. Porter and T. Drummond, PAMI, 2009
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "fast_score.hpp"
#include "fast.hpp"
#include <stdio.h>
#include <iostream>
#include <new>
#include "sorb.h"

#include <opencv2/features2d/features2d.hpp>
#include <string>

using namespace cv;


#if defined _MSC_VER
# pragma warning( disable : 4127)
#endif


namespace cmp
{

  void FASTsaddle_inner(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                        int threshold, int nonmax_suppression, float scale, double responsethr, uchar deltaThr, int scoreType,
						bool allC1feats, bool strictMaximum, int subPixPrecision,  bool gravityCenter, int innerTstType,
						int minArcLength, int maxArcLength );
  void FASTsaddle_shinner(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                          int threshold, int nonmax_suppression, float scale, double responsethr,
                          uchar deltaThr, int scoreType, bool allC1feats, bool strictMaximum,
                          int subPixPrecision, bool gravityCenter, int innerTstType,
                          int minArcLength, int maxArcLength );
  void FASTsaddle_conditioned(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                              int threshold, int nonmax_suppression, float scale, double responsethr, int scoreType,
                              bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
                              int minArcLength, int maxArcLength );
  void FASTblob_conditioned(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                            int threshold, int nonmax_suppression, float scale, double responsethr, int scoreType,
                            bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
                            int minArcLength, int maxArcLength );
  void FASTsaddle_blob(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                       int threshold, int nonmax_suppression, float scale, double responsethr, int scoreType,
                       bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
                       int minArcLength, int maxArcLength );
  double FitQuadratic(double offset[2], const double* resp_up, const double* resp_cent, const double* resp_down, int c);
  inline bool inner_test(int pixel_inner[25], int pixel_mid[25], int pixel_outer[25], const uchar* ptr, double& A, double& B, double& C, double& D, uchar& N, uchar opc);
  inline bool inner_sym_test(int pixel_inner[25], const uchar* ptr, double& A, double& B, double& C, double& D, uchar& N );
  inline void blob_test(int pixel_mid[25], int pixel_outer[25], const uchar* ptr, uchar& N);
  inline void subpixel_precision(int i, int j, const double* curr, const double* prev, const double* pprev, float& x, float& y, float& scoreSc, unsigned char interp_mode);
  inline void add_labelling_array(Mat img, int j, int i, std::vector<SadKeyPoint>& kpts, double v, int threshold, int pixel[25]);


  template<int patternSize>
  void FAST_t(InputArray _img, std::vector<SadKeyPoint>& keypoints, int threshold, bool nonmax_suppression)
  {
    Mat img = _img.getMat();
    const int K = patternSize/2, N = patternSize + K + 1;
#if CV_SSE2
    const int quarterPatternSize = patternSize/4;
    (void)quarterPatternSize;
#endif
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

#if CV_SSE2
    __m128i delta = _mm_set1_epi8(-128), t = _mm_set1_epi8((char)threshold), K16 = _mm_set1_epi8((char)K);
    (void)K16;
    (void)delta;
    (void)t;
#endif
    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
      threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    for(i = 3; i < img.rows-2; i++)
      {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
          {
            j = 3;
#if CV_SSE2
            if( patternSize == 16 )
              {
                for(; j < img.cols - 16 - 3; j += 16, ptr += 16)
                  {
                    __m128i m0, m1;

                    __m128i v0 = _mm_loadu_si128((const __m128i*)ptr); 				// Load the center value
                    //                if (i==3 && j==3)
                    //                	print_m128i_epi8(v0);
                    __m128i v1 = _mm_xor_si128(_mm_subs_epu8(v0, t), delta);		// Lower threshold (v1)
                    v0 = _mm_xor_si128(_mm_adds_epu8(v0, t), delta);				// Upper threshold (v0)

                    __m128i x0 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[0])), delta);						// Load value of the SOUTH pixel (x0)
                    __m128i x1 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[quarterPatternSize])), delta);		// Load value of the EAST pixel  (x1)
                    __m128i x2 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[2*quarterPatternSize])), delta);		// Load value of the NORTH pixel (x2)
                    __m128i x3 = _mm_sub_epi8(_mm_loadu_si128((const __m128i*)(ptr + pixel[3*quarterPatternSize])), delta);		// Load value of the WEST pixel  (x3)

                    m0 = _mm_and_si128(_mm_cmpgt_epi8(x0, v0), _mm_cmpgt_epi8(x1, v0));						// x0,x1 > Upper threshold
                    m1 = _mm_and_si128(_mm_cmpgt_epi8(v1, x0), _mm_cmpgt_epi8(v1, x1));						// x0,x1 < Lower threshold

                    m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x1, v0), _mm_cmpgt_epi8(x2, v0)));	// x1,x2 > Upper threshold
                    m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x1), _mm_cmpgt_epi8(v1, x2)));	// x1,x2 < Lower threshold

                    m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x2, v0), _mm_cmpgt_epi8(x3, v0)));	// x2,x3 > Upper threshold
                    m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x2), _mm_cmpgt_epi8(v1, x3)));	// x2,x3 < Lower threshold

                    m0 = _mm_or_si128(m0, _mm_and_si128(_mm_cmpgt_epi8(x3, v0), _mm_cmpgt_epi8(x0, v0)));	// x3,x0 < Lower threshold
                    m1 = _mm_or_si128(m1, _mm_and_si128(_mm_cmpgt_epi8(v1, x3), _mm_cmpgt_epi8(v1, x0)));	// x3,x0 < Lower threshold

                    m0 = _mm_or_si128(m0, m1);	// At least one pair xi,xj are both significantly brighter or darker

                    int mask = _mm_movemask_epi8(m0);
                    if( mask == 0 )
                      continue;
                    if( (mask & 255) == 0 )
                      {
                        j -= 8;
                        ptr -= 8;
                        continue;
                      }

                    __m128i c0 = _mm_setzero_si128(), c1 = c0, max0 = c0, max1 = c0;
                    for( k = 0; k < N; k++ )
                      {
                        __m128i x = _mm_xor_si128(_mm_loadu_si128((const __m128i*)(ptr + pixel[k])), delta);
                        m0 = _mm_cmpgt_epi8(x, v0);
                        m1 = _mm_cmpgt_epi8(v1, x);

                        c0 = _mm_and_si128(_mm_sub_epi8(c0, m0), m0);
                        c1 = _mm_and_si128(_mm_sub_epi8(c1, m1), m1);

                        max0 = _mm_max_epu8(max0, c0);
                        max1 = _mm_max_epu8(max1, c1);
                      }

                    max0 = _mm_max_epu8(max0, max1);
                    // Creates a 16-bit mask from the most significant bits of the 16 signed or unsigned 8-bit integers in a and zero extends the upper bits.
                    int m = _mm_movemask_epi8(_mm_cmpgt_epi8(max0, K16));

                    for( k = 0; m > 0 && k < 16; k++, m >>= 1 )
                      if(m & 1)
                        {
                          cornerpos[ncorners++] = j+k;
                          if(nonmax_suppression)
                            curr[j+k] = (uchar)cornerScore<patternSize>(ptr+k, pixel, threshold);
                        }
                  }
              }
#endif
            for( ; j < img.cols - 3; j++, ptr++ )
              {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;
                int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

                //return;// Nothing is returned

                if( d == 0 )
                  continue;

                d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
                d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
                d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

                if( d == 0 )
                  continue;

                d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
                d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
                d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
                d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

                if( d & 1 )
                  {
                    int vt = v - threshold, count = 0;

                    for( k = 0; k < N; k++ )
                      {
                        int x = ptr[pixel[k]];
                        if(x < vt)
                          {
                            if( ++count > K )
                              {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                  curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                              }
                          }
                        else
                          count = 0;
                      }
                  }

                if( d & 2 )
                  {
                    int vt = v + threshold, count = 0;

                    for( k = 0; k < N; k++ )
                      {
                        int x = ptr[pixel[k]];
                        if(x > vt)
                          {
                            if( ++count > K )
                              {
                                cornerpos[ncorners++] = j;
                                if(nonmax_suppression)
                                  curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                                break;
                              }
                          }
                        else
                          count = 0;
                      }
                  }
              }
          }

        cornerpos[-1] = ncorners;

        if( i == 3 )
          continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
          {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
              {
                keypoints.push_back(cmp::SadKeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
              }
          }
      }
  }


  template<int patternSize>
  void FASTsaddle_central(InputArray _img, std::vector<SadKeyPoint>& keypoints, int threshold, bool nonmax_suppression)
  {
    Mat img = _img.getMat();
    int i, j, k, pixel[25];
    makeOffsets(pixel, (int)img.step, patternSize);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);

    uchar threshold_tab[512];
    for( i = -255; i <= 255; i++ )
      threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

    AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
    uchar* buf[3];
    buf[0] = _buf; buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
    int* cpbuf[3];
    cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
    cpbuf[1] = cpbuf[0] + img.cols + 1;
    cpbuf[2] = cpbuf[1] + img.cols + 1;
    memset(buf[0], 0, img.cols*3);

    int idx;
    uchar count_elem, n_arcs;
    uchar *labels, *begins, *lengths;
    uchar p_label, p_begin, p_len;


    labels = new uchar[8];
    begins = new uchar[8];
    lengths = new uchar[8];

    for(i = 3; i < img.rows-2; i++)
      {
        const uchar* ptr = img.ptr<uchar>(i) + 3;
        uchar* curr = buf[(i - 3)%3];
        int* cornerpos = cpbuf[(i - 3)%3];
        memset(curr, 0, img.cols);
        int ncorners = 0;

        if( i < img.rows - 3 )
          {
            j = 3;
            // Here I suppressed the fast test (4 cardinal pixels)
            for( ; j < img.cols - 3; j++, ptr++ )
              {
                int v = ptr[0];
                const uchar* tab = &threshold_tab[0] - v + 255;

                // Find the first swap
                k = 1;
                while ( (tab[ptr[pixel[k-1]]] == tab[ptr[pixel[k]]]) && (k < 6) )
                  k++;

                if (k==6)
                  continue;

                uchar n_label[] = {0,0,0};
                p_label=0, p_begin=0, p_len=0, n_arcs = 0, count_elem = 1;

                labels[0] = tab[ptr[pixel[k]]];
                n_label[labels[0]]++;
                begins[0] = k++;


                for (uchar pt=k; pt<k+15; pt++ )
                  {
                    idx = pt % 16;
                    if (labels[p_label] != tab[ptr[pixel[idx]]])
                      {
                        labels[++p_label] = tab[ptr[pixel[idx]]];
                        n_label[labels[p_label]]++;
                        begins[++p_begin] = idx;
                        lengths[p_len++] = count_elem;
                        count_elem = 1;
                        n_arcs++;
                      }
                    else
                      count_elem++;
                  }
                lengths[p_len] = count_elem;
                n_arcs++;

                // ------- Constrains ----------- //

                // Number of arcs constrains
                if ((n_arcs > 8) || (n_arcs < 4))
                  continue;
                if ( (n_label[0] > 4) || (n_label[1] != 2) || (n_label[2] != 2) )
                  continue;


                // Arc length constrains
                bool discard=0;
                uchar red_green_labels[4], *p_redgreen;
                p_redgreen = red_green_labels;
                for ( int m=0; m<n_arcs; m++ )
                  {
                    switch ( labels[m] )
                      {
                      case 0:
                        if ( lengths[m]>2 )
                          discard=1;
                        break;
                      default:
                        *p_redgreen++ = labels[m];
                        if ( (lengths[m]<2) || (lengths[m]>8) )
                          discard=1;
                      }
                  }
                if ( discard )
                  continue;
                // Swapping color constrain
                if ( red_green_labels[0] != red_green_labels[2] )
                  continue;


                // Check the length of the arcs
                cornerpos[ncorners++] = j;
                if(nonmax_suppression)
                  curr[j] = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);

              }
          }

        cornerpos[-1] = ncorners;

        if( i == 3 )
          continue;

        const uchar* prev = buf[(i - 4 + 3)%3];
        const uchar* pprev = buf[(i - 5 + 3)%3];
        cornerpos = cpbuf[(i - 4 + 3)%3];
        ncorners = cornerpos[-1];

        for( k = 0; k < ncorners; k++ )
          {
            j = cornerpos[k];
            int score = prev[j];
            if( !nonmax_suppression ||
                (score > prev[j+1] && score > prev[j-1] &&
                 score > pprev[j-1] && score > pprev[j] && score > pprev[j+1] &&
                 score > curr[j-1] && score > curr[j] && score > curr[j+1]) )
              {
                keypoints.push_back(SadKeyPoint((float)j, (float)(i-1), 7.f, -1, (float)score));
              }
          }
      }
  }
  

  inline bool inner_test(int pixel_inner[25], int pixel_mid[25], int pixel_outer[25], const uchar* ptr, double& A, double& B, double& C, double& D, uchar& N, uchar opc)
  {

	  switch (opc)
	  {
	  	  case 0:
	  		  // Baseline
	  		  if ((ptr[pixel_inner[0]] > ptr[pixel_inner[2]]) &&
	  			  (ptr[pixel_inner[4]] > ptr[pixel_inner[6]]) &&
				  (ptr[pixel_inner[6]] < ptr[pixel_inner[0]]) &&
				  (ptr[pixel_inner[2]] < ptr[pixel_inner[4]]) )
			  {
	  			  N += 2;
	  			  // Marked as new
	  			  A = std::min(ptr[pixel_inner[0]], ptr[pixel_inner[4]]);
	  			  B = std::max(ptr[pixel_inner[2]], ptr[pixel_inner[6]]);
			  }
	  		  else  if ((ptr[pixel_inner[4]]<ptr[pixel_inner[6]]) &&
					  	(ptr[pixel_inner[2]]>ptr[pixel_inner[4]]) &&
					    (ptr[pixel_inner[6]]>ptr[pixel_inner[0]]) &&
					    (ptr[pixel_inner[0]]<ptr[pixel_inner[2]]))
			  {
	  			  N += 2;
	  			  B = std::max(ptr[pixel_inner[0]],ptr[pixel_inner[4]]);
	  			  A = std::min(ptr[pixel_inner[2]],ptr[pixel_inner[6]]);
			  }

	  		  if ((ptr[pixel_inner[1]]<ptr[pixel_inner[3]]) &&
	  			  (ptr[pixel_inner[5]]<ptr[pixel_inner[7]]) &&
				  (ptr[pixel_inner[3]]>ptr[pixel_inner[5]]) &&
				  (ptr[pixel_inner[7]]>ptr[pixel_inner[1]]))
			  {
	  			  N += 2;
	  			  C = std::min(ptr[pixel_inner[3]],ptr[pixel_inner[7]]);
	  			  D = std::max(ptr[pixel_inner[1]],ptr[pixel_inner[5]]);
			  }
	  		  else if ((ptr[pixel_inner[5]]>ptr[pixel_inner[7]]) &&
					   (ptr[pixel_inner[3]]<ptr[pixel_inner[5]]) &&
					   (ptr[pixel_inner[7]]<ptr[pixel_inner[1]]) &&
					   (ptr[pixel_inner[1]]>ptr[pixel_inner[3]]) )
			  {
	  			  N += 2;
	  			  C = std::min(ptr[pixel_inner[1]],ptr[pixel_inner[5]]);
	  			  D = std::max(ptr[pixel_inner[3]],ptr[pixel_inner[7]]);
			  }
	  		  break;

	  	  case 1:
	  		  // Extensions only
	  		  if ((ptr[pixel_mid[0]] > ptr[pixel_mid[3]]) &&
	  			  (ptr[pixel_mid[6]] > ptr[pixel_mid[9]]) &&
				  (ptr[pixel_mid[9]] < ptr[pixel_mid[0]]) &&
	  			  (ptr[pixel_mid[3]] < ptr[pixel_mid[6]]) )
			  {
				  N += 2;
				  // Marked as new
				  A = std::min(ptr[pixel_mid[0]], ptr[pixel_mid[6]]);
				  B = std::max(ptr[pixel_mid[3]], ptr[pixel_mid[9]]);

			  }
			  else  if ((ptr[pixel_mid[6]]<ptr[pixel_mid[9]]) &&
						(ptr[pixel_mid[3]]>ptr[pixel_mid[6]]) &&
						(ptr[pixel_mid[9]]>ptr[pixel_mid[0]]) &&
						(ptr[pixel_mid[0]]<ptr[pixel_mid[3]]))
			  {
				  N += 2;
				  B = std::max(ptr[pixel_mid[0]], ptr[pixel_mid[6]]);
				  A = std::min(ptr[pixel_mid[3]], ptr[pixel_mid[9]]);
			  }

			  if ((ptr[pixel_outer[2 ]]<ptr[pixel_outer[6 ]]) &&
				  (ptr[pixel_outer[10]]<ptr[pixel_outer[14]]) &&
				  (ptr[pixel_outer[6 ]]>ptr[pixel_outer[10]]) &&
				  (ptr[pixel_outer[14]]>ptr[pixel_outer[2 ]]))
			  {
				  N += 2;
				  C = std::min(ptr[pixel_outer[6]],ptr[pixel_outer[14]]);
				  D = std::max(ptr[pixel_outer[2]],ptr[pixel_outer[10]]);
			  }
			  else if ((ptr[pixel_outer[10]]>ptr[pixel_outer[14]]) &&
					   (ptr[pixel_outer[6 ]]<ptr[pixel_outer[10]]) &&
					   (ptr[pixel_outer[14]]<ptr[pixel_outer[2 ]]) &&
					   (ptr[pixel_outer[2 ]]>ptr[pixel_outer[6 ]]) )
			  {
				  N += 2;
				  C = std::min(ptr[pixel_outer[2]],ptr[pixel_outer[10]]);
				  D = std::max(ptr[pixel_outer[6]],ptr[pixel_outer[14]]);
			  }
	  		  break;

	  	  case 2:
	  	  {
	  		  // Extension summation
			 float mid6 = (float)ptr[pixel_mid[6]] + (float)ptr[pixel_inner[4]];
			 float mid0 = (float)ptr[pixel_mid[0]] + (float)ptr[pixel_inner[0]];
			 float mid3 = (float)ptr[pixel_mid[3]] + (float)ptr[pixel_inner[2]];
			 float mid9 = (float)ptr[pixel_mid[9]] + (float)ptr[pixel_inner[6]];
			 if ((mid0 > mid3) &&
				 (mid6 > mid9) &&
				 (mid9 < mid0) &&
				 (mid3 < mid6) )
			 {
				N += 2;
				A = std::min((float)ptr[pixel_mid[0]], (float)ptr[pixel_mid[6]]);
				B = std::max((float)ptr[pixel_mid[3]], (float)ptr[pixel_mid[9]]);
			 }
			 else  if ((mid6<mid9) &&
					   (mid3>mid6) &&
					   (mid9>mid0) &&
					   (mid0<mid3))
			 {
				N += 2;
				B = std::max((float)ptr[pixel_mid[0]], (float)ptr[pixel_mid[6]]);
				A = std::min((float)ptr[pixel_mid[3]], (float)ptr[pixel_mid[9]]);
			 }

			 float out2  = (float)ptr[pixel_outer[2 ]] + (float)ptr[pixel_inner[1]];
			 float out6  = (float)ptr[pixel_outer[6 ]] + (float)ptr[pixel_inner[3]];
			 float out10 = (float)ptr[pixel_outer[10]] + (float)ptr[pixel_inner[5]];
			 float out14 = (float)ptr[pixel_outer[14]] + (float)ptr[pixel_inner[7]];
			 if ((out2<out6) &&
				 (out10<out14) &&
				 (out6>out10) &&
				 (out14>out2))
			 {
				 N += 2;
				 C = std::min((float)ptr[pixel_outer[6]], (float)ptr[pixel_outer[14]]);
				 D = std::max((float)ptr[pixel_outer[2]], (float)ptr[pixel_outer[10]]);
			 }
			 else if ((out10>out14) &&
					  (out6<out10) &&
					  (out14<out2) &&
					  (out2>out6) )
			 {
				 N += 2;
				 C = std::min((float)ptr[pixel_outer[2]], (float)ptr[pixel_outer[10]]);
				 D = std::max((float)ptr[pixel_outer[6]], (float)ptr[pixel_outer[14]]);
			 }
	  	  }
	  		  break;

	  	  case 3:
	  		  // Extension average
	  	  {
	  		  float mid6 = ((float)ptr[pixel_mid[6]] + (float)ptr[pixel_inner[4]])*0.5;
			  float mid0 = ((float)ptr[pixel_mid[0]] + (float)ptr[pixel_inner[0]])*0.5;
			  float mid3 = ((float)ptr[pixel_mid[3]] + (float)ptr[pixel_inner[2]])*0.5;
			  float mid9 = ((float)ptr[pixel_mid[9]] + (float)ptr[pixel_inner[6]])*0.5;
			  if ((mid0 > mid3) &&
				  (mid6 > mid9) &&
				  (mid9 < mid0) &&
				  (mid3 < mid6) )
			 {
				  N += 2;
				 // Marked as new
				 A = std::min(mid0, mid6);
				 B = std::max(mid3, mid9);
			 }
			 else  if ((mid6<mid9) &&
					   (mid3>mid6) &&
					   (mid9>mid0) &&
					   (mid0<mid3))
			 {
				 N += 2;
				 B = std::max(mid0,mid6);
				 A = std::min(mid3,mid9);
			 }

			 float out2  = ((float)ptr[pixel_outer[2 ]] + (float)ptr[pixel_inner[1]])*0.5;
			 float out6  = ((float)ptr[pixel_outer[6 ]] + (float)ptr[pixel_inner[3]])*0.5;
			 float out10 = ((float)ptr[pixel_outer[10]] + (float)ptr[pixel_inner[5]])*0.5;
			 float out14 = ((float)ptr[pixel_outer[14]] + (float)ptr[pixel_inner[7]])*0.5;
			 if ((out2<out6) &&
				 (out10<out14) &&
				 (out6>out10) &&
				 (out14>out2))
			 {
				 N += 2;
				 C = std::min(out6,out14);
				 D = std::max(out2,out10);
			 }
			 else if ((out10>out14) &&
					  (out6<out10) &&
					  (out14<out2) &&
					  (out2>out6) )
			 {
				 N += 2;
				 C = std::min(out2,out10);
				 D = std::max(out6,out14);
			 }
	  	  }
	  		  break;

	  	  case 4:
	  		  // Extension square root of two
	  	  {
	  		  float alpha = sqrt(2)-1;
	  		  float beta = 1-alpha;

	  		  float mid0 = alpha*(float)ptr[pixel_mid[0]] + beta*(float)ptr[pixel_inner[0]];
	  		  float mid3 = alpha*(float)ptr[pixel_mid[3]] + beta*(float)ptr[pixel_inner[2]];
	  		  float mid6 = alpha*(float)ptr[pixel_mid[6]] + beta*(float)ptr[pixel_inner[4]];
			  float mid9 = alpha*(float)ptr[pixel_mid[9]] + beta*(float)ptr[pixel_inner[6]];
			  if ((mid0 > mid3) &&
				  (mid6 > mid9) &&
				  (mid9 < mid0) &&
				  (mid3 < mid6) )
			 {
				  N += 2;
				 // Marked as new
				 A = std::min(mid0, mid6);
				 B = std::max(mid3, mid9);
			 }
			 else  if ((mid6<mid9) &&
					   (mid3>mid6) &&
					   (mid9>mid0) &&
					   (mid0<mid3))
			 {
				 N += 2;
				 B = std::max(mid0,mid6);
				 A = std::min(mid3,mid9);
			 }

			 float inn1  = (float)ptr[pixel_inner[1]];
			 float inn3  = (float)ptr[pixel_inner[3]];
			 float inn5 = (float)ptr[pixel_inner[5]];
			 float inn7 = (float)ptr[pixel_inner[7]];
			 if ((inn1<inn3) &&
				 (inn5<inn7) &&
				 (inn3>inn5) &&
				 (inn7>inn1))
			 {
				 N += 2;
				 C = std::min(inn3,inn7);
				 D = std::max(inn1,inn5);
			 }
			 else if ((inn5>inn7) &&
					  (inn3<inn5) &&
					  (inn7<inn1) &&
					  (inn1>inn3) )
			 {
				 N += 2;
				 C = std::min(inn1,inn5);
				 D = std::max(inn3,inn7);
			 }
	  	  }
	  		  break;

	  }


	  if (N)
		  return true;
	  else
		  return false;

  }


  inline bool inner_sym_test(int pixel_inner[25], const uchar* ptr, double& A, double& B, double& C, double& D, uchar& N )
  {

	  int vert_vals[4];
	  vert_vals[0] = ptr[pixel_inner[0]] + ptr[pixel_inner[1]];
	  vert_vals[1] = ptr[pixel_inner[2]] + ptr[pixel_inner[3]];
	  vert_vals[2] = ptr[pixel_inner[4]] + ptr[pixel_inner[5]];
	  vert_vals[3] = ptr[pixel_inner[6]] + ptr[pixel_inner[7]];

	  int diag_vals[4];
	  diag_vals[0] = ptr[pixel_inner[1]] + ptr[pixel_inner[2]];
	  diag_vals[1] = ptr[pixel_inner[3]] + ptr[pixel_inner[4]];
	  diag_vals[2] = ptr[pixel_inner[5]] + ptr[pixel_inner[6]];
	  diag_vals[3] = ptr[pixel_inner[7]] + ptr[pixel_inner[0]];


	  if ( (vert_vals[0]>vert_vals[1]) && (vert_vals[2]>vert_vals[3]) && (vert_vals[3]<vert_vals[0]) && (vert_vals[1]<vert_vals[2]) )
	  {
		  N += 2;
		  A = std::min(vert_vals[0], vert_vals[2]);
		  B = std::max(vert_vals[1], vert_vals[3]);
	  }
	  else if ( (vert_vals[0]<vert_vals[1]) && (vert_vals[2]<vert_vals[3]) && (vert_vals[3]>vert_vals[0]) && (vert_vals[1]>vert_vals[2]) )
	  {
		  N += 2;
		  A = std::min(vert_vals[1], vert_vals[3]);
		  B = std::max(vert_vals[0], vert_vals[2]);
	  }

	  if ( (diag_vals[0]<diag_vals[1]) && (diag_vals[2]<diag_vals[3]) && (diag_vals[1]>diag_vals[2]) && (diag_vals[0]<diag_vals[3]) )
	  {
		  N += 2;
		  C = std::min(diag_vals[1], diag_vals[3]);
		  D = std::max(diag_vals[0], diag_vals[2]);
	  }
	  else if ( (diag_vals[0]>diag_vals[1]) && (diag_vals[2]>diag_vals[3]) && (diag_vals[1]<diag_vals[2]) && (diag_vals[0]>diag_vals[3]) )
	  {
		  N += 2;
		  C = std::min(diag_vals[0], diag_vals[2]);
		  D = std::max(diag_vals[1], diag_vals[3]);
	  }
	  // The Values exported from this functions are not normalized to 0:255, you need to divide by 2

	  if (N)
		  return true;
	  else
		  return false;

  }

  inline void blob_test(int pixel_mid[25], int pixel_outer[25], const uchar* ptr, uchar& N)
  {
    N = 0;
    if ((ptr[pixel_outer[0]]  > ptr[pixel_mid[0]]) &&
       ( ptr[pixel_outer[4]]  > ptr[pixel_mid[3]]) &&
       ( ptr[pixel_outer[8]]  > ptr[pixel_mid[6]]) &&
       ( ptr[pixel_outer[12]] > ptr[pixel_mid[9]]) )
    {
      N = 1;
    }
      else  if ((ptr[pixel_outer[0 ]] < ptr[pixel_mid[0]]) &&
               ( ptr[pixel_outer[4 ]] < ptr[pixel_mid[3]]) &&
               ( ptr[pixel_outer[8 ]] < ptr[pixel_mid[6]]) &&
               ( ptr[pixel_outer[12]] < ptr[pixel_mid[9]]))
    {
      N = 2;
    }

}

  inline void subpixel_precision(int j, int i, const double* curr, const double* prev, const double* pprev, float& x, float& y, float& scoreSc, unsigned char interp_mode)
  {

    switch(interp_mode)
    {
      case 0:
        // No interpolation
        x = (float)(j + 0.5);
        y = (float)(i - 0.5);
        break;

      case 1:
      {
        // Bilinear
        float sumresp = prev[j] + prev[j + 1] + prev[j-1] + pprev[j] + pprev[j + 1] + pprev[j-1] + curr[j] + curr[j + 1] + curr[j-1];
        x = (j-1)*(pprev[j-1] + prev[j-1] + curr[j-1] ) + (j)*(pprev[j] + prev[j] + curr[j] ) + (j+1)*(pprev[j+1] + prev[j+1] + curr[j+1]);
        y = (i-1)*(prev[j-1] + prev[j] + prev[j+1]) + (i)*(curr[j-1] + curr[j] + curr[j+1]) + (i-2)*(pprev[j-1] + pprev[j] + pprev[j+1]);
        x /= sumresp;
        y /= sumresp;
        break;
      }

      case 2:
      {
        // Quadratic
        double offset[2];
        scoreSc = (float)FitQuadratic( offset, pprev, prev, curr, j);
        x = (float)    j + offset[1];
        y = (float)(i-1) + offset[0];
        break;
      }

      default:
        std::cerr << "Unknown sub-pixel precision estimation" << std::endl;
    }
  }

  inline void add_labelling_array(Mat img, int j, int i, std::vector<SadKeyPoint>& kpts, double v, int threshold, int pixel[25])
  {
    const uchar* ptr;
    double upperThr, lowerThr;
    kpts.back().outLabels.assign(16,0);
    ptr =  img.ptr<uchar>(i - 1) + j;

    for(unsigned l = 0; l < 16; l++)
    {
      kpts.back().intensityPixels[l] = ptr[pixel[l]];
      upperThr = v + (double)threshold;
      lowerThr = v - (double)threshold;

      if (ptr[pixel[l]] > upperThr)
      {
        kpts.back().outLabels.at(l) = 2;
        kpts.back().labels[l] = 2;
      }
      else if (ptr[pixel[l]] < lowerThr)
      {
        kpts.back().outLabels.at(l) = 1;
        kpts.back().labels[l] = 1;
      }
      else
      {
        kpts.back().outLabels.at(l) = 0;
        kpts.back().labels[l] = 0;
      }
    }
  }

  void FASTsaddle_shinner(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                          int threshold, int nonmax_suppression, float scale, double responsethr, uchar deltaThr, int scoreType,
  						bool allC1feats, bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType, int minArcLength, int maxArcLength )
  {

	  const Mat img = _img.getMat();
	  int i, j, k, idx, pixel_inner[25], pixel_outer[25];
	  double threshold2, scEps = 2.0;
	  makeShiftedOffsets(pixel_inner, (int)img.step, 8);
	  makeShiftedOffsets(pixel_outer, (int)img.step, 20);
	  keypoints.clear();


	  // Relating delta and epsilon (there is no adaptation)
	  if (threshold == 0)
	  {
		  threshold = (int)(deltaThr/2);
	    threshold2 = scEps*(double)threshold;
	  }
	  else if (threshold > 0)
	  {
		  threshold = std::min(std::max(threshold, 0), 255);
	    threshold2 = scEps*(double)threshold;
	  }


	  // ----- My try of unification (Scores and Coordinates positions) ----- //
	  AutoBuffer<double> _bufScCp(img.cols*3*(sizeof(double) + sizeof(int) + sizeof(double) + sizeof(uchar)) + 12 );//12 = 3*4(int size)
	  // Set the pointers for SCORES
	  double* bufSc[3];
	  bufSc[0] = _bufScCp;
	  bufSc[1] = bufSc[0] + img.cols;
	  bufSc[2] = bufSc[1] + img.cols;
	  memset(bufSc[0], 0, img.cols*3*sizeof(double));

	  // Set the pointers for COORDINATES POINTS
	  int* bufCp[3];
	  bufCp[0] = (int*)alignPtr(bufSc[2] + img.cols, sizeof(int)) + 1;
	  bufCp[1] = bufCp[0] + img.cols + 1;
	  bufCp[2] = bufCp[1] + img.cols + 1;

	  double* bufV[3];
	  bufV[0] = (double*)alignPtr(bufCp[2] + img.cols, sizeof(double));
	  bufV[1] = bufV[0] + img.cols;
	  bufV[2] = bufV[1] + img.cols;

	  uchar* bufDl[3];
	  bufDl[0] = (uchar*)alignPtr(bufV[2] + img.cols, sizeof(uchar));
	  bufDl[1] = bufDl[0] + img.cols;
	  bufDl[2] = bufDl[1] + img.cols;


	  uchar p_regs, count_elem, maxBlueLength = 2;
	  uchar *labels, *begins, *lengths;
	  labels  = new uchar[9];
	  begins  = new uchar[9];
	  lengths = new uchar[9];

	  // Scanning Y-axis
	  for(i = 3; i < img.rows-3; i++)
	  {

		  const uchar* ptr = img.ptr<uchar>(i) + 3;

		  double* curr   = bufSc[(i - 3)%3];
		  double* currV  = bufV [(i - 3)%3];
		  uchar*  currDl = bufDl[(i - 3)%3];
		  int* cornerpos = bufCp[(i - 3)%3];

		  memset(curr, 0, img.cols*sizeof(double));
		  int ncorners = 0;

		  if( i < img.rows - 3 )
		  {
			  j = 3;
			  // Scanning X-axis
			  for( ; j < img.cols - 4; j++, ptr++)
			  {

				  double v = 0.0, A = 0.0, B = 0.0, C = 0.0, D = 0.0;
				  uchar N = 0;

				  inner_sym_test(pixel_inner, ptr, A, B, C, D, N );

				  if (!N)
					continue;

				  uchar delta = std::max( A-B, C-D );

				  if (N == 4)
				  {
					if ((A >= D) && (B <= C))
						v = std::min(A,C) + std::max (B,D);
					else
						continue;
				  }
				  else
					v = std::max( A+B, C+D );
				  v *= 0.25;

				  double upperThr, lowerThr, upperThr2, lowerThr2;

				  if (threshold > 0)
				  {
					  upperThr = v + (double)threshold;
					  lowerThr = v - (double)threshold;
					  upperThr2 = v + threshold2;
					  lowerThr2 = v - threshold2;
				  }
				  else
				  {
					  upperThr = v + (double)(0.5*delta);
					  lowerThr = v - (double)(0.5*delta);
					  upperThr2 = v + (scEps*0.5*(double)delta);
					  lowerThr2 = v - (scEps*0.5*(double)delta);
				  }

				  int templateLarge[20] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
				  for (k = 0; k < 20; k++)
				  {
					  if ( (double)ptr[pixel_outer[k]] > upperThr ) 		// GREEN
						  templateLarge[k] = 2;
					  else if ( (double)ptr[pixel_outer[k]] < lowerThr )	// RED
						  templateLarge[k] = 1;


					  // FIRST brighter or darker
					  if (((templateLarge[k]==1) && (templateLarge[k-1]==2) && (ptr[pixel_outer[k]] > lowerThr2)) ||
						  ((templateLarge[k]==2) && (templateLarge[k-1]==1) && (ptr[pixel_outer[k]] < upperThr2)))
						  templateLarge[k] = 0;
				  }

				  // Find the position of the first swap
				  k = 1;
				  while ( (k <= maxArcLength) && (templateLarge[k-1] == templateLarge[k]) )
					  k++;

				  if (k > maxArcLength)
					  continue;


				  // Registers for template checking
				  uchar n_label[] = {0,0,0}; // Label counter, position is the label, content is the number of labels seen

				  labels[0] = templateLarge[k];
				  n_label[templateLarge[k]]++;
				  begins[0] = k++;
				  count_elem = 1;
				  p_regs = 0;

				  for (uchar pt=k; pt<k+19; pt++ )
				  {
					  idx = pt % 20;
					  if (labels[p_regs] != templateLarge[idx])
					  {
						  labels[p_regs+1] = templateLarge[idx];
						  n_label[labels[p_regs+1]]++;
						  begins[p_regs+1] = idx;
						  lengths[ p_regs++] = count_elem;
						  count_elem = 1;
						  if (p_regs>7)
							  break;
					  }
					  else
						  count_elem++;
				  }
				  lengths[p_regs++] = count_elem;

				  // ----------------- Constrains ----------------------- //

				  // Number of arcs constrains
				  if ((p_regs > 8) || (p_regs < 4)  || (n_label[0] > 4) || (n_label[1] != 2) || (n_label[2] != 2))
					  continue;

				  // Arc length constrains
				  bool discard=0;
				  uchar red_green_labels[4], *p_redgreen;
				  p_redgreen = red_green_labels;
				  for ( int m=0; m<p_regs && !discard; m++ )
				  {
					if (labels[m] == 0)
					{
						discard = (lengths[m]>maxBlueLength);
					  }
					else
					{
						*p_redgreen++ = labels[m];
						  discard = ( (lengths[m] < minArcLength) || (lengths[m] > maxArcLength) );
					  }
				  }

				  if ( discard || (red_green_labels[0] != red_green_labels[2] ) )  // Swapping color constrain
					continue;

				  // Include the point in the feature set
				  cornerpos[ncorners++] = j;
				  curr[j] = delta;
				  currV[j] = v;
				  currDl[j] = delta;
				  continue;

			  }
		  }
		  cornerpos[-1] = ncorners;


		  bool nmsFlag;
		  const double* prev = bufSc[(i - 4 + 3)%3];
		  const double* pprev = bufSc[(i - 5 + 3)%3];
		  const double* prevV = bufV[(i - 4 + 3)%3];
		  const uchar* prevDl = bufDl[(i - 4 + 3)%3];


		  for( k = 0; k < ncorners; k++ )
		  {
			  // Begin
			  j = cornerpos[k];
			  float scoreSc = prev[j];
			  double v = prevV[j];
			  uchar delta = prevDl[j];

			  // Compute the NMS
			  if (strictMaximum)
				nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
						  scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
						  scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
			  else
				nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
						  scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
						  scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

			  if( !(nonmax_suppression>0) || nmsFlag )
			  {
				  //
				  if (subPixPrecision == 0)
					  keypoints.push_back(SadKeyPoint((float)(j+0.5), (float)(i-0.5), 8.f, -1, (float)scoreSc, 1.f ));
				  else if (subPixPrecision == 1)
				  {
					  float sumresp = prev[j] + prev[j + 1] + prev[j-1] + pprev[j] + pprev[j + 1] + pprev[j-1] + curr[j] + curr[j + 1] + curr[j-1];
					  float thetaX = (j-1)*(pprev[j-1] + prev[j-1] + curr[j-1] ) + (j)*(pprev[j] + prev[j] + curr[j] ) + (j+1)*(pprev[j+1] + prev[j+1] + curr[j+1] );
					  float thetaY = (i-1)*(prev[j-1] + prev[j] + prev[j+1]) + (i)*(curr[j-1] + curr[j] + curr[j+1]) + (i-2)*(pprev[j-1] + pprev[j] + pprev[j+1]) ;
					  thetaX = thetaX/sumresp;
					  thetaY = thetaY/sumresp;
					  keypoints.push_back(SadKeyPoint((float)thetaX+0.5, (float)thetaY+0.5, 8.f, -1, (float)scoreSc, 1.f ));
				  }
				  else if (subPixPrecision == 2)
				  {
					  double offset[2];
					  scoreSc = (float)FitQuadratic( offset, pprev, prev, curr, j);
					  float thetaX = (float)j + offset[1];
					  float thetaY = (float)(i-1) + offset[0];
					  keypoints.push_back(SadKeyPoint((float)(thetaX+0.5), (float)(thetaY+0.5), 8.f, -1, (float)scoreSc, 1.f ));
				  }
				  else
					  std::cerr << "Unknown sub-pixel precision estimation" << std::endl;

				  keypoints.back().intensityCenter = v;
				  keypoints.back().delta = delta;

			  }
		  }

	  }
  }

  void FASTsaddle_inner(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                        int threshold, int nonmax_suppression, float scale, double responsethr,
                        uchar deltaThr, int scoreType, bool allC1feats, bool strictMaximum,
                        int subPixPrecision, bool gravityCenter, int innerTstType, int minArcLength,
                        int maxArcLength )
  {
    double scEps = 2.0, threshold2;
    double st;
    const Mat img = _img.getMat();

    Mat binImg; // Mask of all pixels that fulfill the 1st and 2nd condition
    binImg = Mat::zeros(img.rows, img.cols, CV_8UC1);

    int i, j, k, pixel[25], pixel_inner[25], pixel_mid[25];
    int rej2=0, rej3=0, rej4=0, accp=0;
    makeOffsets(pixel, (int)img.step, 16); //patternSize
    makeOffsets(pixel_inner, (int)img.step, 8);
    makeOffsets(pixel_mid, (int)img.step, 12);

    keypoints.clear();

    // Relating delta and epsilon (there is no adaptation)
    if (threshold == 0)  
    {
      threshold = (int)(deltaThr/2);
      threshold2 = scEps*(double)threshold;
    }
    else if (threshold > 0)
    {
      threshold = std::min(std::max(threshold, 0), 255);
      threshold2 = scEps*(double)threshold;
    }

    // Allocating memory for all the buffer, each sizeof corresponds to one buffer
    AutoBuffer<double> _bufScCp((img.cols+16)*3*(2*sizeof(double) + sizeof(int) + sizeof(uchar)) + 128 );


    // Set the pointers for SCORES
    double* bufSc[3];
    bufSc[0] = _bufScCp;
    bufSc[1] = bufSc[0] + img.cols;
    bufSc[2] = bufSc[1] + img.cols;
    memset(bufSc[0], 0, img.cols*3*sizeof(double));

    // Set the pointers for COORDINATES POINTS
    int* bufCp[3];
    bufCp[0] = (int*)alignPtr(bufSc[2] + img.cols, sizeof(int)) + 1;
    bufCp[1] = bufCp[0] + img.cols + 1;
    bufCp[2] = bufCp[1] + img.cols + 1;
    memset(bufCp[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufV[3];
    bufV[0] = (double*)alignPtr(bufCp[2] + img.cols + 1, sizeof(double));
    bufV[1] = bufV[0] + img.cols;
    bufV[2] = bufV[1] + img.cols;

    // Save Deltas
    uchar* bufDl[3];
    bufDl[0] = (uchar*)alignPtr(bufV[2] + img.cols, sizeof(uchar));
    bufDl[1] = bufDl[0] + img.cols;
    bufDl[2] = bufDl[1] + img.cols;

    int idx;
    uchar p_regs, count_elem;
    uchar *labels, *begins, *lengths;
    int* lbl;

    labels  = new uchar[9];
    begins  = new uchar[9];
    lengths = new uchar[9];

    for(i = 3; i < img.rows-2; i++)
    {
      const uchar* ptr = img.ptr<uchar>(i) + 3;
      double* curr = bufSc[(i - 3)%3];
      double* currV = bufV[(i - 3)%3];
      uchar* currDl = bufDl[(i - 3)%3];
      int* cornerpos = bufCp[(i - 3)%3];
      
      memset(curr, 0, img.cols*sizeof(double));
      int ncorners = 0;
      

      if( i < img.rows - 3 )
      {
        j = 3;

        for( ; j < img.cols - 3; j++, ptr++)
        {        
          double v = 0.0, A = 0.0, B = 0.0, C = 0.0, D = 0.0;
          uchar N = 0, blob_type, delta;
          inner_test(pixel_inner, pixel_mid, pixel, ptr, A, B, C, D, N, innerTstType);

          if (!N)
            continue;

          delta = std::max( A-B, C-D );
          if (N == 4)
          {
          	if ((A >= D) && (B <= C))
          		v = std::min(A,C) + std::max (B,D);
          	else
          		continue;
          }
          else
          	v = std::max( A+B, C+D );
          v *= 0.5;

          if (allC1feats)
          {
  					cornerpos[ncorners++] = j;
  					currV[j] = v;
  					currDl[j] = delta;
  					curr[j] = delta;
  					continue;
          }

          if (delta < deltaThr)
            continue;

          double upperThr, lowerThr, upperThr2, lowerThr2;

          if (threshold > 0)
          {
            upperThr = v + (double)threshold;
            lowerThr = v - (double)threshold;
            upperThr2 = v + threshold2;
            lowerThr2 = v - threshold2;
          }
          else
          {
            upperThr = v + (double)(0.5*delta);
            lowerThr = v - (double)(0.5*delta);
            upperThr2 = v + (scEps*0.5*(double)delta);
            lowerThr2 = v - (scEps*0.5*(double)delta);
          }

          int templateLarge[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
          for (k = 0; k < 16; k++)
          {
            if ( (double)ptr[pixel[k]] > upperThr ) // GREEN
              templateLarge[k] = 2;
            else if ( (double)ptr[pixel[k]] < lowerThr ) // RED
              templateLarge[k] = 1;

            // FIRST brighter or darker
            if (((templateLarge[k]==1) && (templateLarge[k-1]==2) && (ptr[pixel[k]] > lowerThr2)) ||
                ((templateLarge[k]==2) && (templateLarge[k-1]==1) && (ptr[pixel[k]] < upperThr2)))
              templateLarge[k] = 0;
          }

          // Find the position of the first swap
          k = 1;
          while ( (k <= maxArcLength) && (templateLarge[k-1] == templateLarge[k]) )
          	k++;

          if (k > maxArcLength)
          	continue;

          // Registers for template checking
          uchar n_label[] = {0,0,0};

          labels[0] = templateLarge[k];
          n_label[templateLarge[k]]++;
          begins[0] = k++;
          count_elem = 1;
          p_regs = 0;

          for (uchar pt=k; pt<k+15; pt++ )
          {
            idx = pt % 16;
            if (labels[p_regs] != templateLarge[idx])
            {
              labels[p_regs+1] = templateLarge[idx];
              n_label[labels[p_regs+1]]++;
              begins[p_regs+1] = idx;
              lengths[ p_regs++] = count_elem;
              count_elem = 1;

              if (p_regs>7)
                break;
            }
            else
              count_elem++;
          }
          lengths[p_regs++] = count_elem;

          // ----------------- Constrains ----------------------- //

          // Number of arcs constrains
          if ((p_regs > 8) || (p_regs < 4)  || (n_label[0] > 4) || (n_label[1] != 2) || (n_label[2] != 2))
            continue;

          // Arc length constrains
          bool discard=0;
          uchar red_green_labels[4], *p_redgreen;
          p_redgreen = red_green_labels;
          for ( int m=0; m<p_regs && !discard; m++ )
          {
            if (labels[m] == 0)
              discard = (lengths[m]>2);
            else
            {
              *p_redgreen++ = labels[m];
              discard = ( (lengths[m] < minArcLength) || (lengths[m] > maxArcLength) );
            }
          }

          if ( discard || (red_green_labels[0] != red_green_labels[2]))
            continue;

          // Include the feature in the set
          cornerpos[ncorners++] = j;
          currV[j] = v;
          currDl[j] = delta;

          // Compute the feature response
          int* lbl = templateLarge;
          curr[j] = cmpFeatureScore(ptr, pixel, lbl, v, delta, scoreType);

          // Save the point in the binary image
          uchar* ptrBinary = binImg.ptr<uchar>(i);
          ptrBinary[j] = 255;
        }
      }
      cornerpos[-1] = ncorners;

      /*   Collecting the SADDLES   */
      const double*  prev = bufSc[(i - 4 + 3)%3];
      const double* pprev = bufSc[(i - 5 + 3)%3];
      const double* prevV = bufV [(i - 4 + 3)%3];
      const uchar* prevDl = bufDl[(i - 4 + 3)%3];

      double* pr = _resp.ptr<double>(i - 1);
      bool nmsFlag;
      float scoreSc;
      double v;
      unsigned char delta;

      cornerpos = bufCp[(i - 4 + 3)%3];
      ncorners = cornerpos[-1];

      for( k = 0; k < ncorners; k++ )
	    {
        j = cornerpos[k];
        scoreSc = prev[j];
        v = prevV[j];
        delta = prevDl[j];

        // Compute the NMS
        if (strictMaximum)
        	nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
        	nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
        	          scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
        	          scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;

          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().intensityCenter = v;
          keypoints.back().delta = delta;
          keypoints.back().class_id = 0;
          add_labelling_array(img, j, i, keypoints, v, threshold, pixel);
          pr[j] = scoreSc;
        }
      }
    } // Here the Y axis sliding window loop finishes    
  }

  void FASTsaddle_blob(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                              int threshold, int nonmax_suppression, float scale, double responsethr, int scoreType,
                              bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
                              int minArcLength, int maxArcLength )
  {
    double scEps = 2.0, threshold2;
    double st;
    const Mat img = _img.getMat();

    Mat binImg; // Mask of all pixels that fulfill the 1st and 2nd condition
    binImg = Mat::zeros(img.rows, img.cols, CV_8UC1);

    int i, j, k, pixel[25], pixel_inner[25], pixel_mid[25];
    int rej2=0, rej3=0, rej4=0, accp=0;
    makeOffsets(pixel, (int)img.step, 16); //patternSize
    makeOffsets(pixel_inner, (int)img.step, 8);
    makeOffsets(pixel_mid, (int)img.step, 12);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);
    threshold2 = scEps*(double)threshold;

    // Allocating memory for all the buffer, each sizeof corresponds to one buffer
    AutoBuffer<double> _bufScCp((img.cols+16)*3*(3*sizeof(double) + 2*sizeof(int) + 2*sizeof(uchar)) + 128 );

    // Set the pointers for SCORES
    double* bufSc[3];
    bufSc[0] = _bufScCp;
    bufSc[1] = bufSc[0] + img.cols;
    bufSc[2] = bufSc[1] + img.cols;
    memset(bufSc[0], 0, img.cols*3*sizeof(double));

    // Set the pointers for COORDINATES POINTS
    int* bufCp[3];
    bufCp[0] = (int*)alignPtr(bufSc[2] + img.cols, sizeof(int)) + 1;
    bufCp[1] = bufCp[0] + img.cols + 1;
    bufCp[2] = bufCp[1] + img.cols + 1;
    memset(bufCp[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufV[3];
    bufV[0] = (double*)alignPtr(bufCp[2] + img.cols + 1, sizeof(double));
    bufV[1] = bufV[0] + img.cols;
    bufV[2] = bufV[1] + img.cols;

    // Save Deltas
    uchar* bufDl[3];
    bufDl[0] = (uchar*)alignPtr(bufV[2] + img.cols, sizeof(uchar));
    bufDl[1] = bufDl[0] + img.cols;
    bufDl[2] = bufDl[1] + img.cols;


    // Memory allocation for the BLOB locations
    int* bufBlobPos[3];
    bufBlobPos[0] = (int*)alignPtr(bufDl[2] + img.cols, sizeof(int)) + 1;
    bufBlobPos[1] = bufBlobPos[0] + img.cols + 1;
    bufBlobPos[2] = bufBlobPos[1] + img.cols + 1;
    memset(bufBlobPos[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufBlobSc[3];
    bufBlobSc[0] = (double*)alignPtr(bufBlobPos[2] + img.cols + 1, sizeof(double));
    bufBlobSc[1] = bufBlobSc[0] + img.cols;
    bufBlobSc[2] = bufBlobSc[1] + img.cols;
    memset(bufBlobSc[0], 0, img.cols*3*sizeof(double));

    uchar* bufBlobTy[3];
    bufBlobTy[0] = (uchar*)alignPtr(bufBlobSc[2] + img.cols, sizeof(uchar));
    bufBlobTy[1] = bufBlobTy[0] + img.cols;
    bufBlobTy[2] = bufBlobTy[1] + img.cols;

    int idx;
    uchar p_regs, count_elem;
    uchar *labels, *begins, *lengths;
    int* lbl;

    labels  = new uchar[9];
    begins  = new uchar[9];
    lengths = new uchar[9];

    for(i = 3; i < img.rows-2; i++)
    {
      const uchar* ptr = img.ptr<uchar>(i) + 3;
      double* curr = bufSc[(i - 3)%3];
      double* currBlobSc = bufBlobSc[(i - 3)%3];
      double* currV = bufV[(i - 3)%3];
      uchar* currDl = bufDl[(i - 3)%3];
      uchar* currBlobTy = bufBlobTy[(i - 3)%3];
      int* cornerpos = bufCp[(i - 3)%3];
      int* blobpos = bufBlobPos[(i - 3)%3];

      memset(curr, 0, img.cols*sizeof(double));
      memset(currBlobSc, 0, img.cols*sizeof(double));
      int ncorners = 0, nblobs = 0;
      

      if( i < img.rows - 3 )
      {
        j = 3;

        for( ; j < img.cols - 3; j++, ptr++)
        {        
          double v = 0.0, A = 0.0, B = 0.0, C = 0.0, D = 0.0;
          uchar N = 0, blob_type, delta;

          blob_test(pixel_mid, pixel, ptr, blob_type);
          if (blob_type)
          {
            blobpos[nblobs++] = j;
            currBlobSc[j] = cmpFeatureScore(ptr, pixel, lbl, 0.0, 0, SORB::HESS_SCORE);
            currBlobTy[j] = blob_type;
          }

          inner_test(pixel_inner, pixel_mid, pixel, ptr, A, B, C, D, N, innerTstType);
          if (!N)
            continue;

          delta = std::max( A-B, C-D );
          if (N == 4)
          {
            if ((A >= D) && (B <= C))
              v = std::min(A,C) + std::max (B,D);
            else
              continue;
          }
          else
            v = std::max( A+B, C+D );
          v *= 0.5;

          double upperThr, lowerThr, upperThr2, lowerThr2;

          if (threshold > 0)
          {
            upperThr = v + (double)threshold;
            lowerThr = v - (double)threshold;
            upperThr2 = v + threshold2;
            lowerThr2 = v - threshold2;
          }
          else
          {
            upperThr = v + (double)(0.5*delta);
            lowerThr = v - (double)(0.5*delta);
            upperThr2 = v + (scEps*0.5*(double)delta);
            lowerThr2 = v - (scEps*0.5*(double)delta);
          }

          int templateLarge[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
          for (k = 0; k < 16; k++)
          {
            if ( (double)ptr[pixel[k]] > upperThr ) // GREEN
              templateLarge[k] = 2;
            else if ( (double)ptr[pixel[k]] < lowerThr ) // RED
              templateLarge[k] = 1;

            // FIRST brighter or darker
            if (((templateLarge[k]==1) && (templateLarge[k-1]==2) && (ptr[pixel[k]] > lowerThr2)) ||
                ((templateLarge[k]==2) && (templateLarge[k-1]==1) && (ptr[pixel[k]] < upperThr2)))
              templateLarge[k] = 0;
          }

          // Find the position of the first swap
          k = 1;
          while ( (k <= maxArcLength) && (templateLarge[k-1] == templateLarge[k]) )
            k++;

          if (k > maxArcLength)
            continue;

          // Registers for template checking
          uchar n_label[] = {0,0,0};

          labels[0] = templateLarge[k];
          n_label[templateLarge[k]]++;
          begins[0] = k++;
          count_elem = 1;
          p_regs = 0;

          for (uchar pt=k; pt<k+15; pt++ )
          {
            idx = pt % 16;
            if (labels[p_regs] != templateLarge[idx])
            {
              labels[p_regs+1] = templateLarge[idx];
              n_label[labels[p_regs+1]]++;
              begins[p_regs+1] = idx;
              lengths[ p_regs++] = count_elem;
              count_elem = 1;

              if (p_regs>7)
                break;
            }
            else
              count_elem++;
          }
          lengths[p_regs++] = count_elem;

          // ----------------- Constrains ----------------------- //

          // Number of arcs constrains
          if ((p_regs > 8) || (p_regs < 4)  || (n_label[0] > 4) || (n_label[1] != 2) || (n_label[2] != 2))
            continue;

          // Arc length constrains
          bool discard=0;
          uchar red_green_labels[4], *p_redgreen;
          p_redgreen = red_green_labels;
          for ( int m=0; m<p_regs && !discard; m++ )
          {
            if (labels[m] == 0)
              discard = (lengths[m]>2);
            else
            {
              *p_redgreen++ = labels[m];
              discard = ( (lengths[m] < minArcLength) || (lengths[m] > maxArcLength) );
            }
          }

          if ( discard || (red_green_labels[0] != red_green_labels[2]))
            continue;

          // Include the feature in the set
          cornerpos[ncorners++] = j;
          currV[j] = v;
          currDl[j] = delta;

          // Compute the feature response
          int* lbl = templateLarge;
          curr[j] = cmpFeatureScore(ptr, pixel, lbl, v, delta, scoreType);

          // Save the point in the binary image
          uchar* ptrBinary = binImg.ptr<uchar>(i);
          ptrBinary[j] = 255;
        }
      }
      cornerpos[-1] = ncorners;
      blobpos[-1] = nblobs;

      /*   Collecting the SADDLES   */
      const double*  prev = bufSc[(i - 4 + 3)%3];
      const double* pprev = bufSc[(i - 5 + 3)%3];
      const double* prevV = bufV [(i - 4 + 3)%3];
      const uchar* prevDl = bufDl[(i - 4 + 3)%3];
      const uchar* prevBlobType = bufBlobTy[(i - 4 + 3)%3];

      double* pr = _resp.ptr<double>(i - 1);
      bool nmsFlag;
      float scoreSc;
      double v;
      unsigned char delta, blobType;

      cornerpos = bufCp[(i - 4 + 3)%3];
      ncorners = cornerpos[-1];

      for( k = 0; k < ncorners; k++ )
      {
        j = cornerpos[k];
        scoreSc = prev[j];
        v = prevV[j];
        delta = prevDl[j];

        // Compute the NMS
        if (strictMaximum)
          nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
          nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
                    scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
                    scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;

          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().intensityCenter = v;
          keypoints.back().delta = delta;
          keypoints.back().class_id = 0;
          add_labelling_array(img, j, i, keypoints, v, threshold, pixel);
          pr[j] = scoreSc;
        }
      }

      /*   Collecting the BLOBS   */
      prev  = bufBlobSc[(i - 4 + 3)%3];
      pprev = bufBlobSc[(i - 5 + 3)%3];
      blobpos  = bufBlobPos[(i - 4 + 3)%3];
      nblobs = blobpos[-1];

      for( k = 0; k < nblobs; k++ )
      {
        j = blobpos[k];
        scoreSc = prev[j];
        blobType = prevBlobType[j];

        // Compute the NMS
        if (strictMaximum)
          nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
          nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
                    scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
                    scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;
          
          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().class_id = blobType;
          pr[j] = scoreSc;
        }
      }
    } // Here the Y axis sliding window loop finishes    
  }


  void FASTblob_conditioned(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                              int threshold, int nonmax_suppression, float scale, double responsethr, int scoreType,
                              bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
                              int minArcLength, int maxArcLength )
  {
    double scEps = 2.0, threshold2;
    double st;
    const Mat img = _img.getMat();

    Mat binImg; // Mask of all pixels that fulfill the 1st and 2nd condition
    binImg = Mat::zeros(img.rows, img.cols, CV_8UC1);

    int i, j, k, pixel[25], pixel_inner[25], pixel_mid[25];
    int rej2=0, rej3=0, rej4=0, accp=0;
    makeOffsets(pixel, (int)img.step, 16); //patternSize
    makeOffsets(pixel_inner, (int)img.step, 8);
    makeOffsets(pixel_mid, (int)img.step, 12);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);
    threshold2 = scEps*(double)threshold;

    // Allocating memory for all the buffer, each sizeof corresponds to one buffer
    AutoBuffer<double> _bufScCp((img.cols+16)*3*(3*sizeof(double) + 2*sizeof(int) + 2*sizeof(uchar)) + 128 );


    // Set the pointers for SCORES
    double* bufSc[3];
    bufSc[0] = _bufScCp;
    bufSc[1] = bufSc[0] + img.cols;
    bufSc[2] = bufSc[1] + img.cols;
    memset(bufSc[0], 0, img.cols*3*sizeof(double));

    // Set the pointers for COORDINATES POINTS
    int* bufCp[3];
    bufCp[0] = (int*)alignPtr(bufSc[2] + img.cols, sizeof(int)) + 1;
    bufCp[1] = bufCp[0] + img.cols + 1;
    bufCp[2] = bufCp[1] + img.cols + 1;
    memset(bufCp[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufV[3];
    bufV[0] = (double*)alignPtr(bufCp[2] + img.cols + 1, sizeof(double));
    bufV[1] = bufV[0] + img.cols;
    bufV[2] = bufV[1] + img.cols;

    // Save Deltas
    uchar* bufDl[3];
    bufDl[0] = (uchar*)alignPtr(bufV[2] + img.cols, sizeof(uchar));
    bufDl[1] = bufDl[0] + img.cols;
    bufDl[2] = bufDl[1] + img.cols;


    // Memory allocation for the BLOB locations
    int* bufBlobPos[3];
    bufBlobPos[0] = (int*)alignPtr(bufDl[2] + img.cols, sizeof(int)) + 1;
    bufBlobPos[1] = bufBlobPos[0] + img.cols + 1;
    bufBlobPos[2] = bufBlobPos[1] + img.cols + 1;
    memset(bufBlobPos[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufBlobSc[3];
    bufBlobSc[0] = (double*)alignPtr(bufBlobPos[2] + img.cols + 1, sizeof(double));
    bufBlobSc[1] = bufBlobSc[0] + img.cols;
    bufBlobSc[2] = bufBlobSc[1] + img.cols;
    memset(bufBlobSc[0], 0, img.cols*3*sizeof(double));

    uchar* bufBlobTy[3];
    bufBlobTy[0] = (uchar*)alignPtr(bufBlobSc[2] + img.cols, sizeof(uchar));
    bufBlobTy[1] = bufBlobTy[0] + img.cols;
    bufBlobTy[2] = bufBlobTy[1] + img.cols;

    int idx;
    uchar p_regs, count_elem;
    uchar *labels, *begins, *lengths;
    int* lbl;

    labels  = new uchar[9];
    begins  = new uchar[9];
    lengths = new uchar[9];

    for(i = 3; i < img.rows-2; i++)
    {
      const uchar* ptr = img.ptr<uchar>(i) + 3;
      double* curr = bufSc[(i - 3)%3];
      double* currBlobSc = bufBlobSc[(i - 3)%3];
      double* currV = bufV[(i - 3)%3];
      uchar* currDl = bufDl[(i - 3)%3];
      uchar* currBlobTy = bufBlobTy[(i - 3)%3];
      int* cornerpos = bufCp[(i - 3)%3];
      int* blobpos = bufBlobPos[(i - 3)%3];

      memset(curr, 0, img.cols*sizeof(double));
      memset(currBlobSc, 0, img.cols*sizeof(double));
      int ncorners = 0, nblobs = 0;
      

      if( i < img.rows - 3 )
      {
        j = 3;

        for( ; j < img.cols - 3; j++, ptr++)
        {        
          double v = 0.0, A = 0.0, B = 0.0, C = 0.0, D = 0.0;
          uchar N = 0, blob_type, delta;

          inner_test(pixel_inner, pixel_mid, pixel, ptr, A, B, C, D, N, innerTstType);
          if (!N)
          {
            blob_test(pixel_mid, pixel, ptr, blob_type);
            if (blob_type)
            {
              blobpos[nblobs++] = j;
              currBlobSc[j] = cmpFeatureScore(ptr, pixel, lbl, 0.0, 0, SORB::HESS_SCORE);
              currBlobTy[j] = blob_type;
            }
            continue;
          }

          delta = std::max( A-B, C-D );
          if (N == 4)
          {
            if ((A >= D) && (B <= C))
              v = std::min(A,C) + std::max (B,D);
            else
              continue;
          }
          else
            v = std::max( A+B, C+D );
          v *= 0.5;

          double upperThr, lowerThr, upperThr2, lowerThr2;

          if (threshold > 0)
          {
            upperThr = v + (double)threshold;
            lowerThr = v - (double)threshold;
            upperThr2 = v + threshold2;
            lowerThr2 = v - threshold2;
          }
          else
          {
            upperThr = v + (double)(0.5*delta);
            lowerThr = v - (double)(0.5*delta);
            upperThr2 = v + (scEps*0.5*(double)delta);
            lowerThr2 = v - (scEps*0.5*(double)delta);
          }

          int templateLarge[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
          for (k = 0; k < 16; k++)
          {
            if ( (double)ptr[pixel[k]] > upperThr ) // GREEN
              templateLarge[k] = 2;
            else if ( (double)ptr[pixel[k]] < lowerThr ) // RED
              templateLarge[k] = 1;

            // FIRST brighter or darker
            if (((templateLarge[k]==1) && (templateLarge[k-1]==2) && (ptr[pixel[k]] > lowerThr2)) ||
                ((templateLarge[k]==2) && (templateLarge[k-1]==1) && (ptr[pixel[k]] < upperThr2)))
              templateLarge[k] = 0;
          }

          // Find the position of the first swap
          k = 1;
          while ( (k <= maxArcLength) && (templateLarge[k-1] == templateLarge[k]) )
            k++;

          if (k > maxArcLength)
            continue;

          // Registers for template checking
          uchar n_label[] = {0,0,0};

          labels[0] = templateLarge[k];
          n_label[templateLarge[k]]++;
          begins[0] = k++;
          count_elem = 1;
          p_regs = 0;

          for (uchar pt=k; pt<k+15; pt++ )
          {
            idx = pt % 16;
            if (labels[p_regs] != templateLarge[idx])
            {
              labels[p_regs+1] = templateLarge[idx];
              n_label[labels[p_regs+1]]++;
              begins[p_regs+1] = idx;
              lengths[ p_regs++] = count_elem;
              count_elem = 1;

              if (p_regs>7)
                break;
            }
            else
              count_elem++;
          }
          lengths[p_regs++] = count_elem;

          // ----------------- Constrains ----------------------- //

          // Number of arcs constrains
          if ((p_regs > 8) || (p_regs < 4)  || (n_label[0] > 4) || (n_label[1] != 2) || (n_label[2] != 2))
            continue;

          // Arc length constrains
          bool discard=0;
          uchar red_green_labels[4], *p_redgreen;
          p_redgreen = red_green_labels;
          for ( int m=0; m<p_regs && !discard; m++ )
          {
            if (labels[m] == 0)
              discard = (lengths[m]>2);
            else
            {
              *p_redgreen++ = labels[m];
              discard = ( (lengths[m] < minArcLength) || (lengths[m] > maxArcLength) );
            }
          }

          if ( discard || (red_green_labels[0] != red_green_labels[2]))
            continue;

          // Include the feature in the set
          cornerpos[ncorners++] = j;
          currV[j] = v;
          currDl[j] = delta;

          // Compute the feature response
          int* lbl = templateLarge;
          curr[j] = cmpFeatureScore(ptr, pixel, lbl, v, delta, scoreType);

          // Save the point in the binary image
          uchar* ptrBinary = binImg.ptr<uchar>(i);
          ptrBinary[j] = 255;
        }
      }
      cornerpos[-1] = ncorners;
      blobpos[-1] = nblobs;

      /*   Collecting the SADDLES   */
      const double*  prev = bufSc[(i - 4 + 3)%3];
      const double* pprev = bufSc[(i - 5 + 3)%3];
      const double* prevV = bufV [(i - 4 + 3)%3];
      const uchar* prevDl = bufDl[(i - 4 + 3)%3];
      const uchar* prevBlobType = bufBlobTy[(i - 4 + 3)%3];

      double* pr = _resp.ptr<double>(i - 1);
      bool nmsFlag;
      float scoreSc;
      double v;
      unsigned char delta, blobType;

      cornerpos = bufCp[(i - 4 + 3)%3];
      ncorners = cornerpos[-1];

      for( k = 0; k < ncorners; k++ )
      {
        j = cornerpos[k];
        scoreSc = prev[j];
        v = prevV[j];
        delta = prevDl[j];

        // Compute the NMS
        if (strictMaximum)
          nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
          nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
                    scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
                    scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;

          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().intensityCenter = v;
          keypoints.back().delta = delta;
          keypoints.back().class_id = 0;
          add_labelling_array(img, j, i, keypoints, v, threshold, pixel);
          pr[j] = scoreSc;
        }
      }

      /*   Collecting the BLOBS   */
      prev  = bufBlobSc[(i - 4 + 3)%3];
      pprev = bufBlobSc[(i - 5 + 3)%3];
      blobpos  = bufBlobPos[(i - 4 + 3)%3];
      nblobs = blobpos[-1];

      for( k = 0; k < nblobs; k++ )
      {
        j = blobpos[k];
        scoreSc = prev[j];
        blobType = prevBlobType[j];

        // Compute the NMS
        if (strictMaximum)
          nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
          nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
                    scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
                    scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;
          
          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().class_id = blobType;
          pr[j] = scoreSc;
        }
      }
    } // Here the Y axis sliding window loop finishes    
  }


  void FASTsaddle_conditioned(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
                              int threshold, int nonmax_suppression, float scale, double responsethr, int scoreType,
                              bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType,
                              int minArcLength, int maxArcLength )
  {
    double scEps = 2.0, threshold2;
    double st;
    const Mat img = _img.getMat();

    Mat binImg; // Mask of all pixels that fulfill the 1st and 2nd condition
    binImg = Mat::zeros(img.rows, img.cols, CV_8UC1);

    int i, j, k, pixel[25], pixel_inner[25], pixel_mid[25];
    int rej2=0, rej3=0, rej4=0, accp=0;
    makeOffsets(pixel, (int)img.step, 16); //patternSize
    makeOffsets(pixel_inner, (int)img.step, 8);
    makeOffsets(pixel_mid, (int)img.step, 12);

    keypoints.clear();

    threshold = std::min(std::max(threshold, 0), 255);
    threshold2 = scEps*(double)threshold;

    // Allocating memory for all the buffer, each sizeof corresponds to one buffer
    AutoBuffer<double> _bufScCp((img.cols+16)*3*(3*sizeof(double) + 2*sizeof(int) + 2*sizeof(uchar)) + 128 );


    // Set the pointers for SCORES
    double* bufSc[3];
    bufSc[0] = _bufScCp;
    bufSc[1] = bufSc[0] + img.cols;
    bufSc[2] = bufSc[1] + img.cols;
    memset(bufSc[0], 0, img.cols*3*sizeof(double));

    // Set the pointers for COORDINATES POINTS
    int* bufCp[3];
    bufCp[0] = (int*)alignPtr(bufSc[2] + img.cols, sizeof(int)) + 1;
    bufCp[1] = bufCp[0] + img.cols + 1;
    bufCp[2] = bufCp[1] + img.cols + 1;
    memset(bufCp[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufV[3];
    bufV[0] = (double*)alignPtr(bufCp[2] + img.cols + 1, sizeof(double));
    bufV[1] = bufV[0] + img.cols;
    bufV[2] = bufV[1] + img.cols;

    // Save Deltas
    uchar* bufDl[3];
    bufDl[0] = (uchar*)alignPtr(bufV[2] + img.cols, sizeof(uchar));
    bufDl[1] = bufDl[0] + img.cols;
    bufDl[2] = bufDl[1] + img.cols;


    // Memory allocation for the BLOB locations
    int* bufBlobPos[3];
    bufBlobPos[0] = (int*)alignPtr(bufDl[2] + img.cols, sizeof(int)) + 1;
    bufBlobPos[1] = bufBlobPos[0] + img.cols + 1;
    bufBlobPos[2] = bufBlobPos[1] + img.cols + 1;
    memset(bufBlobPos[0], 0, (img.cols + 1)*3*sizeof(int));

    double* bufBlobSc[3];
    bufBlobSc[0] = (double*)alignPtr(bufBlobPos[2] + img.cols + 1, sizeof(double));
    bufBlobSc[1] = bufBlobSc[0] + img.cols;
    bufBlobSc[2] = bufBlobSc[1] + img.cols;
    memset(bufBlobSc[0], 0, img.cols*3*sizeof(double));

    uchar* bufBlobTy[3];
    bufBlobTy[0] = (uchar*)alignPtr(bufBlobSc[2] + img.cols, sizeof(uchar));
    bufBlobTy[1] = bufBlobTy[0] + img.cols;
    bufBlobTy[2] = bufBlobTy[1] + img.cols;

    int idx;
    uchar p_regs, count_elem;
    uchar *labels, *begins, *lengths;
    int* lbl;

    labels  = new uchar[9];
    begins  = new uchar[9];
    lengths = new uchar[9];

    for(i = 3; i < img.rows-2; i++)
    {
      const uchar* ptr = img.ptr<uchar>(i) + 3;
      double* curr = bufSc[(i - 3)%3];
      double* currBlobSc = bufBlobSc[(i - 3)%3];
      double* currV = bufV[(i - 3)%3];
      uchar* currDl = bufDl[(i - 3)%3];
      uchar* currBlobTy = bufBlobTy[(i - 3)%3];
      int* cornerpos = bufCp[(i - 3)%3];
      int* blobpos = bufBlobPos[(i - 3)%3];

      memset(curr, 0, img.cols*sizeof(double));
      memset(currBlobSc, 0, img.cols*sizeof(double));
      int ncorners = 0, nblobs = 0;
      

      if( i < img.rows - 3 )
      {
        j = 3;

        for( ; j < img.cols - 3; j++, ptr++)
        {        
          double v = 0.0, A = 0.0, B = 0.0, C = 0.0, D = 0.0;
          uchar N = 0, blob_type, delta;

          blob_test(pixel_mid, pixel, ptr, blob_type);
          if (blob_type)
          {
            blobpos[nblobs++] = j;
            currBlobSc[j] = cmpFeatureScore(ptr, pixel, lbl, 0.0, 0, SORB::HESS_SCORE);
            currBlobTy[j] = blob_type;
            continue;
          }
          
          inner_test(pixel_inner, pixel_mid, pixel, ptr, A, B, C, D, N, innerTstType);
          if (!N)
            continue;

          delta = std::max( A-B, C-D );
          if (N == 4)
          {
            if ((A >= D) && (B <= C))
              v = std::min(A,C) + std::max (B,D);
            else
              continue;
          }
          else
            v = std::max( A+B, C+D );
          v *= 0.5;

          double upperThr, lowerThr, upperThr2, lowerThr2;

          if (threshold > 0)
          {
            upperThr = v + (double)threshold;
            lowerThr = v - (double)threshold;
            upperThr2 = v + threshold2;
            lowerThr2 = v - threshold2;
          }
          else
          {
            upperThr = v + (double)(0.5*delta);
            lowerThr = v - (double)(0.5*delta);
            upperThr2 = v + (scEps*0.5*(double)delta);
            lowerThr2 = v - (scEps*0.5*(double)delta);
          }

          int templateLarge[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
          for (k = 0; k < 16; k++)
          {
            if ( (double)ptr[pixel[k]] > upperThr ) // GREEN
              templateLarge[k] = 2;
            else if ( (double)ptr[pixel[k]] < lowerThr ) // RED
              templateLarge[k] = 1;

            // FIRST brighter or darker
            if (((templateLarge[k]==1) && (templateLarge[k-1]==2) && (ptr[pixel[k]] > lowerThr2)) ||
                ((templateLarge[k]==2) && (templateLarge[k-1]==1) && (ptr[pixel[k]] < upperThr2)))
              templateLarge[k] = 0;
          }

          // Find the position of the first swap
          k = 1;
          while ( (k <= maxArcLength) && (templateLarge[k-1] == templateLarge[k]) )
            k++;

          if (k > maxArcLength)
            continue;

          // Registers for template checking
          uchar n_label[] = {0,0,0};

          labels[0] = templateLarge[k];
          n_label[templateLarge[k]]++;
          begins[0] = k++;
          count_elem = 1;
          p_regs = 0;

          for (uchar pt=k; pt<k+15; pt++ )
          {
            idx = pt % 16;
            if (labels[p_regs] != templateLarge[idx])
            {
              labels[p_regs+1] = templateLarge[idx];
              n_label[labels[p_regs+1]]++;
              begins[p_regs+1] = idx;
              lengths[ p_regs++] = count_elem;
              count_elem = 1;

              if (p_regs>7)
                break;
            }
            else
              count_elem++;
          }
          lengths[p_regs++] = count_elem;

          // ----------------- Constrains ----------------------- //

          // Number of arcs constrains
          if ((p_regs > 8) || (p_regs < 4)  || (n_label[0] > 4) || (n_label[1] != 2) || (n_label[2] != 2))
            continue;

          // Arc length constrains
          bool discard=0;
          uchar red_green_labels[4], *p_redgreen;
          p_redgreen = red_green_labels;
          for ( int m=0; m<p_regs && !discard; m++ )
          {
            if (labels[m] == 0)
              discard = (lengths[m]>2);
            else
            {
              *p_redgreen++ = labels[m];
              discard = ( (lengths[m] < minArcLength) || (lengths[m] > maxArcLength) );
            }
          }

          if ( discard || (red_green_labels[0] != red_green_labels[2]))
            continue;

          // Include the feature in the set
          cornerpos[ncorners++] = j;
          currV[j] = v;
          currDl[j] = delta;

          // Compute the feature response
          int* lbl = templateLarge;
          curr[j] = cmpFeatureScore(ptr, pixel, lbl, v, delta, scoreType);

          // Save the point in the binary image
          uchar* ptrBinary = binImg.ptr<uchar>(i);
          ptrBinary[j] = 255;
        }
      }
      cornerpos[-1] = ncorners;
      blobpos[-1] = nblobs;

      /*   Collecting the SADDLES   */
      const double*  prev = bufSc[(i - 4 + 3)%3];
      const double* pprev = bufSc[(i - 5 + 3)%3];
      const double* prevV = bufV [(i - 4 + 3)%3];
      const uchar* prevDl = bufDl[(i - 4 + 3)%3];
      const uchar* prevBlobType = bufBlobTy[(i - 4 + 3)%3];

      double* pr = _resp.ptr<double>(i - 1);
      bool nmsFlag;
      float scoreSc;
      double v;
      unsigned char delta, blobType;

      cornerpos = bufCp[(i - 4 + 3)%3];
      ncorners = cornerpos[-1];

      for( k = 0; k < ncorners; k++ )
      {
        j = cornerpos[k];
        scoreSc = prev[j];
        v = prevV[j];
        delta = prevDl[j];

        // Compute the NMS
        if (strictMaximum)
          nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
          nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
                    scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
                    scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;

          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().intensityCenter = v;
          keypoints.back().delta = delta;
          keypoints.back().class_id = 0;
          add_labelling_array(img, j, i, keypoints, v, threshold, pixel);
          pr[j] = scoreSc;
        }
      }

      /*   Collecting the BLOBS   */
      prev  = bufBlobSc[(i - 4 + 3)%3];
      pprev = bufBlobSc[(i - 5 + 3)%3];
      blobpos  = bufBlobPos[(i - 4 + 3)%3];
      nblobs = blobpos[-1];

      for( k = 0; k < nblobs; k++ )
      {
        j = blobpos[k];
        scoreSc = prev[j];
        blobType = prevBlobType[j];

        // Compute the NMS
        if (strictMaximum)
          nmsFlag = scoreSc > responsethr && scoreSc > prev[j+1] && scoreSc > prev[j-1] &&
                    scoreSc > pprev[j-1] && scoreSc > pprev[j] && scoreSc > pprev[j+1] &&
                    scoreSc > curr[j-1] && scoreSc > curr[j] && scoreSc > curr[j+1];
        else
          nmsFlag = scoreSc >= responsethr && scoreSc >= prev[j+1] && scoreSc >= prev[j-1] &&
                    scoreSc >= pprev[j-1] && scoreSc >= pprev[j] && scoreSc >= pprev[j+1] &&
                    scoreSc >= curr[j-1] && scoreSc >= curr[j] && scoreSc >= curr[j+1];

        if(!(nonmax_suppression>0) || nmsFlag)
        {
          float thetaX, thetaY;
          
          subpixel_precision(j, i, curr, prev, pprev, thetaX, thetaY, scoreSc, subPixPrecision);
          keypoints.push_back(SadKeyPoint(thetaX, thetaY, 7.f, -1, scoreSc, 1.f ));
          keypoints.back().class_id = blobType;
          pr[j] = scoreSc;
        }
      }
    } // Here the Y axis sliding window loop finishes    
  }


  /*--------------- My FAST detector for SADDLE with inner pattern with simpler implementation  (End) -------------------*/

  
  void FASTX(InputArray _img, std::vector<SadKeyPoint>& keypoints, int threshold, int nonmax_suppression, int type)
  {
    switch(type) {
      case FastFeatureDetector::TYPE_5_8:
        FAST_t<8>(_img, keypoints, threshold, nonmax_suppression);
        break;
      case FastFeatureDetector::TYPE_7_12:
        FAST_t<12>(_img, keypoints, threshold, nonmax_suppression);
        break;
      case FastFeatureDetector::TYPE_9_16:
#ifdef HAVE_TEGRA_OPTIMIZATION
        if(tegra::FAST(_img, keypoints, threshold, nonmax_suppression))
          break;
#endif
        FAST_t<16>(_img, keypoints, threshold, nonmax_suppression);
        break;

      }
  }

  void FASTX2(InputArray _img, std::vector<SadKeyPoint>& keypoints, Mat& _resp,
              int threshold, int nonmax_suppression, int type, float scale, double responsethr, uchar deltaThr, int scoreType,
			  bool allC1feats, bool strictMaximum, int subPixPrecision, bool gravityCenter, int innerTstType, int minArcLength, int maxArcLength )
  {
    switch(type) {
      case FastFeatureDetector::TYPE_SADDLE_CENTRAL_PIXEL:
        FASTsaddle_central<16>(_img, keypoints, threshold, nonmax_suppression);
        break;
      case FastFeatureDetector::TYPE_SADDLE_INNER_PATTERN:
        FASTsaddle_inner(_img, keypoints, _resp, threshold, nonmax_suppression, scale, responsethr, deltaThr, scoreType,
        					allC1feats, strictMaximum, subPixPrecision, gravityCenter, innerTstType, minArcLength, maxArcLength );
        break;
      case FastFeatureDetector::TYPE_SHADDLE:
    	  FASTsaddle_shinner(_img, keypoints, _resp, threshold, nonmax_suppression, scale, responsethr, deltaThr, scoreType,
              				allC1feats, strictMaximum, subPixPrecision, gravityCenter, innerTstType, minArcLength, maxArcLength );
        break;
      case FastFeatureDetector::TYPE_SADDLE_CONDITIONED:
        FASTsaddle_conditioned(_img, keypoints, _resp, threshold, nonmax_suppression, scale, responsethr, scoreType,
                      strictMaximum, subPixPrecision, gravityCenter, innerTstType, minArcLength, maxArcLength );
        break;
      case FastFeatureDetector::TYPE_BLOB_CONDITIONED:
        FASTblob_conditioned(_img, keypoints, _resp, threshold, nonmax_suppression, scale, responsethr, scoreType,
                      strictMaximum, subPixPrecision, gravityCenter, innerTstType, minArcLength, maxArcLength );
        break;
      case FastFeatureDetector::TYPE_SADDLE_BLOB:
        FASTsaddle_blob(_img, keypoints, _resp, threshold, nonmax_suppression, scale, responsethr, scoreType,
                      strictMaximum, subPixPrecision, gravityCenter, innerTstType, minArcLength, maxArcLength );
        break;
      }
  }

  void FAST(InputArray _img, std::vector<SadKeyPoint>& keypoints, int threshold, int nonmax_suppression)
  {
    cmp::FASTX(_img, keypoints, threshold, nonmax_suppression, FastFeatureDetector::TYPE_9_16);
  }

  /*
 *   FastFeatureDetector
 */
  FastFeatureDetector::FastFeatureDetector( int _threshold, int _nonmaxSuppression )
    : threshold(_threshold), nonmaxSuppression(_nonmaxSuppression)
  {}

  void FastFeatureDetector::detect2( const Mat& image, vector<SadKeyPoint>& keypoints,
                                     Mat& resp, const Mat& mask ) const
  {
    keypoints.clear();

    if( image.empty() )
      return;

    CV_Assert( resp.empty() || mask.empty() || (mask.type() == CV_8UC1 && mask.size() == image.size()) );

    // detectImpl( image, keypoints, mask );
    detectImpl2( image, keypoints, resp, mask);
  }

  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type(cmp::FastFeatureDetector::TYPE_9_16), scale(1.0), responsethr(0.0), deltaThr(0), scoreType(cmp::SORB::DELTA_SCORE), allC1feats(false), strictMaximum(false), subPixPrecision(0), gravityCenter(false), innerTstType(0), minArcLength(2), maxArcLength(8)
  {}

  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression, int _type )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type((short)_type), scale(1.0), responsethr(0.0), deltaThr(0), scoreType(cmp::SORB::DELTA_SCORE), allC1feats(false), strictMaximum(false), subPixPrecision(0), gravityCenter(false), innerTstType(0), minArcLength(2), maxArcLength(8)
  {}
  // -------- Javier Aldana --------
  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression, int _type, float _scale )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type((short)_type), scale((float)_scale), responsethr(0.0), deltaThr(0), scoreType(cmp::SORB::DELTA_SCORE), allC1feats(false), strictMaximum(false), subPixPrecision(0), gravityCenter(false), innerTstType(0), minArcLength(2), maxArcLength(8)
  {}

  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression, int _type, float _scale, double _responsethr )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type((short)_type), scale((float)_scale), responsethr((double)_responsethr), deltaThr(0), scoreType(cmp::SORB::DELTA_SCORE), allC1feats(false), strictMaximum(false), subPixPrecision(0), gravityCenter(false), innerTstType(0), minArcLength(2), maxArcLength(8)
  {}

  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression, int _type, float _scale, double _responsethr, uchar _deltaThr )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type((short)_type), scale((float)_scale), responsethr((double)_responsethr), deltaThr((uchar)_deltaThr), scoreType(cmp::SORB::DELTA_SCORE), allC1feats(false), strictMaximum(false), subPixPrecision(0), gravityCenter(false), innerTstType(0), minArcLength(2), maxArcLength(8)
  {}
  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression, int _type, float _scale, double _responsethr, uchar _deltaThr, int _scoreType )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type((short)_type), scale((float)_scale), responsethr((double)_responsethr), deltaThr((uchar)_deltaThr), scoreType((int)_scoreType), allC1feats(false), strictMaximum(false), subPixPrecision(0), gravityCenter(false), innerTstType(0), minArcLength(2), maxArcLength(8)
  {}
  FastFeatureDetector2::FastFeatureDetector2( int _threshold, int _nonmaxSuppression, int _type, float _scale, double _responsethr, uchar _deltaThr, int _scoreType, bool _allC1feats, bool _strictMaximum, int _subPixPrecision, bool _gravityCenter, int _innerTstType, int _minArcLength, int _maxArcLength )
    : FastFeatureDetector(_threshold, _nonmaxSuppression), type((short)_type), scale((float)_scale), responsethr((double)_responsethr), deltaThr((uchar)_deltaThr), scoreType((int)_scoreType), allC1feats((bool)_allC1feats), strictMaximum((bool)_strictMaximum), subPixPrecision((int)_subPixPrecision), gravityCenter((bool)_gravityCenter), innerTstType((int) _innerTstType), minArcLength((int)_minArcLength), maxArcLength((int)_maxArcLength)
  {}
  // I changed here (FastFeatureDetector2, KeyPoint)
  void FastFeatureDetector2::detectImpl( const Mat& image, vector<SadKeyPoint>& keypoints, const Mat& mask ) const
  {
    Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );
    cmp::FASTX( grayImage, keypoints, threshold, nonmaxSuppression, type );
    // KeyPointsFilter::runByPixelsMask( keypoints, mask );
  }

  void FastFeatureDetector2::detectImpl2( const Mat& image, vector<SadKeyPoint>& keypoints,
                                          Mat& resp, const Mat& mask ) const
  {
    Mat grayImage = image;
    // The image is already in gray scale from SORB functions
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );
    cmp::FASTX2( grayImage, keypoints, resp, threshold, nonmaxSuppression, type, scale, responsethr, deltaThr, scoreType,
    			 allC1feats, strictMaximum, subPixPrecision, gravityCenter, innerTstType, minArcLength, maxArcLength );
  }



double FitQuadratic(double offset[2], const double* resp_up, const double* resp_cent, const double* resp_down, int c)
{

     double g[2];
     double H[2][2];

     /* Fill in the values of the gradient from pixel differences. */
     g[0] = (resp_up[c]     - resp_down[c]  ) / 2.0f;
     g[1] = (resp_cent[c+1] - resp_cent[c-1]) / 2.0f;

     /* Fill in the values of the Hessian from pixel differences. */
     H[0][0] = resp_up[c]     - 2.0 * resp_cent[c] + resp_down[c];
     H[1][1] = resp_cent[c-1] - 2.0 * resp_cent[c] + resp_cent[c+1];

     H[0][1] = H[1][0] = ((resp_down[c+1] - resp_down[c-1]) -
    		 	 	 	  (resp_up[c+1]   - resp_up[c-1]) ) / 4.0f;
     Mat Hmat( 2, 2, CV_64FC1, &H);

     /* Solve the 3x3 linear sytem, Hx = -g.  Result gives peak offset.
        Note that SolveLinearSystem destroys contents of H. */
     offset[0] = - g[0];
     offset[1] = - g[1];
     Mat offsetmat( 2, 1, CV_64FC1, offset);

     Mat shifting(2,1,CV_64FC1);
     solve(Hmat, offsetmat, shifting );

     double* ptrShift = shifting.ptr<double>(0);
     offset[0] = ptrShift[0];
     offset[1] = ptrShift[1];

     /* Also return value of DOG at peak location using initial value plus
        0.5 times linear interpolation with gradient to peak position
        (this is correct for a quadratic approximation). */

     float v=0; for(int i=0;i<2;i++) v+=offset[i]*g[i];
     return (resp_cent[c] + 0.5f * v);

}

} // namespace cmp
