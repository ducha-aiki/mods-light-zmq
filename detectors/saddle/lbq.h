#ifndef LBQ_H
#define LBQ_H

namespace Binpat{

enum { ORB_GV = 0,
	   ORB_ORIENTED = 1,
	   Saddle_GV = 2,
	   Saddle_ORIENTED = 3,
	   SURF_GV = 4,
	   SURF_ORIENTED = 5,
	   OCV = 6 };

class BitPatterns
{
public:

    BitPatterns(int pat_id)
        : pat_id_(pat_id)
    {}

    void set_pattern_id(const int pat_id);
    int* get_pattern();

private:
    int pat_id_;
    int* bit_pattern_;
};
} //End namespace Binpat


#endif // LBQ_H