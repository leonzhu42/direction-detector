#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

class Double2 {
public:
    double c[2];

    Double2(double x, double y);
};

class Size_t2 {
public:
    size_t s[2];

    Size_t2(size_t a, size_t b);
};

double lineSlope(Vec4i line);

double lineBias(Vec4i line);

double lineLength(Vec4i line);

Double2 linesIntersect(Vec4i line1, Vec4i line2);

//bool pointInImage(Double2 point, int height, int width);

bool intersectIsPillar(Vec4i line1, Vec4i line2, Double2 intersection);

double pointsDistance(Double2 point1, Double2 point2);