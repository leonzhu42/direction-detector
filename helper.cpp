#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "helper.h"

using namespace cv;

Double2::Double2(double x, double y) {
    c[0] = x;
    c[1] = y;
}

Size_t2::Size_t2(size_t a, size_t b) {
    s[0] = a;
    s[1] = b;
}

double lineSlope(Vec4i line) {
    /*
    double dy = line[1] - line[3];
    double dx = line[0] - line[2];
    return dy / dx;
    */
    return double(line[1] - line[3]) / double(line[0] - line[2]);
}

double lineBias(Vec4i line) {
    /*
    double a = line[0];
    double b = line[1];
    double c = line[2];
    double d = line[3];
    return (a * d - b * c) / (a - c);
    */
    return double(line[0] * line[3] - line[1] * line[2]) / double(line[0] - line[2]);
}

double lineLength(Vec4i line) {
    return sqrt((line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3]));
}

Double2 linesIntersect(Vec4i line1, Vec4i line2) {
    double a = lineSlope(line1);
    double b = lineBias(line1);
    double c = lineSlope(line2);
    double d = lineBias(line2);
    double x = (d - b) / (a - c);
    double y = a * x + b;
    return Double2(x, y);
}

/*
bool pointInImage(Double2 point, int height, int width) {
    double x = point.c[0];
    double y = point.c[1];
    return x >= 0 && x <= width && y >= 0 && y <= height;
}
*/

bool intersectIsPillar(Vec4i line1, Vec4i line2, Double2 intersection) { 
    if (intersection.c[1] - std::max(std::max(line1[1], line1[3]), std::max(line2[1], line2[3])) >= -10)
        return true;
    else
        return false;
}

double pointsDistance(Double2 point1, Double2 point2) {
    return sqrt((point1.c[0] - point2.c[0]) * (point1.c[0] - point2.c[0]) + (point1.c[1] - point2.c[1]) * (point1.c[1] - point2.c[1]));
}