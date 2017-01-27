// Copyright[2017] <Zhu Zeyu>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;

int canny_low_threshold = 100;
const int CANNY_MAX_LOW_THRESHOLD = 100;
const int canny_ratio = 3;

int hough_low_threshold = 30;
const int HOUGH_MAX_LOW_THRESHOLD = 500;

const double MIN_SLOPE_THRESHOLD = 0.1;
const double MAX_SLOPE_THRESHOLD = 10;

const double CAM_SPIN_DIS = 0;
const double ANGLE_OF_VIEW = 90 * CV_PI / 180;

Mat src, dst, color_dst;

class Double2 {
public:
    double c[2];

    Double2(double x, double y) {
        c[0] = x;
        c[1] = y;
    }

    ~Double2() {}
};

class Size_t2 {
public:
    size_t s[2];

    Size_t2(size_t a, size_t b) {
        s[0] = a;
        s[1] = b;
    }

    ~Size_t2() {}
};

/*
Angle Formula:
$$\alpha=\arctan{\frac{d}{s+\frac{h}{2}\cot{{\theta/2}}}}$$

where $\alpha$ is the desired angle,
$d$ is the distance between vanishing point and the center of image (in pixel),
$s$ is the distance between camera and spinning origin of the robot,
$h$ is the height of the image, and
$\theta$ is the angle of view of the camera.
*/

double posAngle(Double2 vp, int height, int width) {
    /*
    double dx = vp.c[0] - double(width) / 2;
    return atan(dx / (CAM_SPIN_DIS + float(height) / 2 / tan(ANGLE_OF_VIEW / 2)));
    */
    return atan(vp.c[0] - double(width) / 2 / (CAM_SPIN_DIS + float(height) / 2 / tan(ANGLE_OF_VIEW / 2)));
}

/*
Steps for Vanishing Point Detection:
1. Drop the lines with slopes that are too great or too little; other lines are denoted as L_i
2. Calculate V_{i,j}=\intersect{L_i, L_j}
3. Drop V_{i,j} that is outside the border of the image
4. For the rest, the result V=\frac{\sum{V_{i,j}\times{{Len_i}\times{Len_j}}}}{\sum{{Len_i}\times{Len_j}}
*/

double lineSlope(Vec4i line) {
    /*
    double dy = line[1] - line[3];
    double dx = line[0] - line[2];
    return dy / dx;
    */
    return (line[1] - line[3]) / (line[0] / line[2]);
}

double lineBias(Vec4i line) {
    /*
    double a = line[0];
    double b = line[1];
    double c = line[2];
    double d = line[3];
    return (a * d - b * c) / (a - c);
    */
    return (line[0] * line[3] - line[1] * line[2]) / (line[0] - line[2]);
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

bool pointInImage(Double2 point, int height, int width) {
    double x = point.c[0];
    double y = point.c[1];
    return x >= 0 && x <= width && y >= 0 && y <= height;
}

Double2 vanishingPoint(std::vector<Vec4i> lines, int height, int width) {
    // Step 1
    std::vector<size_t> ind_filtered_lines;
    for (size_t i = 0; i < lines.size(); ++i)
        if (!(lines[i][0] == lines[i][2] ||
            abs(lineSlope(lines[i])) < MIN_SLOPE_THRESHOLD ||
            abs(lineSlope(lines[i])) > MAX_SLOPE_THRESHOLD))
                ind_filtered_lines.push_back(i);

    // Sketch the filtered lines on color_dst
    std::cout << "Filtered lines: " << ind_filtered_lines.size() << std::endl;
    for (size_t i = 0; i < ind_filtered_lines.size(); i++) {
        line(color_dst, Point(lines[ind_filtered_lines[i]][0], lines[ind_filtered_lines[i]][1]),
        Point(lines[ind_filtered_lines[i]][2], lines[ind_filtered_lines[i]][3]), Scalar(255, 0, 0), 3, 8);
    }

    // Step 2 & 3
    std::vector<Double2> vps;
    std::vector<Size_t2> ind_vps_lines;
    for (size_t i = 0; i < ind_filtered_lines.size(); ++i)
        for (size_t j = 0; j < i; ++j) {
            Double2 vp = linesIntersect(lines[ind_filtered_lines[i]], lines[ind_filtered_lines[j]]);
            if (pointInImage(vp, height, width)) {
                vps.push_back(vp);
                ind_vps_lines.push_back(Size_t2(j, i));
            }
        }

    // Mark the intersections of filtered lines
    for (size_t i = 0; i < vps.size(); ++i) {
        Point pt = Point(int(vps[i].c[0]), int(vps[i].c[1]));
        circle(color_dst, pt, 2, Scalar(0, 255, 0));
    }

    // Step 4
    double x_up = 0, x_down = 0, y_up = 0, y_down = 0;
    for (size_t i = 0; i < vps.size(); ++i) {
        double x = vps[i].c[0];
        double y = vps[i].c[1];
        size_t ind_line1 = ind_vps_lines[i].s[0];
        size_t ind_line2 = ind_vps_lines[i].s[1];
        double mul_length = lineLength(lines[ind_line1]) * lineLength(lines[ind_line2]);
        x_up += x * mul_length;
        y_up += y * mul_length;
        x_down += mul_length;
        y_down += mul_length;
    }
    return Double2(x_up / x_down, y_up / y_down);
}

void detectLines(int = 0, void* = 0) {
    Canny(src, dst, canny_low_threshold, canny_low_threshold * canny_ratio, 3);

    cvtColor(dst, color_dst, COLOR_GRAY2BGR);

    std::vector<Vec4i> lines;
    HoughLinesP(dst, lines, 1, CV_PI/180, 80, hough_low_threshold, 10);

    for (size_t i = 0; i < lines.size(); i++) {
        line(color_dst, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
    }

    Double2 vp = vanishingPoint(lines, src.rows, src.cols);
    std::cout << "(x, y) = (" << vp.c[0] << ", " << vp.c[1] << ")" << std::endl;
    Point pt = Point(vp.c[0], vp.c[1]);
    circle(color_dst, pt, 4, Scalar(255, 0, 255));

    std::cout << "Position angle: " << posAngle(vp, src.rows, src.cols)  * 180 / CV_PI << std::endl;

    imshow("Source", src);
    imshow("Detected Lines", color_dst);
}

int main(int argc, char** argv) {
    namedWindow("Source", WINDOW_NORMAL);
    namedWindow("Detected Lines", WINDOW_NORMAL);

    if (argc == 1) {
        std::cout << "Entering camera mode..." << std::endl;
        createTrackbar("Canny Min Threshold:", "Detected Lines", &canny_low_threshold, CANNY_MAX_LOW_THRESHOLD, detectLines);
        createTrackbar("Hough Min Threshold:", "Detected Lines", &hough_low_threshold, HOUGH_MAX_LOW_THRESHOLD, detectLines);
        VideoCapture cap;
        cap.open(0);
        while (true) {
            cap.read(src);

            detectLines();

            imshow("Source", src);
            imshow("Detected Lines", color_dst);

            if (waitKey(100) < 255)
                break;
        }
    } else {
        std::cout << "Entering picture mode..." << std::endl;
        createTrackbar("Canny Min Threshold:", "Detected Lines", &canny_low_threshold, CANNY_MAX_LOW_THRESHOLD, detectLines);
        createTrackbar("Hough Min Threshold:", "Detected Lines", &hough_low_threshold, HOUGH_MAX_LOW_THRESHOLD, detectLines);
        src = imread(argv[1], 1);

        detectLines();

        imshow("Source", src);
        imshow("Detected Lines", color_dst);
        waitKey(0);
    }
    return 0;
}
