// Copyright[2017] <Zhu Zeyu>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "helper.h"

using namespace cv;

// const int BLUR_KERNEL = 3;

int canny_low_threshold = 100;
const int CANNY_MAX_LOW_THRESHOLD = 100;
const int canny_ratio = 2;

int hough_low_threshold = 200;
const int HOUGH_MAX_LOW_THRESHOLD = 500;

const double MIN_SLOPE_THRESHOLD = 0.1;
const double MAX_SLOPE_THRESHOLD = 10;

const double CLUSTER_DIS_THRESHOLD = 0.1;

const double CAM_SPIN_DIS = 0;
const double ANGLE_OF_VIEW = 90 * CV_PI / 180;

Mat src, dst, color_dst;

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
    return atan((vp.c[0] - double(width) / 2) / (CAM_SPIN_DIS + float(height) / 2 / tan(ANGLE_OF_VIEW / 2)));
}

/*
Steps for Vanishing Point Detection:
1. Drop the lines with slopes that are too great or too little; other lines are denoted as L_i
2. Calculate V_{i,j}=\intersect{L_i, L_j}
3. Drop V_{i,j} that is outside the border of the image
4. Decide the perspective mode (1/2 vanishing point(s))
5. For the rest, the result V=\frac{\sum{V_{i,j}\times{{Len_i}\times{Len_j}}}}{\sum{{Len_i}\times{Len_j}}
*/

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
            Vec4i line1 = lines[ind_filtered_lines[i]];
            Vec4i line2 = lines[ind_filtered_lines[j]];
            Double2 vp = linesIntersect(line1, line2);
            if (!intersectIsPillar(line1, line2, vp)) {
                vps.push_back(vp);
                ind_vps_lines.push_back(Size_t2(j, i));
            }
        }

    // Mark the intersections of filtered lines
    for (size_t i = 0; i < vps.size(); ++i) {
        Point pt = Point(int(vps[i].c[0]), int(vps[i].c[1]));
        circle(color_dst, pt, 2, Scalar(0, 255, 0));
    }

    // Step 4 & 5
    size_t num_clusters = 0;
    std::vector<long> cluster_belonging(vps.size(), -1);
    std::vector<std::vector<size_t> > ind_clusters_vps;
    std::vector<Double2> cluster_center;
    std::vector<bool> visited(vps.size(), false);

	for (size_t i = 0; i < vps.size(); ++i) {
        if (visited[i])
            continue;
        else {
            visited[i] = true;
            cluster_belonging[i] = num_clusters++;
            cluster_center.push_back(vps[i]);
            ind_clusters_vps.push_back(std::vector<size_t>());
            ind_clusters_vps[cluster_belonging[i]].push_back(i);
        }
        for (size_t j = i + 1; j < vps.size(); ++j) {
            if (pointsDistance(cluster_center[cluster_belonging[i]], vps[j]) < CLUSTER_DIS_THRESHOLD * width) {
                // Include vps[j] into this cluster
                visited[j] = true;
                cluster_belonging[j] = cluster_belonging[i];
                ind_clusters_vps[cluster_belonging[i]].push_back(j);

                // Update cluster_center
                double x_up = 0, x_down = 0, y_up = 0, y_down = 0;
                for (size_t k = 0; k < ind_clusters_vps[cluster_belonging[i]].size(); ++k) {
                    double x = vps[ind_clusters_vps[cluster_belonging[i]][k]].c[0];
                    double y = vps[ind_clusters_vps[cluster_belonging[i]][k]].c[1];
                    size_t ind_line1 = ind_vps_lines[ind_clusters_vps[cluster_belonging[i]][k]].s[0];
                    size_t ind_line2 = ind_vps_lines[ind_clusters_vps[cluster_belonging[i]][k]].s[1];
                    double mul_length = lineLength(lines[ind_line1]) * lineLength(lines[ind_line2]);
                    x_up += x * mul_length;
                    y_up += y * mul_length;
                    x_down += mul_length;
                    y_down += mul_length;
                }
                cluster_center[cluster_belonging[i]] = Double2(x_up / x_down, y_up / y_down);
            }
        }
    }

    std::vector<size_t> cluster_sizes(num_clusters, 0);
    for (size_t i = 0; i < vps.size(); ++i)
        cluster_sizes[cluster_belonging[i]]++;
    
    size_t max_cluster_size = 0;
    size_t ind_largest_cluster = -1;
    for (size_t i = 0; i < num_clusters; ++i)
        if (cluster_sizes[i] > max_cluster_size) {
            max_cluster_size = cluster_sizes[i];
            ind_largest_cluster = i;
        }

    if (ind_largest_cluster == -1)
        return Double2(-1, -1);
    else    
        return cluster_center[ind_largest_cluster];
}

void detectLines(int = 0, void* = 0) {
    // blur(src, dst, Size(BLUR_KERNEL, BLUR_KERNEL));

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

    if (vp.c[0] == -1 && vp.c[1] == -1)
        return;

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
            std::cout << src.rows << " * " << src.cols << std::endl;

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
        std::cout << src.rows << " * " << src.cols << std::endl;

        detectLines();

        imshow("Source", src);
        imshow("Detected Lines", color_dst);
        waitKey(0);
    }
    return 0;
}
