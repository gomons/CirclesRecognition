#include <iostream>
#include <tchar.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc, char **argv)
{
    if (argc <= 1) 
    {
        std::cout << "Usage: Circles.exe file_path" << std::endl;
        return 1;
    }

    std::string pictureFilepath(argv[1]);
    Mat inPictureMatrix = imread(pictureFilepath, CV_LOAD_IMAGE_GRAYSCALE);
    Mat outPictureMatrix = inPictureMatrix;

    if (inPictureMatrix.empty())
    {
        std::cout << "Error: invalid picture " << pictureFilepath << std::endl;
        return 2;
    }

    int apertureSize = 11;
    medianBlur(inPictureMatrix, outPictureMatrix, apertureSize);

    cv::Mat contoursPictureMatrix;
    double firstThreshold = 50.0f;
    double secondThreshold = 150.0f;
    cv::Canny(inPictureMatrix, contoursPictureMatrix, firstThreshold, secondThreshold);

    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    int retrivalMode = CV_RETR_EXTERNAL;
    int approximationMethod = CV_CHAIN_APPROX_SIMPLE;
    findContours(contoursPictureMatrix, contours, hierarchy, retrivalMode, approximationMethod);

    std::cout << "Found " << contours.size() << " circles" << std::endl;
   
    return 0;
}