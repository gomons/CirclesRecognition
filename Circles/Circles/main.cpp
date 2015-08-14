#include <iostream>
#include <tchar.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;


// Параметры алгоритма
const int       kMedianBlureApertureSize    = 11;

const double    kCannyFirstThreshold        = 50.0f;
const double    kCannySecondThreshold       = 150.0f;

const double    kRectSideMaxRation          = 2.0f;
const double    kMinRectArea                = 500.0f;

const int       kHoughCirclesParam1         = 200;
const int       kHoughCirclesParam2         = 20;


// Константы
const int kColorWhite1 = 255;
const int kColorBlack1 = 0;
const Scalar kColorBlue3 = Scalar(255, 0, 0);
const Scalar kColorRed3 = Scalar(0, 0, 255);


// 1-й аргумент обязательный - путь к файлу с изображением
// 2-й аргумент необязательный - показать результат
int main(int argc, char **argv)
{
    if (argc <= 1) 
    {
        std::cout << "Usage: Circles.exe <file_path> [show]" << std::endl;
        return 1;
    }

    bool showResult = false;
    if (argc > 2) {
        std::string argName(argv[2]);
        std::string showArgName("show");
        if (argName == showArgName)
            showResult = true;
    }

    std::string imageFilepath(argv[1]);
    Mat originImage = imread(imageFilepath);
    if (originImage.empty())
    {
        std::cout << "Error: invalid grayImage " << imageFilepath << std::endl;
        return 2;
    }
    
    // Большинству алгоритмов нужно чернобелое изображение, поэтому сразу делаем его серым.
    Mat grayImage;
    cvtColor(originImage, grayImage, CV_BGR2GRAY);

    // 1. Размываем изображение, чтобы убрать большую часть мелких шумов.
    medianBlur(grayImage, grayImage, kMedianBlureApertureSize);

    // 2. Оставляем только контуры на изображении, чтобы найти на нем нужные фигуры.
    Canny(grayImage, grayImage, kCannyFirstThreshold, kCannySecondThreshold);

    // 3. Получаем список конутров
    std::vector< std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(grayImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    // 4. Среди контуров оставляем только те, которые по форме могут оказаться кружками (но необязательно)
    std::vector< std::vector<Point> > possibleCircles;
    for (size_t i = 0; i < contours.size(); i++)
    { 
        RotatedRect minRect = minAreaRect(Mat(contours[i]));
        float height = minRect.size.height; 
        float width = minRect.size.width;
        float area = minRect.size.area();

        // 4.1. Если соотношение сторон описывающего контур прямоугольника сильно отличается от kRectSideMaxRation,
        //      то, скорее всего, контур не является кружочком, в крайнем случае, это сильно вытянутый эллипс.
        // 4.2. Если площадь описывающего контур прямоугольника меньше kMinRectArea, то, скорее всего, это шум.
        if (max(height,width)/min(height,width) > kRectSideMaxRation || area < kMinRectArea) 
            continue;

        // Конутр может оказаться кружком
        possibleCircles.push_back(contours[i]);
    }

    // 5. Проверяем оставшиеся контуры на наличие признаков кружков
    size_t circlesCount = 0;
    for (size_t i = 0; i< possibleCircles.size(); ++i)
    {
        Mat drawing = Mat::zeros(grayImage.size(), CV_8UC1);
        drawContours(drawing, possibleCircles, i, kColorWhite1);       

        std::vector<cv::Vec3f> circles;
        int minRadius = 0, maxRadius = 0;
        HoughCircles(drawing, circles, CV_HOUGH_GRADIENT, 1, drawing.rows, 
                     kHoughCirclesParam1, kHoughCirclesParam2, minRadius, maxRadius);
        if (!circles.empty()) 
        { // Найдены признаки круга. Будет считать, что контур является кружочком.
            circlesCount += 1;

            drawContours(originImage, possibleCircles, i, kColorBlue3, 2);
            for (size_t i = 0; i < circles.size(); ++i) 
            {
                Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                circle(originImage, center, 3, kColorRed3);
                circle(originImage, center, radius, kColorRed3, 3);
            }
        }
    }

    std::cout << "Found " << circlesCount << " circles" << std::endl;

    if (showResult)
    {
        namedWindow("Result", CV_WINDOW_AUTOSIZE);
        imshow("Result", originImage);
        waitKey(0);
    }

    return 0;
}