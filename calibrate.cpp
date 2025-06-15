#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

int main() {
    // 设置棋盘格参数
    const Size boardSize(9, 6);        // 棋盘格内部角点数量 (宽 x 高)
    const float squareSize = 0.02f;   // 单个方格的实际尺寸（单位：米）
    const int minFrames = 15;          // 最小标定帧数

    VideoCapture cap(0);  // 打开默认相机
    cap.set(CAP_PROP_FPS, 30);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    if (!cap.isOpened()) {
        cerr << "无法打开摄像头!" << endl;
        return -1;
    }

    vector<vector<Point3f>> objectPoints;  // 3D世界坐标
    vector<vector<Point2f>> imagePoints;   // 2D图像坐标
    vector<Point2f> cornerPoints;          // 临时存储角点

    // 生成棋盘格3D坐标
    vector<Point3f> obj;
    for (int i = 0; i < boardSize.height; ++i) {
        for (int j = 0; j < boardSize.width; ++j) {
            obj.push_back(Point3f(j * squareSize, i * squareSize, 0));
        }
    }

    Mat frame, gray, frame_left,frame_right;
    int framesCount = 0;
    bool calibrated = false;
    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;

    cout << "按空格键捕获帧，按'c'开始标定，按ESC退出" << endl;

    while (true) {
        cap >> frame;
        frame_right = frame(Rect(640,0,640,480));
        frame_left = frame(Rect(0,0,640,480));
        
        if (frame.empty()) break;
         
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        bool found = findChessboardCorners(
            gray, boardSize, cornerPoints,
            CALIB_CB_ADAPTIVE_THRESH
        );

        // 优化角点位置
        if (found) {
            cornerSubPix(
                gray, cornerPoints, Size(11, 11), Size(-1, -1),
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1)
            );
            drawChessboardCorners(frame, boardSize, cornerPoints, found);
        }

        putText(frame, format("Captured Image: %d/%d", framesCount, minFrames),
                Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        
        imshow("相机标定 - 棋盘格 左目图像", frame_left);
        imshow("相机标定 - 棋盘格 右目图像", frame_right);

        int key = waitKey(1);
        if (key == 27) break;  // ESC退出

        if (key == ' ' && found) {  // 空格键捕获有效帧
            imagePoints.push_back(cornerPoints);
            objectPoints.push_back(obj);
            framesCount++;
            cout << "捕获帧 #" << framesCount << endl;
        }

        if ((key == 'c' || framesCount >= minFrames) && !calibrated) {
            if (framesCount < 3) {
                cerr << "需要至少3帧进行标定!" << endl;
                continue;
            }

            // 执行相机标定
            double rms = calibrateCamera(
                objectPoints, imagePoints, frame.size(),
                cameraMatrix, distCoeffs, rvecs, tvecs
            );

            cout << "\n标定成功! 重投影误差: " << rms << endl;
            cout << "相机内参矩阵:\n" << cameraMatrix << endl;
            cout << "畸变系数: " << distCoeffs.t() << endl;

            // 保存标定结果
            FileStorage fs("camera_calibration.yml", FileStorage::WRITE);
            fs << "camera_matrix" << cameraMatrix;
            fs << "distortion_coefficients" << distCoeffs;
            fs.release();
            cout << "标定结果已保存到 camera_calibration.yml" << endl;

            calibrated = true;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}