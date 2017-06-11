#include "svo.h"

using namespace cv;
using namespace std;
//using namespace cv::ximgproc;

#define MAX_FRAME_NUM 5000
#define MIN_FEAT_NUM 50


int main(int argc,char** argv){

    //get perspective transformation matrix between two cameras
    //cv::Mat R1,R2,P1,P2,Q;
    //cv::stereoRectify(K,distCoeffs1,K,distCoeffs2,Size(376,1241),leftRright,leftTright,R1,R2,P1,P2,Q);

    char fileNameLeft_1[200];
    char fileNameLeft_2[200];
    char fileNameRight_1[200];
    char fileNameRight_2[200];
    sprintf(fileNameLeft_1,"/home/dhp/DATASET/svo/image_0/%06d.png",0);
    sprintf(fileNameLeft_2,"/home/dhp/DATASET/svo/image_0/%06d.png",1);
    sprintf(fileNameRight_1,"/home/dhp/DATASET/svo/image_1/%06d.png",0);
    sprintf(fileNameRight_2,"/home/dhp/DATASET/svo/image_1/%06d.png",1);

    cout<<fileNameLeft_1<<endl;
    cout<<fileNameLeft_2<<endl;
    cout<<fileNameRight_1<<endl;
    cout<<fileNameRight_2<<endl;

    cv::Mat inImage_1_left = imread(fileNameLeft_1);
    cv::Mat inImage_2_left = imread(fileNameLeft_2);
    cv::Mat inImage_1_right = imread(fileNameRight_1);
    cv::Mat inImage_2_right = imread(fileNameRight_2);


    if ( !inImage_1_left.data || !inImage_2_left.data || !inImage_1_right.data || !inImage_2_right.data ){
        std::cout<< " Error reading images " << std::endl;
        return -1;
    }

    cv::Mat disMap_1,disMap_2;
    getDisMap(inImage_1_left,inImage_1_right,disMap_1);
    getDisMap(inImage_2_left,inImage_2_right,disMap_2);

    vector<Point2f>keypoints_left_1,keypoints_left_2;
    vector<uchar> status;
    featureDetection(inImage_1_left,keypoints_left_1);
    cout<<"feature detection done"<<endl;
    featureTracking(inImage_1_left,inImage_2_left,keypoints_left_1,keypoints_left_2,status);

    Mat img_show = inImage_2_left.clone();
    for ( auto kp:keypoints_left_2 )
        cv::circle(img_show, kp, 2, cv::Scalar(0, 240, 0), 2);
    cv::imshow("Motion Corners", img_show);
    cv::waitKey(0);

    cout<<"Tracking done"<<endl;

    vector<Point3f> pts3d_1,pts3d_2;
    cv::Mat R,t;
    getDisMap(inImage_1_left,inImage_1_right,disMap_1);
    getDisMap(inImage_2_left,inImage_2_right,disMap_2);
    get3dPairs(disMap_1,disMap_2,keypoints_left_1,keypoints_left_2,pts3d_1,pts3d_2);
    for(int i=0;i<pts3d_1.size();i++){
        cout<<pts3d_1.at(i).x<<endl;
        cout<<pts3d_1.at(i).y<<endl;
        cout<<pts3d_1.at(i).z<<endl;
    }
    getPose3d3d(pts3d_1,pts3d_2,R,t);
    Mat traj(1600, 1600, CV_8UC3,Scalar(255,255,255));
    cv::Mat R_f,t_f;
    R_f = R.clone();
    t_f = t.clone();
    char filenameL[200];
    char filenameR[200];
    cv::Mat prevImage = inImage_2_left;
    cv::Mat prevDismap = disMap_2;
    cv::Mat currDismap;
    vector<Point2f> prevFeatures = keypoints_left_2;

    cv::Vec3f prevPosition;
    prevPosition[0]= 0;
    prevPosition[1] = 0;
    prevPosition[2] = 0;
    cv::Mat positionMat = (Mat_<double>(3,1)<<prevPosition[0],prevPosition[1],prevPosition[2]);

    //cv::Mat currImage;
    clock_t begin = clock();
    for(int frameNum = 2;frameNum < MAX_FRAME_NUM;frameNum++){
        sprintf(filenameL,"/home/dhp/DATASET/svo/image_0/%06d.png",frameNum);
        sprintf(filenameR,"/home/dhp/DATASET/svo/image_1/%06d.png",frameNum);
        cout<<filenameL<<endl;
        cout<<filenameR<<endl;

        Mat currImage_l = imread(filenameL);
        Mat currImage_r = imread(filenameR);
        //cvtColor(currImage_l, currImage_l, COLOR_BGR2GRAY);
        //cvtColor(currImage_r, currImage_r, COLOR_BGR2GRAY);
        //cout<<"cvtColor done"<<endl;

        vector<uchar> status;
        vector<Point2f> currFeatures;
        if(!currImage_l.data || !currImage_r.data){
            cout<<"Errors read stereo images"<<endl;
        }
        getDisMap(currImage_l,currImage_r,currDismap);
        cout<<"Disparity map done"<<endl;
        featureTracking(prevImage, currImage_l, prevFeatures, currFeatures, status);

        Mat img_show = currImage_l.clone();
        for ( auto kp:currFeatures )
            cv::circle(img_show, kp, 2, cv::Scalar(0, 240, 0), 2);
        cv::imshow("Motion Corners", img_show);
        cv::waitKey(0);
        cv::imshow("Right Images",currImage_r);
        get3dPairs(prevDismap,currDismap,prevFeatures,currFeatures,pts3d_1,pts3d_2);
        getPose3d3d(pts3d_1,pts3d_2,R,t);

        //absolute orientation
        //cv::Mat positionMat = (Mat_(3,1)<<prevPosition[0],prevPosition[1],prevPosition[2]);
        R_f = R*R_f;
        t_f = t_f + R*t;
        positionMat = R_f*positionMat + t_f;

        //redetect features if feature num less than the thresold
        if(prevFeatures.size() < MIN_FEAT_NUM){
            featureDetection(prevImage,prevFeatures);
            featureTracking(prevImage,currImage_l,prevFeatures,currFeatures,status);
            Mat img_show = currImage_l.clone();
            for ( auto kp:currFeatures )
                cv::circle(img_show, kp, 2, cv::Scalar(0, 240, 0), 2);
            cv::imshow("Motion Corners", img_show);
        }

        prevImage = currImage_l.clone();
        prevFeatures = currFeatures;
        //drawTrajectory(traj,R,t);

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;

        char text[200];
        circle(traj,Point(x,y),1,CV_RGB(0,255,0),2);
        rectangle(traj,Point(10,30),Point(550,50),CV_RGB(0,0,0),CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", positionMat.at<double>(0,0), positionMat.at<double>(1,0), positionMat.at<double>(2,0));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
        imshow( "Trajectory", traj );
    }

    return 0;
}

