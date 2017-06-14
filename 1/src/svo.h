#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>
#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
//#include <eigen3/Eigr>


using namespace cv;
using namespace std;

const double f = 718.85;
const double cx = 607.19;
const double cy = 185.21;
const double b = 0.5371;  //meters

int fontFace = FONT_HERSHEY_PLAIN;
double fontScale = 1;
int thickness = 1;
cv::Point textOrg(10, 50);

const Mat K = (Mat_<double>(3,3)<<f,0,cx,0,f,cy,0,0,1);
//const cv::Mat distCoeffs1=Mat(1,5,CV_32FC1,Scalar::all(0));
//const cv::Mat distCoeffs2=Mat(1,5,CV_32FC1,Scalar::all(0));
//const cv::Mat leftRright = (Mat_<double>(3,3)<<1,0,0,0,1,0,0,0,1);
//const cv::Mat leftTright = (Mat_<double>(3,1)<<-b,0,0);
const cv::Mat Q = (Mat_<double>(4,4)<<1.0, 0.0, 0.0, -cx,
                                      0.0, 1.0, 0.0, -cy,
                                      0.0, 0.0, 0.0, f,
                                      0.0, 0.0, -1.0/b, 0.0);

const double lambda = 8000.0;
const double sigma = 1.5;
const int winSize = 15;
const int sgbmWinSize = 3;
const int numberOfDisparities = 80;

//get perspective transformation matrix between two cameras
//cv::Mat R1,R2,P1,P2,Q;
//cv::stereoRectify(K,distCoeffs1,K,distCoeffs2,Size(376,1241),leftRright,leftTright,R1,R2,P1,P2,Q);

//normalization
Point2d pixel2cam(const Point2d& p,const Mat& K){
    return Point2d(
        (p.x-K.at<double>(0,2))/K.at<double>(0,0),
        (p.y-K.at<double>(1,2))/K.at<double>(1,1)
    );
}


void featureDetection(Mat img_1, vector<Point2f>& points1){
    vector<KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    KeyPoint::convert(keypoints_1, points1, vector<int>());
}


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status){
    vector<float> err;
    Size winSize=Size(21,21);
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++){
        Point2f pt = points2.at(i- indexCorrection);

        //get rid fo features tracking failed
     	if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)){
     		  if((pt.x<0)||(pt.y<0)){
     		  	status.at(i) = 0;
     		  }
     		  points1.erase (points1.begin() + (i - indexCorrection));
     		  points2.erase (points2.begin() + (i - indexCorrection));
     		  indexCorrection++;
     	}
    }
    /*
    Mat img_show = img_2.clone();
    for ( auto kp:points2 )
        cv::circle(img_show, kp, 2, cv::Scalar(0, 240, 0), 2);
    cv::imshow("Motion Corners", img_show);
    */
}

void getPose3d3d(const vector<Point3f>& pts1,const vector<Point3f>& pts2,Mat& R,Mat& t){
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for ( int i=0; i<N; i++ )
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f( Vec3f(p1) /  N);
    p2 = Point3f( Vec3f(p2) / N);
    vector<Point3f>     q1 ( N ), q2 ( N ); // remove the center
    for ( int i=0; i<N; i++ )
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3f W = Eigen::Matrix3f::Zero();
    for ( int i=0; i<N; i++ )
    {
        W += Eigen::Vector3f ( q1[i].x, q1[i].y, q1[i].z ) * Eigen::Vector3f ( q2[i].x, q2[i].y, q2[i].z ).transpose();
    }
    //cout<<"W="<<W<<endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3f> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();
    //cout<<"U="<<U<<endl;
    //cout<<"V="<<V<<endl;

    Eigen::Matrix3f R_ = U* ( V.transpose() );
    Eigen::Vector3f t_ = Eigen::Vector3f ( p1.x, p1.y, p1.z ) - R_ * Eigen::Vector3f ( p2.x, p2.y, p2.z );

    // convert to cv::Mat
    R = ( Mat_<double> ( 3,3 ) <<
          R_ ( 0,0 ), R_ ( 0,1 ), R_ ( 0,2 ),
          R_ ( 1,0 ), R_ ( 1,1 ), R_ ( 1,2 ),
          R_ ( 2,0 ), R_ ( 2,1 ), R_ ( 2,2 )
        );
    t = ( Mat_<double> ( 3,1 ) << t_ ( 0,0 ), t_ ( 1,0 ), t_ ( 2,0 ) );
}

void _get3dPairs(const Mat& mat1, const Mat& mat2, const vector<Point2f>feat_1,
                const vector<Point2f>feat_2, vector<Point3f>& pts3d_1, vector<Point3f>& pts3d_2){
    const double max_z = 1.0e4;
    for(int i = 0; i < feat_1.size(); i++){
        Vec3f point_1, point_2;
        point_1 = mat1.at<Vec3f>(feat_1.at(i).x,feat_1.at(i).y);
        point_2 = mat2.at<Vec3f>(feat_2.at(i).x,feat_2.at(i).y);
        if(fabs(point_1[2] - max_z) < FLT_EPSILON || fabs(point_1[2]) > max_z)
                continue;
        if(fabs(point_2[2] - max_z) < FLT_EPSILON || fabs(point_2[2]) > max_z)
                continue;
        pts3d_1.push_back(Point3f(point_1[0],point_1[1],point_1[2]));
        pts3d_2.push_back(Point3f(point_2[0],point_2[1],point_2[2]));
    }
    /*
    for(int y = 0; y < mat1.rows; y++){
        for(int x = 0; x < mat1.cols; x++){
            Vec3f point_1 = mat1.at<Vec3f>(y,x);
            Vec3f point_2 = mat2.at<Vec3f>(y,x);
            if(fabs(point_1[2] - max_z) < FLT_EPSILON || fabs(point_1[2]) > max_z)
                continue;
            if(fabs(point_2[2] - max_z) < FLT_EPSILON || fabs(point_2[2]) > max_z)
                continue;
            pts3d_1.push_back(Point3f(point_1[0],point_1[1],point_1[2]));
            pts3d_2.push_back(Point3f(point_2[0],point_2[1],point_2[2]));
        }
    }
    */
    //cout<<"3d Points:"<<pts3d.size()<<endl;
}

/*
void get3dPairs(const Mat& disMap_1,const Mat& disMap_2,const vector<Point2f>keypoints_1,
                const vector<Point2f>keypoints_2,vector<Point3f>& pts3d_1,vector<Point3f>& pts3d_2){
    cv::Mat _3dPointsImage_1,_3dPointsImage_2;
    cv::Mat R1,R2,P1,P2,Q;
    stereoRectify(K,distCoeffs1,K,distCoeffs2,Size(376,1241),leftRright,leftTright,R1,R2,P1,P2,Q);
    reprojectImageTo3D(disMap_1,_3dPointsImage_1,Q,true);
    reprojectImageTo3D(disMap_2,_3dPointsImage_2,Q,true);
    for(int i=0;i<keypoints_1.size();i++){
        float x1,y1,z1,x2,y2,z2;
        x1 = _3dPointsImage_1.at<Vec3f>(keypoints_1.at(i).y,keypoints_1.at(i).x)[0];
        y1 = _3dPointsImage_1.at<Vec3f>(keypoints_1.at(i).y,keypoints_1.at(i).x)[1];
        z1 = _3dPointsImage_1.at<Vec3f>(keypoints_1.at(i).y,keypoints_1.at(i).x)[2];
        x2 = _3dPointsImage_2.at<Vec3f>(keypoints_1.at(i).y,keypoints_1.at(i).x)[0];
        y2 = _3dPointsImage_2.at<Vec3f>(keypoints_1.at(i).y,keypoints_1.at(i).x)[1];
        z2 = _3dPointsImage_2.at<Vec3f>(keypoints_1.at(i).y,keypoints_1.at(i).x)[2];
        if(z1 == 0 || z2 == 0)
            continue;
        if(x1!=x1 || y1!=y1 || z1!=z1 || x2!=x2 || y2!=y2 || z2!=z2)  //not a number
            continue;
        if(x1+1==x1 || y1+1==y1 || z1+1==z1 || x2+1==x2 || y2+1==y2 || z2+1==z2) //inf
            continue;
        pts3d_1.push_back(Point3f(x1,y1,z1));
        pts3d_2.push_back(Point3f(x2,y2,z2));
    }
    /*
    for(int i=0;i<keypoints.size();i++){
        double x,y,z;
        x = _3dPointsImage.<Vec3f>at(i).x;
        y = _3dPointsImage.<Vec3f>at(i).y;
        z = _3dPointsImage.<Vec3f>at(i).z;
        if(z == 0)
            continue;
        if(x!=x || y!=y || z!=z)  //not a number
            continue;
        pts3d.push_back(Point3f(x,y,z));
        pts2d.push_back(Point2f(x,y));
    }

}
*/

/*
void getDisMap(const Mat& leftImage,const Mat& rightImage,Mat& disMap){
    cv::Mat leftForMatcher,rightForMatcher;
    cv::Mat leftDisp,rightDisp;
    cv::Mat filteredDisp;
    if(leftImage.empty() || rightImage.empty()){
        cout<<"Cannot read images correctly."<<endl;
        exit(0);
    }

    int maxDist = 160;
    maxDist /= 2;
    if(maxDist%16 != 0){
        maxDist += 16-(maxDist%16);
    }
    cv::resize(leftImage,leftForMatcher,Size(),1.0,1.0);
    cv::resize(rightImage,rightForMatcher,Size(),1.0,1.0);

    cv::Ptr<cv::StereoBM> leftMatcher = cv::StereoBM::create(maxDist,winSize);
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> WLSFilter = cv::ximgproc::createDisparityWLSFilter(leftMatcher);
    cv::Ptr<cv::StereoMatcher> rightMatcher = cv::ximgproc::createRightMatcher(leftMatcher);

    cvtColor(leftForMatcher,leftForMatcher,COLOR_BGR2GRAY);
    cvtColor(rightForMatcher,rightForMatcher,COLOR_BGR2GRAY);

    leftMatcher-> compute(leftForMatcher,rightForMatcher,leftDisp);
    rightMatcher-> compute(rightForMatcher,leftForMatcher,rightDisp);

    WLSFilter-> setLambda(lambda);
    WLSFilter-> setSigmaColor(sigma);
    WLSFilter->filter(leftDisp,leftImage,filteredDisp,rightDisp);

    cv::ximgproc::getDisparityVis(filteredDisp,disMap);
}
*/

void getDismapSGBM(const Mat& leftImage,const Mat& rightImage,Mat& disMap){
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,80,3);
    if(leftImage.empty() || rightImage.empty()){
        cout<<"Errors getting disparity map using SGBM"<<endl;
        exit(0);
    }
    sgbm->setPreFilterCap(63);

    sgbm->setBlockSize(sgbmWinSize);

    int cn = leftImage.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);
    sgbm->compute(leftImage,rightImage,disMap);
}

void drawTrajectory(Mat& traj,const Mat R,const Mat t){
    int x = int(t.at<double>(0)) + 300;
    int y = int(t.at<double>(2)) + 100;

    char text[200];
    circle(traj,Point(x,y),1,CV_RGB(0,255,0),2);
    rectangle(traj,Point(10,30),Point(550,50),CV_RGB(255,255,255),CV_FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", t.at<double>(0), t.at<double>(1), t.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
    imshow( "Trajectory", traj );
}







