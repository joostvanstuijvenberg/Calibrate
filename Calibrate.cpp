/*
* Joost van Stuijvenberg
* Avans Hogeschool Breda
* September 2018
*
* CC BY-SA 4.0, see:      https://creativecommons.org/licenses/by-sa/4.0/
* sources & updates:      https://github.com/joostvanstuijvenberg/OpenCV
* Based on code found at: http://aishack.in/tutorials/calibrating-undistorting-opencv-oh-yeah/
*
* You are free to:
*    Share � copy and redistribute the material in any medium or format
*    Adapt � remix, transform, and build upon the material for any purpose, even commercially.
*
* The licensor cannot revoke these freedoms as long as you follow the license terms.
*
* Under the following terms:
*    Attribution � You must give appropriate credit, provide a link to the license, and indicate
*                  if changes were made. You may do so in any reasonable manner, but not in any
* 	                way that suggests the licensor endorses you or your use.
*    ShareAlike  � If you remix, transform, or build upon the material, you must distribute your
*                  contributions under the same license as the original.
*
* No additional restrictions � You may not apply legal terms or technological measures that
* legally restrict others from doing anything the license permits.
*
* Notices:
*    You do not have to comply with the license for elements of the material in the public domain
*    or where your use is permitted by an applicable exception or limitation. No warranties are
*    given. The license may not give you all of the permissions necessary for your intended use.
*    For example, other rights such as publicity, privacy, or moral rights may limit how you use
*    the material.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
* IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
* AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
* OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define CALIBRATE_VERSION			"1.0.0"

#define DEFAULT_CAMERA				"0"
#define DEFAULT_CALIBRATION_FILE	"../CalibrationData.yml"

#define PATTERN_NUMBER_OF_BOARDS	10
#define PATTERN_CORNERS_HOR			9
#define PATTERN_CORNERS_VER			6

#define WINDOW_LEFT_TITLE			"Left"
#define WINDOW_RIGHT_TITLE			"Right"

void performCalibration(cv::VideoCapture& capture, cv::Mat& intrinsic, cv::Mat& distCoeffs);
void loadCalibrationData(std::string filePath, cv::Mat& intrinsic, cv::Mat& distCoeffs);
void saveCalibrationData(std::string filePath, cv::Mat& intrinsic, cv::Mat& distCoeffs);

/*
* ----------------------------------------------------------------------------------------------- *
* main()                                                                                          *
* ----------------------------------------------------------------------------------------------- *
*/
int main(int argc, char** argv)
{
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "Calibrate " << CALIBRATE_VERSION;
    std::cout << " - Image and video capture utility" << std::endl;
    std::cout << "Joost van Stuijvenberg" << std::endl;
    std::cout << "Avans Hogeschool Breda" << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    // First optional parameter: camera number (expecting a valid integer).
    std::string camera = argc > 1 ? argv[1] : DEFAULT_CAMERA;

    // Second optional parameter: calibration file for images and videos (expecting a valid path).
    std::string calibrationFile = argc > 2 ? argv[2] : DEFAULT_CALIBRATION_FILE;

    // See if we can access the camera using the given camera number.
    cv::VideoCapture capture(std::stoi(camera));
    if (!capture.isOpened())
    {
        std::cerr << "Could not access the camera. Press Enter to quit." << std::endl;
        std::cin.get();
        return -1;
    }

    std::cout << "Press <l> to load calibration data, <c> to perform the calibration" << std::endl;
    std::cout << "routine, <s> to save the calibration data, <r> to reset the" << std::endl;
    std::cout << "calibration data, <Esc> to quit. Make sure to press keys while one" << std::endl;
    std::cout << "of the image windows has focus." << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "Using camera . . . . : " + camera << std::endl;
    std::cout << "Calibration file . . : " + calibrationFile << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;

    cv::namedWindow(WINDOW_LEFT_TITLE, CV_WINDOW_AUTOSIZE);
    cv::namedWindow(WINDOW_RIGHT_TITLE, CV_WINDOW_AUTOSIZE);
    cv::Mat normal, calibrated, distCoeffs, intrinsic; // = cv::Mat(3, 3, CV_32FC1);
    int c = 0;
    while (c != 27)
    {
        capture >> normal;
        cv::imshow(WINDOW_LEFT_TITLE, normal);
        cv::moveWindow(WINDOW_LEFT_TITLE, 50, 50);

        calibrated = normal.clone();
        if (!intrinsic.empty() && !distCoeffs.empty())
            cv::undistort(normal, calibrated, intrinsic, distCoeffs);
        cv::imshow(WINDOW_RIGHT_TITLE, calibrated);
        cv::moveWindow(WINDOW_RIGHT_TITLE, 750, 50);

        // With a delay of 40 msec, our theoretical framerate is 25 fps.
        c = cv::waitKey(40);
        switch (c)
        {
            // C: perform calibration. In the next iteration the calibration data is used
            // to undistort the live image.
            case 'c':
            case 'C':
                intrinsic.data = nullptr;
                distCoeffs.data = nullptr;
                std::cout << "Performing calibration." << std::endl;
                performCalibration(capture, intrinsic, distCoeffs);
                break;
                // L: load calibration data from a file. Previously gathered calibration data
                // is lost.
            case 'l':
            case 'L':
                intrinsic.data = nullptr;
                distCoeffs.data = nullptr;
                loadCalibrationData(calibrationFile, intrinsic, distCoeffs);
                std::cout << "Loaded calibration data from " << calibrationFile << '.' << std::endl;
                break;
                // S: save calibration data to a file.
            case 's':
            case 'S':
                saveCalibrationData(calibrationFile, intrinsic, distCoeffs);
                std::cout << "Saved calibration data to " << calibrationFile << '.' << std::endl;
                break;
                // R: reset calibration data.
            case 'r':
            case 'R':
                intrinsic.data = nullptr;
                distCoeffs.data = nullptr;
                std::cout << "Calibration data has been reset." << std::endl;
                break;
            default:
                break;
        }
    }

    capture.release();
    return 0;
}

/*
* ----------------------------------------------------------------------------------------------- *
* performCalibration()                                                                            *
* ----------------------------------------------------------------------------------------------- *
*/
void performCalibration(cv::VideoCapture& capture, cv::Mat& intrinsic, cv::Mat& distCoeffs)
{
    assert(intrinsic.empty());
    assert(distCoeffs.empty());

    cv::Mat image, gray;
    cv::Size boardSize(PATTERN_CORNERS_HOR, PATTERN_CORNERS_VER);
    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point3f> obj;
    std::vector<cv::Mat> rvecs, tvecs;

    for (int j = 0; j < PATTERN_CORNERS_HOR * PATTERN_CORNERS_VER; j++)
        obj.emplace_back(cv::Point3f(j / PATTERN_CORNERS_HOR, j % PATTERN_CORNERS_HOR, 0.0f));
    int successes = 0;
    while (successes < PATTERN_NUMBER_OF_BOARDS)
    {
        capture >> image;
        cv::cvtColor(image, gray, CV_BGR2GRAY);
        bool found = cv::findChessboardCorners(image, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
        if (found)
        {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0));
            cv::drawChessboardCorners(gray, boardSize, corners, true);
        }
        cv::imshow(WINDOW_LEFT_TITLE, image);
        cv::imshow(WINDOW_RIGHT_TITLE, gray);
        int key = cv::waitKey(40);
        if (key == ' ' && found)
        {
            image_points.push_back(corners);
            object_points.push_back(obj);
            std::cout << "Snap stored!" << std::endl;
            successes++;
        }
    }

    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    cv::calibrateCamera(object_points, image_points, image.size(), intrinsic, distCoeffs, rvecs, tvecs);
}

/*
* ----------------------------------------------------------------------------------------------- *
* loadCalibrationData()                                                                           *
* ----------------------------------------------------------------------------------------------- *
*/
void loadCalibrationData(std::string filePath, cv::Mat& intrinsic, cv::Mat& distCoeffs)
{
    assert(filePath.length() > 0);
    assert(intrinsic.empty());
    assert(distCoeffs.empty());

    cv::FileStorage fs2(filePath, cv::FileStorage::READ);
    fs2["intrinsic"] >> intrinsic;
    fs2["distCoeffs"] >> distCoeffs;
    fs2.release();
}

/*
* ----------------------------------------------------------------------------------------------- *
* saveCalibrationData()                                                                           *
* ----------------------------------------------------------------------------------------------- *
*/
void saveCalibrationData(std::string filePath, cv::Mat& intrinsic, cv::Mat& distCoeffs)
{
    assert(!intrinsic.empty());
    assert(!distCoeffs.empty());

    cv::FileStorage fs(filePath, cv::FileStorage::WRITE);
    fs << "intrinsic" << intrinsic << "distCoeffs" << distCoeffs;
    fs.release();
}