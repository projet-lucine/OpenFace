#include "LandmarkCoreIncludes.h"

#include <boost/python.hpp>
#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>

#define INFO_STREAM( stream )                   \
  std::cout << stream << std::endl

#define WARN_STREAM( stream )                     \
  std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream )                  \
  std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
  std::cout << error << std::endl;
}

#define FATAL_STREAM( stream )                                  \
  printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

// Convert a vector into a py list
template <class T>
inline
boost::python::list std_vector_to_py_list(std::vector<T> vector) {
  typename std::vector<T>::iterator iter;
  boost::python::list list;
  for (iter = vector.begin(); iter != vector.end(); ++iter) {
    list.append(*iter);
  }
  return list;
}

// Converts a C++ map to a python dict
template <class K, class V>
boost::python::dict toPythonDict(std::map<K, V> map) {
    typename std::map<K, V>::iterator iter;
	boost::python::dict dictionary;
	for (iter = map.begin(); iter != map.end(); ++iter) {
		dictionary[iter->first] = std_vector_to_py_list(iter->second);
	}
	return dictionary;
}



boost::python::dict load_folder(string path)
{
  vector<string> arguments;
  std::map<string, vector<double>> csv;


  arguments.push_back("./FeatureExtraction");
  arguments.push_back("-fdir");
  arguments.push_back(path);

  // Load the modules that are being used for tracking and face analysis
  // Load face landmark detector
  LandmarkDetector::FaceModelParameters det_parameters(arguments);
  // Always track gaze in feature extraction
  LandmarkDetector::CLNF face_model(det_parameters.model_location);

  // Load facial feature extractor and AU analyser
  FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
  face_analysis_params.OptimizeForImages();
  FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

	if (!face_model.eye_model)
	{
		cout << "WARNING: no eye model found" << endl;
	}

	if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
	{
		cout << "WARNING: no Action Unit models found" << endl;
	}

  Utilities::SequenceCapture sequence_reader;

  // The sequence reader chooses what to open based on command line arguments provided

  cv::Mat captured_image;

  bool test = sequence_reader.Open(arguments);
  captured_image = sequence_reader.GetNextFrame();

  // For reporting progress
  double reported_completion = 0;

  INFO_STREAM("Starting tracking");
  while (!captured_image.empty())
  {


    // Converting to grayscale
    cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

    // The actual facial landmark detection / tracking
    bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image,
                                                                      face_model,
                                                                      det_parameters);

    // Gaze tracking, absolute gaze direction
    cv::Point3f gazeDirection0(0, 0, 0);
    cv::Point3f gazeDirection1(0, 0, 0);
    cv::Vec2d gazeAngle(0, 0);

    if (detection_success && face_model.eye_model)
    {
      GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx,
                                 sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
      GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx,
                                 sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
      gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
    }

    // Do face alignment
    cv::Mat sim_warped_img;
    cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;


    face_analyser.AddNextFrame(captured_image, face_model.detected_landmarks, face_model.detection_success, sequence_reader.time_stamp, sequence_reader.IsWebcam());


    // Work out the pose of the head from the tracked model
    cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

    auto aus_intensity = face_analyser.GetCurrentAUsReg();
    auto aus_presence = face_analyser.GetCurrentAUsClass();

    csv["confidence"].push_back(face_model.detection_certainty);

		for (auto au : aus_presence)
		{
      csv[au.first + "_presence"].push_back(au.second);
		}

		for (auto au : aus_intensity)
		{
      csv[au.first + "_intensity"].push_back(au.second);
		}

		for (auto lm : face_model.detected_landmarks)
		{
      csv["2d_points"].push_back(lm);
		}

		for (auto lm : face_model.GetShape(sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy))
		{
      csv["3d_points"].push_back(lm);
		}

    csv["gaze_direction_0_x"].push_back(gazeDirection0.x);
    csv["gaze_direction_0_y"].push_back(gazeDirection0.y);
    csv["gaze_direction_0_z"].push_back(gazeDirection0.z);

    csv["gaze_direction_1_x"].push_back(gazeDirection1.x);
    csv["gaze_direction_1_y"].push_back(gazeDirection1.y);
    csv["gaze_direction_1_z"].push_back(gazeDirection1.z);


    csv["gaze_angle_0"].push_back(gazeAngle[0]);
    csv["gaze_angle_1"].push_back(gazeAngle[1]);



    captured_image = sequence_reader.GetNextFrame();

  }

  // Reset the models for the next video
  face_analyser.Reset();
  face_model.Reset();
  return toPythonDict(csv);
}

BOOST_PYTHON_MODULE(OpenFace)
{
  using namespace boost::python;
  def("load_folder", load_folder);
}
