#pragma once

//#define KCB_ENABLE_FT
//#define KCB_ENABLE_SPEECH

#include "KinectCommonBridgeLib.h"
#include "NuiSensor.h"
#include "ofMain.h" // this MUST come after KCB!!! Not sure you need NuiSensor.h if using KCB
#include "ofxCv.h"
#include "opencv2/cuda.hpp"

#pragma comment (lib, "KinectCommonBridge.lib") // add path to lib additional dependency dir $(TargetDir)

#ifdef KCB_ENABLE_FT
#pragma comment(lib, "FaceTrackLib.lib")
#endif

class ofxKCBFace  {

public:

	enum FACE_POSITIONS {

	};

	enum FEATURE {
		LEFT_EYE, RIGHT_EYE, MOUTH, NOSE, CHIN, LEFT_EAR, RIGHT_EAR
	};

	ofxKCBFace & operator=(const ofxKCBFace & rhs) {
		rotation = rhs.rotation;
		translation = rhs.translation;
		mesh = rhs.mesh;
		rect = rhs.rect;

		return *this;
	};

	ofVec3f rotation, translation;
	ofRectangle rect;
	ofMesh mesh;

	ofVec3f getLocationByIdentifier(FACE_POSITIONS position);
	ofRectangle getFeatureBounding(FACE_POSITIONS position);

};

// if you want to use events, subscribe to this, otherwise
class ofxKCBSpeechEvent : public ofEventArgs
{

public:

	std::string detectedSpeech;
	int confidence;

	static ofEvent<ofxKCBSpeechEvent> event;

};

// poll for this very simple data object for speech data
class SpeechData {
public:
	std::string detectedSpeech;
	int confidence;
};

class SkeletonBone
{
public:
	enum TrackingState {
		NotTracked,
		Tracked,
		Inferred
	};
	// lots of constness because we're putting these in a map and that
	// copies stuff all over the place
	const ofQuaternion getCameraRotation();
	const ofMatrix4x4 getCameraRotationMatrix();

	const ofVec3f& getStartPosition();
	const ofVec3f getScreenPosition();
	const ofQuaternion&	getRotation();
	const ofMatrix4x4& getRotationMatrix();

	const int getStartJoint();
	int getEndJoint();

	TrackingState getTrackingState();

	SkeletonBone( const Vector4& inPosition, const _NUI_SKELETON_BONE_ORIENTATION& bone, const NUI_SKELETON_POSITION_TRACKING_STATE& trackingState );

private:

	ofMatrix4x4 cameraRotation;
	int	endJoint;
	int	startJoint;
	ofVec3f	position;
	ofMatrix4x4	rotation;
	ofVec2f screenPosition;
	TrackingState trackingState;
};

typedef map<_NUI_SKELETON_POSITION_INDEX, SkeletonBone> Skeleton;

class ofxKinectCommonBridge : protected ofThread {
  public:
	static const int HORIZONTAL_VIEWING_ANGLE = 57;
	static const float HORIZONTAL_FOCAL_LENGTH;
	static const float HORIZONTAL_FOCAL_LENGTH_INV;
	static const int VERTICAL_VIEWING_ANGLE = 43;
	static const float VERTICAL_FOCAL_LENGTH;
	static const float VERTICAL_FOCAL_LENGTH_INV;

	ofxKinectCommonBridge();

	// new API
	bool initSensor( int id = 0 );
	bool createDepthPixels( int width = 0, int height = 0 );
	bool initDepthStream( int width, int height, bool nearMode = false, bool mapColorToDepth = false);
	bool createColorPixels( int width = 0, int height = 0 );
	bool initColorStream( int width, int height, bool mapColorToDepth = false );
	bool initIRStream( int width, int height );
	bool initSkeletonStream( bool seated );
	bool initAudio();
	bool start();

	KCBHANDLE getHandle();
	INuiSensor & getNuiSensor();

#ifdef KCB_ENABLE_FT
	bool initFaceTracking(); // no params, can't use with other stuff either.
#endif

	// audio functionality
#ifdef KCB_ENABLE_SPEECH
	bool startAudioStream();
	bool initSpeech();
	bool loadGrammar(string filename);

	// speech
	bool hasNewSpeechData();
	SpeechData getNewSpeechData();

#endif

	void stop();

  	/// is the current frame new?
	bool isFrameNew();
	bool isFrameNewVideo();
	bool isFrameNewDepth();
	bool isNewSkeleton();
	bool isFaceNew() {
		return bIsFaceNew;
	}

	/// updates the pixel buffers and textures
	///
	/// make sure to call this to update to the latest incoming frames
	void update();

	void setDepthClipping(float nearClip=500, float farClip=4000);
	void setComputeDepthImage(bool bCompute);
	void setComputeNuiFullDepth(bool bCompute);
	void setUseTexture(bool bUse);
	void setUseStreams(bool bUse);
	void setSpeechGrammarFile(string path) {
		grammarFile = path;
	}

	ofPixels& getColorPixelsRef();
	ofPixels & getDepthImageRef();       	///< grayscale values
	ofShortPixels & getDepthPixelsRef();	///< raw 11 bit values
	const ofShortPixels & getDepthPixelsRef() const;	///< raw 11 bit values
	cv::cuda::GpuMat & ofxKinectCommonBridge::getDepthPixelsGpuRef();
	ofShortPixels getDepthPixels() const;	///< raw 11 bit values
	cv::cuda::GpuMat ofxKinectCommonBridge::getDepthPixelsGpu() const;
	ofShortPixels & getDepthPlayerPixelsRef();	///< raw 11 bit values
	NUI_DEPTH_IMAGE_PIXEL* getNuiDepthPixelsRef();
	vector<Skeleton> &getSkeletons();
	ofTexture &getDepthTexture() {
		return depthTex;
	}

	ofTexture &getDepthImageTexture() {
		if (!bComputingDepthImage) {
			ofLogError("ofxKinectCommonBridge::getDepthImageTexture") << "The computation of the depth image has not been set";
			return ofTexture();
		}
		return depthImageTex;
	}

	ofTexture &getColorTexture() {
		return videoTex;
	}
	float getDepthAt(int xColor, int yColor) const;
	void CropAndSmooth(const cv::cuda::GpuMat &mask, int size, float sigma);
	ofVec3f project(ofVec3f worldPoint) const;
	ofVec3f getWorldCoordinates(int xColor, int yColor) const;
	ofVec3f getWorldCoordinates(int xColor, int yColor, float depth) const;
	ofTexture &getFaceTrackingTexture()
	{
		return faceTrackingTexture;
	}

	ofxKCBFace& getFaceData();

	/// draw the video texture
	void draw(float x, float y, float w, float h);
	void draw(float x, float y);
	void draw(const ofPoint& point);
	void draw(const ofRectangle& rect);

	/// draw the grayscale depth image texture
	void drawDepthImage(float x, float y, float w, float h);
	void drawDepthImage(float x, float y);
	void drawDepthImage(const ofPoint& point);
	void drawDepthImage(const ofRectangle& rect);

	/// draw the depth texture
	void drawDepth(float x, float y, float w, float h);
	void drawDepth(float x, float y);
	void drawDepth(const ofPoint& point);
	void drawDepth(const ofRectangle& rect);

	/// draw IR
	void drawIR( float x, float y, float w, float h );

	/// draw skeleton
	void drawSkeleton(int index);

  private:

    KCBHANDLE hKinect;
	KINECT_IMAGE_FRAME_FORMAT depthFormat;
	KINECT_IMAGE_FRAME_FORMAT colorFormat;
	NUI_SKELETON_FRAME k4wSkeletons;

  	bool bInitedColor, bInitedDepth, bInitedIR;
	bool bStarted;
	vector<Skeleton> skeletons;

	//quantize depth buffer to 8 bit range
	vector<unsigned char> depthLookupTable;
	void updateDepthLookupTable();
	void updateDepthPixels();
	void updateIRPixels();
	bool bNearWhite;
	float nearClipping, farClipping;

#ifdef KCB_ENABLE_FT
	void updateFaceTrackingData( IFTResult* ftResult );
#endif

  	bool bUseTexture;
	ofTexture depthImageTex; ///< the depth texture
	ofTexture depthTex; ///<
	ofTexture videoTex; ///< the RGB texture

	// face
	ofTexture faceTrackingTexture;
	//ofTexture irTex;

	ofPixels videoPixels;
	ofPixels videoPixelsBack;			///< rgb back

	bool bComputingDepthImage; /// Computing a depth image can be avoided as it is costly
	ofPixels depthImage;

	ofShortPixels depthPixelsPacked;
	ofShortPixels depthPixelsPackedBack;	///< depth back
	NUI_DEPTH_IMAGE_PIXEL* depthPixelsNuiWrap; // Unpack without full depth
	ofShortPixels depthPlayerPixels;
	ofShortPixels depthPixels;
	cv::cuda::GpuMat depthPixels_gpu;

	bool bComputingNuiFullDepth; /// Can ask for a full depth retrieval
	NUI_DEPTH_IMAGE_PIXEL* depthPixelsNui;

	ofPixels irPixels;
	ofPixels irPixelsBack;

	bool bIsFrameNewVideo;
	bool bNeedsUpdateVideo;
	bool bIsFrameNewDepth;
	bool bNeedsUpdateDepth;
	bool bNeedsUpdateSkeleton;
	bool bIsSkeletonFrameNew;
	bool bUsingFaceTrack;

	bool bUpdateSpeech;
	bool bUpdateFaces;
	bool bIsFaceNew;
	bool bUseStreams;

	bool bProgrammableRenderer;

	// speech
	bool bHasLoadedGrammar;
	bool bUsingSpeech;
	string grammarFile;
	SpeechData speechData;
	float speechConfidenceThreshold; // make a way to set this

	// audio
	bool bUsingAudio;

	// face
	bool bIsTrackingFace;

#ifdef KCB_ENABLE_FT

	// double buffer for faces
	ofxKCBFace faceDataBack, faceData; // only single face at the moment
	// camera params for face tracking
	FT_CAMERA_CONFIG depthCameraConfig;
	FT_CAMERA_CONFIG videoCameraConfig;
    FLOAT pSUCoef;
    FLOAT zoomFactor;

#endif

#ifdef KCB_ENABLE_SPEECH
	bool bInitedSpeech;
#endif

	bool bVideoIsInfrared;
	bool bUsingSkeletons;
	bool bUsingDepth;

	BYTE *irPixelByteArray;

	void threadedFunction();

	bool bMappingColorToDepth;
	bool bMappingDepthToColor;
	bool beginMappingColorToDepth, beginMappingDepthToColor;

	NUI_IMAGE_RESOLUTION colorRes;
	NUI_IMAGE_RESOLUTION depthRes;

	INuiSensor *nuiSensor;
	INuiCoordinateMapper *mapper;

};
