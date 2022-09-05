
#include "stdafx.h"
#include "Scene.h"
#include "GlobalAppState.h"
#include "LabelUtil.h"

using namespace cv;

void Scene::load(GraphicsDevice& g, const ScanInfo& scanInfo, bool bGenerateTestOnly, mat4f transform /*= mat4f::identity()*/) {
	clear();
	transform = mat4f::identity();
	m_name = scanInfo.sceneName;
	m_sensFiles = scanInfo.sensFiles;

	MeshDataf mesh = MeshIOf::loadFromFile(scanInfo.meshFile);

	if (!scanInfo.alnFile.empty()) {
		mat4f align = mat4f::identity();
		std::ifstream ifs(scanInfo.alnFile); std::string line;
		for (unsigned int i = 0; i < 3; i++) //read header
			std::getline(ifs, line);
		for (unsigned int r = 0; r < 4; r++) {
			for (unsigned int c = 0; c < 4; c++)
				ifs >> align(r, c);
		}
		ifs.close();
		mesh.applyTransform(transform * align);
		mat4f translation = mat4f::translation(-mesh.computeBoundingBox().getMin());
		mesh.applyTransform(translation);
		transform = translation * transform * align;
	}
	else {
		mesh.applyTransform(transform);
	}

	m_bb = mesh.computeBoundingBox();
	m_obb = OBB3f(mesh.m_Vertices, vec3f::eZ);

	m_transformFromOrig = transform;


	m_sensDatas.resize(m_sensFiles.size(), nullptr);
	for (unsigned int i = 0; i < m_sensFiles.size(); i++) {
		m_sensDatas[i] = new SensorData(m_sensFiles[i]);
		for (unsigned int f = 0; f < m_sensDatas[i]->m_frames.size(); f++) {
			const mat4f frameTransform = m_sensDatas[i]->m_frames[f].getCameraToWorld();
			m_sensDatas[i]->m_frames[f].setCameraToWorld(transform * frameTransform);
			m_linearizedSensFrameIds.push_back(vec2ui(i, f));
		}
	}

	if (!bGenerateTestOnly) {
		Aggregation agg; Segmentation seg;
		agg.loadFromJSONFile(scanInfo.aggregationFile);
		seg.loadFromFile(scanInfo.segmentationFile);

		//assign semantics to mesh
		computeObjectIdsAndColorsPerVertex(agg, seg, mesh);
	}
	m_mesh.init(g, TriMeshf(mesh));

	// init rendering
	m_cbCamera.init(g);
	std::vector<DXGI_FORMAT> formats = {
		DXGI_FORMAT::DXGI_FORMAT_R32G32B32A32_FLOAT
	};
	const unsigned int width = GlobalAppState::get().s_renderWidth;
	const unsigned int height = GlobalAppState::get().s_renderHeight;
	m_renderTarget.init(g.castD3D11(), width, height, formats, true);
}

void Scene::computeObjectIdsAndColorsPerVertex(const Aggregation& aggregation, const Segmentation& segmentation,
	MeshDataf& meshData)
{
	m_objectBBs.clear();
	std::vector<vec4f> colorsPerVertex(meshData.m_Vertices.size(), vec4f(0.0f, 0.0f, 255.0f, 255.0f));

	const auto& aggregatedSegments = aggregation.getAggregatedSegments();
	const auto& objectIdsToLabels = aggregation.getObjectIdsToLabels();
	const auto& objectIds = aggregation.getObjectIds();
	//generate some random colors
	std::unordered_map<unsigned char, vec4f> objectColors;
	std::unordered_map<unsigned int, unsigned char> objectIdsToLabelIds;
	std::vector<bool> bValidObject(aggregatedSegments.size(), false);
	for (unsigned int i = 0; i < aggregatedSegments.size(); i++) {
		const unsigned int objectId = objectIds[i];
		const auto itl = objectIdsToLabels.find(objectId);
		MLIB_ASSERT(itl != objectIdsToLabels.end());
		unsigned char labelId;
		if (LabelUtil::get().getIdForLabel(itl->second, labelId)) {
			objectIdsToLabelIds[objectId] = labelId;
			auto itc = objectColors.find(objectId + 1); // Plus one here because the json file is zero based
			RGBColor c = RGBColor::colorPalette((int)labelId);
			objectColors[objectId + 1] = vec4f(c.x / 255.0f, c.y / 255.0f, objectId + 1, labelId); //objectid -> instance, labelid-> label

			if (itl->second != "floor" && itl->second != "wall" && itl->second != "ceiling")
				bValidObject[i] = true;
		}
	}
	//assign object ids and colors
	std::unordered_map< unsigned int, std::vector<unsigned int> > verticesPerSegment = segmentation.getSegmentIdToVertIdMap();
	for (unsigned int i = 0; i < aggregatedSegments.size(); i++) {
		const unsigned int objectId = objectIds[i];
		//const auto itl = objectIdsToLabelIds.find(objectId);
		//MLIB_ASSERT(itl != objectIdsToLabelIds.end());
		const vec4f& color = objectColors[objectId + 1];
		bbox3f objectBbox;
		for (unsigned int seg : aggregatedSegments[i]) {
			const std::vector<unsigned int>& vertIds = verticesPerSegment[seg];
			for (unsigned int v : vertIds) {
				colorsPerVertex[v] = color;
				objectBbox.include(meshData.m_Vertices[v]);
			}
		}

		if (bValidObject[i])
			m_objectBBs.push_back(BBInfo{ objectBbox, (unsigned short)std::round(color.w), (unsigned short)std::round(color.z) });
	}
	meshData.m_Colors = colorsPerVertex;

}

bool Scene::renderSemanticsFrame(GraphicsDevice& g, unsigned int idx, const DepthImage32& origDepth, BaseImage<unsigned char>& semantics, BaseImage<unsigned char>& instance)
{
	const vec2ui& sensFrameId = m_linearizedSensFrameIds[idx];
	const SensorData& sd = *m_sensDatas[sensFrameId.x];
	mat4f intrinsic = sd.m_calibrationDepth.m_intrinsic;
	const mat4f extrinsic = sd.m_frames[sensFrameId.y].getCameraToWorld();
	if (extrinsic[0] == -std::numeric_limits<float>::infinity()) return false;

	const unsigned int width = semantics.getWidth();
	const unsigned int height = semantics.getHeight();
	const float zNear = 0.4f;
	const float zFar = 4.0f; //TODO FIX HARDCODING (corresponded to min/max for fuse)
	// adapt intrinsics
	intrinsic._m00 *= (float)width / (float)sd.m_depthWidth;
	intrinsic._m11 *= (float)height / (float)sd.m_depthHeight;
	intrinsic._m02 *= (float)(width - 1) / (float)(sd.m_depthWidth - 1);
	intrinsic._m12 *= (float)(height - 1) / (float)(sd.m_depthHeight - 1);
	const mat4f proj = Cameraf::visionToGraphicsProj(width, height, intrinsic(0, 0), intrinsic(1, 1), zNear, zFar);
	const float fov = 2.0f * 180.0f / math::PIf * std::atan(0.5f * sd.m_depthWidth / intrinsic(0, 0));
	Cameraf cam = Cameraf(extrinsic, fov, (float)sd.m_depthWidth / (float)sd.m_depthHeight, zNear, zFar);

	ConstantBufferCamera cbCamera;
	cbCamera.worldViewProj = proj * cam.getView();
	m_cbCamera.updateAndBind(cbCamera, 0);
	g.castD3D11().getShaderManager().registerShader("shaders/drawAnnotations.hlsl", "drawAnnotations", "vertexShaderMain", "vs_4_0", "pixelShaderMain", "ps_4_0");
	g.castD3D11().getShaderManager().bindShaders("drawAnnotations");

	m_renderTarget.clear();
	m_renderTarget.bind();
	m_mesh.render();
	m_renderTarget.unbind();

	ColorImageR32G32B32A32 colorBuffer;
	m_renderTarget.captureColorBuffer(colorBuffer);
	//annotations
	for (unsigned int i = 0; i < colorBuffer.getNumPixels(); i++) {
		const vec4f& c = colorBuffer.getData()[i];
		float label = c.w;			float id = c.z;
		label = std::round(label);	id = std::round(id);
		MLIB_ASSERT(label >= 0 && label < 65535 && id >= 0 && id <= 255);
		instance.getData()[i] = (unsigned char)id;
		semantics.getData()[i] = (unsigned char)label;
	}

	DepthImage32 depthBuffer;
	m_renderTarget.captureDepthBuffer(depthBuffer);
	depthBuffer.setInvalidValue(-std::numeric_limits<float>::infinity());
	mat4f projToCamera = cam.getProj().getInverse();
	for (auto& p : depthBuffer) {
		vec3f posWorld = vec3f(-std::numeric_limits<float>::infinity());
		if (p.value != 0.0f && p.value != 1.0f) {
			vec3f posProj = vec3f(g.castD3D11().pixelToNDC(vec2i((int)p.x, (int)p.y), depthBuffer.getWidth(), depthBuffer.getHeight()), p.value);
			vec3f posCamera = projToCamera * posProj;
			if (posCamera.z >= 0.4f && posCamera.z <= 4.0f) {
				p.value = posCamera.z;
				posWorld = extrinsic * posCamera;
			}
			else {
				p.value = -std::numeric_limits<float>::infinity();
			}
		}
		else {
			p.value = -std::numeric_limits<float>::infinity();
		}
	} //depth pixels

	// filter by orig depth
	for (const auto& p : depthBuffer) {
		const float d = p.value;
		const float od = origDepth(p.x, p.y);

		if (od == -std::numeric_limits<float>::infinity() && d != -std::numeric_limits<float>::infinity()) {
			semantics(p.x, p.y) = 0;
			instance(p.x, p.y) = 0;
		}
		else if (std::fabs(d - od) > 0.5f) {
			semantics(p.x, p.y) = 0;	// no unannotated cat here
			instance(p.x, p.y) = 0;	// no unannotated cat here
		}
	}

	return true;
}

//void colorToMat(BaseImage<vec3f>& colorImage, Mat& _color) {
//	uchar max = 0;
//	int numPixels = colorImage.getWidth() * colorImage.getHeight();
//
//	for (int i = 0; i < colorImage.getWidth(); ++i)
//		for (int j = 0; j < colorImage.getHeight(); ++j) {
//			vec3f color = colorImage(i, j);
//			Vec3b __color((uchar)(color.x * 255.0), (uchar)(color.y * 255.0), (uchar)(color.z * 255.0));
//			std::cout << __color << "\n";
//			_color.at<Vec3b>(Point(i, j)) = __color;
//		}
//}


bool Scene::renderDepthSemantics(GraphicsDevice& g, unsigned int idx, mat4f& intrinsic, mat4f& extrinsic, DepthImage32& depth, BaseImage<unsigned char>& semantics, BaseImage<unsigned char>& instance, BaseImage<unsigned char>& instance_gt,
	BaseImage <vec3f>& colorImage)
{
	const vec2ui& sensFrameId = m_linearizedSensFrameIds[idx];
	const SensorData& sd = *m_sensDatas[sensFrameId.x];
	intrinsic = sd.m_calibrationDepth.m_intrinsic;
	extrinsic = sd.m_frames[sensFrameId.y].getCameraToWorld();

	if (extrinsic[0] == -std::numeric_limits<float>::infinity()) return false;

	const unsigned int width = semantics.getWidth();
	const unsigned int height = semantics.getHeight();
	const float zNear = 0.4f;
	const float zFar = 4.0f; //TODO FIX HARDCODING (corresponded to min/max for fuse)
	// adapt intrinsics
	intrinsic._m00 *= (float)width / (float)sd.m_depthWidth;
	intrinsic._m11 *= (float)height / (float)sd.m_depthHeight;
	intrinsic._m02 *= (float)(width - 1) / (float)(sd.m_depthWidth - 1);
	intrinsic._m12 *= (float)(height - 1) / (float)(sd.m_depthHeight - 1);
	const mat4f proj = Cameraf::visionToGraphicsProj(width, height, intrinsic(0, 0), intrinsic(1, 1), zNear, zFar);
	const float fov = 2.0f * 180.0f / math::PIf * std::atan(0.5f * sd.m_depthWidth / intrinsic(0, 0));
	Cameraf cam = Cameraf(extrinsic, fov, (float)sd.m_depthWidth / (float)sd.m_depthHeight, zNear, zFar);

	ConstantBufferCamera cbCamera;
	cbCamera.worldViewProj = proj * cam.getView();
	m_cbCamera.updateAndBind(cbCamera, 0);
	g.castD3D11().getShaderManager().registerShader("shaders/drawAnnotations.hlsl", "drawAnnotations", "vertexShaderMain", "vs_4_0", "pixelShaderMain", "ps_4_0");
	g.castD3D11().getShaderManager().bindShaders("drawAnnotations");

	m_renderTarget.clear();
	m_renderTarget.bind();
	m_mesh.render();
	m_renderTarget.unbind();



	


	
	
	vec3uc* m_colorFrame;
	
	m_colorFrame = sd.decompressColorAlloc(idx);
	//Mat _color(Size(sd.m_colorWidth,sd.m_colorHeight), CV_8UC3);
	BaseImage <vec3f> _colorImage(sd.m_colorWidth, sd.m_colorHeight);
	for (int i = 0; i < sd.m_colorWidth; i++) {
		for (int j = 0; j < sd.m_colorHeight; j++) {
			Vec3b c;
			uchar r = m_colorFrame[j * sd.m_colorWidth + i].r;
			uchar g = m_colorFrame[j * sd.m_colorWidth + i].g;
			uchar b = m_colorFrame[j * sd.m_colorWidth + i].b;


			c[0] = r;
			c[1] = g;
			c[2] = b;
			vec3f color_normalized((float)r, (float)g, (float)b);

			//_color.at<Vec3b>(Point(i, j)) = c;

			_colorImage(i, j) = color_normalized;
		}
	}
	delete[] m_colorFrame;
	_colorImage.resize(width, height);
	colorImage = _colorImage;

	//ColorImageR32G32B32A32 colorBuffer;
	//m_renderTarget.captureColorBuffer(colorBuffer);

	//read instance-filtered & semantic images with the corresponding frame 
	Mat image_instance, image_instance_gt;

	// https://answers.opencv.org/question/19272/unable-to-properly-read-16bit-tiff-image/
	std::string instance_base_path_gt = "scene0000_00_2d-instance-filt\\instance-filt\\";
	std::string instance_base_path = "D:\\RealTimeSegmentation\\checkpoints_new__\\predictions\\scene0000_00\\";
	std::string image_instance_path = instance_base_path + std::to_string(idx) + ".png";
	std::string image_instance_path_gt = instance_base_path_gt + std::to_string(idx) + ".png";
	image_instance = imread(image_instance_path, CV_LOAD_IMAGE_UNCHANGED);
	image_instance_gt = imread(image_instance_path_gt, CV_LOAD_IMAGE_UNCHANGED);

	if (!image_instance.data)                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		return false;
	}

	if (!image_instance_gt.data)                              // Check for invalid input
	{
		std::cout << "Could not open or find the image" << std::endl;
		std::cout << "A7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" << std::endl;
		return false;
	}

	std::string semantic_base_path = "scene0000_00_2d-label-filt\\label-filt\\";
	std::string image_semantic_path = semantic_base_path + std::to_string(idx) + ".png";
	FreeImageWrapper fiw;
	BaseImage<unsigned short> image_semantic; // TODO: This is made on the stack, so no need to free, right?
	fiw.loadImage(image_semantic_path, *(DepthImage16*)&image_semantic, false);

	resize(image_instance, image_instance, Size(width, height), 0, 0, INTER_NEAREST);
	resize(image_instance_gt, image_instance_gt, Size(width, height), 0, 0, INTER_NEAREST);
	image_semantic.resize(width, height);

	int render_num_pixels = width * height;
	for (unsigned int i = 0; i < render_num_pixels; ++i) {
		unsigned char label;
		if (!LabelUtil::get().getNyuForObjId(image_semantic.getData()[i], label)) label = 0;

		unsigned char id = image_instance.data[i];
		unsigned char id_gt = image_instance_gt.data[i];

		MLIB_ASSERT(label >= 0 && label < 65535 && id >= 0 && id <= 255)

		instance.getData()[i] = id;
		semantics.getData()[i] = (unsigned char)label;
		instance_gt.getData()[i] = id_gt;
	}

	image_instance.release();
	

	/*unsigned char* _depth = sd.m_frames[sensFrameId.y].getDepthCompressed();
	unsigned short* __depth;
	int len;
	__depth = (unsigned short*)stb::stbi_zlib_decode_malloc((const char*)_depth, (int)sd.m_frames[sensFrameId.y].getDepthSizeBytes(), &len);
	
	depth.allocate(sd.m_depthWidth, sd.m_depthHeight);
	int num_depth_pixels = sd.m_depthWidth * sd.m_depthHeight;
	for (int i = 0; i < num_depth_pixels; ++i)
		depth.getData()[i] = __depth[i];

	depth.resize(width, height);*/
	m_renderTarget.captureDepthBuffer(depth);
	depth.setInvalidValue(-std::numeric_limits<float>::infinity());

	// delete[] _depth; // No need to free since we didn't allocate it.
	//delete[]  __depth;
	mat4f projToCamera = cam.getProj().getInverse();
	for (auto& p : depth) {
		if (p.value != 0.0f && p.value != 1.0f) {
			vec3f posProj = vec3f(g.castD3D11().pixelToNDC(vec2i((int)p.x, (int)p.y), depth.getWidth(), depth.getHeight()), p.value);
			vec3f posCamera = projToCamera * posProj;
			if (posCamera.z >= 0.4f && posCamera.z <= 4.0f) {
				p.value = posCamera.z;
			}
			else {
				p.value = -std::numeric_limits<float>::infinity();
			}
		}
		else {
			p.value = -std::numeric_limits<float>::infinity();
		}
	} //depth pixels
	return true;
}

bool Scene::getDepthFrame(unsigned int idx, DepthImage32& depth, mat4f& intrinsic, mat4f& extrinsic, float minDepth /*= 0.0f*/, float maxDepth /*= 12.0f*/) const {
	const vec2ui& sensFrameId = m_linearizedSensFrameIds[idx];
	const SensorData& sd = *m_sensDatas[sensFrameId.x];
	intrinsic = sd.m_calibrationDepth.m_intrinsic;
	extrinsic = sd.m_frames[sensFrameId.y].getCameraToWorld();
	if (extrinsic[0] == -std::numeric_limits<float>::infinity()) return false;

	const unsigned int newWidth = depth.getWidth();
	const unsigned int newHeight = depth.getHeight();
	const float factorX = (float)(sd.m_depthWidth - 1) / (float)(newWidth - 1);
	const float factorY = (float)(sd.m_depthHeight - 1) / (float)(newHeight - 1);

	//adapt intrinsics
	intrinsic._m00 *= (float)newWidth / (float)sd.m_depthWidth;
	intrinsic._m11 *= (float)newHeight / (float)sd.m_depthHeight;
	intrinsic._m02 *= (float)(newWidth - 1) / (float)(sd.m_depthWidth - 1);
	intrinsic._m12 *= (float)(newHeight - 1) / (float)(sd.m_depthHeight - 1);

	unsigned short* depthVals = sd.decompressDepthAlloc(sensFrameId.y);
	const float depthShift = 1.0f / sd.m_depthShift;
	for (unsigned int j = 0; j < newHeight; j++) {
		for (unsigned int i = 0; i < newWidth; i++) {
			const unsigned x = std::round((float)i * factorX);
			const unsigned y = std::round((float)j * factorY);
			const unsigned short d = depthVals[y * sd.m_depthWidth + x];
			if (d == 0) depth(i, j) = -std::numeric_limits<float>::infinity();
			else {
				float fd = depthShift * d;
				if (fd < minDepth || fd > maxDepth)
					depth(i, j) = -std::numeric_limits<float>::infinity();
				else
					depth(i, j) = fd;
			}
		}
	}

	std::free(depthVals);
	return true;
}