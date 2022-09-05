#include "stdafx.h"
#include "VoxelGrid.h"

#include <functional> //https://blog.csdn.net/chao56789/article/details/80076956 The Chinese people really helped us in this project
#include "densecrf.h"
#include <fstream>
#include <math.h>
#include <algorithm>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

template<typename T>
inline void hash_combine(std::size_t& seed, const T& val)
{
	seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename T>
inline void hash_val(std::size_t& seed, const T& val)
{
	hash_combine(seed, val);
}

template<typename T, typename... Types>
inline void hash_val(std::size_t& seed, const T& val, const Types&... args)
{
	hash_combine(seed, val);
	hash_val(seed, args...);
}

template<typename... Types>
inline std::size_t hash_val(const Types& ...args)
{
	std::size_t seed = 0;
	hash_val(seed, args...);
	return seed;
}

class Hasher {
public:
	size_t operator()(const vec2i& vec) const {
		return hash_val(vec.x, vec.y);
	}
};

class Equal {
public:
	bool operator()(const vec2i& vec1, const vec2i& vec2) const {
		return vec1 == vec2;
	}
};

void medianFilter(BaseImage<int>& img, BaseImage<unsigned char>& isInstance, BaseImage<int>& _img, BaseImage<unsigned char>& _isInstance) {
	const int width = img.getWidth();
	const int height = img.getHeight();
	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			//std::vector<std::pair<int, unsigned char>> window;
			std::map<int, unsigned char> kernel;

			for (int i = -1; i <= 1; ++i) 
				for (int j = -1; j <= 1; ++j) 
//					kernel[img(min(max(u + i, 0), width), min(max(v + j, 0), height))] = isInstance(min(max(u + i, 0), width), min(max(v + j, 0), height));
					if(u + i < width && u + i >= 0 && v + j < height && v + j >= 0)
						kernel[img(u + i, v + j)] = isInstance(u + i, v + j);

			/*window.resize(9);

			window[0] = std::make_pair(img(max(u - 1, 0), max(v - 1, 0)), isInstance(max(u - 1, 0), max(v - 1, 0)));
			window[1] = std::make_pair(img(max(u - 1, 0), v), isInstance(max(u - 1, 0), v));
			window[2] = std::make_pair(img(max(u - 1, 0), min(v + 1, height)), isInstance(max(u - 1, 0), min(v + 1, height)));
			window[3] = std::make_pair(img(u, max(v - 1, 0)), isInstance(u, max(v - 1, 0)));
			window[4] = std::make_pair(img(u, v), isInstance(u, v));
			window[5] = std::make_pair(img(u, min(v + 1, height)), isInstance(u, min(v + 1, height)));
			window[6] = std::make_pair(img(min(u + 1, width), max(v - 1, 0)), isInstance(min(u + 1, width), max(v - 1, 0)));
			window[7] = std::make_pair(img(min(u + 1, width), v), isInstance(min(u + 1, width), v));
			window[8] = std::make_pair(img(min(u + 1, width), min(v + 1, height)), isInstance(min(u + 1, width), min(v + 1, height)));

			std::sort(window.begin(), window.end());

			_img(u, v) = window[4].first;
			_isInstance(u, v) = window[4].second;*/

			auto& iter = kernel.begin();
			std::advance(iter, kernel.size() / 2);
			_img(u, v) = iter->first;
			_isInstance(u, v) = iter->second;
		}
	}
}


void medianFilter(BaseImage<uchar>& img, BaseImage<unsigned char>& isInstance, BaseImage<uchar>& _img, BaseImage<unsigned char>& _isInstance) {
	const int width = img.getWidth();
	const int height = img.getHeight();
	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			//std::vector<std::pair<int, unsigned char>> window;
			std::map<uchar, unsigned char> kernel;

			for (int i = -1; i <= 1; ++i)
				for (int j = -1; j <= 1; ++j)
					//					kernel[img(min(max(u + i, 0), width), min(max(v + j, 0), height))] = isInstance(min(max(u + i, 0), width), min(max(v + j, 0), height));
					if (u + i < width && u + i >= 0 && v + j < height && v + j >= 0)
						kernel[img(u + i, v + j)] = isInstance(u + i, v + j);

			/*window.resize(9);

			window[0] = std::make_pair(img(max(u - 1, 0), max(v - 1, 0)), isInstance(max(u - 1, 0), max(v - 1, 0)));
			window[1] = std::make_pair(img(max(u - 1, 0), v), isInstance(max(u - 1, 0), v));
			window[2] = std::make_pair(img(max(u - 1, 0), min(v + 1, height)), isInstance(max(u - 1, 0), min(v + 1, height)));
			window[3] = std::make_pair(img(u, max(v - 1, 0)), isInstance(u, max(v - 1, 0)));
			window[4] = std::make_pair(img(u, v), isInstance(u, v));
			window[5] = std::make_pair(img(u, min(v + 1, height)), isInstance(u, min(v + 1, height)));
			window[6] = std::make_pair(img(min(u + 1, width), max(v - 1, 0)), isInstance(min(u + 1, width), max(v - 1, 0)));
			window[7] = std::make_pair(img(min(u + 1, width), v), isInstance(min(u + 1, width), v));
			window[8] = std::make_pair(img(min(u + 1, width), min(v + 1, height)), isInstance(min(u + 1, width), min(v + 1, height)));

			std::sort(window.begin(), window.end());

			_img(u, v) = window[4].first;
			_isInstance(u, v) = window[4].second;*/

			auto& iter = kernel.begin();
			std::advance(iter, kernel.size() / 2);
			_img(u, v) = iter->first;
			_isInstance(u, v) = iter->second;
		}
	}
}

void VoxelGrid::generateReferenceLabel(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, BaseImage<int>& referencePanoptic, BaseImage<unsigned char>& referenceIsInstance) const{
	const int width = depthImage.getWidth();
	const int height = depthImage.getHeight();

	std::map<int, int> cnt;

	BaseImage<int> _referencePanoptic(width, height);
	BaseImage<unsigned char> _referenceIsInstance(width, height);

	float last_valid_depth;
	int last_valid_id;
	char last_valid;
	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			float d = depthImage(u, v);
			
			 //if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) { // checking for depth min and max doesn't matter
			if (d != depthImage.getInvalidValue()) {
				//last_valid_depth = d;
				vec3f backProjectedPoint = depthToSkeleton(intrinsic, u, v, d);
				backProjectedPoint = cameraToWorld * backProjectedPoint;
				vec3i voxel_index = worldToVoxel(backProjectedPoint); // This doesn't make a difference when tested on only one frame
				//vec3i voxel_index = math::round(backProjectedPoint);
				const Voxel& voxel = (*this)(voxel_index.x, voxel_index.y, voxel_index.z);


				if ((voxel.panoptic_weight != std::numeric_limits<float>::min()) && (voxel.panoptic_id <= 40000)) {
				//if ((voxel.panoptic_weight != std::numeric_limits<float>::min())){
					referencePanoptic(u, v) = voxel.panoptic_id;
					referenceIsInstance(u, v) = voxel.isInstance;
					last_valid_id = voxel.panoptic_id;
					last_valid = voxel.isInstance;
					++cnt[voxel.panoptic_id];
				}
				else {
					referencePanoptic(u, v) = last_valid_id;
					referenceIsInstance(u, v) = last_valid;

					/*referencePanoptic(u, v) = 0;
					referenceIsInstance(u, v) = 0;*/
				}
			}
			else {
				referencePanoptic(u, v) = last_valid_id;
				referenceIsInstance(u, v) = last_valid;
				/*referencePanoptic(u, v) = 360000;
				referenceIsInstance(u, v) = 1;*/
			}
		}
	}

	//medianFilter(_referencePanoptic, _referenceIsInstance, referencePanoptic, referenceIsInstance);

	/*for (auto& it : cnt) {
		std::cout << "\n" << it.first << " -> " << it.second << "\n";
	}*/
	//exit(0);
}

void unique(BaseImage<unsigned char>& img, std::unordered_set<uchar>& unq) {
	int num_pixels = img.getWidth() * img.getHeight();
	for (int i = 0; i < num_pixels; ++i)
			unq.insert(img.getData()[i]);
}

void unique(BaseImage<unsigned char>& panotpic, BaseImage<unsigned char>& isInstance, std::unordered_set<uchar>& unq) {
	int num_pixels = panotpic.getWidth() * panotpic.getHeight();
	for (int i = 0; i < num_pixels; ++i) 
		if(isInstance.getData()[i])
			unq.insert(panotpic.getData()[i]);
}

void unique(BaseImage<int>& panotpic, BaseImage<unsigned char>& isInstance, std::unordered_set<int>& unq) {
	int num_pixels = panotpic.getWidth() * panotpic.getHeight();
	for (int i = 0; i < num_pixels; ++i)
		if (isInstance.getData()[i])
			unq.insert(panotpic.getData()[i]);
}

void unique(BaseImage<unsigned char>& panotpic, BaseImage<unsigned char>& isInstance, std::unordered_set<uchar>& unqInstance, std::unordered_set<uchar>& unqSemantic) {
	int num_pixels = panotpic.getWidth() * panotpic.getHeight();
	for (int i = 0; i < num_pixels; ++i)
		if (isInstance.getData()[i])
			unqInstance.insert(panotpic.getData()[i]);
		else
			unqSemantic.insert(panotpic.getData()[i]);
}

void unique(BaseImage<int>& panotpic, BaseImage<unsigned char>& isInstance, std::unordered_set<uchar>& unqInstance, std::unordered_set<uchar>& unqSemantic) {
	int num_pixels = panotpic.getWidth() * panotpic.getHeight();
	for (int i = 0; i < num_pixels; ++i)
		if (isInstance.getData()[i])
			unqInstance.insert(panotpic.getData()[i]);
		else
			unqSemantic.insert(panotpic.getData()[i]);
}

double computeIOU(std::unordered_set < vec2i, Hasher, Equal> & a, std::unordered_set < vec2i, Hasher, Equal> & b, bool debug=false) {
	double unionCount = (double) b.size(), intersectionCount = 0.0;

	for (vec2i point : a)
		if (b.find(point) != b.end())
			intersectionCount += 1.0;
		else
			unionCount += 1.0;

	if (unionCount == 0.0)
		return 0.0;

	if (debug)
		std::cout << "\nIntersection count " << intersectionCount << " Union count " << unionCount << "\n";

	return intersectionCount / unionCount;
}

void insertCoordInMap(std::unordered_map<unsigned char, std::unordered_set < vec2i, Hasher, Equal>> &map, unsigned char id, unsigned char isInstance, int u, int v) {
	if (isInstance) {
		auto& it = map.find(id);

		if (it == map.end()) {
			std::unordered_set < vec2i, Hasher, Equal> set;
			set.insert(vec2i(u, v));
			map[id] = set;
		}
		else {
			map[id].insert(vec2i(u, v));
		}
	}
}

void insertCoordInMap(std::unordered_map<int, std::unordered_set < vec2i, Hasher, Equal>>& map, int id, unsigned char isInstance, int u, int v) {
	if (isInstance) {
		auto& it = map.find(id);

		if (it == map.end()) {
			std::unordered_set < vec2i, Hasher, Equal> set;
			set.insert(vec2i(u, v));
			map[id] = set;
		}
		else {
			map[id].insert(vec2i(u, v));
		}

		
		/*if (id > 40000) {
			std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl;
			std::cout << id << std::endl;
			std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << std::endl << std::endl << std::endl << std::endl << std::endl;
		}*/
	}
}

void mask(BaseImage<int>& in, int val, BaseImage<int>& mask) {
	int numPixels = in.getWidth() * in.getHeight();

	for (int i = 0; i < numPixels; ++i)
		mask.getData()[i] = in.getData()[i] == val ? val : 0;
}

void mask(BaseImage<uchar>& in, int val, BaseImage<uchar>& mask) {
	int numPixels = in.getWidth() * in.getHeight();

	for (int i = 0; i < numPixels; ++i)
		mask.getData()[i] = in.getData()[i] == val ? val : 0;
}

void computeMaskInformation(BaseImage<unsigned char>& liveLabel, BaseImage<unsigned char>& liveIsInstance, BaseImage<int>& referenceLabel, BaseImage<unsigned char>& referenceIsInstance,
	std::unordered_map<unsigned char, std::unordered_set < vec2i, Hasher, Equal>> &live_idToCoords, std::unordered_map<int, std::unordered_set < vec2i, Hasher, Equal>> &reference_idToCoords, std::map<unsigned char, int>& liveIdToSize) {

	const int width = liveLabel.getWidth();
	const int height = liveLabel.getHeight();
	
	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			insertCoordInMap(live_idToCoords, liveLabel(u, v), liveIsInstance(u, v), u, v);
			insertCoordInMap(reference_idToCoords, referenceLabel(u, v), referenceIsInstance(u, v), u, v);
		}
	}

	for (const auto& entry : live_idToCoords)
		liveIdToSize[entry.first] = entry.second.size();
}

void VoxelGrid::createConsistentLabel(BaseImage<unsigned char>& liveLabel, BaseImage<unsigned char>& liveIsInstance, BaseImage<int>& referenceLabel, BaseImage<unsigned char>& referenceIsInstance, BaseImage<int>& consistentLabel,
	BaseImage<unsigned char>& consistentIsInstance) {

	std::unordered_map<unsigned char, std::unordered_set < vec2i, Hasher, Equal>> live_idToCoords; // 2D coordinates of pixels that contain a certain id, of the live panoptic segmentation frame. Maps from the id to the set of coordinates.
	std::unordered_map<int, std::unordered_set < vec2i, Hasher, Equal>> reference_idToCoords;

	std::map<unsigned char, int> liveIdToSize; // The live ids sorted by their occurence number (i.e. size of the mask)

	std::unordered_map<int, unsigned char> associations; // Reference to live
	std::unordered_map<unsigned char, int> idLookUp; // Live to reference

	const int width = liveLabel.getWidth();
	const int height = liveLabel.getHeight();

	std::map<int, int> h, y;

	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			++h[referenceLabel(u, v)];
			++y[liveLabel(u, v)];
		}
	}

	//std::cout << "\n###########################################################################\n";
	//std::cout << "\nReference Count\n";
	//std::cout << "{";
	//for (auto& i : h)
	//	std::cout << i.first << " -> " << i.second << "\n";
	//std::cout << "}\n";
	//std::cout << "###########################################################################\n";


	//std::cout << "\n###########################################################################\n";
	//std::cout << "\nLive Count\n";
	//std::cout << "{";
	//for (auto& i : y)
	//	std::cout << i.first << " -> " << i.second << "\n";
	//std::cout << "}\n";
	//std::cout << "###########################################################################\n";

	//system("pause");

	std::unordered_set<int> toDelete;

	computeMaskInformation(liveLabel, liveIsInstance, referenceLabel, referenceIsInstance, live_idToCoords, reference_idToCoords, liveIdToSize);

	for (auto& iter : liveIdToSize) {
		if (iter.second < 200) {
			toDelete.insert(iter.first);
		}
	}

	for (auto& iter : toDelete) {
		live_idToCoords.erase(iter);
		liveIdToSize.erase(iter);
	}

	/*std::cout << "\n###########################################################################\n";
	std::cout << "\nliveIdToSize\n";
	std::cout << liveIdToSize.size() << "\n";
	std::cout << "{";
	for (auto& r : liveIdToSize)
		std::cout << (int)r.first << " - " << (int)r.second << "\n";
	std::cout << "}\n";
	std::cout << "###########################################################################\n";*/

	//system("pause");
	/*std::unordered_set<uchar> unqLive;
	std::unordered_set<int> unqRef, unqCons;
	unique(referenceLabel, referenceIsInstance, unqRef);
	unique(liveLabel, liveIsInstance, unqLive);

	std::cout << "\n###########################################################################\n";
	std::cout << "\nReference\n";
	std::cout << unqRef.size() << "\n";
	std::cout << "{";
	for (int i : unqRef)
		std::cout << (int)i << "\n";
	std::cout << "}\n";
	std::cout << "###########################################################################\n";


	std::cout << "\n###########################################################################\n";
	std::cout << "\nLive\n";
	std::cout << unqLive.size() << "\n";
	std::cout << "{";
	for (uchar i : unqLive)
		std::cout << (int)i << "\n";
	std::cout << "}\n";
	std::cout << "###########################################################################\n";*/


	for (auto& it = liveIdToSize.rbegin(); it != liveIdToSize.rend(); ++it) { // Reverse iterator on sorted map
		bool foundMatch = false;
		double maxIou = std::numeric_limits<double>::min();
		int argmax = std::numeric_limits<int>::max(); // THIS IS WHERE WE GET WEIRD NUMBERS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (maybe)
		for (auto& entry : reference_idToCoords) {
			if (associations.find(entry.first) == associations.end()) { // If this reference id hasn't been associated yet
				double currIou = computeIOU(live_idToCoords[it->first], entry.second, it->first == 59 && entry.first == 59);
				/*if (currIou > 0.1) {
					std::cout << "\n" << currIou << "\n";
				}*/
				/*if (it->first == 59 && entry.first == 59) {
					std::cout << "\n@@@@@@@@@@@@@@@@\n";
					std::wcout << entry.first << "\n";
					std::cout << currIou << "\n";
					std::cout << "\n@@@@@@@@@@@@@@@@\n";

					for (auto& y : entry.second) {
						std::cout << y << "\n";
					}
				}*/
				if (currIou > consistencyThreshold && currIou > maxIou) {
					/*std::cout << "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
					std::cout << currIou << "\n";
					std::cout << argmax << "\n";
					std::cout << it->first << "\n";
					std::cout << "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";*/
					maxIou = currIou;
					argmax = entry.first;
					foundMatch = true;
				}
			}
		}
		if (foundMatch) {
			/*std::cout << "\n1111111111111111111111111111111111111111111111111111111111111111111111111\n";
			std::cout << (int)it->first << "\n";
			std::cout << (int)argmax << "\n";
			std::cout << "\n1111111111111111111111111111111111111111111111111111111111111111111111111\n";*/

			// no need to register it in the id set since it should have been already
			associations[argmax] = it->first;
			idLookUp[it->first] = argmax;
		}else {
			bool foundNewId = false;
			//if (instanceIds.find(it->first) == instanceIds.end()) {
			if(false){
				associations[it->first] = it->first;
				idLookUp[it->first] = it->first;
				instanceIds[it->first] = 0;
				foundNewId = true;
			}
			else {
				for (int i = 40000; i > 0; --i) { // i > 0, because 0 is the void label
					if (instanceIds.find(i) == instanceIds.end()) {
						associations[i] = it->first;
						/*std::cout << "\n22222222222222222222222222222222222222222222222222222222222222222222222222222222222\n";
						std::cout << (int)it->first << "\n";
						std::cout << (int)i<< "\n";
						std::cout << (int)associations[i] << "\n";
						std::cout << "\n22222222222222222222222222222222222222222222222222222222222222222222222222222222222\n";*/
						idLookUp[it->first] = i;
						//instanceIds.insert(i);
						instanceIds[i] = 0;
						foundNewId = true;
						//if (i < 70) {
						//	std::cout << "A7AAAAAAAAAAAAAAAAAAAAAAAAAAAA22222222222222" << std::endl;
						//}
						//std::cout << "######" << i << "#####" << std::endl;
						////instanceIds.insert(i);
						//std::cout << "\n###########################################################################\n";
						//std::cout << "\nINSTANCE\n";
						//std::cout << instanceIds.size() << "\n";
						//std::cout << "{";
						//for (int i : instanceIds)
						//	std::cout << (int)i << " - ";
						//std::cout << "}\n";
						break;
					}
				}
				if (!foundNewId) {
					std::cout << "\n###########################################################################\n";
					std::cout << "warning: exceeded id range, overwriting a new id now.\n";
					std::cout << "\n###########################################################################\n";

					int rand_id = 1 + (std::rand() % (40000 - 1 + 1));
					/*std::cout << "\n333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333\n";
					std::cout << (int)it->first << "\n";
					std::cout << (int)rand_id << "\n";
					std::cout << "\n333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333\n";*/
					associations[rand_id] = it->first;
					idLookUp[it->first] = rand_id; // Assign random number from 1-255
					//instanceIds.insert(rand_id);
					instanceIds[rand_id] = 0;
				}
			}
		}
	}


	/*std::cout << "\n###########################################################################\n";
	std::cout << "\ASSOCIATIONS\n";
	std::cout << associations.size() << "\n";
	std::cout << "{";
	for (auto& r : associations)
		std::cout << (int)r.first << " - " << (int)r.second << "\n";
	std::cout << "}\n";
	std::cout << "###########################################################################\n";*/

	//system("pause");

	//std::cout << "\n**************************************************************************\n";
	//std::cout << "\IDLOOKUP\n";
	//std::cout << idLookUp.size() << "\n";
	//std::cout << "{";
	//for (auto& r : idLookUp)
	//	std::cout << (int)r.first << " - " << (int)r.second << "\n";
	//std::cout << "}\n";
	//std::cout << "\n**************************************************************************\n";


	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			if (liveIsInstance(u, v) && !idLookUp[liveLabel(u, v)]) {
				consistentIsInstance(u, v) = 0;
				consistentLabel(u, v) = 0;
				continue;
			}
			consistentIsInstance(u, v) = liveIsInstance(u, v);

			if (!liveIsInstance(u, v)) 
				consistentLabel(u, v) = liveLabel(u, v);
			else {
				/*std::cout << "\n###########################################################################\n";
				std::cout << (int)idLookUp[liveLabel(u, v)] << "\n";
				std::cout << "\n###########################################################################\n";*/
				consistentLabel(u, v) = idLookUp[liveLabel(u, v)];
			}
		}
	}

	//unique(consistentLabel, consistentIsInstance, unqCons);
	//std::cout << "\n###########################################################################\n";
	//std::cout << "\nConsistent\n";
	//std::cout << unqCons.size() << "\n";
	//std::cout << "{";
	//for (int i : unqCons)
	//	std::cout << (int)i << "\n";
	//std::cout << "}\n";
	//std::cout << "###########################################################################\n";

	/*std::unordered_map<int, int> cc, ic;

	for (int u = 0; u < width; ++u) {
		for (int v = 0; v < height; ++v) {
			++cc[consistentLabel(u, v)];
			++ic[liveLabel(u, v)];
		}
	}

	std::cout << "\n###########################################################################\n";
	std::cout << "\nConsistent Count\n";
	std::cout << "{";
	for (auto& i : cc)
		std::cout << i.first << " -> " << i.second << "\n";
	std::cout << "}\n";
	std::cout << "###########################################################################\n";

	std::cout << "\n###########################################################################\n";
	std::cout << "\nConsistent Count\n";
	std::cout << "{";
	for (auto& i : ic)
		std::cout << i.first << " -> " << i.second << "\n";
	std::cout << "}\n";
	std::cout << "###########################################################################\n";*/

	//live_idToCoords.clear(); // On the stack, should be deleted automatically, clear has linear time complexity too
	//reference_idToCoords.clear();
	//liveIdToSize.clear();
	//associations.clear();
}

void baseImageToMat(const BaseImage<unsigned char>& label, Mat& out) {
	for (int i = 0; i < label.getWidth() * label.getHeight(); ++i)
		out.data[i] = label.getData()[i];
}

void idToRGB(unsigned int id, unsigned char isInstance, Vec3b& color) {
	if (isInstance == 0) {
		color[0] = abs(100 -( (unsigned char)(id & 0xF000 >> 16)));
		color[1] = abs(100 - ( (unsigned char)(id & 0xF00 >> 8)));
		color[2] = abs(100 - ( (unsigned char)(id & 0xFF)));
	}
	else {
		color[0] = abs(100 - (unsigned char)(id & 0xFF));
		color[1] = abs(100 - (unsigned char)(id & 0xF000 >> 16));
		color[2] = abs(100 - (unsigned char)(id & 0xF00 >> 8));
	}
}

void colorPanoptic(BaseImage<int>& panoptic, BaseImage<unsigned char>& isInstance, Mat& _panoptic) {
	const int width = panoptic.getWidth();
	const int height = panoptic.getHeight();

	int numPixels = width * height;

	for (int i = 0; i < numPixels; ++i) {
		Vec3b color;
		int id = panoptic.getData()[i];
		int flag = isInstance.getData()[i];

		auto& _color = RGBColor::colorPalette(id);
		color[0] = _color.r;
		color[1] = _color.g;
		color[2] = _color.b;

		_panoptic.at<Vec3b>(Point(i%width,i/width)) = color;
	}
}

void colorPanoptic(BaseImage<unsigned char>& panoptic, BaseImage<unsigned char>& isInstance, Mat& _panoptic) {
	const int width = panoptic.getWidth();
	const int height = panoptic.getHeight();

	int numPixels = width * height;

	for (int i = 0; i < numPixels; ++i) {
		Vec3b color;
		int id = panoptic.getData()[i];
		int flag = isInstance.getData()[i];

		auto& _color = RGBColor::colorPalette(id);
		color[0] = _color.r;
		color[1] = _color.g;
		color[2] = _color.b;

		_panoptic.at<Vec3b>(Point(i % width, i / width)) = color;
	}
}

void colorPanoptic(const BaseImage<unsigned char>& panoptic, BaseImage<unsigned char>& isInstance, Mat& _panoptic) {
	const int width = panoptic.getWidth();
	const int height = panoptic.getHeight();

	int numPixels = width * height;

	for (int i = 0; i < numPixels; ++i) {
		Vec3b color;
		int id = panoptic.getData()[i];
		int flag = isInstance.getData()[i];

		auto& _color = RGBColor::colorPalette(id);
		color[0] = _color.r;
		color[1] = _color.g;
		color[2] = _color.b;

		_panoptic.at<Vec3b>(Point(i % width, i / width)) = color;
	}
}

void colorPanoptic(const BaseImage<int>& panoptic, BaseImage<unsigned char>& isInstance, Mat& _panoptic) {
	const int width = panoptic.getWidth();
	const int height = panoptic.getHeight();

	int numPixels = width * height;

	for (int i = 0; i < numPixels; ++i) {
		Vec3b color;
		int id = panoptic.getData()[i];
		int flag = isInstance.getData()[i];

		auto& _color = RGBColor::colorPalette(id);
		color[0] = _color.r;
		color[1] = _color.g;
		color[2] = _color.b;

		_panoptic.at<Vec3b>(Point(i % width, i / width)) = color;
	}
}

void colorToMat(BaseImage<vec3f>& colorImage, Mat& _color) {
	uchar max = 0;
	int numPixels = colorImage.getWidth() * colorImage.getHeight();

	for(int i = 0; i < colorImage.getWidth(); ++i)
		for (int j = 0; j < colorImage.getHeight(); ++j) {
			vec3f color = colorImage(i, j);
			Vec3b __color((uchar)(color.x ), (uchar)(color.y ), (uchar)(color.z ));
			//std::cout << __color << "\n";
			_color.at<Vec3b>(Point(i, j)) = __color;
		}
}

void panopticToMat(BaseImage<unsigned char>& panoptic, BaseImage<unsigned char>& isInstance, Mat& _panoptic) {
	uchar max = 0;
	int numPixels = panoptic.getWidth() * panoptic.getHeight();
	std::unordered_set<uchar> unqInstance, unqSemantic;
	std::unordered_map<uchar, uchar> associations;

	unique(panoptic, isInstance, unqInstance, unqSemantic);



	//std::cout << "\n###########################################################################\n";
	//std::cout << "\nINSTANCE\n";
	//std::cout << unqInstance.size() << "\n";
	//std::cout << "{";
	//for (uchar i : unqInstance)
	//	std::cout << (int)i << " - ";
	//std::cout << "}\n";

	//std::cout << "\nSEMANTIC\n";
	//std::cout << unqSemantic.size() << "\n";
	//std::cout << "{";
	//for (uchar i : unqSemantic)
	//	std::cout << (int)i << " - ";
	//std::cout << "}\n";
	//std::cout << "\n###########################################################################\n";


	for (int i = 0; i < numPixels; ++i) {
		uchar panId = panoptic.getData()[i];
		if (!isInstance.getData()[i]) {
			_panoptic.data[i] = panId;
		}
		else if (unqSemantic.find(panId) == unqSemantic.end()) {
			_panoptic.data[i] = panId;
		}
		else if (associations.find(panId) != associations.end()) {
			_panoptic.data[i] = associations[panId];
		}
		else {
			bool foundMatch = false;
			for (int j = 1; j < 256; ++j) {
				if ((unqInstance.find(j) == unqInstance.end()) && (unqSemantic.find(j) == unqSemantic.end())) {
					unqInstance.insert(j);
					associations[panId] = j;
					foundMatch = true;
					_panoptic.data[i] = j;
					break;
				}
			}

			if (!foundMatch) {
				std::cout << "\n###########################################################################\n";
				std::cout << "WARNING: Exceeded id range, overwriting a new id now.\n";
				std::cout << "\n###########################################################################\n";
				system("pause");
			}
		}
	}
}

void panopticToMat(BaseImage<int>& panoptic, BaseImage<unsigned char>& isInstance, Mat& _panoptic) {
	uchar max = 0;
	int numPixels = panoptic.getWidth() * panoptic.getHeight();
	std::unordered_set<uchar> unqInstance, unqSemantic;
	std::unordered_map<uchar, uchar> associations;

	unique(panoptic, isInstance, unqInstance, unqSemantic);



	//std::cout << "\n###########################################################################\n";
	//std::cout << "\nINSTANCE\n";
	//std::cout << unqInstance.size() << "\n";
	//std::cout << "{";
	//for (uchar i : unqInstance)
	//	std::cout << (int)i << " - ";
	//std::cout << "}\n";

	//std::cout << "\nSEMANTIC\n";
	//std::cout << unqSemantic.size() << "\n";
	//std::cout << "{";
	//for (uchar i : unqSemantic)
	//	std::cout << (int)i << " - ";
	//std::cout << "}\n";
	//std::cout << "\n###########################################################################\n";


	for (int i = 0; i < numPixels; ++i) {
		uchar panId = panoptic.getData()[i];
		if (!isInstance.getData()[i]) {
			_panoptic.data[i] = panId;
		}
		else if (unqSemantic.find(panId) == unqSemantic.end()) {
			_panoptic.data[i] = panId;
		}
		else if (associations.find(panId) != associations.end()) {
			_panoptic.data[i] = associations[panId];
		}
		else {
			bool foundMatch = false;
			for (int j = 1; j < 256; ++j) {
				if ((unqInstance.find(j) == unqInstance.end()) && (unqSemantic.find(j) == unqSemantic.end())) {
					unqInstance.insert(j);
					associations[panId] = j;
					foundMatch = true;
					_panoptic.data[i] = j;
					break;
				}
			}

			if (!foundMatch) {
				std::cout << "\n###########################################################################\n";
				std::cout << "WARNING: Exceeded id range, overwriting a new id now.\n";
				std::cout << "\n###########################################################################\n";
				system("pause");
			}
		}
	}
}


static int frame_index = 0;

float weightCalculate(float z, float truncDist, float voxelSize, float sdf) {
	if (-voxelSize < sdf) {
		return 1.0 / z * z;
	}
	if (sdf < -truncDist) {
		return 0;
	}
	return (1.0 / z * z) * (1.0 / truncDist - voxelSize) * (truncDist + sdf);
}

void ucharToInt(BaseImage<uchar>& in, BaseImage<int>& out) {
	const int width = in.getWidth();
	const int height = in.getHeight();

	int numPixels = width * height;

	for (int i = 0; i < numPixels; ++i)
		out.getData()[i] = (int)in.getData()[i];
}
void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned char>& label, const BaseImage<unsigned char>& instance,
	const BaseImage<unsigned char>& instance_gt, BaseImage<unsigned char>& panoptic, BaseImage<unsigned char>& isInstance, BaseImage<vec3f>& colorImage, bool debugOut /*= false*/)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	// Backprojected the camera boundary points to world space. So that anything outside of this bbox is not seen
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	if (debugOut) { // Only for debugging
		PointCloudf pc;
		for (const auto& p : depthImage) {
			if (p.value > 0 && p.value != -std::numeric_limits<float>::infinity()) {
				vec3f cam = depthToSkeleton(intrinsic, p.x, p.y, p.value); // Backprojecting pixel to camera space
				vec3f worldpos = cameraToWorld * cam; // Transforming point in camera space to world space
				pc.m_points.push_back(m_worldToGrid * worldpos); // Appending to point cloud, and transforming from world to grid space
				RGBColor c = RGBColor::colorPalette(instance(p.x, p.y)); // This is a color pallette with apparently a max number of colors = 64
				pc.m_colors.push_back(vec4f(c)); // Adding the color information of this point based on the instance id
			}
		}
		PointCloudIOf::saveToFile("points.ply", pc);

		TriMeshf trimesh = computeInstanceMesh(m_voxelSize);
		MeshIOf::saveToFile("before-integrate.ply", trimesh.computeMeshData());
		trimesh = computeInstanceMesh(m_voxelSize * 2.0f);
		MeshIOf::saveToFile("before-integrate-1.5.ply", trimesh.computeMeshData());
	}

	const int width = label.getWidth();
	const int height = label.getHeight();
	BaseImage<int> consistentPanoptic(width, height), referencePanoptic(width, height);
	BaseImage<unsigned char>  consistentIsInstance(width, height), referenceIsInstance(width, height);


	generateReferenceLabel(intrinsic, cameraToWorld, depthImage, referencePanoptic, referenceIsInstance);


	/*Mat _panoptic(Size(panoptic.getWidth(), panoptic.getHeight()), CV_8UC1), _instance(Size(panoptic.getWidth(), panoptic.getHeight()), CV_8UC1), _reference(Size(panoptic.getWidth(), panoptic.getHeight()), CV_8UC1), _consistent(Size(panoptic.getWidth(), panoptic.getHeight()), CV_8UC1),
		_instance_colored, _reference_colored, _consistent_colored;*/

	Mat _consistent(Size(instance.getWidth(), instance.getHeight()), CV_8UC3), _reference(Size(instance.getWidth(), instance.getHeight()), CV_8UC3), _instance(Size(instance.getWidth(), instance.getHeight()), CV_8UC3), _panoptic(Size(instance.getWidth(), instance.getHeight()), CV_8UC3);
	Mat _color(Size(instance.getWidth(), instance.getHeight()), CV_8UC3);
	Mat _gt(Size(instance.getWidth(), instance.getHeight()), CV_8UC3);
	//createConsistentLabel(panoptic, isInstance, referencePanoptic, referenceIsInstance, consistentPanoptic, consistentIsInstance);
	//BaseImage<int> __panoptic(panoptic.getWidth(), panoptic.getHeight());
	//ucharToInt(panoptic, __panoptic);
	createConsistentLabel(panoptic, isInstance, referencePanoptic,  referenceIsInstance, consistentPanoptic, consistentIsInstance);


	/*BaseImage<int> maskRef(instance.getWidth(), instance.getHeight());
	BaseImage<uchar> maskLive(instance.getWidth(), instance.getHeight()), live_(instance.getWidth(), instance.getHeight()), _(instance.getWidth(), instance.getHeight());*/
	//medianFilter(panoptic, isInstance, live_, _);
	/*Mat _maskRef(Size(instance.getWidth(), instance.getHeight()), CV_8UC3), _maskLive(Size(instance.getWidth(), instance.getHeight()), CV_8UC3);*/


	/*mask(referencePanoptic, 21, maskRef);
	mask(panoptic, 21, maskLive);*/

	bool vis = 0;
	if (vis && frame_index >= 1135) {
		colorPanoptic(consistentPanoptic, consistentIsInstance, _consistent);
		colorPanoptic(referencePanoptic, referenceIsInstance, _reference);
		colorPanoptic(panoptic, isInstance, _panoptic);
		colorPanoptic(instance, isInstance, _instance);
		colorPanoptic(instance_gt, isInstance, _gt);


		/*colorPanoptic(maskRef, referenceIsInstance, _maskRef);
		colorPanoptic(maskLive, isInstance, _maskLive);*/

		///*panopticToMat(panoptic, isInstance, _panoptic);
		//panopticToMat(referencePanoptic, referenceIsInstance, _reference);
		colorToMat(colorImage, _color);
		///*baseImageToMat(instance, _instance);*/

		///*applyColorMap(_panoptic, _panoptic_colored, COLORMAP_JET);
		//applyColorMap(_reference, _reference_colored, COLORMAP_JET);
		//applyColorMap(_instance, _instance_colored, COLORMAP_JET);
		//applyColorMap(_consistent, _consistent_colored, COLORMAP_JET);*/

		/*namedWindow("Panoptic", CV_WINDOW_AUTOSIZE);
		imshow("Panoptic", pc);*/
		namedWindow("Instance", CV_WINDOW_AUTOSIZE);
		imshow("Instance", _instance);

		namedWindow("Reference", CV_WINDOW_AUTOSIZE);
		imshow("Reference", _reference);

		namedWindow("consistent", CV_WINDOW_AUTOSIZE);
		imshow("consistent", _consistent);

		namedWindow("Panoptic", CV_WINDOW_AUTOSIZE);
		imshow("Panoptic", _panoptic);

		namedWindow("RGB", CV_WINDOW_AUTOSIZE);
		imshow("RGB", _color);

		namedWindow("GT", CV_WINDOW_AUTOSIZE);
		imshow("GT", _gt);

		/*namedWindow("MaskRef", CV_WINDOW_AUTOSIZE);
		imshow("MaskRef", _maskRef);

		namedWindow("MaskLive", CV_WINDOW_AUTOSIZE);
		imshow("MaskLive", _maskLive);*/

		//std::unordered_set<uchar> unq;
		//unique(consistentPanoptic, unq);

		////std::cout << "\n###########################################################################\n";
		////std::cout << "\CONSISTENT\n";
		////std::cout << unq.size() << "\n";
		////std::cout << "{";
		////for (uchar i : unq)
		////	std::cout << (int)i << "\n";
		////std::cout << "}\n";
		////std::cout << "###########################################################################\n";

		waitKey(0);
	}

	// Loops on each visible voxel with respect to the camera or the frame
	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {

				//transform to current frame
				vec3f p = worldToCamera * voxelToWorld(vec3i(i, j, k));

				//project into depth image
				p = skeletonToDepth(intrinsic, p);
				vec3i pi = math::round(p); // Rounding for pixel coordinates
				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) { // Within image dimensions

					// See the current voxel from the perspective of the current frame
					const float d = depthImage(pi.x, pi.y);
					unsigned char lbl = label(pi.x, pi.y);
					//unsigned char inst = instance(pi.x, pi.y);
					unsigned char inst = instance_gt(pi.x, pi.y);
					vec3f rgb = colorImage(pi.x, pi.y);

					// We can also get isInstance here from the above images, but do we need it elsewhere, like in the iou calculation?
					int panopticId = consistentPanoptic(pi.x, pi.y);
					unsigned char panopticIsInstance = consistentIsInstance(pi.x, pi.y);

					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {

						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);

						if (sdf > -truncation) {
							if (sdf >= 0.0f) {
								sdf = fminf(truncation, sdf);
							}
							else {
								sdf = fmaxf(-truncation, sdf);
							}
							const float integrationWeightSample = 3.0f;
							const float depthWorldMin = 0.4f;
							const float depthWorldMax = 4.0f;
							float depthZeroOne = (d - depthWorldMin) / (depthWorldMax - depthWorldMin);
							float weightUpdate = std::max(integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);
							float colorUpdate = weightCalculate(depthZeroOne, 4.0 * m_voxelSize, m_voxelSize, sdf);

							Voxel& v = (*this)(i, j, k);
							if (v.sdf == -std::numeric_limits<float>::infinity()) { // If this is the first time visiting this voxel
								v.sdf = sdf;
							}
							else { // else, update its sdf: curless & levoy
								v.sdf = (v.sdf * (float)v.weight + sdf * weightUpdate) / (float)(v.weight + weightUpdate);
							}
							v.weight = (uchar)std::min((int)v.weight + (int)weightUpdate, (int)std::numeric_limits<unsigned char>::max());

							if (std::fabs(v.sdf) <= 2.0f * m_voxelSize) { // If it's on the surface. The surface exists where the sdf = 0, but we give some more space for the surface around it in order to not be too thin
								if (std::fabs(sdf) <= 2.0f * m_voxelSize) {
									if (v.color.r == 0 || (lbl != 0)) {
										v.color.r = lbl;
										//v.color.g = lbl == inst ? 0 : inst; // remove sem ids to do evaluation
										//v.color.g = inst;
										//if (lbl == 1 || lbl == 2 || lbl == 22) {
										if(stuff.find(lbl) != stuff.end()){
											v.color.g = 0;
										}
										else {
											v.color.g = inst;
										}
									}

									//std::cout << "v.rgb: " << v.rgb << " " << " rgb:" << rgb << "color weight:"<< colorUpdate<< "voxel weight:"<< v.color_weight<<"\n";

									v.rgb = (v.rgb * (float)v.color_weight + rgb * colorUpdate) / (float)(v.color_weight + colorUpdate);
									v.color_weight = v.color_weight != std::numeric_limits<float>::min() ? colorUpdate + v.color_weight : colorUpdate;

									//v.panoptic_id = panopticId; // This gives 100 AP
									//v.isInstance = panopticIsInstance;
									//v.panoptic_weight = 23;

									// Commenting out label fusion doesn't make a difference (or a big difference, don't remember), signaling that the problem is in the consistent label
									if (colorUpdate > v.panoptic_weight) {
									/*if (instanceIds.find(panopticId) != instanceIds.end() && instanceIds[panopticId] > instanceIds[v.panoptic_id]) {*/
										// update id counts
										if (panopticIsInstance) {
											/*if (!panopticId) {
												std::cout << "\nA7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";
												std::cout << (int)panopticIsInstance << " id " << panopticId << "\n";
											}*/
											++instanceIds[panopticId];
											if (v.panoptic_weight != std::numeric_limits<float>::min() && v.isInstance) { // There was a valid instance id in this voxel
												--instanceIds[v.panoptic_id];
											}
										}
										v.panoptic_id = panopticId;
										v.isInstance = panopticIsInstance;
										v.panoptic_weight = v.panoptic_weight != std::numeric_limits<float>::min() ? colorUpdate - v.panoptic_weight : colorUpdate; // Handle weight update for the first time we see a voxel
									}else {
										if (v.panoptic_id != panopticId || v.isInstance != panopticIsInstance) // Different label
											v.panoptic_weight -= colorUpdate;
										else
											v.panoptic_weight += colorUpdate;
									}
								}
							}
						}
					}
				}

			}
		}
	}
	//std::wcout << "\n###################################\n";
	//std::cout << "\n instance ids\n";
	//for (auto& y : instanceIds) {
	//	std::cout << y.first << " -> " << y.second << "\n";
	//}
	//std::wcout << "\n###################################\n";
	// check pan id == l # Only true for 0 and 2 for some reason...
	// check weight values > min float && > 0 # True
	// If it's an untouched voxel, shouldn't the void id have prob = 1
	// Check 0/void and isInstance # We check for the void to be the size of the instanceIds set, but the map returns all zeros which we don't consider as void (since void is instanceIds.size()), so that's why the 0 gets a true isInstance
	// Consider only adding voxels with valid weights, and make a map to map from the crf id to the grid space
	/*actualR = pixelIndexInResultingArray_i % sizeOfImages[0];
	actualC = (pixelIndexInResultingArray_i / sizeOfImages[0]) % sizeOfImages[1];
	actualZ = (pixelIndexInResultingArray_i / (sizeOfImages[0] * sizeOfImages[1])) % sizeOfImages[2];*/

	//if ((frame_index >= 90 && frame_index % 90 == 0 && frame_index < 2000) || (frame_index > 2000 && frame_index % 300 == 0 && frame_index < 4000) || (frame_index > 4000 && frame_index % 500 == 0) && frame_index < 5000) {
	//if (false) {
	if(frame_index % 1000 == 0 && frame_index < 5000){
	//if (frame_index > 0) {

		//int m = 40000 + 24;
		//int m = instanceIds.size() + 24 + 1;
		//std::unordered_map<int, int> crf_to_pan;
		//std::unordered_map<int, int> pan_to_crf;
		//crf_to_pan.reserve(m);

		//int y = 1;
		//for (auto& id : instanceIds) {
		//	crf_to_pan[y] = id.first;
		//	pan_to_crf[id.first] = y++;
		//}
		//crf_to_pan[0] = 0;
		//crf_to_pan[instanceIds.size() + 1 + 1] = 0;
		//crf_to_pan[instanceIds.size() + 1 + 1 + 1] = 1;
		//crf_to_pan[instanceIds.size() + 1 + 2 + 1] = 2;
		//crf_to_pan[instanceIds.size() + 1 + 22 + 1] = 22;

		//pan_to_crf[0] = instanceIds.size() + 1 + 1;
		//pan_to_crf[1] = instanceIds.size() + 1 + 1 + 1;
		//pan_to_crf[2] = instanceIds.size() + 1 + 2 + 1;
		//pan_to_crf[22] = instanceIds.size() + 1 + 22 + 1;

		int m = instanceIds.size() + 3 + 1;
		std::unordered_map<int, int> crf_to_pan;
		std::unordered_map<int, int> pan_to_crf;
		crf_to_pan.reserve(m);
		pan_to_crf.reserve(m);

		crf_to_pan[0] = 0;
		pan_to_crf[0] = 0;

		int y = 1;
		for (auto& id : instanceIds) {
			if (id.second > 250) {
				crf_to_pan[y] = id.first;
				pan_to_crf[id.first] = y++;
			}
		}

		crf_to_pan[y] = 1;
		pan_to_crf[1] = y++;

		crf_to_pan[y] = 2;
		pan_to_crf[2] = y++;

		crf_to_pan[y] = 22;
		pan_to_crf[22] = y++;

		m = y + 3; // y is 1-based

		//std::cout << "\ncrf to pan\n";
		//for (auto& dd : crf_to_pan)
		//	std::cout << dd.first << " " << dd.second << "\n";

		//std::cout << "\npan to crf\n";
		//for (auto& dd : pan_to_crf)
		//	std::cout << dd.first << " " << dd.second << "\n";

		//std::cout << m << "\n";

		int _Z = voxelBounds.getMaxZ() + 1, _Y = voxelBounds.getMaxY() + 1, _X = voxelBounds.getMaxX() + 1;


		int all_cnt = 0, valid_cnt = 0;
		std::unordered_map<int, int> crf_voxel_idx;
		std::unordered_map<int, int> voxel_crf_idx;
		std::unordered_set<int> taken;
		for (auto& it = this->begin(); it != this->end(); ++it) {
			Voxel& v = (*this)(it.x, it.y, it.z);
			//if ((v.panoptic_weight != std::numeric_limits<float>::min()) && (v.panoptic_id <= 40000)) {
			//	crf_voxel_idx[valid_cnt] = all_cnt;
			//	voxel_crf_idx[all_cnt] = valid_cnt++;
			//}
			//++all_cnt;
			if ((v.panoptic_weight != std::numeric_limits<float>::min()) && (v.panoptic_id <= 40000)) {
				for (int i = -1; i <= 1; ++i) {
					for (int j = -1; j <= 1; ++j) {
						for (int k = -1; k <= 1; ++k) {
							int _i = it.x + i, _j = it.y + j, _k = it.z + k;
							int index = _k * this->getDimY() * this->getDimX() + _j * this->getDimX() + _i;
							if (_i < this->getDimX() && _j < this->getDimY() && _k < this->getDimZ() && voxel_crf_idx.find(index) == voxel_crf_idx.end()) {
								crf_voxel_idx[valid_cnt] = index;
								voxel_crf_idx[index] = valid_cnt++;
							}
						}
					}
				}
			}
		}

		//DenseCRF3D crf(_X, _Y, _Z, m); // TODO: CHANGE 3 TO BE 
		DenseCRF3D crf(valid_cnt, 1, 1, m);
		/*MatrixXf  unary(m, _X * _Y * _Z);*/
		//std::cout << std::distance(this->begin(), this->end()) << "\n


		// MatrixXf unary(m, this->m_dimX * this->m_dimY * this->m_dimZ);
		MatrixXf unary(m, valid_cnt);
		//std::cout << "after unary\n";
		std::ofstream myfile1;
		std::ofstream myfile2;
		myfile1.open("Before.txt");
		/*for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
			for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
				for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {
					Voxel& v = (*this)(i, j, k);*/


		int q = 0;
		/*for (auto& it = this->begin(); it != this->end(); ++it) {
			Voxel& v = (*this)(it.x, it.y, it.z);*/
		for(auto& v_idx: voxel_crf_idx){
			Voxel v = this->m_data[v_idx.first];

			//if(v.panoptic_weight != std::numeric_limits<float>::min())
			myfile1 << v.panoptic_id << " " << (int)v.isInstance << "\n";

			//if (v.panoptic_id == 0 && v.isInstance == 1) // True after first time applying crf
			//	std::cout << "A7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n";

			double prob_true = std::numeric_limits<double>::max();
			double prob_else = std::numeric_limits<double>::max();
			if (v.color_weight > 0 && v.color_weight != std::numeric_limits<float>::min() && v.panoptic_weight != std::numeric_limits<float>::min() && pan_to_crf.find(v.panoptic_id) != pan_to_crf.end()) {
			//if(true){
				//						if (((0.5 * (1 + (v.panoptic_weight / v.color_weight))) > 0) && ((1 / (m - 1)) * (1 - prob_true) > 0)) {
				double p = (0.5 * (1 + (v.panoptic_weight / v.color_weight)));
				double ep = (1 / (m - 1)) * (1 - p);
				//if (p > 0) {
					/*std::cout << "HEREEEEEEEEEEEEEEEEEEE\n";
					std::cout << v.panoptic_weight << " " << v.color_weight << "\n";*/
					prob_true = -1 * log(p + 1e-10);
					/*std::cout << prob_true << "\n";
					std::cout << prob_else << "\n";*/
				//}

				//if (ep > 0) {
					prob_else = -1 * log(ep + 1e-10);
				//}
				//std::cout << prob_true << "\n";
				//	std::cout << prob_else << "\n";
			}
			else {
				prob_true = -1 * log(1e-10);
				prob_else = -1 * log(1.0 - 1e-10);
			}

			//std::cout << "PROB TRUE: " << prob_true << std::endl;
			//std::cout << "PROB ELSE: " << prob_else << std::endl;

			for (int l = 0; l < m; ++l) { // l would never reach the 40000 indexed ids we use...
				if (l == 0) {
					/*unary(l, k* _Y* _X + j * _X + i) = std::numeric_limits<float>::max();*/
					/*unary(l, q) = std::numeric_limits<float>::max();*/
//					unary(l, v_idx.second) = std::numeric_limits<float>::max();
					unary(l, v_idx.second) = -1 * log(1e-10);
					continue;
				}
				if (v.isInstance) {
					/*unary(l, k * _Y * _X + j * _X + i) = l == pan_to_crf[v.panoptic_id] ? prob_true : prob_else;*/
					/*unary(l, q) = l == pan_to_crf[v.panoptic_id] ? prob_true : prob_else;*/
					unary(l, v_idx.second) = l == pan_to_crf[v.panoptic_id] ? prob_true : prob_else; // if not found give very low probability
				}
				else {
					//unary(l, k * _Y * _X + j * _X + i) = l == pan_to_crf[v.panoptic_id] ? prob_true : prob_else; // Handle pan_to_crf that takes non instance ids here
					//unary(l, q) = l == pan_to_crf[v.panoptic_id] ? prob_true : prob_else; // Handle pan_to_crf that takes non instance ids here
					unary(l, v_idx.second) = l == pan_to_crf[v.panoptic_id] ? prob_true : prob_else; // Handle pan_to_crf that takes non instance ids here
				}
			}
			++q;
		}
		/*		}
			}*/
		myfile1.close();
		crf.setUnaryEnergy(unary);
		//std::cout << "After set energy\n";
		crf.addPairwiseGaussian(0.05, 0.05, 0.05, this, crf_voxel_idx, new PottsCompatibility(15.0));
		crf.addPairwiseBilateral(0.05, 0.05, 0.05, 20, 20, 20, this, crf_voxel_idx, new PottsCompatibility(10.0));

		//std::cout << "right before  map energy\n";
		VectorXs map = crf.map(5); // TODO: MAKE 
		//std::cout << "After map energy\n";
		//std::cout << "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\n";

		/*std::cout << "GRID X: " << _X << std::endl;
		std::cout << "GRID Y: " << _Y << std::endl;
		std::cout << "GRID Z: " << _Z << std::endl;
		std::cout << "MAP SIZE: " << map.size() << std::endl;
		std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;*/
		bool zero = 0;
		bool one = 0;
		bool others = 0;
		std::unordered_map<int, int> debug_map;
		for (int i = 0; i < map.size(); i++) {
			++debug_map[map[i]];
			if (map[i] == 0) {
				zero = 1;
			}
			else if (map[i] == 1) {
				one = 1;
			}
			else {
				others = 1;
			}
		}
		std::cout << "\nZERO STATUS: " << zero << std::endl;
		std::cout << "ONE STATUS: " << one << std::endl;
		std::cout << "OTHERS STATUS: " << others << std::endl;
		for (auto& dd : debug_map)
			std::cout << dd.first << " " << dd.second << "\n";
		std::cout << "MAP SIZE: " << map.size() << std::endl;
		std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl << std::endl << std::endl;
		myfile2.open("After.txt");
		/*for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
			for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
				for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {
					Voxel& v = (*this)(i, j, k);*/
		q = 0;
		/*for (auto& it = this->begin(); it != this->end(); ++it) {*/
		for(q; q < valid_cnt; ++q){
			//Voxel& v = (*this)(it.x, it.y, it.z);
			Voxel& v = this->m_data[crf_voxel_idx[q]];
			/*int result_label = map[k * _Y * _X + j * _X + i];*/
			int result_label = map[q];
			//					v.panoptic_id = (result_label >= instanceIds.size()) ? (result_label - instance.size()) : result_label;
			//if (!result_label) {
			//	v.panoptic_id = 0;
			//	v.isInstance = 0;
			//}
			//else {
			//	v.panoptic_id = crf_to_pan[result_label];
			//	v.isInstance = (result_label >= instanceIds.size() + 1) ? 0 : 1;
			//}

			if (result_label) {
				v.panoptic_id = crf_to_pan[result_label];
				//v.isInstance = (result_label >= instanceIds.size() + 1) ? 0 : 1;
				v.isInstance = crf_to_pan[result_label] == 1 || crf_to_pan[result_label] == 2 || crf_to_pan[result_label] == 22 || crf_to_pan[result_label] == 0 ? 0 : 1;
			}

			myfile2 << v.panoptic_id << " " << (int)v.isInstance << " ha " << result_label << "\n";
			++q;
		}
		/*	}
		}*/
		myfile2.close();

	}
	//
	frame_index++;
	//std::cout << frame_index << "\n";
	////if (debugOut) {
	////	TriMeshf trimesh = computeInstanceMesh(m_voxelSize);
	////	MeshIOf::saveToFile("integrated.ply", trimesh.computeMeshData());
	////	trimesh = computeInstanceMesh(m_voxelSize * 2.0f);
	////	MeshIOf::saveToFile("integrated-1.5.ply", trimesh.computeMeshData());
	////	std::cout << "waiting..." << std::endl;
	////	getchar();
	////}
}


void VoxelGrid::evaluate() {
	std::unordered_map<int, int> idCountGT, idCountPred, associations, newCountGT; // gt -> pred
	std::unordered_map<int, std::unordered_map<int, int>> overlap; // gt -> (pred -> count)

	//std::unordered_map<int, int> tnp, tng;
	int all_cnt = 0, valid_cnt = 0;
	for (auto& it = this->begin(); it != this->end(); ++it){
		Voxel& v = (*this)(it.x, it.y, it.z);
		++all_cnt;
		if ((v.panoptic_weight != std::numeric_limits<float>::min()) && (v.panoptic_id <= 40000)) {
			++valid_cnt;
			int gt = v.color.g, pred = v.panoptic_id;
			/*++tng[gt];
			if(v.isInstance)
			++tnp[pred];*/

			if(v.color.g)
				++idCountGT[v.color.g];
			if (v.isInstance) {
				++idCountPred[v.panoptic_id];
				if (overlap.find(gt) == overlap.end()) {
					std::unordered_map<int, int> _cnt;
					_cnt[pred] = 1;
					overlap[gt] = _cnt;
				}
				else {
					++overlap[gt][pred];
				}
			}
		}
	}

	std::cout << "\nall " << all_cnt << " valid " << valid_cnt << "\n";

	std::vector<int> toDelete;

	for (auto& it : idCountGT) {
		if (it.second <= 250) {
			toDelete.push_back(it.first);
		}
	}

	for (int& a : toDelete) {
		idCountGT.erase(a);
	}

	toDelete.clear();

	for (auto& it : idCountPred) {
		if (it.second <= 250) {
			toDelete.push_back(it.first);
		}
	}

	for (int& a : toDelete) {
		idCountPred.erase(a);
	}

	/*std::cout << "\n pred found\n";
	for (auto& it : tnp)
		std::cout << it.first << " " <<it.second << "\n";

	std::cout << "\n existing ids\n";
	for (auto& it : instanceIds)
		std::cout << it.first << " " << it.second << "\n";*/

	//exit(0);
	
	double thresholds[10] = {0.2, 0.4, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9};
	double recall_segments[101];

	double aps[10];
	double ap = 0.0;

	double r = 0.0;
	for (int i = 0; i < 101; ++i) {
		recall_segments[i] = r;
		r += 0.01;
	}

	std::unordered_map<int, double> gt_pred_map; // gt -> iou. In this case, we keep track of max_iou, so far we don't need to keep track of the iou
	std::map<double, double> recall_precision_map; // recall -> precision
	

	std::cout << "\n GT count\n";
	for (auto& y : idCountGT) {
		std::cout << y.first << " -> " << y.second << "\n";
	}

	std::cout << "\n pred count\n";
	for (auto& y : idCountPred) {
		std::cout << y.first << " -> " << y.second << "\n";
	}

	/*std::cout << "\n instance ids\n";
	for (auto& y : instanceIds) {
		std::cout << y.first << " -> " << y.second << "\n";
	}*/

	//exit(0);
	for (int t = 0; t < 10; ++t) {
		double thresh = thresholds[t];
		int tp = 0, fp = 0, fn = idCountGT.size();
		for (auto& pred : idCountPred) {
			//std::cout << "\npred " << pred.first << " count " << pred.second << "\n"; // According to this, we're not considering all pred ids, or at least some pred ids are being overwritten for some reason
			bool found_match = false;
			for (auto& gt : idCountGT) {
				double iou = 0.0;
				if (overlap.find(gt.first) != overlap.end() && overlap[gt.first].find(pred.first) != overlap[gt.first].end()) { // if gt overlaps with anything, and if gt overlaps with pred in particular
					iou = ((double)overlap[gt.first][pred.first]) / ((double)(gt.second + pred.second - overlap[gt.first][pred.first])); // handle weird case when iou is 1 for pred count = 1
					/*if (pred.second == 1) {
						std::cout << "\noverlap " << overlap[gt.first][pred.first] << " gt count " << gt.second <<"\n";
						std::cout << iou << "\n";
					}*/
				}

				//std::cout << "\niou " << iou << "\n";

				if (iou >= thresh) {
					found_match = true; // in case of fp, we should set it to true as well since we increment fp in this if block, to avoid double increment
					if (gt_pred_map.find(gt.first) == gt_pred_map.end()) { // if gt was never associated
						++tp;
						--fn;
						gt_pred_map[gt.first] = iou;
						associations[gt.first] = pred.first;
					}else {
						++fp; // double matching
						if (iou > gt_pred_map[gt.first]) {
							gt_pred_map[gt.first] = iou;
							associations[gt.first] = pred.first;
						}
					}
				}
			}
			if (!found_match) { // pred never exceeded threshold
				//std::cout << "\niou failure\n";
				++fp;
			}
			double curr_precision = (double)tp / ((double)tp + (double)fp);
			double curr_recall = (double)tp / ((double)tp + (double)fn);

			recall_precision_map[curr_recall] = curr_precision;
		}

		double average_precision = 0.0;
		std::cout << "\ngt count " << idCountGT.size() << " pred count " << idCountPred.size() << "\n";
		std::cout << "\n" << "fn " << fn << " fp " << fp << " tp " << tp << "\n";

		int _r = 0;
		for (_r = 0; _r < 101; ++_r) {
			double r = recall_segments[_r];
			auto& low = recall_precision_map.lower_bound(r);
			
			if (low == recall_precision_map.end()) {
				//std::cout << "\nHEREEEEEEEEEEEEEEEEEEEEEEEE\n";
				//std::cout << "Recall " << r << "\n";
				//std::cout << "Thresh " << thresh << "\n";

				//std::cout << "\n Recall precision map\n";
				//for (auto& y : recall_precision_map) {
				//	std::cout << y.first << " -> " << y.second << "\n";
				//}
				average_precision += recall_precision_map.rbegin()->second;
				break;
			}

			double curr_max_precision = std::numeric_limits<double>::min();

			for (auto& it = low; it != recall_precision_map.end(); ++it) 
				curr_max_precision = it->second > curr_max_precision ? it->second : curr_max_precision;
			
			average_precision += curr_max_precision;
		}

		//average_precision /= (float)_r;
		average_precision /= (float)101.0;
		aps[t] = average_precision;
		ap += average_precision;

		recall_precision_map.clear();
		gt_pred_map.clear();

		std::cout << "\n Associations\n";
		for (auto& y : associations) {
			std::cout << y.first << " -> " << y.second << "\n";
		}

		associations.clear();
	}


	ap /= 10.0;
	std::cout << "\n" << ap << "\n";

	for (int t = 0; t < 10; ++t)
		std::cout << aps[t] << "\n";

}


vec2ui VoxelGrid::countOccupancyAABB(const bbox3f& aabb, unsigned int weightThresh, float sdfThresh, unsigned short instanceId, MaskType& mask) const {
	vec2ui occ(0, 0);
	bbox3i bounds(math::floor(aabb.getMin()), math::ceil(aabb.getMax()));
	MLIB_ASSERT(bounds.getExtent() != vec3f(0.0f));
	bounds.setMin(math::min(math::max(bounds.getMin(), 0), vec3i((int)getDimX() - 1, (int)getDimY() - 1, (int)getDimZ() - 1)));
	bounds.setMax(math::min(bounds.getMax(), vec3i((int)getDimX(), (int)getDimY(), (int)getDimZ())));
	mask.allocate(bounds.getExtent()); mask.setValues(0);

	const vec3i boundsMin = bounds.getMin();
	const vec3i boundsMax = bounds.getMax();
	for (int k = boundsMin.z; k < boundsMax.z; k++) {
		for (int j = boundsMin.y; j < boundsMax.y; j++) {
			for (int i = boundsMin.x; i < boundsMax.x; i++) {
				const Voxel& v = (*this)(i, j, k);
				if (v.weight >= weightThresh && std::fabs(v.sdf) <= sdfThresh) {
					occ.y++;
					const unsigned short instId = v.color.g;
					if (instId == instanceId) {
						occ.x++;
						vec3i coordBg = vec3i(i, j, k) - boundsMin;
						if (!mask.isValidCoordinate(coordBg))
							throw MLIB_EXCEPTION("bad coord compute for mask");
						mask(coordBg) = 1;
					}
				}
			}  // i
		}  // j
	}  // k

	return occ;
}

TriMeshf VoxelGrid::computeColorMesh(float sdfThresh) const {
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (unsigned int z = 0; z < getDimZ(); z++) {
		for (unsigned int y = 0; y < getDimY(); y++) {
			for (unsigned int x = 0; x < getDimX(); x++) {
				if (std::fabs((*this)(x, y, z).sdf) < sdfThresh) nVoxels++;
			}
		}
	}
	size_t nVertices = nVoxels * 8; //no normals
	size_t nIndices = nVoxels * 12;
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < getDimZ(); z++) {
		for (size_t y = 0; y < getDimY(); y++) {
			for (size_t x = 0; x < getDimX(); x++) {
				const Voxel& v = (*this)(x, y, z);
				if (std::fabs(v.sdf) < sdfThresh) {
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						triMesh.m_vertices.back().color = vec4f(v.rgb / 255.0);
					}
					for (size_t i = 0; i < 12; i++) {
						indices[i] += vertIdxBase;
						triMesh.m_indices.emplace_back(indices[i]);
					}
				}
			}
		}
	}
	triMesh.setHasColors(true);

	return triMesh;
}


TriMeshf VoxelGrid::computePanopticMesh(float sdfThresh) const {
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (unsigned int z = 0; z < getDimZ(); z++) {
		for (unsigned int y = 0; y < getDimY(); y++) {
			for (unsigned int x = 0; x < getDimX(); x++) {
				if (std::fabs((*this)(x, y, z).sdf) < sdfThresh) nVoxels++;
			}
		}
	}
	size_t nVertices = nVoxels * 8; //no normals
	size_t nIndices = nVoxels * 12;
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < getDimZ(); z++) {
		for (size_t y = 0; y < getDimY(); y++) {
			for (size_t x = 0; x < getDimX(); x++) {
				const Voxel& v = (*this)(x, y, z);
				if (std::fabs(v.sdf) < sdfThresh) {
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);
					unsigned int sem = v.panoptic_id;

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for empty
						}
						else if (sem == 255) {
							triMesh.m_vertices.back().color = vec4f(0.5f, 0.5f, 0.5f, 1.0f); //gray for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette((unsigned int)sem);
							triMesh.m_vertices.back().color = vec4f(vec3f(c.x, c.y, c.z) / 255.0f);
						}
					}
					for (size_t i = 0; i < 12; i++) {
						indices[i] += vertIdxBase;
						triMesh.m_indices.emplace_back(indices[i]);
					}
				}
			}
		}
	}
	triMesh.setHasColors(true);

	return triMesh;
}

TriMeshf VoxelGrid::computeLabelMesh(float sdfThresh) const {
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (unsigned int z = 0; z < getDimZ(); z++) {
		for (unsigned int y = 0; y < getDimY(); y++) {
			for (unsigned int x = 0; x < getDimX(); x++) {
				if (std::fabs((*this)(x, y, z).sdf) < sdfThresh) nVoxels++;
			}
		}
	}
	size_t nVertices = nVoxels * 8; //no normals
	size_t nIndices = nVoxels * 12;
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < getDimZ(); z++) {
		for (size_t y = 0; y < getDimY(); y++) {
			for (size_t x = 0; x < getDimX(); x++) {
				const Voxel& v = (*this)(x, y, z);
				if (std::fabs(v.sdf) < sdfThresh) {
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);
					const unsigned char sem = v.color.r;

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for empty
						}
						else if (sem == 255) {
							triMesh.m_vertices.back().color = vec4f(0.5f, 0.5f, 0.5f, 1.0f); //gray for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette((unsigned int)sem);
							triMesh.m_vertices.back().color = vec4f(vec3f(c.x, c.y, c.z) / 255.0f);
						}
					}
					for (size_t i = 0; i < 12; i++) {
						indices[i] += vertIdxBase;
						triMesh.m_indices.emplace_back(indices[i]);
					}
				}
			}
		}
	}
	triMesh.setHasColors(true);

	return triMesh;
}


TriMeshf VoxelGrid::computeInstanceMesh(float sdfThresh) const {
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (unsigned int z = 0; z < getDimZ(); z++) { // Dims are initialized in fuser, basically they are the grid dimensions, scene bounding box/voxel size -> number of voxels in each dimensions
		for (unsigned int y = 0; y < getDimY(); y++) {
			for (unsigned int x = 0; x < getDimX(); x++) {
				if (std::fabs((*this)(x, y, z).sdf) < sdfThresh) nVoxels++; // gets a voxel, checks its sdf
			}
		}
	}

	size_t nVertices = nVoxels * 8; //no normals A voxel/cube has 8 vertices
	size_t nIndices = nVoxels * 12; // I think the indices are the faces/triangulations
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < getDimZ(); z++) {
		for (size_t y = 0; y < getDimY(); y++) {
			for (size_t x = 0; x < getDimX(); x++) {
				const Voxel& v = (*this)(x, y, z);
				if (std::fabs(v.sdf) < sdfThresh) {
					vec3f p(x, y, z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f; // Is this the voxel size?
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices); // This creates the voxel
					const unsigned char sem = v.color.g;

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size()); // This gets the current size of m_vertices, not nVertices
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for empty
						}
						else if (sem == 255) {
							triMesh.m_vertices.back().color = vec4f(0.5f, 0.5f, 0.5f, 1.0f); //gray for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette((unsigned int)sem);
							triMesh.m_vertices.back().color = vec4f(vec3f(c.x, c.y, c.z) / 255.0f);
						}
					}
					for (size_t i = 0; i < 12; i++) {
						indices[i] += vertIdxBase;
						triMesh.m_indices.emplace_back(indices[i]);
					}
				}
			}
		}
	}
	triMesh.setHasColors(true);

	return triMesh;
}