/***  File Header  ************************************************************/
/**
* generate.cpp
*
* Tiny ML interpreter on ONNX runtime
* @author   Shozo Fukuda
* @date		create Wed Jun 28 17:19:28 JST 2023
* System    Windows10<br>
*
**/
/**************************************************************************{{{*/

#pragma warning(disable : 4996)

#include "stdafx.h"
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;
#include <string>
#include <vector>
#include <random>
#include <memory>

#include "getopt/getopt.h"
#include "tf2/tf2_interp.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "CImgEx.h"
using namespace cimg_library;


typedef std::vector<int> Seeds;

#define MAX_LATANT	512
typedef std::vector<std::unique_ptr<float>> LatantIn;

/***  Module Header  ******************************************************}}}*/
/**
* <title>
* @par DESCRIPTION
*   <description>
**/
/**************************************************************************{{{*/
Seeds parse_seeds(const std::string& str)
{
	Seeds seeds;

	if (str.empty()) {
		return seeds;
	}

	size_t pos = std::string::size_type(0);
	do {
		std::string chunk;

		// split to chunk
		size_t end = str.find(',', pos);
		if (end != std::string::npos) {
			chunk = str.substr(pos, end - pos);
			pos = end + 1;
		}
		else {
			chunk = str.substr(pos);
			pos = std::string::npos;
		}

		// translate seeds
		size_t dash = chunk.find('-');
		if (dash != std::string::npos) {
			// range seeds
			int beg = std::stoi(chunk.substr(0, dash));
			int end = std::stoi(chunk.substr(dash + 1));
			for (int i = beg; i <= end; i++) {
				seeds.push_back(i);
			}
		}
		else {
			// single seed
			seeds.push_back(std::stoi(chunk));
		}
	} while (pos != std::string::npos);

	return seeds;
}

LatantIn latants_from_seeds(std::string str)
{
	LatantIn latants;

	std::uniform_real_distribution<float> dist(0.0, 1.0);

	for (auto seed : parse_seeds(str)) {
		std::mt19937 engine(seed);

		float* latant = new float[MAX_LATANT];
		for (int i = 0; i < MAX_LATANT; i++) {
			latant[i] = dist(engine);
		}
		latants.emplace_back(latant);
	}

	return latants;
}

/***  Module Header  ******************************************************}}}*/
/**
* display model card
* @par DESCRIPTION
*   show i/o specification of the model.
**/
/**************************************************************************{{{*/
void
model_card(Tf2Interp& interp)
{
	/*SUBROUTINE*/
	auto print_tensor_spec = [](json& spec) {
		std::cout
		<< "  "
		<< spec["index"] << " : "
		<< spec["name"] << ", "
		<< spec["type"] << ", [";

		for (auto& n : spec["dims"]) {
			std::cout << n << ", ";
		}
		std::cout << "]," << std::endl;
	};
	/**/

	json res;
	interp.info(res);

	std::cout << "inputs:" << std::endl << "{" << std::endl;
	for (auto& item : res["inputs"]) {
		print_tensor_spec(item);
	}
	std::cout << "}" << std::endl;

	std::cout << "outputs:" << std::endl << "{" << std::endl;
	for (auto& item : res["outputs"]) {
		print_tensor_spec(item);
	}
	std::cout << "}" << std::endl;
}

/***  Module Header  ******************************************************}}}*/
/**
* save result image
* @par DESCRIPTION
*   convert the result tensor to the image and save it.
**/
/**************************************************************************{{{*/
void
save_to_image(const std::string& bin, const fs::path& outdir, const char* format, int n)
{
	char basename[32];
	sprintf(basename, format, n);
	fs::path fname = outdir / basename;

	int hw = sqrt(bin.size() / 4 / 3);
	CImg<uint8_t> img = 255*(CImg<float>((float*)bin.data(), hw, hw, 1, 3) + 1.0)/2.0;

	img.save(fname.string().c_str());
}

/***  Module Header  ******************************************************}}}*/
/**
* prit usage
* @par DESCRIPTION
*   print usage to terminal
**/
/**************************************************************************{{{*/
void
usage()
{
    std::cout
    << "generate [opts] <model> [<outdir>]\n"
    << "\toption:\n"
    << "\t  -s <seeds> : random seeds - \"f4,1,3,224,224\"\n"
	<< "\t  -d <path>  : dlatants file\n"
	<< "\t  -p         : print model card\n"
    ;
}

/***  Module Header  ******************************************************}}}*/
/**
* main
* @par DESCRIPTION
*   generate images
*
* @return exit status
**/
/**************************************************************************{{{*/
int
main(int argc, char* argv[])
{
	int opt;
	const struct option longopts[] = {
	    {"seeds",     required_argument, NULL, 's'},
		{"print",     no_argument,       NULL, 'p'},
		{0,0,0,0}
	};

	fs::path model;
	fs::path outdir;

	std::string seeds;

	bool do_inspect = false;

	for (;;) {
		opt = getopt_long(argc, argv, "s:p", longopts, NULL);
		if (opt == -1) {
			break;
		}
		else switch (opt) {
		case 's':
			seeds = optarg;
		    break;
		case 'p':
			do_inspect = true;
			break;
		case '?':
		case ':':
			std::cerr << "error: unknown options\n\n";
			usage();
			return 1;
		}
	}
	if ((argc - optind) < 1) {
		// argument error
		std::cerr << "error: expect <model>\n\n";
		usage();
		return 1;
	}

	model = fs::absolute(argv[optind]);
	if (!fs::exists(model)) {
		std::cerr << "Error: model file isn't exist: " << model << std::endl;
		exit(1);
	}

	outdir = fs::absolute(((argc - optind) == 2) ? argv[optind + 1] : "./out");
	if (!fs::exists(outdir)) {
		fs::create_directories(outdir);
	}

	// 85,265,297,849 

	LatantIn latant_in;

	if (!seeds.empty()) {
		latant_in = latants_from_seeds(seeds);
	}
	else {
		std::cerr << "Error: needs --seeds option." << std::endl;
		exit(1);
	}

	try {
		Tf2Interp interp(model.string(), "Gs/latents_in,f32,1,512", "Gs/images_out,f32,1,3,512,512");
		
		if (do_inspect) { model_card(interp); }

		for (int i = 0; i < latant_in.size(); i++) {
			float* latant = latant_in[i].get();
			//std::cout << latant[0] << "," << latant[10] << "," << latant[511] << std::endl;

			interp.set_input_tensor(0, reinterpret_cast<uint8_t*>(latant), MAX_LATANT*sizeof(float));
			interp.invoke();
			std::string bin = interp.get_output_tensor(0);

			save_to_image(bin, outdir, "result_%d.jpg", i);
		}
	}

	catch (...) {
		std::cerr << "Error: can't launch interp.";
		exit(1);
	}

    return 0;
}

/*** generate.cpp *********************************************************}}}*/
