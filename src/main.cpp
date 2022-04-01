/* Program driver for a series of Parallel Fourier Tranformations.
 *
 * Authors:
 *	Brett Sumser
 *	Justin Spidell
 */
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <fstream>
#include <pthread.h>
#include <omp.h>
//#include <mpi.h>
#include "../include/timelord.h"
#include "../include/wavio.h"
#include "../include/imageLoader.h"


using namespace std::chrono;
using namespace std;


int processArgs(int argc, char* argv[]);
void writeToFile(string file, vector<complex<double>> output);
void writeVectorToFile(string file, vector<double> output, int N);
vector<complex<double>> DFT(vector<double> input);
vector<complex<double>> PDFT(vector<double> input);
vector<complex<double>> CT(vector<double> input);
vector<complex<double>> CTP1(vector<double> input);
vector<complex<double>> CTP2(vector<double> input);
void MPIDFT(double* input, int low, int high, double** real, double** imag);
vector<double> signalGenerator(int sampleSize);
vector<double> detectPitches(vector<complex<double>> output);
unsigned int bitReverse(unsigned int input, int log2n);
void discreteCosineTransform(ImageLoader imageloader);


int n = 1024;
double test = 0.000000001;
bool wav = false;
bool image = false;
bool num = false;
bool mpi = true;
bool err = false;
string inFile = "wavs/a.wav";
vector<omp_lock_t> locks;


int main(int argc, char* argv[]) {
    if (processArgs(argc, argv))
		return 1;

    // Init Variables
    high_resolution_clock::time_point start, stop;
    Timelord newTimeLord();
    duration<double> duration;
	vector<double> durations;
    vector<double> input;
	vector<complex<double>> dftout, pdftout, ctout, ctp1out, ctp2out;

	if (wav) {
		cout << "Reading " << inFile << ":" << endl;
		start = high_resolution_clock::now();
		input = read_wav(inFile.c_str());
		stop = high_resolution_clock::now();
		cout << "done in " << duration.count() << " seconds" << endl;
		durations.push_back(duration.count());
		n = input.size();
		writeVectorToFile("sig.txt", input, n);
	}

	if (image) {
		cout << "Looking at " << inFile << " ... ";
		//attempting to load image
		ImageLoader imageLoader(inFile.c_str());
		start = high_resolution_clock::now();
		imageLoader.grayscaler();
		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		cout << "grayscaler done in " << duration.count() << " microseconds" << endl;
		imageLoader.doubleVector();
		cout << imageLoader.imageDoubles.size() << endl;
		vector<double> imageInput = imageLoader.imageDoubles;
		cout << imageInput.size() << endl;
		start = high_resolution_clock::now();
		vector<complex<double>> output = CTP1(imageInput);
		stop = high_resolution_clock::now();
		cout << "done in " << duration.count() << " seconds" << endl;
		durations.push_back(duration.count());
		cout << "saving the transformed image" << endl;
		imageLoader.doubleVectorConvert(output);
		n = input.size();
	}

	cout << "N = " << n << endl << endl;

	if (!wav && !image) {
	    // Signal Generator
		cout << "Generating signal ... ";
	    start = high_resolution_clock::now();
	    input = signalGenerator(n);
	    stop = high_resolution_clock::now();
	    duration = stop - start;
	    cout << "done in " << duration.count() << " seconds" << endl;
		durations.push_back(duration.count());
		writeVectorToFile("sig.txt", input, n);
	}

	/*if (mpi) {
	    MPI_Init(NULL, NULL);

		// Current rank's ID
	    int world_rank;
	    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	    // Total number of ranks
	    int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);

		int n_per_rank = 0;
		vector<double> input;
		input.resize(n);
		high_resolution_clock::time_point start, stop;
    	Timelord newTimeLord();
    	duration<double> duration;

		if (world_rank == 0) {
			if (processArgs(argc, argv))
				return 1;

    		// Signal Generator
			cout << "Generating signal ... ";
	    	start = high_resolution_clock::now();
	    	input = signalGenerator(n);
	    	stop = high_resolution_clock::now();
	    	duration = stop - start;
	    	cout << "done in " << duration.count() << " seconds" << endl;
			writeVectorToFile("sig.txt", input, n);

			n_per_rank = n / world_size;
			cout << "World size = " << world_size << endl;
			cout << "n per rank = " << n_per_rank << endl;
		}

		// Broadcast this to everyone
		MPI_Bcast(&(input[0]), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&n_per_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);

		int low = world_rank * n_per_rank;
		int high = (world_rank + 1) * n_per_rank;

		cout << "Rank: " << world_rank << " from: " << low << " -> to: " << high << endl;

		double* real = (double *)malloc(sizeof(double) * n);
		double* imag = (double *)malloc(sizeof(double) * n);
		//memset(real, 0.0, sizeof(double) * n);
		//memset(real, 0.0, sizeof(double) * n);

		if (world_rank == 0) {
			cout << "Starting MPI DFT ... ";
			start = high_resolution_clock::now();
		}
		MPI_Barrier(MPI_COMM_WORLD);
		MPIDFT(&(input[0]), low, high, &real, &imag);
		MPI_Barrier(MPI_COMM_WORLD);
		if (world_rank == 0) {
			stop = high_resolution_clock::now();
			cout << "done in " << duration.count() << " seconds" << endl;
		}

		double* real_final = NULL;
		double* imag_final = NULL;

		if (world_rank == 0) {
			real_final = (double *)malloc(sizeof(double) * n);
			imag_final = (double *)malloc(sizeof(double) * n);
			//memset(real_final, 0.0, sizeof(double) * n);
			//memset(imag_final, 0.0, sizeof(double) * n);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (world_rank == 0) {
			cout << "Gathering ... ";
		}
		if (MPI_Gather(real, n_per_rank, MPI_DOUBLE, real_final, n_per_rank, MPI_DOUBLE, 0, MPI_COMM_WORLD) == MPI_SUCCESS)
			cout << "asdf" << endl;
		if (MPI_Gather(imag, n_per_rank, MPI_DOUBLE, imag_final, n_per_rank, MPI_DOUBLE, 0, MPI_COMM_WORLD) == MPI_SUCCESS)
			cout << "fdsa" << endl;
		if (world_rank == 0) {
			cout << "done" << endl;
		}

		if (world_rank == 0) {
			for (int i = 0; i < n; i++)
				cout << imag_final[i] << " ";
			cout << endl << endl;;
		}
		if (world_rank == 0) {
			for (int i = 0; i < n; i++)
				cout << real_final[i] << " ";
			cout << endl << endl;
		}

		if (world_rank == 0) {
			vector<complex<double>> res;
			res.reserve(n);
			for (int i = 0; i < n; i++) {
				complex<double> tmp;
				tmp.real(real[i]);
				tmp.imag(imag[i]);
				res.push_back(tmp);
			}
			writeToFile("mpiout.txt", res);
		}

		free(real);
		free(imag);
		if (world_rank == 0) {
			free(real_final);
			free(imag_final);
		}

		MPI_Finalize();
		return 0;
	} */
	

	// Locks
	cout << "Setting up locks ... ";
    start = high_resolution_clock::now();
	locks.reserve(n);
	for (int i = 0; i < n; i++)
		omp_init_lock(&(locks[i]));
    stop = high_resolution_clock::now();
	duration = stop - start;
	cout << "done in " << duration.count() << " seconds" << endl << endl;;
	durations.push_back(duration.count());

    // DFT
    cout << "Starting DFT ... ";
    start = high_resolution_clock::now();
    dftout = DFT(input);
    stop = high_resolution_clock::now();
    duration = stop - start;
    cout << "done in " << duration.count() << " seconds" << endl;
	durations.push_back(duration.count());
    writeToFile("dft.txt", dftout);


    // DFTP1
    cout << "Starting // DFT ... ";
    start = high_resolution_clock::now();
    pdftout = PDFT(input);
    stop = high_resolution_clock::now();
    duration = stop - start;
    cout << "done in " << duration.count() << " seconds" << endl;
	durations.push_back(duration.count());
    writeToFile("pdft.txt", pdftout);


    //CT
    cout << "Starting Cooley-Turkey ... ";
    start = high_resolution_clock::now();
    ctout = CT(input);
    stop = high_resolution_clock::now();
    duration = stop - start;
    cout << "done in " << duration.count() << " seconds" << endl;
	durations.push_back(duration.count());
	writeToFile("ct.txt", ctout);


    //CTP1
    cout << "Starting V1 // Cooley-Turkey ... ";
    start = high_resolution_clock::now();
    ctp1out = CTP1(input);
    stop = high_resolution_clock::now();
    duration = stop - start;
    cout << "done in " << duration.count() << " seconds" << endl;
	durations.push_back(duration.count());
	writeToFile("ctp1.txt", ctp1out);

	
    //CTP2
	cout << "Starting V2 // Cooley-Turkey ... ";
    start = high_resolution_clock::now();
    ctp2out = CTP2(input);
    stop = high_resolution_clock::now();
    duration = stop - start;
    cout << "done in " << duration.count() << " seconds" << endl;
	durations.push_back(duration.count());
	if (!err)
		writeToFile("ctp2.txt", ctp2out);

	for (int i = 0; i < n; i++)
		omp_destroy_lock(&(locks[i]));

	if (wav) {
		// Pitch Detection
		cout << "Detecting Pitches ... ";
		start = high_resolution_clock::now();
		vector<double> pitches = detectPitches(dftout);
		stop = high_resolution_clock::now();
		duration = stop - start;
		cout << "done in " << duration.count() << " seconds" << endl;
		durations.push_back(duration.count());
		cout << "Pitches detected:" << endl;
		for (unsigned int i = 0; i < pitches.size(); i++)
			cout << pitches[i] << endl;
		cout << endl;
	}

	writeVectorToFile("dur.txt", durations, (int)durations.size());

	return 0;
}

int processArgs(int argc, char* argv[])
{
	int opt;
	while ((opt = getopt(argc, argv, "w::i::m::n:")) != -1) {
		switch (opt) {
			case 'w':
				wav = true;
				if (optarg == NULL && optind < argc && argv[optind][0] != '-')
					optarg = argv[optind++];
				if (optarg != NULL)
					inFile = optarg;
				break;
			case 'i':
				image = true;
				if (optarg == NULL && optind < argc && argv[optind][0] != '-')
					optarg = argv[optind++];
				if (optarg != NULL)
					inFile = optarg;
				break;
			case 'n':
				num = true;	
				n = stoi(optarg);
				break;
			case 'm':
				wav = true;
				if (optarg == NULL && optind < argc && argv[optind][0] != '-')
					optarg = argv[optind++];
				if (optarg != NULL)
					n = stoi(optarg);
			case '?':
				return 1;
		}
	}
	if (wav && image) {
		cout << "I can't do both an image and a wav file... sorry" << endl;
		return 1;
	}
	return 0;
}

void writeToFile(string file, vector<complex<double>> vec)
{
    #ifdef PRINT
		cout << "Writing to file ... ";
		ofstream f;
		f.open("out/" + file, ios::trunc);
		for (int i = 0 ; i < n; i++)
			f << vec[i].real() << ", " << vec[i].imag() << endl;
		f.close();
		cout << "done" << endl << endl;
    #endif
}

void writeVectorToFile(string file, vector<double> vec, int N)
{
    #ifdef PRINT
		cout << "Writing to file ... ";
		ofstream f;
		f.open("out/" + file, ios::trunc);
		for (int i = 0 ; i < N; i++)
			f << vec[i] << endl;
		f.close();
		cout << "done" << endl << endl;
    #endif
}

void discreteCosineTransform(ImageLoader imageloader)
{
    int length = imageloader.image.size();
    // pow? org: (2/length)^(0.5)
	//int coeff = pow((2 / length), (0.5));
    for (int i = 0; i < length; i++) {
        //For each pixel in image, apply the discrete cosine transform
        //to access pixels use (int)imageloader.image[i]
        //the image data is a 1d vector of char style pixels

    }
    //for (int i = 0; i < imageloader.image.size(); i+=4)
    //{
    //    std::cout << "Pixel " << i << " is : " 
    //    << "r:" << (int)imageloader.image[i] << " " 
    //    << "g:" << (int)imageloader.image[i + 1] << " "
    //    << "b:" << (int)imageloader.image[i + 2] << " "
    //    << "a:" << (int)imageloader.image[i + 3] << " "
    //    << std::endl;
    //}
}

vector<complex<double>> DFT(vector<double> input)
{
    cout << "starting dft" << endl;
    //initialize sizes of samples
    int N = input.size();
    int K = input.size();

	int k;
	int n;


    //init variable for internal loop
    complex<double> innerSum = complex<double>(0.0,0.0);

    //init vector of std::complex doubles for results
    vector<complex<double>> output;

    //set output vector to have enough space
    output.resize(N);

	complex<double> tmp = complex<double>(0.0,0.0);

	double real = 0.0;
	double imag = 0.0;

	//sigma notation algorithm start
    for (k = 0; k < K; k++) {
        innerSum = complex<double>(0.0,0.0);
		for (n = 0; n < N; n++) {
	        real = cos((2 * M_PI * k * n) / N);
            imag = -sin((2 * M_PI * k * n) / N);

			tmp.real(real);
			tmp.imag(imag);
			tmp *= input[n];
			innerSum += tmp;
		}
		if (fabs(innerSum.real()) < test)
			innerSum.real(0.0);
		if (fabs(innerSum.imag()) < test)
			innerSum.imag(0.0);
		output.at(k) = innerSum;
    }
    cout << "ending dft" << endl;
    return output;
}

vector<complex<double>> PDFT(vector<double> input)
{
    //initiliaze sizes of samples
    int N = input.size();
    int K = input.size();

    //loop indexes
    int k;
    int n;

    //init variable for internal loop
    complex<double> innerSum = complex<double>(0.0,0.0);

    //init vector of std::complex doubles for results
    vector<complex<double>> output;

    //set output vector to have enough space
    output.resize(N);

	complex<double> tmp = complex<double>(0.0,0.0);

	double real = 0.0;
	double imag = 0.0;

	#pragma omp parallel for default(none) private(real, imag, innerSum, k, tmp, n) shared(input, output, K, N, test)
	for (k = 0; k < K; k++) {
		innerSum = complex<double>(0.0,0.0);	
		for (n = 0; n < N; n++) {
			real = cos((2 * M_PI * k * n) / N);
			imag = -sin((2 * M_PI * k * n) / N);

			tmp.real(real);
			tmp.imag(imag);
			tmp *= input[n];
			innerSum += tmp;
		}
		if (fabs(innerSum.real()) < test)
			innerSum.real(0.0);
		if (fabs(innerSum.imag()) < test)
			innerSum.imag(0.0);
		output.at(k) = innerSum;
	}

    return output;
}

void MPIDFT(double* input, int low, int high, double** real, double** imag)
{
	int k;
    int i;

    //init variable for internal loop
	double realSum = 0.0;
	double imagSum = 0.0;	

	double real_tmp = 0.0;
	double imag_tmp = 0.0;

	//#pragma omp parallel for default(none) private(real, imag, innerSum, k, tmp, n) shared(input, output, K, N, test)
	for (k = low; k < high; k++) {
		for (i = 0; i < n; i++) {
			real_tmp = cos((2 * M_PI * k * i) / n);
			imag_tmp = -sin((2 * M_PI * k * i) / n);

			real_tmp *= input[i];
			imag_tmp *= input[i];
			realSum += real_tmp;
			imagSum += imag_tmp;
		}
		if (fabs(realSum) < test)
			realSum = 0.0;
		if (fabs(imagSum) < test)
			imagSum = 0.0;
		*(*real + k) = realSum;
		*(*imag + k) = imagSum;
	}
}

vector<complex<double>> CT(vector<double> input)
{
    vector<complex<double>> output;

	int high = ceil(log(input.size()) / log(2));
	uint32_t pad = pow(2, high) - input.size();
	vector<double>::iterator it = input.begin();
	int half = floor(input.size() / 2);
	
	input.insert(it + half, pad, (double)0.0);

	unsigned int N = (unsigned int)input.size();
    output.reserve(N);

	int top = input.size();

	for (unsigned int i = 0; i < N; ++i) {
		unsigned int reversed = bitReverse(i, high);
		complex<double> tmp;
		tmp.real(input[reversed]);
		tmp.imag(0.0);
		output.push_back(tmp);
	}

	const complex<double> iota (0, 1);
	for (int s = 1; s <= high; ++s) {
		int m = 1 << s;
		int m2 = m >> 1;
		complex<double> w (1.0, 0.0);
		double real = cos(M_PI / m2);
		double imag = -sin(M_PI / m2);
		complex<double> wm (real, imag);
		if (fabs(wm.real()) < test)
			wm.real(0.0);
		if (fabs(wm.imag()) < test)
			wm.imag(0.0);
		for (int j = 0; j < m2; ++j) {
			complex<double> t, u, tmpk, tmpm;
			int k;
			for (k = j; k < top; k += m) {
				t = (w * output[k + m2]);
				u = output[k];
				tmpk = u + t;
				tmpm = u - t;
				if (fabs(tmpk.real()) < test)
					tmpk.real(0.0);
				if (fabs(tmpk.imag()) < test)
					tmpk.imag(0.0);
				if (fabs(tmpm.real()) < test)
					tmpm.real(0.0);
				if (fabs(tmpm.imag()) < test)
					tmpm.imag(0.0);
				output[k] = tmpk;
				output[k + m2] = tmpm;
			}
			w *= wm;
		}
	}
    return output;
}

vector<complex<double>> CTP1(vector<double> input)
{
    vector<complex<double>> output;

	int high = ceil(log(input.size()) / log(2));
	uint32_t pad = pow(2, high) - input.size();
	vector<double>::iterator it = input.begin();
	int half = floor(input.size() / 2);
	
	input.insert(it + half, pad, (double)0.0);

	unsigned int N = (unsigned int)input.size();
    output.reserve(N);

	int top = input.size();

	for (unsigned int i = 0; i < N; ++i) {
		unsigned int reversed = bitReverse(i, high);
		complex<double> tmp;
		tmp.real(input[reversed]);
		tmp.imag(0.0);
		output.push_back(tmp);
	}

	const complex<double> iota (0, 1);
	for (int s = 1; s <= high; ++s) {
		int m = 1 << s;
		int m2 = m >> 1;
		complex<double> w (1.0, 0.0);
		double real = cos(M_PI / m2);
		double imag = -sin(M_PI / m2);
		complex<double> wm (real, imag);
		if (fabs(wm.real()) < test)
			wm.real(0.0);
		if (fabs(wm.imag()) < test)
			wm.imag(0.0);
		for (int j = 0; j < m2; ++j) {
			complex<double> t, u, tmpk, tmpm;
			int k;
			#pragma omp parallel for default(none) shared(j, output, test, top, m, m2, w) private(t, u, k, tmpk, tmpm)
			for (k = j; k < top; k += m) {
				t = (w * output[k + m2]);
				u = output[k];
				tmpk = u + t;
				tmpm = u - t;
				if (fabs(tmpk.real()) < test)
					tmpk.real(0.0);
				if (fabs(tmpk.imag()) < test)
					tmpk.imag(0.0);
				if (fabs(tmpm.real()) < test)
					tmpm.real(0.0);
				if (fabs(tmpm.imag()) < test)
					tmpm.imag(0.0);
				#pragma omp critical
				{
					output[k] = tmpk;
					output[k + m2] = tmpm;
				}
			}
			w *= wm;
		}
	}
    return output;
}

vector<complex<double>> CTP2(vector<double> input)
{
    vector<complex<double>> output;

	int org = input.size();
	int high = log(org) / log(2);

	if (pow(2, high) != input.size()) {
		cout << "CTP2 only takes power of 2 inputs .... sorry!" << endl;
		err = true;
		return output;
	}

	unsigned int N = (unsigned int)input.size();
    output.reserve(N);

	int top = input.size();


	for (unsigned int i = 0; i < N; ++i) {
		unsigned int reversed = bitReverse(i, high);
		complex<double> tmp;
		tmp.real(input[reversed]);
		tmp.imag(0.0);
		output.push_back(tmp);
	}


	const complex<double> iota (0, 1);
	for (int s = 1; s <= high; ++s) {
		int m = 1 << s;
		int m2 = m >> 1;
		complex<double> w (1.0, 0.0);
		double real = cos(M_PI / m2);
		double imag = -sin(M_PI / m2);
		complex<double> wm (real, imag);
		if (fabs(wm.real()) < test)
			wm.real(0.0);
		if (fabs(wm.imag()) < test)
			wm.imag(0.0);
		for (int j = 0; j < m2; ++j) {
			complex<double> t, u, tmpk, tmpm;
			int k;
			#pragma omp parallel for default(none) shared(j, output, test, top, m, m2, w, locks) private(t, u, k, tmpk, tmpm)
			for (k = j; k < top; k += m) {
				t = (w * output[k + m2]);
				u = output[k];
				tmpk = u + t;
				tmpm = u - t;
				if (fabs(tmpk.real()) < test)
					tmpk.real(0.0);
				if (fabs(tmpk.imag()) < test)
					tmpk.imag(0.0);
				if (fabs(tmpm.real()) < test)
					tmpm.real(0.0);
				if (fabs(tmpm.imag()) < test)
					tmpm.imag(0.0);
				omp_set_lock(&(locks[k]));
				output[k] = tmpk;
				omp_unset_lock(&(locks[k]));
				omp_set_lock(&(locks[k + m2]));
				output[k + m2] = tmpm;
				omp_unset_lock(&(locks[k + m2]));
			}
			w *= wm;
		}
	}
    return output;
}

unsigned int bitReverse(unsigned int num, int log2n)
{
	unsigned int rev = 0;

	for (int i = 0; i < log2n; i++) {
		rev <<= 1;
		rev |= (num & 1);
		num >>= 1;
	}

	return rev;
}

vector<double> signalGenerator(int sampleSize)
{
	double test = 0.00000000001;

    vector<double> output; //output vector to store the test signal
    output.resize(sampleSize);  //allocate proper size for vector

	double tmp;
	int i;

    #pragma omp parallel for default(none) shared(test, output, sampleSize) private(i, tmp)
    for (i = 0; i < sampleSize; i++) {
		tmp = sin(880.0 * M_PI * (double)((double)i / (double)sampleSize));
		if (fabs(tmp) < test) 
			tmp = 0.0;
        output.at(i) = tmp;
    }
    
    return output;
}

vector<double> detectPitches(vector<complex<double>> output)
{
	vector<double> pitches;
	for (int i = 0; i < (int)output.size() / 2; i++) {
		if (fabs(output[i].imag()) > 700.0)
			pitches.push_back(i);
	}
	return pitches;
}

