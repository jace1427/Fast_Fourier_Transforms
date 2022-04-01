#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>
#include <unistd.h>

using namespace std;


typedef struct  WAV_HEADER
{
    /* RIFF Chunk Descriptor */
    uint8_t         RIFF[4];        // RIFF Header Magic header
    uint32_t        ChunkSize;      // RIFF Chunk Size
    uint8_t         WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t         fmt[4];         // FMT header
    uint32_t        Subchunk1Size;  // Size of the fmt chunk
    uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Sterio
    uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t        bytesPerSec;    // bytes per second
    uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t         Subchunk2ID[4]; // "data"  string
    uint32_t        Subchunk2Size;  // Sampled data length
} wav_hdr;


int getFileSize(FILE* inFile);


vector<double> read_wav(const char* inFile)
{
    wav_hdr wavHeader;
    int headerSize = sizeof(wav_hdr), filelength = 0;

    FILE* wavFile = fopen(inFile, "r");

	if (wavFile == nullptr)
		cout << "Unable to open wave file: " << inFile << endl;

	vector<double> data;

    //Read the header
    size_t bytesRead = fread(&wavHeader, 1, headerSize, wavFile);
    cout << "Header Read " << bytesRead << " bytes." << endl;
    if (bytesRead > 0) {
        //Read the data
        uint16_t bytesPerSample = wavHeader.bitsPerSample / 8;      //Number     of bytes per sample
        uint64_t numSamples = wavHeader.Subchunk2Size / bytesPerSample; //How many samples are in the wav file?
        short buffer;
        
		data.reserve((int)numSamples);
		while ((bytesRead = fread((unsigned char*)&buffer, sizeof(buffer), 1, wavFile)) > 0) {
			data.push_back(buffer / 32768.0);
		}

        filelength = getFileSize(wavFile);

        cout << "File is                    : " << filelength << " bytes." << endl;
        cout << "RIFF header                : " << wavHeader.RIFF[0] << wavHeader.RIFF[1] << wavHeader.RIFF[2] << wavHeader.RIFF[3] << endl;
        cout << "WAVE header                : " << wavHeader.WAVE[0] << wavHeader.WAVE[1] << wavHeader.WAVE[2] << wavHeader.WAVE[3] << endl;
        cout << "FMT                        : " << wavHeader.fmt[0] << wavHeader.fmt[1] << wavHeader.fmt[2] << wavHeader.fmt[3] << endl;
        cout << "Data size                  : " << wavHeader.ChunkSize << endl;

        // Display the sampling Rate from the header
        cout << "Samples per second         : " << wavHeader.SamplesPerSec << endl;
        cout << "Bits per sample            : " << wavHeader.bitsPerSample << endl;
        cout << "Number of channels         : " << wavHeader.NumOfChan << endl;
        cout << "Number of bytes per second : " << wavHeader.bytesPerSec << endl;
        cout << "Data length                : " << wavHeader.Subchunk2Size << endl;
        cout << "Audio Format               : " << wavHeader.AudioFormat << endl;
        // Audio format 1=PCM,6=mulaw,7=alaw, 257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM

        cout << "Block align                : " << wavHeader.blockAlign << endl;
        cout << "Data string                : " << wavHeader.Subchunk2ID[0] << wavHeader.Subchunk2ID[1] << wavHeader.Subchunk2ID[2] << wavHeader.Subchunk2ID[3] << endl;
    }
    fclose(wavFile);

	return data;
}

// find the file size
int getFileSize(FILE* inFile)
{
    int fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}
