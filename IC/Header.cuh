//blocks   x threadsPerBlock
//clusters x UEsPerCluster = numberOfUEs
#define numberOfUEs 2560
#define clusters 32
#define UEsPerCluster 80
#define ic 'P'
#define modulation 16

double getGeneratedRandom();
cuDoubleComplex * getChannel();
cuDoubleComplex getSignal(cuDoubleComplex * channel);
cuDoubleComplex getQAM4();
cuDoubleComplex getQAM16();