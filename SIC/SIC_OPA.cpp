#include "SIC_header.cuh"
class SIC_OPA
{
  public:
    FILE *fp1, *fp2, *fp3;
    float powerCoefficientMatrix[numberOfUEs];
  float * getOPA()
  {
    char fileLocation[256] = "/home/talgat/github/SIC/OPA/";
  	char a[8] = "10";
  	char fileLocationEnd[8] = ".txt";
  	strcat(a, fileLocationEnd);
  	strcat(fileLocation, a);
  	//printf("File Location %s \n", fileLocation);
  	char line[255];
  	int lengthOfLine = 0;
  	lengthOfLine = lengthOfLineFunction(fp1, fileLocation, line, lengthOfLine);
  	//printf("\nlengthOfLine %d \n", lengthOfLine);
  	//array for optimum power allocation coefficients    
  	//iterate through each value in a line
  	fp2 = fopen(fileLocation, "r");
  	fgets(line, lengthOfLine, (FILE*)fp2);
  	//printf("\nline: %s\n", line);
  	char *p = strtok(line, " ");
  	for (int m = 0; m < numberOfUEs; m++)
  	{
  		powerCoefficientMatrix[m] = (float)atof(p);
  		p = strtok(NULL, " ");
  	}
  	fclose(fp2);
    return powerCoefficientMatrix;
  }

  int lengthOfLineFunction(FILE *fp, char *fileLocation, char *line, int lengthOfLine)
  {
  	fp = fopen(fileLocation, "r");
  	int numberOfUsers = 1;
  	for (int i = 0; i < 1; i++)
    {
  		fgets(line, 511, (FILE*)fp);
  		lengthOfLine = strlen(line);
  		for (int j = 0; line[j] != '\0'; j++)
      {
  			if (line[j] == ' ') numberOfUsers++;
  		}
  	}
  	//printf("1. There are %d chars in a line \n", lengthOfLine);
  	//printf("2. There are %d UEs in a BS", numberOfUsers);
  	lengthOfLine = lengthOfLine + 3;
  	fclose(fp);
  	return lengthOfLine;
  }
};
