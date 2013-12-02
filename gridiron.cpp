#include <cv.h>
#include <highgui.h>
#include <stdint.h>
#include "stack.h"
#include <cv.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
extern "C" {
#include "colorspace.h"
}
#define BLUE 0
#define GREEN 1
#define RED 2
#define TOLERANCE 20
#define TOLERANCE_H .5
#define GRASS_HUE_TOLERANCE 10
#define GRASS_LIGHT_MIN 5
#define GRASS_LIGHT_MAX 20
#define W 400 //See MyFilledCircle
#define PI 3.141592653

/* TO-DO
   -17.) i.) Can limit angles to 0-180, get rid of -180 - 180 to get rid of dual line detection, maybe I can use this?
		ii.) Can run hough lines on a range of theta values, only consider angles within angleCount range!
   -16.) 
	  ii.)Need to indentify both teams, and put LOS between both of them
		Correctly implemented derivative convolution matrix to indentify the MOST COMMON slope of lines
			--how do I use this? I need to identify the sideline
			--I need to use the other team identify the LOS, what line best separate the two teams into 2 units? (use different color for each)
			--isLeft = !isRight
			--elminate MOST COMMON slope of lines, maybe I will be left with just the sidelines
   -15.) Scanline-fill until a grass-pixel is reached, in this way we can pick out the actual players/player blobs
   -14.) Now that the LOS is good, and theta gives good distribution, we need to separate the LOS from the SIDELINE
			ii.) Use the SIDELINE to eliminate false positives outside of it.
			iii.)What about the bottom sideline?
			iv.) a.) Find the median of the x-axis
				 b.) Whichever x-intercept is closest to the median is the LOS, the other is the SIDELINE!
                 c.) Get line intersections of all lines, find distances of all intersections within the cluster. Which every avgDistance is lower, those are the yardlines!  
   -13.)Line of scrimmage must be the line which has the most blobs closest to it (i.e. the D-Linemen)
   -12.) Issue with the players on the sideline! False Positives!
   -11.) Which cluster is the yard lines? What if the yard-lines have a positive slope? What if yard-lines aren't evenly spaces?
   -10.) Now I can order lines by SLOPE to find median, or order lines by X-INTERCEPT so I can divide into 5 yard increments
   -9.) Use MEDIAN Line slope or average the middle two!
   -8!.) Can hierarchically cluster area of players, or rectangular-surrounding box, divide the bigger boxes into two boxes, place dot in middle
   -6.) Vector Quantization => Reduce number of colors in palette to K, by clustering colors into K colors
   -5.) If we can find where yard-lines meet green, or end, then we can adjust field to top-down view(rotate around LOS)
   -4.) It seems that delta_h is much more effective than delta_e when identifying jerseys
   -3.) Write function removeBrightness, which sets brightness to a flat value, thus showing only hue/saturation?
   -2.) Look up pixels by Delta_H instead of Delta_E!!!!
  -1.) Right now, lines are considered parallel if they have similiar acute
		angles at their point of intersection, what if two lines are very far
		apart, but intersect with very small acute angle?  Need to calculate
		adjusted distance between two parallel lines.
		-- To do so, using both midpoints, find two perpendicular lines to opposite
		line through midpoint. Add the two perp. line distances, and divide by two.
   0.) Re-structure code and put each step into functions
   1.) Fix the issue of L*a*b getting represented as INTS, losing
   the data and specificty of FLOATS
   2.) How do I detect shapes? I find the most similiar shapes, need
   to use some sort of a clustering algorithm.
   3.) Turn the Tolerance way up, and find the largest block of pixels, find center of it,
   use GRIDLINES to draw LOS
   4.) Find the GRIDLINES!
*/


using namespace cv;


typedef struct{
	Point p1, p2;
	float x, y;
	float slope;
	int ID;
}pointSlopeLine;

typedef struct{
	float x, y;
}floatPoint;

typedef struct floatSlopeNode{
	floatSlopeNode* next;
    short ID;
	int size;
	float avgSlope;
	float avgTheta;
	pointSlopeLine* cluster;
};

typedef struct fPointList{
	floatPoint p1;
	floatPoint p2;
	fPointList* next;
};

typedef struct{
    Point p1, p2;
}myLine;

typedef struct{
	int x, y, pixels;
	float delta_e;
}blob;

typedef struct{
	unsigned short L;
	unsigned short C;
	unsigned short H;
}pixel_LCH_16;

typedef struct{
	unsigned short b;
	unsigned short g;
	unsigned short r;
}pixel_RGB_16;

typedef struct{
	unsigned char magnitude_int;   //1 byte
	unsigned char magnitude_mant;//1 byte
	unsigned char angle;   //1 byte
	short hue; //2 bytes
}pixel_gradient;

typedef struct{
	uint8_t b;
	uint8_t g;
	uint8_t r;
}pixel;

typedef struct{
	uint8_t magnitude;
	uint8_t angle;
}gradient_pixel;

typedef struct{
	float L;
	float a;
	float b;
}lab_pixel;

typedef struct
{
	floatSlopeNode* yardlines;
    floatSlopeNode* sidelines;
}linesPair;

// Convolution stuff

void mapGradients(Mat src);
double applyConvolutionMat(int* convolutionMat, pixel* pxl);
// ********************************************************** //


linesPair* findYardSideLines(floatSlopeNode* lines);

int compar (const void* p1, const void* p2);

float Dist(float p1, float p2);
floatSlopeNode* hierarchicalCluster(floatSlopeNode* head, int numLines, int k);
float avgYardlineSlope(fPointList* lines, int numLines, Mat image, blob* players);
float getSlopeF(floatPoint p1, floatPoint f2);
float getYintF(floatPoint p1, floatPoint p2);
float getYintFSlope(floatPoint p1, float m);
float R2XY(float theta);

float XY2R(floatPoint p1, floatPoint p2);


void findLines(Mat image, blob* players);
void printLines(fPointList* lines, int lineCount, Mat image, blob* players);

bool isAlmostParallel(myLine l1, myLine l2);
float getSlope(Point pt1, Point pt2);
float getYint(Point pt1, Point pt2);
floatPoint getXint(float slope, float b);
floatPoint getIntersection(Point pt1, Point pt2, Point pt3, Point pt4);
float distParLines(float b1, float b2, float m);

bool isLine(lab_pixel* grass, lab_pixel* pxl);
bool isWhite(pixel* rgb);

void Short2RGB(short colorID, pixel* grassPix);
short RGB2Short(pixel* rgb_pix);

void RGB2Lab(pixel* rgb, lab_pixel* lab);
void Lab2RGB(lab_pixel* lab, pixel* rgb);

float delta_e(lab_pixel* ref, lab_pixel* pxl);
float delta_h(lab_pixel* ref, lab_pixel* pix);
int getHue(lab_pixel* pxl);

blob* scanline_fill(pixel* seed, lab_pixel* refPxl, Mat img, int x, int y, Stack* stack);

void insertionSort(blob* b, blob* blobs);

int a_ref, b_ref, L_ref;


void MyFilledCircle( Mat img, Point center,int r, int g, int b );
// GLOBALS
FILE* fp = NULL;
int imgWidth, imgHeight;



int main(int argc, char** argv)
{
	//char* filename = argv[1];
	//char* filename = "C:/Users/Administrator/Documents/Visual Studio 2010/Projects/Gridiron_Analyst/debug/wesleyan.png";
	char* clusterFile = "C:/Users/Administrator/Documents/Visual Studio 2010/Projects/Gridiron_Analyst/Gridiron_Analyst/debug/cluster.csv";
	fp = fopen(clusterFile, "w");

	char* filename = "C:/Users/Administrator/Documents/Visual Studio 2010/Projects/Gridiron_Analyst/Gridiron_Analyst/debug/wesleyan.png";
	Mat image, img;
	image = imread(filename, 1);
	imgWidth = image.cols;
	imgHeight = image.rows;
	/* Mode Color Array */
	//int* colors = (int*)calloc(4096, sizeof(*colors));
	int* hues = (int*) calloc(360, sizeof(*hues));

	Stack* stk = new Stack(10000);
	
	blob* topHundred = (blob*)calloc(100, sizeof(*topHundred));
	for(int k=0;k<100;k++)
	{
		topHundred[k].delta_e = TOLERANCE + 1; //initialize to something one greater than TOLERANCE
	}
    /* grass pixel */
	pixel* grassPix = (pixel*) malloc(sizeof(*grassPix));
	
	/* jersey pixel */
	pixel rgbPix;

	/*/TUFTS JERSEY
	/rgbPix.r = 150;
	rgbPix.g = 149;
	rgbPix.b = 191;
	*/
	
	//Middlebury jerseys
	rgbPix.r = 44;
	rgbPix.g = 41;
	rgbPix.b = 49;

	// ATLANTA FALCONS JERSEY
	/*
	rgbPix.r = 100;
	rgbPix.g = 38;
	rgbPix.b = 51;
	*/

	// Williams Jersey 
	//rgbPix.r = 64;
	//rgbPix.g = 67;
	//rgbPix.b = 77;
	

	//mapGradients(image);
	
	lab_pixel refLabPxl;

	Rgb2Lab(&refLabPxl.L, &refLabPxl.a, &refLabPxl.b, rgbPix.r/255.0, rgbPix.g/255.0, rgbPix.b/255.0);

	pixel* pixPtr = (pixel*)image.data;

	int channels = image.channels();
	
	
	/*
	for(int i=0; i < image.rows; i++)
	{ //row-major
		for(int j=0; j<image.cols; j++)
		{ //horizontal traversal
		    pixel* rgb_pxl = pixPtr + (image.cols*i) + j;
	        
	//		colors[RGB2Short(rgb_pxl)]++;
			
			lab_pixel lab_pxl;
		    lab_pixel lab_grass;
			//RGB2Lab(rgb_pxl, &lab_pxl);
			
			num R, G, B;
			R = rgb_pxl->r/255.0;
			G = rgb_pxl->g/255.0;
			B = rgb_pxl->b/255.0;
			num D[3];
			Rgb2Lch(&D[0], &D[1], &D[2], R, G, B);
			int H = D[2];
			//fprintf(stderr, "L: %f a: %f b: %f\n", D[0], D[1], D[2]);
		    hues[H]++;
			
			Lch2Rgb(&R, &G, &B, D[0], D[1], D[2]);
			
			rgb_pxl->r = R*255;
			rgb_pxl->g = G*255;
			rgb_pxl->b = B*255;
			
		
			//float L, a, b;

			Rgb2Lab(&lab_pxl.L, &lab_pxl.a, &lab_pxl.b, R, G, B);
						
			if(delta_e(&lab_pxl, &refLabPxl) < TOLERANCE)
            { 
				fprintf(stderr, "SCANLINE FILL\n");
				insertionSort(scanline_fill(rgb_pxl, &refLabPxl, image, j, i, stk), topHundred);
			}
		}
    }*/

	
   
	/* Find MODE color 
	int max = -1;
	int hID = 0;
	for(short j=0; j<360; j++)
	{
		if(hues[j] > max)
		{
			max = hues[j];
			hID = j;
		}
	}
	*/
    mapGradients(image);
	
//	fprintf(stderr, "Hue: %d\n", hID);
	
	//Short2RGB(cID, grassPix);
	//grassPix.r = 96;
	//grassPix.g = 128;
	//grassPix.b = 64;
	//fprintf(stderr, "FIELD COLOR\nR: %d\nG: %d\nB: %d\n\n", (grassPix->r), (grassPix->g), (grassPix->b));
/********************/

	// GRASS LINES
	
	// find all lines, or places where green remains, but brightness increases 
	/*for(int i=0; i < image.rows; i++)
	{ //row-major
		for(int j=0; j<image.cols; j++)
		{ //horizontal traversal
		    pixel* rgb_pxl = pixPtr + (image.cols*i) + j;
			lab_pixel lab_pxl;
		    lab_pixel lab_grass;
			RGB2Lab(rgb_pxl, &lab_pxl);
			RGB2Lab(grassPix, &lab_grass);
			
			num L, C, H;

			Rgb2Lch(&L, &C, &H, rgb_pxl->r, rgb_pxl->g, rgb_pxl->b);
			
			
			//if(isLine(&lab_grass, &lab_pxl))
			
			if(abs(hID - H) > 1)
			{
			
				rgb_pxl->r=255;
				rgb_pxl->g=255;
				rgb_pxl->b=255;
				
			}else
			{
                rgb_pxl->r=0;
				rgb_pxl->g=0;
				rgb_pxl->b=0;
			}
		}
	}
	imshow("Black and White", image);
	waitKey();
	exit(1);
	*/

	//findLines(image, topHundred);
	/*
	for(int k=0; k<11; k++)
	{
		Point p = Point(topHundred[k].x, topHundred[k].y);
		//fprintf(stderr, "DONE x: %d, y: %d\n", topHundred[k].x, topHundred[k].y);
		MyFilledCircle(image, p, 0, 0, 255);
	}
	for(int l=11; l<100; l++)
	{
		Point p = Point(topHundred[l].x, topHundred[l].y);
		MyFilledCircle(image, p, 255, 0, 0);
	}*/

	
	imshow("Color Distance", image);
	waitKey();
}

float getY(float m, int yInt, int x)
{
	return m*x + yInt;

}

float solveForY(float x1, float x, float y, float m)
{
	return ((-m * (x - x1)) + y);
}

float R2XY(float theta)
{ 
	return tan(theta);
}

float XY2R(floatPoint p1, floatPoint p2)
{
	float theta = atan2(p2.y-p1.y, p2.x-p1.x);
	return theta;
}


bool blobIsLeft(blob player, float m, float x, float y)
{
	if(solveForY(player.x, x, y, m) > player.y && m > 0)
	{
		return false;
	}
	if(solveForY(player.x, x, y, m) > player.y && m < 0)
	{
		return true;
	}
	if(solveForY(player.x, x, y, m) < player.y && m < 0)
	{
		return false;
	}else return true;
}

float avgYardlineSlope(fPointList* lines, int numLines, Mat image, blob* players)
{
	floatSlopeNode* head = (floatSlopeNode*) malloc(sizeof(*head));
	floatSlopeNode* cur = head;
	
	for(int j=0;j<numLines;j++)
	{
		cur->cluster = (pointSlopeLine*) calloc(numLines, sizeof(*cur->cluster));
		cur->ID = j;
		cur->size = 1;
		Point p1, p2;
		p1.x = lines->p1.x; p1.y=lines->p1.y;
		p2.x = lines->p2.x; p2.y=lines->p2.y;
		//line(image, p1, p2, Scalar(100,100,100), 9);
		cur->avgSlope = getSlopeF(lines->p1, lines->p2);
		cur->avgTheta = XY2R(lines->p1, lines->p2);
		
		cur->cluster[0].p1.x = p1.x;
		cur->cluster[0].p1.y = p1.y;
		cur->cluster[0].p2.x = p2.x;
		cur->cluster[0].p2.y = p2.y;

		cur->cluster[0].slope = getSlopeF(lines->p1, lines->p2);
		cur->cluster[0].x = getXint(cur->cluster[0].slope, getYintF(lines->p1, lines->p2)).x;
		cur->cluster[0].y = 0; 
		//fprintf(stderr, "Slope: %f\n", cur->cluster[0].slope);
		cur->next = (floatSlopeNode*) malloc(sizeof(*cur->next));
		cur=cur->next;
		lines = lines->next;
	}
	floatSlopeNode* pointSlopeYardlines = hierarchicalCluster(head, numLines, 2);
	cur = pointSlopeYardlines;

	for(int i=0;i<2;i++)
	{
		float avgM=0;
		float totM=0;

		/* Get AVG Slope */
		for(int m=0; m<cur->size; m++)
		{
			totM += cur->cluster[m].slope;
		}
		avgM = R2XY(cur->avgTheta);
		/*Get AVG Slope */


		qsort (cur->cluster, cur->size, sizeof(*cur->cluster), compar);
		float totDist = 0;
		
		for(int h=0;h<cur->size-1;h++)
		{
			floatPoint fp1, fp2;
			fp1.x = cur->cluster[h].x;
			fp1.y = cur->cluster[h].y;
			fp2.x = cur->cluster[h+1].x;
			fp2.y = cur->cluster[h+1].y;
			

			line(image, cur->cluster[h].p1, cur->cluster[h].p2, Scalar(255, i*200, 100), 3);
			totDist += distParLines(getYintFSlope(fp1, avgM), getYintFSlope(fp2, avgM), avgM);
		}
		//perhaps cluster these distances into two groups
		float avgDist = totDist/cur->size-1;
		float distance = 0;
		
		int maxCount = 0;
		pointSlopeLine* max = (pointSlopeLine*) malloc(sizeof(*max));

		for(int j=0;j<cur->size-1;j++)
		{
			floatPoint fp1, fp2;
			fp1.x = cur->cluster[j].x;
			fp1.y = cur->cluster[j].y;
			fp2.x = cur->cluster[j+1].x;
			fp2.y = cur->cluster[j+1].y;
			distance += distParLines(getYintFSlope(fp1, avgM), getYintFSlope(fp2, avgM), avgM);
			float d = abs(fp2.x - fp1.x);
						
			if(distance > avgDist)
			{
				fprintf(stderr, "Distance: %f\n", distance);
				for(int y=0;y<5;y++)
				{
					floatPoint newPt1, newPt2;
					newPt1.x = fp1.x + (d/5 * y);
					newPt1.y = 0;
					newPt2.y = getYintFSlope(newPt1, avgM);
					newPt2.x = 0;
					
					// Iterate over players, find how many are to the right of line
					int playerCount = 0;
					for(int k=0;k<22;k++)
					{
						if(players[k].delta_e < TOLERANCE)
						{
							//consider the pixel for analysis of position with respect to line
							if(blobIsLeft(players[k], avgM, newPt1.x, newPt1.y))
							{
								playerCount++;
							}
						}
					}
					if(playerCount > maxCount)
					{//new line of scrimmage is found
						fprintf(stderr, "++++\nplayerCount : %d\n+++\n", playerCount);
						maxCount = playerCount;
						max->slope = avgM;
						max->x = newPt1.x + d/5; //grab next yardline, linemen are about 2 yards apart!
						max->y = newPt1.y;
						
					}

					Point p1, p2;
					p1.x = newPt1.x;
					p1.y = newPt1.y;
					p2.x = newPt2.x;
					p2.y = newPt2.y;
				   	line(image,p1,p2,Scalar(100,100,100));
			       // MyFilledCircle(image, p1, 124, 48, 200);
				}
				distance = 0;
			}
		}
		//PRINT LOS!!!
		Point p1, p2;
		p1.x = max->x;
		p1.y = max->y;
		MyFilledCircle(image, p1, 100, 200, 150);
		p2.y = p1.y - (avgM*p1.x);
		p2.x = 0;
		line(image, p1, p2, Scalar(56, 189, 200), 6);
		cur = cur->next;
	}

	return 1.0;
}

float Dist(float p1, float p2)
{
	return sqrt((p1 - p2)*(p1 - p2));
}

//yardlines passed by reference to be filled with sorted yardlines
floatSlopeNode* hierarchicalCluster(floatSlopeNode* head, int numLines, int k)
{    //numLines is numClusters
	float min;
	int id1, id2;
	int size1;
	float avgSlope;
	float avgTheta;
	floatSlopeNode* top = head;

    while(numLines >= k)
	{

		//test
		floatSlopeNode* YO;
		YO = top;
		fprintf(stderr, "\n=================\nnumLines : %d\n", numLines);
		for(int h=0;h<numLines;h++)
		{
	    	fprintf(stderr, "Line: %d\tSlope: %f Size: %d\n", h, YO->avgTheta, YO->size);
			YO=YO->next;
		}
		if(k == numLines){break;}
		//test
		
		min = 1000;//need to reinitialize

		head = top;
		for(int i=0;i<(numLines-1);i++)
		{
			floatSlopeNode* cur = head->next;
			for(int j=i+1;j<numLines;j++)
			{
				if(Dist(head->avgTheta, cur->avgTheta) < min)
				{   //we want to merge the two clusters
					id1 = head->ID;
					id2 = cur->ID;
					size1 = head->size;
					avgSlope = (head->avgSlope*head->size + cur->avgSlope*cur->size) / (head->size + cur->size);
					avgTheta = (head->avgTheta*head->size + cur->avgTheta*cur->size) / (head->size + cur->size);
					min = Dist(head->avgTheta, cur->avgTheta);
					fprintf(stderr, "HeadSlope: %f\tCurSlope: %f\nDist: %f\n", head->avgTheta, cur->avgTheta, min);
				}
				cur = cur->next;
			}
			head = head->next;
		}
		
		//Replace two nodes with one node of avgeraged slopes
	    if(top->ID == id1)
		{
			fprintf(stderr, "Slope1: %f\n", top->avgTheta);
			floatSlopeNode* temp = top;
			top = top->next;
			
			numLines--;
			floatSlopeNode* cur = top;
			for(int k=0;k<numLines;k++)
			{
				if(cur->ID == id2)
				{
					fprintf(stderr, "Slope2: %f\n", cur->avgTheta);
					cur->avgSlope = avgSlope;
					cur->avgTheta = avgTheta;
					for(int z=0;z<temp->size;z++)
					{   //combine the two clusters into one cluster
						cur->cluster[cur->size + z].p1.x = temp->cluster[z].p1.x;
						cur->cluster[cur->size + z].p1.y = temp->cluster[z].p1.y;
						cur->cluster[cur->size + z].p2.x = temp->cluster[z].p2.x;
						cur->cluster[cur->size + z].p2.y = temp->cluster[z].p2.y;
						//Retain original line information!
						cur->cluster[cur->size + z].x = temp->cluster[z].x;
						cur->cluster[cur->size + z].y = temp->cluster[z].y;
						cur->cluster[cur->size + z].slope = temp->cluster[z].slope;
					}
					    cur->size += temp->size;
						free(temp->cluster);
						free(temp);
				}
				cur = cur->next;
			}
		}else
		{
			floatSlopeNode* cur = top->next;
			floatSlopeNode* prev = top;
			floatSlopeNode* temp;
		    for(int k=0;k<numLines-1;k++)
		    {
		        if(cur->ID == id1)
				{
					fprintf(stderr, "Slope1: %f\n", cur->avgTheta);
					temp = cur;
					prev->next = cur->next; //re-link the list
					cur = prev->next;
				} else
			    if(cur->ID == id2)
				{
					fprintf(stderr, "Slope2: %f\n", cur->avgTheta);
					cur->avgSlope = avgSlope;
					cur->avgTheta = avgTheta;
					for(int z=0;z<temp->size;z++)
					{   //combine the two clusters into one cluster
						cur->cluster[cur->size + z].p1.x = temp->cluster[z].p1.x;
						cur->cluster[cur->size + z].p1.y = temp->cluster[z].p1.y;
						cur->cluster[cur->size + z].p2.x = temp->cluster[z].p2.x;
						cur->cluster[cur->size + z].p2.y = temp->cluster[z].p2.y;
						//Retain original line information!
						cur->cluster[cur->size + z].x = temp->cluster[z].x;
						cur->cluster[cur->size + z].y = temp->cluster[z].y;
						cur->cluster[cur->size + z].slope = temp->cluster[z].slope;
					}
					    cur->size += temp->size;
						free(temp->cluster);
						free(temp);
					break;
				}else
				{
					prev = cur;
					cur = cur->next;
				}
			}
			numLines--;
		}
			
	}
     return top;
}
void printLines(fPointList* lines, int lineCount, Mat image, blob* players) //lines is a linked list
{
	
	/* Print lines */
	fPointList* current = lines;
	avgYardlineSlope(lines, lineCount, image, players);

	pointSlopeLine* pointSlopeLinesArr = (pointSlopeLine*) calloc(lineCount, sizeof(*pointSlopeLinesArr));
	
	for(int i=0;i<lineCount;i++)
	{
		/*
		floatPoint xInt = getXint(getSlopeF(current->p1, current->p2), getYintF(current->p1, current->p2));
	    pointSlopeLinesArr[i].x = xInt.x;
		pointSlopeLinesArr[i].y = xInt.y;
		pointSlopeLinesArr[i].slope = getSlopeF(current->p1, current->p2);
		Point center;
		center.x = (int) xInt.x;
		center.y = (int) xInt.y;
        MyFilledCircle( image, center, 123, 234, 234 );

		fprintf(stderr, "XXXX: %f\nYYYY:%f\nSLOPE: %f\n\n", pointSlopeLinesArr[i].x, pointSlopeLinesArr[i].y, pointSlopeLinesArr[i].slope);
		//line(image, current->p1, current->p2, Scalar(199,255,0), 3, CV_AA);
		//fprintf(stderr, "X1: %d Y1: %d\nX2: %d Y2: %d\n\n", current->p1.x, current->p1.y, current->p2.x, current->p2.y);
		*/
		current = current->next;
	}
}

floatPoint getXint(float slope, float b)
{
	floatPoint toReturn;
	toReturn.y = 0;
	toReturn.x = -b/slope;
	return toReturn;
}

void findLines(Mat image, blob* players)
{
	vector<Vec2f> lines;
	Mat img;
    cvtColor(image,img,CV_RGB2GRAY);
	// What if we get less specific with range of angles accepted?
	HoughLines(img, lines, 1, CV_PI/180, 400, 0, 0 );
	fprintf(stderr, "Lines: %d\n", lines.size());
	myLine* grid = (myLine*) calloc(lines.size(), sizeof(myLine)); //Contains all of the grid-lines as two end points
	for( size_t i = 0; i < lines.size(); i++ )
   {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
		grid[i].p1 = pt1;
		grid[i].p2 = pt2;
        //line(image, pt1, pt2, Scalar(199,255,0), 3, CV_AA);
    }

	int lineCount = 0;
	fPointList* head = (fPointList*) malloc(sizeof(*head));
	fPointList* current = head;
	
	// Eliminate redundant parallel lines 
	for(int i=0; i<lines.size()-1; i++)
	{
		for(int p=i+1; p<lines.size(); p++)
		{
			floatPoint fp1;
			fp1 = getIntersection(grid[i].p1, grid[i].p2, grid[p].p1, grid[p].p2);
			fprintf(fp, "%f,%f\n", fp1.x, fp1.y);
		    fprintf(stderr, "%f,%f\n", fp1.x, fp1.y);
		}

		//fprintf(fp, "%d,%f\n", i, getSlope(grid[i].p1, grid[i].p2));


		bool isUnique = true;
		for(int j=i+1; j<lines.size(); j++)
		{
			if(isAlmostParallel(grid[i], grid[j]))
			{
				isUnique = false;
				fprintf(stderr, "FOUND ONE THATS PARALELL: %d - %d\n",i,j);
				break;
			}else
			{
				fprintf(stderr, "NOT PARALLEL\n");
			}
		}
		if(isUnique == true)
		{
			lineCount++;
			current->p1.x = grid[i].p1.x;
			current->p1.y = grid[i].p1.y;
			current->p2.x = grid[i].p2.x;
			current->p2.y = grid[i].p2.y;
			current->next = (fPointList*) malloc(sizeof(*current->next));
			current = current->next;
		}
	}
	printLines(head, lineCount, image, players);
}

bool isAlmostParallel(myLine l1, myLine l2)
{

	// NEED MORE PRECISON IN "POINT" DATA TYPE
	if(getSlope(l1.p1, l1.p2) == getSlope(l2.p1, l2.p2)){return true;} //truly parallel

	floatPoint inter = getIntersection(l1.p1, l1.p2, l2.p1, l2.p2);
	fprintf(stderr, "X: %f\nY: %f\n\n", inter.x, inter.y);
	float slope1 = (l1.p1.y - inter.y) / (l1.p1.x - inter.x);
	float slope2 = (l2.p1.y - inter.y) / (l2.p1.x - inter.x);

	/* This sucks, need a better statistical measure of how far apart two ~perp. lines are */
	floatPoint midpoint1, midpoint2;
	midpoint1.y = (l1.p1.y + l1.p2.y) / 2;
	midpoint1.x = (l1.p1.x + l1.p2.x) / 2;
	midpoint2.y = (l2.p1.y + l2.p2.y) / 2;
	midpoint2.x = (l2.p1.x + l2.p2.x) / 2;
	double distance = sqrt( (midpoint1.y - midpoint2.y)*(midpoint1.y - midpoint2.y) + (midpoint1.x - midpoint2.x)*(midpoint1.x - midpoint2.x) );

	float theta1 = atan(slope1) * (180 / PI);
	float theta2 = atan(slope2) * (180 / PI);
	fprintf(stderr, "Theta1: %f\nTheta2: %f\nDistance: %f\n\n", theta1, theta2, distance);

	float diffTheta = abs(theta1 - theta2);
	float diffThetaComp = abs(180 - diffTheta);
	if(diffTheta > diffThetaComp)
	{
	    fprintf(stderr, "Acute angle: %f\n", diffThetaComp);
		if(diffThetaComp < 3 && distance < 10)
		{
			return true;
		}else return false;
	}else
	{
	    fprintf(stderr, "Acute angle: %f\n", diffTheta);
		if(diffTheta < 3 && distance < 10)		                 
		{
			return true;
		}else return false;
	}//what if they are equal????
}

linesPair* findYardSideLines(floatSlopeNode* lines)
{
	linesPair* yslines = (linesPair*) malloc(sizeof(*yslines));
	float avgXint1;
	float avgXint2;

	for(int i=0;i<lines->size;i++)
	{
		avgXint1 += lines->cluster[i].x;
	}
	avgXint1 /= lines->size;

	for(int j=0; j<lines->size; j++)
	{
		avgXint2 += lines->next->cluster[j].x;
	}
	avgXint2 /= lines->size;

	if(abs(imgWidth/2 - avgXint1) < abs(imgWidth/2 - avgXint2))
	{   //the first set of lines intersect the x-axis somewhere near the middle of
		yslines->yardlines = lines;
		yslines->sidelines = lines->next;
	}else
	{
		yslines->sidelines = lines;
		yslines->yardlines = lines->next;
	}
	return yslines;
	
}

float getSlopeF(floatPoint p1, floatPoint p2)
{
	return (p1.y-p2.y)/(p1.x-p2.x);
}
float getSlope(Point pt1, Point pt2)
{
	return (float) ((float)pt1.y-(float)pt2.y)/((float)pt1.x-(float)pt2.x);
}
float getYintFSlope(floatPoint p1, float m)
{
	return (p1.y - (m*p1.x));
}
float getYintF(floatPoint p1, floatPoint p2)
{
	return (p1.y - (getSlopeF(p1, p2)*p1.x));
}
float getYint(Point pt1, Point pt2)
{
	return (float)((float)pt1.y - getSlope(pt1, pt2)*(float)pt1.x);
}
float distParLines(float b1, float b2, float m)
{
	return (abs(b2-b1)) / sqrt(m*m + 1);
}
floatPoint getIntersection(Point pt1, Point pt2, Point pt3, Point pt4)
{
	float slope1 = getSlope(pt1, pt2);
	float slope2 = getSlope(pt3, pt4);
	
	float yInt1  = getYint(pt1, pt2);
	float yInt2  = getYint(pt3, pt4);
	floatPoint intersection;
	intersection.x = (yInt2-yInt1)/(slope1-slope2);
	intersection.y = slope1*intersection.x + yInt1;

    return intersection;
}
bool isWhite(pixel* rgb)
{
	if(rgb->r==255 && rgb->g==255 && rgb->b==255)
	{
		return true;
	}else return false;
}

bool isLine(lab_pixel* grass, lab_pixel* pxl)
{
	if(delta_h(pxl, grass) < GRASS_HUE_TOLERANCE)
	{
		    if((pxl->L - grass->L) > GRASS_LIGHT_MIN && (pxl->L - grass->L) < GRASS_LIGHT_MAX)
			{
				return true;
			}
	}
	return false;
}

void Short2RGB(short colorID, pixel* pxl)
{
	pxl->b     = (colorID & 0xF) << 4;
	pxl->g     = ((colorID >> 4) & 0xF) << 4;
	pxl->r     = ((colorID >> 8) & 0xF) << 4;
}

short RGB2Short(pixel* rgb_pix)
{
	short colorID = ((rgb_pix->r / 16) << 8) + ((rgb_pix->g / 16) << 4) + (rgb_pix->b / 16);
    if(colorID > 4095)
	{
		fprintf(stderr, "ColorID: %d\n", colorID);
		exit(1);
	}
	return colorID;
}

void MyFilledCircle( Mat img, Point center,int r, int g, int b )
{
  int thickness = -1;
  int lineType = 8;

  circle( img,
      center,
      W/32,
      Scalar( r, g, b ),
      thickness,
      lineType );
}

void insertionSort(blob* b, blob* blobs)
{
	if(b->pixels < 50)
	{
		return;
	}
	for(int i=0; i<100; i++)
	{
		if(b->delta_e < blobs[i].delta_e) // insert here
		{
			for(int j=21; j>i; j--)
			{
				blobs[j].delta_e = blobs[j-1].delta_e;
				blobs[j].x       = blobs[j-1].x;
				blobs[j].y       = blobs[j-1].y;
			}
			blobs[i].delta_e = b->delta_e;
			blobs[i].x = b->x;
			blobs[i].y = b->y;
			break;
		}
	}
}

blob* scanline_fill(pixel* seed, lab_pixel* refPxl, Mat img, int x, int y, Stack* stack) 
{
	
     int w = img.cols;
	 
	 bool spanUp, spanDown;
	 pixel* origin = (pixel*) img.data;
	 pixel* Seed;
	 stack->emptyStack();
	 // BLOB //
	 blob* b = (blob*) malloc(sizeof(*b));
	 b->x = 0;
	 b->y = 0;
	 b->delta_e = 0;
	
	 int x_total = 0;
	 int y_total = 0;
	 float delta_e_total = 0;
	 int pixels = 1;

	 stack->push(x, y, w);
	 //scanline_fill is called with x and y
	 //which are invariantly going to be less than TOLERANCE
	 while(! stack->isEmpty())
	 {
		 int p = stack->pop();
		 int x1 = p % w;
		 int y1 = p / w;
		 lab_pixel lab_seed;
		 lab_pixel up;
		 lab_pixel down;
		 int count = 0;
		  if(y1 >= img.rows - 1){return b;}
		 Seed = origin + (y1 * w) + x1;
	     Rgb2Lab(&lab_seed.L, &lab_seed.a, &lab_seed.b, Seed->r/255.0, Seed->g/255.0, Seed->b/255.0);
		 
		 while(x1 < (w - 1) && (delta_e(&lab_seed, refPxl) < TOLERANCE))
		 {
			  x1++;
		      Seed = origin + (y1 * w) + x1;
		  	  Rgb2Lab(&lab_seed.L, &lab_seed.a, &lab_seed.b, Seed->r/255.0, Seed->g/255.0, Seed->b/255.0);  
    	 } 
		  				
		 x1--;
		 Seed = origin + (y1 * w) + x1;
		 spanUp = spanDown = false; // could be the issue
		 Rgb2Lab(&lab_seed.L, &lab_seed.a, &lab_seed.b, Seed->r/255.0, Seed->g/255.0, Seed->b/255.0); //L*a*b pixel is not floating points, rounded to ints
		  /* After changing the origin to img.data, this should be working! */	 
		 while(x1 >= 0 && (delta_e(&lab_seed, refPxl) < TOLERANCE)) 
		 {
		     Seed->r = 199;
			 Seed->g = 255;
			 Seed->b = 0;
			 //found a pixel, add it to blob
			 pixels++;
			 x_total += x1;
			 y_total += y1;
			 	
			 delta_e_total += delta_e(&lab_seed, refPxl); //Can optimize, (two calls to delta_e see while loop)
		      
			 if(y1 > 0) //the seed has a pixel above it
			 {
				 pixel* target = Seed - w;

				 Rgb2Lab(&up.L, &up.a, &up.b, target->r/255.0, target->g/255.0, target->b/255.0);
				if(!spanUp && y1 >= 0 && (delta_e(&up, refPxl) < TOLERANCE))
				{
					 stack->push(x1, y1-1, w);
					spanUp = true;
				}else if(spanUp && y1 >= 0 && !(delta_e(&up, refPxl) < TOLERANCE))
				{
					 spanUp = false;
				}
			 }
			 if(y1 < img.rows) //the seed has a pixel below it
			 {
				 pixel* target = Seed + w;

				 Rgb2Lab(&down.L, &down.a, &down.b, target->r/255.0, target->g/255.0, target->b/255.0);
				 
				 if(!spanDown && y1 < w - 1 && (delta_e(&down, refPxl) < TOLERANCE)) //may need to be <=
				 {
					 stack->push(x1, y1+1, w);
					 spanDown = true;
				 }else if(spanDown && y1 < w - 1 && !(delta_e(&down, refPxl) < TOLERANCE))
				 {
					 spanDown = false;
				 }
				
			 }else{break;}//can't travel passed img.rows
		     x1--; //return and back-track across scanline
	    	 Seed = origin + (y1 * w) + x1;

			 Rgb2Lab(&lab_seed.L, &lab_seed.a, &lab_seed.b, Seed->r/255.0, Seed->g/255.0, Seed->b/255.0);
	     }	 
	 }
	    b->x = x_total / pixels;
	    b->y = y_total / pixels;
		b->delta_e = delta_e_total / pixels;
		b->pixels  = pixels;
	    
	    return b;
}

float delta_e(lab_pixel* ref, lab_pixel* pix)
{
	return 
		sqrt(
		  (pow(ref->a - pix->a, (float)2.0)) +
		  (pow(ref->b - pix->b, (float)2.0)) +
		  (pow(ref->L - pix->L, (float)2.0))
		);
}


int getHue(lab_pixel* pxl)
{
	int num = (atan2(pxl->b, pxl->a) * 180/PI);
	return num;
}

float delta_h(lab_pixel* ref, lab_pixel* pix)
{
	 float c_ref = sqrt((ref->a)*(ref->a) + (ref->b)*(ref->b));
	 float c_pix = sqrt((pix->a)*(pix->a) + (pix->b)*(pix->b));	 
     return sqrt( ((ref->a)-(pix->a))*((ref->a)-(pix->a))
		        + ((pix->b)-(ref->b))*((pix->b)-(ref->b))
			    - ((c_pix-c_ref)*(c_pix-c_ref)));
}
void RGB2Lab(pixel* rgbPix, lab_pixel* labPix)
{
  unsigned char R = rgbPix->r;
  unsigned char G = rgbPix->g;
  unsigned char B = rgbPix->b;
  float a, b, L;
  float X, Y, Z, fX, fY, fZ;
  float oneThird = (float)1.0/3.0;
  X = 0.412453*R + 0.357580*G + 0.180423*B;
  Y = 0.212671*R + 0.715160*G + 0.072169*B;
  Z = 0.019334*R + 0.119193*G + 0.950227*B;

  X /= (255 * 0.950456);
  Y /=  255;
  Z /= (255 * 1.088754);

  if (Y > 0.008856)
    {
      fY = pow(Y, oneThird);
      L = (int)(116.0*fY - 16.0 + 0.5);
    }
  else
    {
      fY = 7.787*Y + 16.0/116.0;
      L = (float)(903.3*Y + 0.5);
    }

  if (X > 0.008856)
      fX = pow(X, oneThird); //(float, double)
  else
      fX = 7.787*X + 16.0/116.0;

  if (Z > 0.008856)
      fZ = pow(Z, oneThird);
  else
      fZ = 7.787*Z + 16.0/116.0;

  a = (float)(500.0*(fX - fY) + 0.5);
  b = (float)(200.0*(fY - fZ) + 0.5);

  labPix->a = a;
  labPix->b = b;
  labPix->L = L;
//printf("RGB=(%d,%d,%d) ==> Lab(%d,%d,%d)\n",R,G,B,*L,*a,*b);
}

int compar (const void* p1, const void* p2)
{
	if( (*(pointSlopeLine*)p1).x < (*(pointSlopeLine*)p2).x){return -1;}
	else if( (*(pointSlopeLine*)p1).x > (*(pointSlopeLine*)p2).x){return  1;}
	else{ return 0;}
}

int round(float n)
{
	if(abs(n - floor(n)) >= .5)
	{
		return floor(n) + 1;
	}else
		return floor(n);
}
//fuck microsoft

int findModeHue(int* hues)
{ //maybe print this graph out?
	int modeHue = -1;
	double maxHueCount = -1;
	for(int x=0;x<360;x++)
	{
		if(hues[x] > maxHueCount)
		{
			maxHueCount = hues[x];
			modeHue = x;
		}
	}
	return modeHue;
}

void MatRgb2Gradients(Mat src, pixel_gradient* gradientArr, int* hues, double* angleCount, double* magnitudes, double* magCount)
{
	pixel* origin = (pixel*) src.data;
	pixel* cur;
	pixel_gradient* cur_grad = gradientArr;
	
	int dYmat[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	int dXmat[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
	float L, C, H;

	double size = (src.rows - 4) * (src.cols - 4);
	fprintf(stderr, "Size: %d, %d\n", src.rows, src.cols);

	for(int i=2; i<src.rows - 2; i++)
	{
		for(int j = 2; j<src.cols - 2; j++)
		{
			pixel* center = origin + (src.cols * i) + j;
			Rgb2Lch(&L, &C, &H, center->r/255.0, center->g/255.0, center->b/255.0);
			
		    double Dy = applyConvolutionMat(dYmat, center);
			double Dx = applyConvolutionMat(dXmat, center);
			double magnitude =   sqrt((Dy * Dy) + (Dx * Dx));
			int gradientAngle =  (round(atan2(Dy, Dx) * 180/PI) + 180) % 180;

			cur_grad->angle = gradientAngle;
			cur_grad->hue   = H;
			cur_grad->magnitude_int  = floor(magnitude);
			cur_grad->magnitude_mant = round((magnitude - floor(magnitude)) * 255);

			angleCount[gradientAngle] += 1;
			magnitudes[gradientAngle] += (cur_grad->magnitude_int + (cur_grad->magnitude_mant / 255.0));
			magCount[round(magnitude)]++;
			hues[round(H)]++;

			cur_grad++;
		}
	}
}

void mapGradients(Mat img)
{
	Mat src = Mat(img.rows, img.cols, CV_8UC3, Scalar(0,0,0));
	
	GaussianBlur(img, src, Size(5, 5), 0, 0);
	img.release();

	int* hues = (int*) calloc(360, sizeof(*hues));
	pixel_gradient* gradientArr = (pixel_gradient*) calloc((src.rows * src.cols - 1), sizeof(*gradientArr));
	double* angleCount  = (double*) calloc(180, sizeof(*angleCount));
	double* magnitudes  = (double*) calloc(180, sizeof(*magnitudes));
	float* avgMagnitudes = (float*) calloc(180, sizeof(*avgMagnitudes));
	double* magnitudeCount = (double*) calloc(144, sizeof(*magnitudeCount));

	MatRgb2Gradients(src, gradientArr, hues, angleCount, magnitudes, magnitudeCount);

	int modeHue = findModeHue(hues);

	FILE* f = NULL;
	f = fopen("C:/Users/Seano/Documents/GridIron Analyst/gradients.csv", "w");
	if(f == NULL)
	{
		fprintf(stderr, "DEAD\n");
		waitKey();
		exit(0);
	}
	
	double maxAvgMag = -1;
	int AvgMagIndex = 0;
	double maxAngleCount = -1;
	int AngleCountIndex = 0;
	double maxMagnitude = -1;
	int MagnitudeIndex = 0;

	for(int theta=0; theta<180; theta++)
	{
		avgMagnitudes[theta] = magnitudes[theta] / angleCount[theta];

		if(magnitudes[theta] > maxMagnitude)
		{
			maxMagnitude = magnitudes[theta];
			MagnitudeIndex = theta;
		}
		if(avgMagnitudes[theta] > maxAvgMag)
		{
			maxAvgMag = avgMagnitudes[theta];
			AvgMagIndex = theta;
		}
		if(angleCount[theta] > maxAngleCount)
		{
			maxAngleCount = angleCount[theta];
			AngleCountIndex = theta;
		}
		
		fprintf(f, "%d,%f,%f,%f\n", theta, magnitudes[theta], angleCount[theta], avgMagnitudes[theta]);
	}

	fprintf(f, "%d, %d, %d, %d\n\n", AvgMagIndex, MagnitudeIndex, AngleCountIndex, modeHue);

	for(int h=0;h<142;h++)
	{
		fprintf(f, "%d,%f\n", h, magnitudeCount[h]);
	}

	//exit(1);

	pixel* cur               = (pixel*)src.data;
	pixel_gradient* gradient = gradientArr;
	for(int i=0; i<src.rows; i++)
	{
		for(int j=0; j<src.cols; j++)
		{
			
			if(i < 2 || i >= src.rows - 2 || j < 2 || j >= src.cols - 2) // turn the pixel black
			{
				//fprintf(stderr, "%d, %d\n", i, j);
				cur->r = 0;
				cur->g = 0;
				cur->b = 0;
			}else
			{ 
				float L, C, H;
				if(abs(gradient->hue - modeHue) > 10)
				{
					cur->r = 0;
					cur->g = 0;
					cur->b = 0;
				}else
				{
					int range = 10 ;
					float mag = gradient->magnitude_int + (gradient->magnitude_mant / 255.0);
				    if(abs(gradient->angle - MagnitudeIndex) < range && (mag > maxAvgMag))
					{
						int min = MagnitudeIndex - range;
						int space = 360 / (range*2); //number of different hues that can be mapped to
						float L, C, H;
						
						H = (gradient->angle - min) * space;
						C = 75;
						L = 75;
						float R, G, B;
						Lch2Rgb(&R, &G, &B, L, C, H);
						cur->r = round(R*255);
						cur->g = round(G*255);
						cur->b = round(B*255);
					}else
					{
						cur->r = 0;
						cur->g = 0;
						cur->b = 0;
					}
				}
				gradient++;
			}
			cur++;
		}
	}
	Mat n;
	cvtColor(src, n, CV_BGR2GRAY );
	vector<Vec2f> lines;
    HoughLines( n, lines, 1, CV_PI/180, 400);
	for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        Point pt1(cvRound(x0 + 1000*(-b)),
                  cvRound(y0 + 1000*(a)));
        Point pt2(cvRound(x0 - 1000*(-b)),
                  cvRound(y0 - 1000*(a)));
		
		int lineTheta = 0;
		lineTheta = (round(theta * 180 / CV_PI));
		fprintf(stderr, "%d..%d\n", (180 - lineTheta), ((MagnitudeIndex+90) % 180));
		if(abs(lineTheta - ((MagnitudeIndex+90) % 180)) < 4) //BROKEN, is theta already in degrees?
		{
			//MyFilledCircle(src, pt1, 255, 0, 0);
			//MyFilledCircle(src, pt2, 0, 255, 0);
            line(src, pt1, pt2, Scalar(0, 0,255), 3, 8 );
		}
	}



	imshow("Edge Detect", src);
	waitKey();
	exit(1);
}

double applyConvolutionMat(int* convolutionMat, pixel* pxl)
{
	double weightSum = 0;
	float L, C, H;

	for(int i=0;i<9;i++)
	{
		pixel* cur = pxl + (i - 4);

		Rgb2Lch(&L, &C, &H, cur->r/255.0, cur->g/255.0, cur->b/255.0);
		weightSum += convolutionMat[i] * L;
	}
	return weightSum;
}

