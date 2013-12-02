#include "stack.h"
#include <stdio.h>
#include <cstdlib>

using namespace std;

Stack::Stack(int Size)
{
	stack = (int*)calloc(Size, sizeof(int));
	size = Size;
	stackPtr = 0;
}


int Stack::pop()
{
	if(stackPtr == 0)
	{
		fprintf(stderr, "Stack Underflow, exitin...\n");
		exit(1);
	}
	stackPtr--;
	return stack[stackPtr];
}

void Stack::push(int x, int y, int w)
{
	if(stackPtr < size - 1)
	{
	   //fprintf(stderr, "stackPtr: %d\n", stackPtr);
	   stack[stackPtr] = y * w + x;
	   stackPtr++;
	   return;
	}else
	{
		fprintf(stderr, "Stack Overflow...Exiting...\n");
	    exit(1);
	}
}

bool Stack::isEmpty()
{
	if(stackPtr == 0)
	{
		return true;
	}
	else return false;
}

void Stack::emptyStack()
{
	while(stackPtr > 0)
	{
		pop();
	}
}
