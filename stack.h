class Stack
{
public:
	Stack(int size);
	int pop();
	void push(int x, int y, int w);
	bool isEmpty();
	void emptyStack();

private:
	int* stack;
	int size;
	int stackPtr;
};