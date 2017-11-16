#include<bits/stdc++.h>
using namespace std;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

/*The parameter dir indicates the sorting direction, ASCENDING
or DESCENDING; if (a[i] > a[j]) agrees with the direction,
then a[i] and a[j] are interchanged.*/
void compAndSwap(int a[], int i, int j, int dir)
{
	if (dir==(a[i]>a[j]))
		swap(a[i],a[j]);
}

/*It recursively sorts a bitonic sequence in ascending order,
if dir = 1, and in descending order otherwise (means dir=0).
The sequence to be sorted starts at index position low,
the parameter cnt is the number of elements to be sorted.*/
void bitonicMerge(int a[], int low, int cnt, int dir)
{
	if (cnt>1)
	{
		int k = cnt/2;
		for (int i=low; i<low+k; i++)
			compAndSwap(a, i, i+k, dir);
		bitonicMerge(a, low, k, dir);
		bitonicMerge(a, low+k, k, dir);
	}
}

/* This function first produces a bitonic sequence by recursively
	sorting its two halves in opposite sorting orders, and then
	calls bitonicMerge to make them in the same order */
void bitonicSort(int a[],int low, int cnt, int dir)
{
	if (cnt>1)
	{
		int k = cnt/2;

		// sort in ascending order since dir here is 1
		bitonicSort(a, low, k, 1);

		// sort in descending order since dir here is 0
		bitonicSort(a, low+k, k, 0);

		// Will merge wole sequence in ascending order
		// since dir=1.
		bitonicMerge(a,low, cnt, dir);
	}
}

/* Caller of bitonicSort for sorting the entire array of
length N in ASCENDING order */
void sort(int a[], int N, int up)
{
	bitonicSort(a,0, N, up);
}

// Driver code
int main()
{

	int n=0;
	while(n!=11)
	{
		n = rand()%30 + 1;
	}
	cout<<n<<endl;
	int size = pow(2,n);
	int a[size];

	int up = 1; // means sort in ascending order
	
	clock_t start, stop;
	start = clock();
	sort(a, size, up);
	stop = clock();

	/*printf("Sorted array: \n");
	for (int i=0; i<N; i++)
		printf("%d ", a[i]);
		*/
	print_elapsed(start, stop);
	return 0;
}
