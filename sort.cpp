// C++ program for implementation of Heap Sort
// The sort is in descending order.
// Reference:
// https://www.tutorialspoint.com/cplusplus/cpp_return_arrays_from_functions.htm 
#include <iostream>
#include <vector>
using namespace std;
 
 /* A utility function to print a vector */
void printVector(vector<int> vec)
{   
    for (unsigned int i = 0; i < vec.size(); ++i)
        cout << vec[i] << " ";
    cout << "\n";
} 

// To heapify a subtree rooted with node i which is
// an index in vec[]. n is size of heap
void heapify(vector<int> &vec, vector<int> &idx, int n, int i)
{
    int smallest = i;  // Initialize largest as root
    int l = 2 * i + 1;  // left = 2*i + 1
    int r = 2 * i + 2;  // right = 2*i + 2
 
    // If left child is larger than root
    if (l < n && vec[l] < vec[smallest])
        smallest = l;
 
    // If right child is larger than largest so far
    if (r < n && vec[r] < vec[smallest])
        smallest = r;
 
    // If largest is not root
    if (smallest != i)
    { 
        swap(vec[i], vec[smallest]);
        swap(idx[i], idx[smallest]);
 
        // Recursively heapify the affected sub-tree
        heapify(vec, idx, n, smallest);
    }
}
 
// main function to do heap sort
vector<int> heapSort(vector<int> &vec)
{   
    int n = int(vec.size());
    vector<int> idx(vec.size());
    for (int i = 0; i < n; i++)
        idx[i] = i;
    // Build heap (revecange vector)
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(vec, idx, n, i);
    // One by one extract an element from heap
    for (int i = n - 1; i >= 0; i--)
    {
        // Move current root to end
        swap(vec[0], vec[i]);
        swap(idx[0], idx[i]);
        heapify(vec, idx, i, 0);
    }
    return idx;
}
 

// Driver program
int main()
{
    vector<int> vec = {1, 3, 2, 4};
 
    vector<int> idx = heapSort(vec);
 
    cout << "Sorted vector is \n";
    printVector(vec);
    printVector(idx);
}