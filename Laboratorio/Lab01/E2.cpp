#include "iostream"
#include <cstdlib>
#include <vector>
#include <math.h>
#include <chrono>

using namespace std;
const int N=1000;
std::vector<float> A(N), B(N), C(N);
void CrearVectores(vector<float>& A, vector<float>& B){
    srand(time(NULL));
    for(int i=0; i<N ; i++){
        A[i] = (rand() % 1000)/10.0;
        B[i] = (rand() % 1000)/10.0;

    }
}
float Tiempo(vector<float>& A, vector<float>& B , vector<float>& C){
    float time=0.0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

   //std::cout << "Tiempo de ejecucion: " << duration.count() << " segundos" << std::endl;
    time=duration.count();
    //return round(time * 1000000.0f) / 1000000.0f; // 6 decimales (microsegundos)
    return time;

}

int main(){
    CrearVectores(A,B);
    
    float time =Tiempo(A,B,C);
    

    for(int i=0; i<100 ; i++)
    {
        cout<<"A"<<i<<": "<<A[i]<<" + "<<"B"<<i<<": "<<B[i]<<" = "<<"C"<<i<<": "<<C[i]<<endl;
    }

    cout<<"El tiempo en segundos fue: "<<time<<endl;
    return 0;
}