#include <iostream>
#include <locale>
using namespace std;

class Circle
{
public:
    float pi = 3.14;
    float radius;
    float degrees;
    float length(){
        float length = 2 * pi * radius;
        return length;
    } 
    float area(){
        float square = pi * radius * radius;
        return square;
    }
    float areaSector(){
        float sqSector = pi * radius * radius * (degrees / 360);
        return sqSector;
    }
};


int main(){
    Circle circle;
    cin >> circle.radius;
    cin >> circle.degrees;
    float length = circle.length();
    float square = circle.area();
    float sqSector = circle.areaSector();
    cout << length << endl;
    cout << square << endl;
    cout << sqSector << endl;
    return 1;
}