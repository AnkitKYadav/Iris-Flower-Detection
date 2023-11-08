#include <stdio.h>
#include <stdlib.h>

int main() {
    // Run the Flask application using Python
    system("python iris_classification.py &");
    
    // Open the web browser to the specified URL
    system("xdg-open http://localhost:5000/");
    
    return 0;
}
