#include <iostream>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h> // for errno



using namespace std;

#if !defined(filesystem_INCLUDED)
#define filesystem_INCLUDED
#define LOGNAME_FORMAT "%Y%m%d_%H%M%S"
#define LOGNAME_SIZE 20

class filesystem{
public:
    static char* timeStampDir(char*);
    static char* getTimeStamp(char*);

    static bool create_directory(const char*);

};
char* filesystem::getTimeStamp(char * out) {
    char* rc;
    char timestamp[LOGNAME_SIZE];
    time_t rawtime = time(0);
    tm *now = localtime(&rawtime);

    if(rawtime != -1) {
        strftime(timestamp, LOGNAME_SIZE, LOGNAME_FORMAT,now);
        rc = strcat(out,timestamp);
    }
    return(rc);
}
char* filesystem::timeStampDir(char* out_dir){
    char* rc;
    char timestamp[LOGNAME_SIZE];
    time_t rawtime = time(0);
    tm *now = localtime(&rawtime);

    if(rawtime != -1) {
        strftime(timestamp, LOGNAME_SIZE, LOGNAME_FORMAT,now);
        rc = strcat(out_dir,timestamp);
    }
    return(rc);
}

bool filesystem::create_directory(const char* dir){
    struct stat st;

    #if defined(_WIN32)
        _mkdir(dir);
    #elif defined(_WIN64) 
        mkdir(dir, 0700); 
    #else
        cout<<"linux/unix system."<<endl;
        if (stat(dir, &st) == 0) {
            printf("%s already exists\n", dir);
            return false;
        }
        if (mkdir(dir, S_IRWXU|S_IRWXG) != 0) {
            printf("Error creating directory: %s\n", strerror(errno));
            return false;
        }
        printf("%s successfully created\n", dir);
        return true;
    #endif


}


#endif