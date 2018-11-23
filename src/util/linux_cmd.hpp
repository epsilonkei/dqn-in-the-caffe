#ifndef LINUX_CMD_HPP
#define LINUX_CMD_HPP

#include <sys/stat.h>
#include <iostream>

void CreateDir(const char* directory) {
  struct stat sb;
  if (!stat(directory, &sb) == 0 || !S_ISDIR(sb.st_mode)) {
    const int dir_err = mkdir(directory, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
      if (dir_err == -1) {
         std::cerr <<"Error creating log directory!" << std::endl;
         exit(1);
      }
  }
}

void CopyFile( const char *from_file_name, const char *to_file_name )
{
  std::ifstream is( from_file_name, std::ios::in | std::ios::binary );
  std::ofstream os( to_file_name, std::ios::out | std::ios::binary );
  // Copy file
  os << is.rdbuf();
}

void CopyFile( const std::string& from_file_name, const std::string& to_file_name )
{
  CopyFile(from_file_name.c_str(), to_file_name.c_str());
}

#endif // LINUX_CMD_HPP
