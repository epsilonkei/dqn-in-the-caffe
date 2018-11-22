#ifndef COPY_HPP
#define COPY_HPP

#include <iostream>

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

#endif
